import torch
from torch.distributions import Normal
from transformers import AutoTokenizer

from .watermark_interface import Watermark

class BDLMWatermark(Watermark):
    def __init__(
        self,
        gamma: float = 0.5,
        delta: float = 2.0,
        offset: int = 32,
        context_len: int = 32,
        topk: int = 40,
        tokenizer: AutoTokenizer = None,
        device: str = "cuda",
    ):
        self.gamma = gamma
        self.delta = delta
        self.offset = offset
        self.context_len = context_len
        self.topk = topk
        self.device = device
        self.vocab_size = len(tokenizer.get_vocab())
        self._initialize_fixed_table()

    def _initialize_fixed_table(self):
        rng = torch.Generator(device=self.device)
        rng.manual_seed(2971215073)  # fib47 is prime
        self.table_size = 1_000_003
        self.fixed_table = torch.randperm(1_000_003, device=self.device, generator=rng)

    def _hash_tensor(self, x: torch.LongTensor) -> torch.LongTensor:
        indices = x % self.table_size
        return self.fixed_table[indices] + 1 
        
    def _hash_tensor(self, x: torch.LongTensor) -> torch.LongTensor:
        indices = x % self.table_size
        return self.fixed_table[indices] + 1 

    def _conpute_min_hash(
        self, 
        full_seq: torch.LongTensor, # [B, L]
        candidates: torch.LongTensor, # [B, L, num_candidates]
        salt_key: int = 15485863,
        mask_id: int = 126336,
    ): 
        B, L = full_seq.shape
        # pad seq with zeros
        padded_seq = torch.cat(
            [torch.zeros((B, self.offset + self.context_len - 1), dtype=full_seq.dtype, device=full_seq.device), full_seq],
            dim=-1,
        )[:, :-self.offset] # [B, context_len - 1 + L]

        # sliding window to get contexts
        contexts = padded_seq.unfold(dimension=1, size=self.context_len, step=1) # [B, L, context_len]

        h_ctx = self._hash_tensor(contexts)  # [B, L, context_len]
        h_cand = self._hash_tensor(candidates)  # [B, L, num_candidates]

        h_ctx_expanded = h_ctx.unsqueeze(2) # [B, L, 1, context_len]
        h_cand_expanded = h_cand.unsqueeze(3) # [B, L, num_candidates, 1]

        # simple hash function: multiply and take min
        product = salt_key * h_ctx_expanded * h_cand_expanded # [B, L, num_candidates, context_len]
        min_values = product.min(dim=-1).values # [B, L, num_candidates]

        return min_values

    def _compute_green_mask(self, x: torch.LongTensor):
        # x: [B, L, num_candidates]
        x = torch.bitwise_xor(x, x >> 33)
        x = x * 0xff51afd7ed558ccd
        x = torch.bitwise_xor(x, x >> 33)
        x = x * 0xc4ceb9fe1a85ec53
        x = torch.bitwise_xor(x, x >> 33)

        max_int = 2**63 - 1
        threshold = int(self.gamma * max_int)
        green_mask = (x & 0x7FFFFFFFFFFFFFFF) < threshold  # [B, L, num_candidates]

        return green_mask
    
    def get_key_params(self):
        out = {
            "gamma": self.gamma,
            "delta": self.delta,
            "offset": self.offset, 
            "context_len": self.context_len,
            "topk": self.topk,
            "watermark_type": "BDLM",
        }
        return out

    def watermark_logits(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ):
        B, L = input_ids.shape
        top_probs, top_indices = logits.topk(self.topk, dim=-1) # [B, L, topk]
        min_hash_values = self._conpute_min_hash(
            full_seq=input_ids,
            candidates=top_indices
        )  # [B, L, num_candidates]

        green_mask = self._compute_green_mask(min_hash_values)  # [B, L, topk]
        
        delta_tensor = torch.zeros_like(logits, dtype=logits.dtype)  # [B, L, vocab_size]
        src_values = (green_mask * self.delta).to(dtype=logits.dtype) # [B, L, topk]
        # 只在 TopK 的位置加上 delta
        delta_tensor.scatter_add_(
            dim = -1, 
            index = top_indices, 
            src = src_values
        )
        # 计算 watermarked_logits
        watermarked_logits = logits + delta_tensor

        return watermarked_logits, watermarked_logits

    def detect(
        self,
        input_ids: torch.LongTensor,  # (seq_len)
        attention_mask: torch.FloatTensor = None,  # (seq_len)
    ) -> torch.FloatTensor:

        input_ids = input_ids.view(1, -1)
        attention_mask = (
            attention_mask.view(1, -1) if attention_mask is not None else None
        )

        input_ids_device = input_ids.device

        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).bool()
        attention_mask = attention_mask.to(self.device)

        zscores, token_color, ignored_token_mask = self._detect(
            input_ids, attention_mask
        )
        cdf = Normal(0, 1).cdf(zscores)
        p_values = 1 - cdf

        zscore = zscores.to(input_ids_device).item()  # Convert to scalar
        p_value = p_values.to(input_ids_device).item()

        out = {
            "total_tokens": int(ignored_token_mask[0].sum().detach().cpu().item()),
            "total_green": int((token_color[0] * ignored_token_mask[0]).sum().detach().cpu().item()),
            "z_score": zscore,
            "p_value": p_value,
            "token_color": token_color[0].detach().cpu().tolist(),
            "mask": ignored_token_mask[0].detach().cpu().tolist(),
        }

        return out

    def _detect(
        self,
        input_ids: torch.LongTensor,  #  [B, L]
        attention_mask: torch.FloatTensor = None,  # [B, L]
    ) -> torch.FloatTensor:

        B, L = input_ids.shape

        min_hash_values = self._conpute_min_hash(
            full_seq=input_ids,  # [B, L]
            candidates=input_ids.unsqueeze(-1),  # [B, L, 1]
        )
        token_color = self._compute_green_mask(min_hash_values).squeeze(-1)  # [B, L]
        ignored_tokens_mask = torch.ones_like(input_ids, dtype=torch.bool)  # [B, L]
        T = ignored_tokens_mask.sum(dim=1)  # (batch_size,)

        zscore = token_color.sum(dim=1)  # [B,]

        zscore = (zscore - self.gamma * T) / torch.sqrt(
            self.gamma * T * (1 - self.gamma)
        )

        return zscore, token_color, ignored_tokens_mask