import torch
import torch.nn as nn


class MemoryEnhancementModule(nn.Module):
    """
    Two-step cross-attention memory enhancement module.

    Step 1: 16 learnable queries attend to instruction embeddings → understands "what to do"
    Step 2: Step-1 output attends to full odometry history embeddings → aligns "where we are"

    Output: fixed num_mem_tokens memory tokens [B, num_mem_tokens, llm_dim]
    """

    def __init__(self, llm_dim: int, num_mem_tokens: int = 16, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_mem_tokens = num_mem_tokens

        # Learnable queries + independent positional encodings (added to prevent representation collapse)
        self.learnable_queries = nn.Parameter(torch.randn(1, num_mem_tokens, llm_dim) * 0.02)
        self.query_pos_enc     = nn.Parameter(torch.randn(1, num_mem_tokens, llm_dim) * 0.02)

        # Cross-Attn 1: queries → instruction embeddings
        self.cross_attn_1 = nn.MultiheadAttention(llm_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(llm_dim)

        # Cross-Attn 2: step-1 output → full odometry history embeddings
        self.cross_attn_2 = nn.MultiheadAttention(llm_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_2 = nn.LayerNorm(llm_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Linear(llm_dim * 4, llm_dim),
        )
        self.norm_3 = nn.LayerNorm(llm_dim)

    def forward(
        self,
        instruct_emb: torch.Tensor,    # [B, L_inst, D]
        full_hist_emb: torch.Tensor,   # [B, T_full, D]
        instruct_mask: torch.Tensor,   # [B, L_inst]  bool, True = real token
        full_hist_mask: torch.Tensor,  # [B, T_full]  bool, True = real token
    ) -> torch.Tensor:
        B = instruct_emb.shape[0]
        dtype = instruct_emb.dtype

        # Learnable queries with positional encodings, cast to input dtype
        queries = (
            self.learnable_queries.to(dtype).expand(B, -1, -1)
            + self.query_pos_enc.to(dtype)
        )  # [B, num_mem_tokens, D]

        # Step 1: attend to instruction embeddings
        # key_padding_mask: True means "ignore this key position" (i.e., padding tokens)
        step1_out, _ = self.cross_attn_1(
            queries,
            instruct_emb,
            instruct_emb,
            key_padding_mask=~instruct_mask,
        )
        step1_out = self.norm_1(queries + step1_out)

        # Step 2: attend to full odometry history
        step2_out, _ = self.cross_attn_2(
            step1_out,
            full_hist_emb,
            full_hist_emb,
            key_padding_mask=~full_hist_mask,
        )
        step2_out = self.norm_2(step1_out + step2_out)

        # FFN with residual
        mem_tokens = self.norm_3(step2_out + self.ffn(step2_out))

        return mem_tokens  # [B, num_mem_tokens, D]


class InstructionAlignmentHead(nn.Module):
    """
    Cross-attention classification head for instruction alignment.

    Query:   instruction word embeddings (shared from LLM embed_layer, no detach)
    Key/Val: LLM output hidden states at memory token positions
    Output:  per-token 3-class logits (0=done / 1=active / 2=upcoming)
    """

    def __init__(self, llm_dim: int, num_heads: int = 8, num_classes: int = 3, dropout: float = 0.0):
        super().__init__()
        self.query_proj = nn.Linear(llm_dim, llm_dim)
        self.cross_attn = nn.MultiheadAttention(llm_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm       = nn.LayerNorm(llm_dim)
        self.classifier = nn.Linear(llm_dim, num_classes)

    def forward(
        self,
        instruct_emb: torch.Tensor,   # [B, L_inst, D]
        mem_hidden: torch.Tensor,     # [B, 16, D]
        instruct_mask: torch.Tensor,  # [B, L_inst] bool
    ) -> torch.Tensor:
        queries = self.query_proj(instruct_emb)  # [B, L_inst, D]

        # K/V = mem_hidden (all 16 tokens are real, no key_padding_mask needed)
        attn_out, _ = self.cross_attn(queries, mem_hidden, mem_hidden)

        # Zero out Q-side padding positions to prevent any gradient leakage.
        # (loss-side ignore_index=-100 already handles this, this is belt-and-suspenders.)
        attn_out = attn_out * instruct_mask.unsqueeze(-1).to(attn_out.dtype)

        attn_out = self.norm(queries + attn_out)
        logits = self.classifier(attn_out)  # [B, L_inst, 3]

        return logits
