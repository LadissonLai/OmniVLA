"""
modeling_langpark_align_only.py

LangParkAlignOnlyVLAForActionPrediction — ablation #3 model: KEEP the
InstructionAlignmentHead, REMOVE the MemoryEnhancementModule.

The full model produces 16 memory tokens with the MEM cross-attention module and
the alignment head attends to their hidden states. This ablation keeps the 16
"memory slots" in the sequence but fills them with ZERO placeholder embeddings
(no MEM module, no full-history projection). The alignment head still attends to
the LLM hidden states at those 16 slot positions, so this isolates the
contribution of the MEM module's structured computation from the alignment task.

Everything else (sequence layout, tail offsets) is identical to
LangParkVLAForActionPrediction, so the dec/act/mem hidden-state indexing in the
training/eval scripts is unchanged.

Sequence layout:
  BOS | sys_prompt | instruct | p_front | front | p_rear | rear | p_left | left |
  p_right | right | p_hist | hist | p_slots | MEM_ZERO(16) | dec(1) | act(32) | EOS(1)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .configuration_prismatic import OpenVLAConfig
from .constants import ACTION_DIM, FUTURE_ACTION_WAYPOINTS
from .modeling_prismatic import ExpVLAForActionPrediction
from .modeling_langpark import LangParkVLAOutput   # reuse output dataclass (import only)


class LangParkAlignOnlyVLAForActionPrediction(ExpVLAForActionPrediction):
    """
    VLA model with the instruction-alignment head but NO memory-enhancement module.

    The 16 memory slots are zero placeholders; the alignment head supervises the
    LLM hidden states read out at those positions.

    External modules (passed into forward, not stored in this class):
      past_traj_projector : nn.Module  – projects smart-sampled history → LLM tokens
    """

    config_class = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig):
        super().__init__(config)

    def forward(
        self,
        batch_inputs,
        past_traj_projector: nn.Module,
        num_mem_tokens: int = 16,
        use_cache: bool = False,
    ) -> LangParkVLAOutput:

        device      = self.language_model.device
        dtype       = self.language_model.dtype
        embed_layer = self.get_input_embeddings()
        llm_dim     = self.config.text_config.hidden_size

        history_list = batch_inputs['history_traj']
        B            = len(history_list)
        pad_id       = batch_inputs.get('pad_token_id', 0)

        # ── 1. Vision feature extraction ───────────────────────────────────

        def encode_img(img_tensor: torch.Tensor) -> torch.Tensor:
            patches = self.vision_backbone(img_tensor.to(dtype).to(device))
            return self.projector(patches)

        front_embeds = encode_img(batch_inputs['pixel_values_front'])
        rear_embeds  = encode_img(batch_inputs['pixel_values_rear'])
        left_embeds  = encode_img(batch_inputs['pixel_values_left'])
        right_embeds = encode_img(batch_inputs['pixel_values_right'])

        # ── 2. Smart-sampled history → LLM context tokens ──────────────────

        max_hist = max(t.shape[0] for t in history_list) if history_list else 0
        padded_hist, hist_masks = [], []
        for t in history_list:
            seq = t.shape[0]
            if seq < max_hist:
                pad = torch.zeros((max_hist - seq, 4), dtype=dtype, device=device)
                padded_hist.append(torch.cat([t.to(dtype).to(device), pad], dim=0))
                hist_masks.append(torch.cat([
                    torch.ones(seq,           dtype=torch.bool, device=device),
                    torch.zeros(max_hist - seq, dtype=torch.bool, device=device),
                ]))
            else:
                padded_hist.append(t.to(dtype).to(device))
                hist_masks.append(torch.ones(seq, dtype=torch.bool, device=device))

        if max_hist > 0:
            hist_embeds   = past_traj_projector(torch.stack(padded_hist, dim=0))  # [B, max_hist, D]
            hist_masks_t  = torch.stack(hist_masks, dim=0)                         # [B, max_hist]
        else:
            hist_embeds  = torch.empty((B, 0, llm_dim), dtype=dtype, device=device)
            hist_masks_t = torch.empty((B, 0), dtype=torch.bool, device=device)

        # ── 3. Output placeholder tokens ────────────────────────────────────

        dec_placeholder = torch.zeros((B, 1,                              llm_dim), dtype=dtype, device=device)
        act_placeholder = torch.zeros((B, FUTURE_ACTION_WAYPOINTS * ACTION_DIM, llm_dim), dtype=dtype, device=device)

        # ── 4. Sequence assembly ─────────────────────────────────────────────

        bos_id = torch.tensor([[self.config.text_config.bos_token_id]] * B, device=device)
        eos_id = torch.tensor([[self.config.text_config.eos_token_id]] * B, device=device)

        embeds_list: list = []
        masks_list:  list = []

        def add_component(emb: torch.Tensor, mask: Optional[torch.Tensor] = None):
            embeds_list.append(emb)
            if mask is None:
                mask = torch.ones(emb.shape[:2], dtype=torch.bool, device=device)
            masks_list.append(mask)

        def get_text_emb_and_mask(key: str):
            ids  = batch_inputs[key].to(device)
            emb  = embed_layer(ids)
            mask = (ids != pad_id)
            return emb, mask

        # BOS
        add_component(embed_layer(bos_id))

        # System prompt
        te, tm = get_text_emb_and_mask('sys_prompt_ids')
        add_component(te, tm)

        # Instruction — save emb & mask for the alignment head
        instruct_emb, instruct_mask = get_text_emb_and_mask('instruct_ids')
        add_component(instruct_emb, instruct_mask)

        # Four camera views
        te, tm = get_text_emb_and_mask('p_front_ids')
        add_component(te, tm)
        add_component(front_embeds)

        te, tm = get_text_emb_and_mask('p_rear_ids')
        add_component(te, tm)
        add_component(rear_embeds)

        te, tm = get_text_emb_and_mask('p_left_ids')
        add_component(te, tm)
        add_component(left_embeds)

        te, tm = get_text_emb_and_mask('p_right_ids')
        add_component(te, tm)
        add_component(right_embeds)

        # Smart-sampled history trajectory
        te, tm = get_text_emb_and_mask('p_hist_ids')
        add_component(te, tm)
        add_component(hist_embeds, hist_masks_t)

        # Parking slots text
        te, tm = get_text_emb_and_mask('p_slots_ids')
        add_component(te, tm)

        # ── 5. Zero-placeholder memory slots (no MEM module) ────────────────
        #
        # Same position/count as the full model's MEM tokens, but filled with
        # zeros. The alignment head attends to these positions' LLM hidden states.
        mem_tokens = torch.zeros((B, num_mem_tokens, llm_dim), dtype=dtype, device=device)
        add_component(mem_tokens)   # all-True mask (default)

        # Placeholders and EOS
        add_component(dec_placeholder)
        add_component(act_placeholder)
        add_component(embed_layer(eos_id))

        # ── 6. LLM forward ────────────────────────────────────────────────────

        inputs_embeds  = torch.cat(embeds_list, dim=1)   # [B, Seq, D]
        attention_mask = torch.cat(masks_list,  dim=1)   # [B, Seq]

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        return LangParkVLAOutput(
            logits        = outputs.logits,
            hidden_states = outputs.hidden_states,
            instruct_emb  = instruct_emb,
            instruct_mask = instruct_mask,
        )
