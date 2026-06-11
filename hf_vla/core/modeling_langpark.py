"""
modeling_langpark.py

LangParkVLAForActionPrediction — extends ExpVLAForActionPrediction with the
Instruction-Aligned Memory (IAM) module.

Key difference from the base class forward:
  1. Projects *all* historical odometry frames via `full_hist_projector` for the MEM module
     (separate from the smart-sampled `hist_embeds` that go into the LLM context).
  2. Computes 16 memory tokens via `mem_module` and inserts them AFTER `p_slots` text,
     just before dec / act / EOS placeholders.
  3. Returns `LangParkVLAOutput`, which additionally carries `instruct_emb` and
     `instruct_mask` so the training script can compute the instruction alignment loss.

Sequence layout:
  BOS | sys_prompt | instruct | p_front | front_embeds | p_rear | rear_embeds |
  p_left | left_embeds | p_right | right_embeds | p_hist | hist_embeds |
  p_slots | MEM(16) | dec(1) | act(32) | EOS(1)

Tail offsets (unchanged from ExpVLAForActionPrediction):
  decision_hidden  = last_hidden[:, -(N_act+3), :]              # MEM[15] position (shift)
  actions_hidden   = last_hidden[:, -(N_act+2):-2, :]           # dec → act[30] (shift)
  mem_hidden       = last_hidden[:, -(N_act+2+16):-(N_act+2), :] # MEM[0..15]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .configuration_prismatic import OpenVLAConfig
from .constants import ACTION_DIM, FUTURE_ACTION_WAYPOINTS
from .modeling_prismatic import ExpVLAForActionPrediction


# ── Output dataclass ───────────────────────────────────────────────────────────

@dataclass
class LangParkVLAOutput:
    logits:        torch.Tensor   # LLM vocab logits  [B, Seq, vocab_size]
    hidden_states: tuple          # all LLM hidden states (num_layers + 1,)
    instruct_emb:  torch.Tensor   # instruction word embeddings  [B, L_inst, D]
    instruct_mask: torch.Tensor   # bool mask for instruct_emb  [B, L_inst]


# ── Model ─────────────────────────────────────────────────────────────────────

class LangParkVLAForActionPrediction(ExpVLAForActionPrediction):
    """
    VLA model with Instruction-Aligned Memory (IAM).

    External modules (passed into forward, not stored in this class):
      past_traj_projector : nn.Module  – projects smart-sampled history → LLM tokens
      full_hist_projector : nn.Module  – projects ALL historical odometry → MEM input
      mem_module          : nn.Module  – MemoryEnhancementModule → 16 memory tokens
    """

    config_class = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig):
        super().__init__(config)

    def forward(
        self,
        batch_inputs,
        past_traj_projector: nn.Module,
        full_hist_projector: nn.Module,
        mem_module: nn.Module,
        use_cache: bool = False,
    ) -> LangParkVLAOutput:

        device     = self.language_model.device
        dtype      = self.language_model.dtype
        embed_layer = self.get_input_embeddings()
        llm_dim    = self.config.text_config.hidden_size

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

        # ── 3. Full history → MEM module input ─────────────────────────────

        full_hist_list = batch_inputs['full_history_traj']
        max_full = max(t.shape[0] for t in full_hist_list) if full_hist_list else 0
        padded_full, full_masks = [], []
        for t in full_hist_list:
            seq = t.shape[0]
            if seq < max_full:
                pad = torch.zeros((max_full - seq, 4), dtype=dtype, device=device)
                padded_full.append(torch.cat([t.to(dtype).to(device), pad], dim=0))
                full_masks.append(torch.cat([
                    torch.ones(seq,            dtype=torch.bool, device=device),
                    torch.zeros(max_full - seq, dtype=torch.bool, device=device),
                ]))
            else:
                padded_full.append(t.to(dtype).to(device))
                full_masks.append(torch.ones(seq, dtype=torch.bool, device=device))

        if max_full > 0:
            full_hist_emb = full_hist_projector(torch.stack(padded_full, dim=0))  # [B, max_full, D]
            full_masks_t  = torch.stack(full_masks, dim=0)                         # [B, max_full]
        else:
            full_hist_emb = torch.empty((B, 0, llm_dim), dtype=dtype, device=device)
            full_masks_t  = torch.empty((B, 0), dtype=torch.bool, device=device)

        # ── 4. Output placeholder tokens ────────────────────────────────────

        dec_placeholder = torch.zeros((B, 1,                              llm_dim), dtype=dtype, device=device)
        act_placeholder = torch.zeros((B, FUTURE_ACTION_WAYPOINTS * ACTION_DIM, llm_dim), dtype=dtype, device=device)

        # ── 5. Sequence assembly ─────────────────────────────────────────────

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

        # Instruction — save emb & mask for IAM
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

        # ── 6. Memory tokens (after p_slots, before dec/act/eos) ────────────
        #
        # MEM module takes instruction word embeddings and full odometry history,
        # returns 16 fixed memory tokens that encode "what to do" × "where we are".
        mem_tokens = mem_module(instruct_emb, full_hist_emb, instruct_mask, full_masks_t)
        add_component(mem_tokens)   # all-True mask (default, all 16 tokens are real)

        # Placeholders and EOS
        add_component(dec_placeholder)
        add_component(act_placeholder)
        add_component(embed_layer(eos_id))

        # ── 7. LLM forward ────────────────────────────────────────────────────

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
