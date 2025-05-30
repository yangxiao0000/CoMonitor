#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers.models.llama import LlamaForCausalLM
import os
DEVICE_ID= 0
if os.getenv('DEVICE_ID') and str.isdigit(os.getenv('DEVICE_ID')):
    DEVICE_ID= int(os.getenv('DEVICE_ID'))


def freeze_non_embeds_parameters(model: LlamaForCausalLM) -> None:
    """Freeze all parameters except embeddings."""
    for name, params in model.named_parameters():
        if "embed_tokens" not in name and "lm_head" not in name:
            params.requires_grad = False
        else:
            params.requires_grad = True


def unfreeze_parameters(model: LlamaForCausalLM) -> None:
    for name, params in model.named_parameters():
        params.requires_grad = False
