# -*- coding: utf-8 -*-
"""
Continual Pre-training/Supervised fine-tuning of Colossal-LLaMA-2 developed by Colossal-AI Team
"""

import argparse
import json
import os
import resource
from contextlib import nullcontext
import torch_npu

import torch
from colossal_llama.dataset.dummy_dataset import RandomDataset
from colossal_llama.dataset.loader import (
    DataCollatorForSupervisedDataset,
    StatefulDistributedSampler,
    load_tokenized_dataset,
)
from colossal_llama.utils.ckpt_io import load_checkpoint, save_checkpoint
from colossal_llama.utils.froze import freeze_non_embeds_parameters
from colossal_llama.utils.neftune_patch import activate_neftune, deactivate_neftune
from colossal_llama.utils.utils import all_reduce_mean, format_numel_str, get_model_numel
from peft import LoraConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device

from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download

import torch.npu
import os


ASCEND_DEVICE_ID= 0
if os.getenv('ASCEND_DEVICE_ID') and str.isdigit(os.getenv('ASCEND_DEVICE_ID')):
    ASCEND_DEVICE_ID= int(os.getenv('ASCEND_DEVICE_ID'))
if torch.npu.current_device() != ASCEND_DEVICE_ID:
    torch.npu.set_device(f'npu:{ASCEND_DEVICE_ID}')
# print(ASCEND_DEVICE_ID)
RANK_SIZE = int(os.getenv('RANK_SIZE'))
RANK_ID = int(os.getenv('RANK_ID'))
# torch.distributed.init_process_group('hccl', rank=RANK_ID, world_size=RANK_SIZE)

def train(args) -> None:
    # print(111111111111111111111)
    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch()
    accelerator = get_accelerator()
    coordinator = DistCoordinator()


    # ==============================
    # Initialize Tensorboard and Save Config
    # ==============================
    if coordinator.is_master():
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)

        with open(args.config_file, "w") as f:
            json.dump(args.__dict__, f, indent=4)

    # ==============================
    # Initialize Booster
    # ==============================
    if args.plugin == "ddp":
        plugin = TorchDDPPlugin(find_unused_parameters=True if args.use_grad_checkpoint is False else False)
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(
            precision=args.mixed_precision,
            initial_scale=2**16,
            max_norm=args.grad_clip,
            enable_gradient_accumulation=(args.accumulation_steps > 1),
            enable_fused_normalization=get_accelerator().is_available(),
            enable_flash_attention=args.use_flash_attn,
        )
    elif args.plugin == "gemini_auto":
        plugin = GeminiPlugin(
            precision=args.mixed_precision,
            placement_policy="auto",
            initial_scale=2**16,
            max_norm=args.grad_clip,
            enable_gradient_accumulation=(args.accumulation_steps > 1),
            enable_fused_normalization=get_accelerator().is_available(),
            enable_flash_attention=args.use_flash_attn,
        )
    elif args.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=args.mixed_precision,
            initial_scale=2**16,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "zero2_cpu":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=args.mixed_precision,
            initial_scale=2**16,
            cpu_offload=True,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "3d":
        plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=args.pp,
            sp_size=args.sp,
            sequence_parallelism_mode=args.sp_mode,
            zero_stage=args.zero_stage,
            enable_flash_attention=args.use_flash_attn,
            enable_fused_normalization=get_accelerator().is_available(),
            enable_sequence_parallelism=args.enable_sequence_parallelism,
            cpu_offload=True if args.zero_stage >= 1 and args.zero_cpu_offload else False,
            max_norm=args.grad_clip,
            precision=args.mixed_precision,
            microbatch_size=args.microbatch_size,
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    booster = Booster(plugin=plugin)

    # ======================================================
    # Initialize Tokenizer, Dataset, Collator and Dataloader
    # ======================================================
    # model_dir = snapshot_download('colossalai/Colossal-LLaMA-2-7b-base', revision='v1.0.1')
    model_dir='/root/.cache/modelscope/hub/colossalai/Colossal-LLaMA-2-7b-base'
    # model_dir = snapshot_download('colossalai/Colossal-LLaMA-2-7b-base', revision='v1.0.1')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()
    generation_kwargs = {"max_new_tokens": 256, 
                        "top_p": 0.95, 
                        "temperature": 0.3
                        }

    input = '明月松间照，\n\n->\n\n'
    inputs = tokenizer(input, return_token_type_ids=False, return_tensors='pt')
    # inputs = inputs.to('cuda:0')
    inputs = inputs.to('npu:0')
    output = model.generate(**inputs, **generation_kwargs)
    print(tokenizer.decode(output.cpu()[0], skip_special_tokens=True)[len(input):])


    # print(222222222222222)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Basic training information.
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Address of the pre-trained model",
    )
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Load checkpoint for continuous training.")
    parser.add_argument("--dataset", nargs="+", default=[])
    parser.add_argument(
        "--plugin",
        type=str,
        default="gemini",
        choices=["gemini", "gemini_auto", "zero2", "zero2_cpu", "3d", "ddp"],
        help="Choose which plugin to use",
    )
    parser.add_argument("--save_interval", type=int, default=1000, help="Save interval")
    parser.add_argument("--save_dir", type=str, default="checkpoint_dir", help="Checkpoint directory")
    parser.add_argument("--tensorboard_dir", type=str, default="logs_dir", help="Tensorboard directory")
    parser.add_argument("--config_file", type=str, default="config_file", help="Config file")
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of accumulation steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Global Batch size of each process")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=8192, help="Model max length")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Mixed precision",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")
    parser.add_argument(
        "--use_grad_checkpoint",
        action="store_true",
        default=False,
        help="Use gradient checkpointing",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        default=False,
        help="Use flash-attention",
    )
    parser.add_argument(
        "--use_neft",
        action="store_true",
        default=False,
        help="Use NEFTune",
    )
    parser.add_argument(
        "--freeze_non_embeds_params",
        action="store_true",
        default=False,
        help="Freeze non embeddings parameters",
    )
    parser.add_argument("--pad_token", choices=["eos", "unk"], default="eos")
    parser.add_argument("--padding_mode", choices=["max_length", "longest"], default="max_length")
    parser.add_argument(
        "--skip_save_each_epoch",
        action="store_true",
        default=False,
        help="Skip saving the model checkpoint after each epoch is completed.",
    )

    # Additional arguments for 3d plugin.
    parser.add_argument("--tp", type=int, default=1, help="TP size, used for 3d plugin.")
    parser.add_argument("--pp", type=int, default=1, help="PP size, used for 3d plugin.")
    parser.add_argument("--sp", type=int, default=1, help="SP size, used for 3d plugin.")
    parser.add_argument("--zero_stage", type=int, default=0, help="Zero stage, used for 3d plugin.", choices=[0, 1, 2])
    parser.add_argument(
        "--sp_mode",
        type=str,
        default="split_gather",
        choices=["split_gather", "ring", "all_to_all"],
        help="SP mode, used for 3d plugin.",
    )
    parser.add_argument(
        "--enable_sequence_parallelism",
        default=False,
        action="store_true",
        help="Whether to enable SP, used for 3d plugin.",
    )
    parser.add_argument(
        "--zero_cpu_offload", default=False, action="store_true", help="Whether to use offloading, used for 3d plugin."
    )
    parser.add_argument(
        "--microbatch_size", type=int, default=1, help="Batch size for each process in PP, used for 3d plugin."
    )
    parser.add_argument("--lora_rank", type=int, default=0, help="lora rank when using lora to train.")

    # Additional arguments for benchmark.
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples for benchmarking.")
    parser.add_argument(
        "--benchmark", action="store_true", default=False, help="Benchmark performance using random dataset."
    )
    args = parser.parse_args()
    train(args)
