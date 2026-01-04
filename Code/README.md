# MERA: Meta-Cognitive Reasoning Architecture

MERA is a novel reasoning optimization framework that equips Large Reasoning Models (LRMs) with meta-cognitive self-regulation capabilities. It significantly reduces redundant reasoning and generation overhead without sacrificing accuracy. MERA explicitly decouples the reasoning process and the control mechanism, allowing fine-grained optimization of model behavior.

## 🔍 Key Features

- **Reasoning–Control Separation**: Introduces an independent control mechanism while preserving the interpretability of the original reasoning path.
- **Takeover-Based Data Construction**: Utilizes auxiliary LLMs to generate high-quality control labels, covering diverse decision points.
- **Control Segment Policy Optimization (CSPO)**: Combines Group Relative Policy Optimization (GRPO) with a control masking mechanism for localized learnability and globally stable control policy training.
- **Significant Redundancy Reduction**: Particularly effective for long-chain reasoning tasks such as AIME and AMC, improving both efficiency and accuracy.

------

## 🚀 Getting Started

### 1. Fine-Tune DeepSeek-1.5B with QLoRA

```
accelerate launch qlora.py \
--model_name_or_path deepseek-ai/deepseek-1.5B-base \
--dataset_path ./data/deepscaler_sft.json \
--output_dir ./checkpoints/mera_deepseek1.5B \
--do_train True \
--learning_rate 1e-6 \
--source_max_len 4096 \
--target_max_len 4096 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--lora_modules all \
--lora_r 64 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--bf16 True \
--bits 4 \
--double_quant True \
--quant_type nf4 \
--gradient_checkpointing True \
--save_strategy steps \
--report_to wandb
```

------

### 2. Run MERA Alternating Control Module with Inference Configuration

```
target_model_name: deepseek-ai/deepseek-14B
alter_model_name: meta-llama/Llama-3-70B-Instruct
target_model_gpu: 2
alter_model_gpu: 1
max_target_tokens: 64
TRIGGER_TOKENS: ["<control>"]
TARGET_VALIDATION_KEYWORDS:
    [ "verify",
      "think again",
      "recap",
      "check",
      "wait",
      "alternatively",
      "hold on",
      "another",
      "yeah",
      "yes",
      "final answer",
      "confident"]
```

Use other YAML configuration as input to the main control script ( main_cspo.py) .

------

## ⚙️ Notes

- Recommended environment: Python 3.12, CUDA 12.1, PyTorch 2.3.0.
- Hardware: Experiments are best conducted on a compute cluster with NVIDIA A100 GPUs (80GB VRAM).
