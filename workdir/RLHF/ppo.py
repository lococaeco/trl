import shutil
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import trl
print(f"TRL import path: {trl.__file__}")

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


'''
accelerate launch --config_file /workspace/trl/examples/accelerate_configs/deepspeed_zero2.yaml \
    ppo.py \
    --dataset_name trl-lib/ultrafeedback-prompt \
    --dataset_train_split train \
    --output_dir ppo \
    --num_ppo_epochs 2 \
    --num_mini_batches 1 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --sft_model_path Qwen/Qwen2.5-0.5B-Instruct \
    --reward_model_path /workspace/trl/workdir/reward_model/checkpoint-646 \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0 \
    --report_to wandb \
    --response_length 512 \
    --trust_remote_code True \
    --stop_token_id 151645
'''
# --total_episodes 1000 \
# --num_train_epochs 2.0

'''
accelerate launch --config_file /workspace/trl/examples/accelerate_configs/deepspeed_zero2.yaml \
    ppo.py \
    --dataset_name trl-lib/ultrafeedback-prompt \
    --dataset_train_split train \
    --output_dir ppo-lora \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --sft_model_path Qwen/Qwen2.5-0.5B-Instruct \
    --reward_model_path /workspace/trl/workdir/reward_model/checkpoint-646 \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0 \
    --report_to wandb \
    \
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj

'''

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )

    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )
    policy.generation_config.max_new_tokens = 1024        # 응답 길이 제한
    policy.generation_config.max_length = 4096           # 프롬프트 + 응답 전체 길이 제한
    # policy.generation_config.temperature = 0.7           # 샘플링 온도
    # policy.generation_config.top_p = 0.9                 # nucleus sampling
    # policy.generation_config.do_sample = True            # 탐색적 샘플링 활성화

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split
    )
    eval_samples = 10
    train_dataset_full = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))

    reduce_ratio = 1 / 20
    num_train = int(len(train_dataset_full) * reduce_ratio)

    print(f"train dataset num:{len(train_dataset_full)}")

    train_dataset = train_dataset_full.shuffle(seed=42).select(range(num_train))
    print(f"reduced train dataset num:{len(train_dataset)}")

    dataset_text_field = "prompt"


    # print(dataset.column_names)
    # print(dataset[0])

    def prepare_dataset(dataset, tokenizer):
        def tokenize(element):
            prompts = []
            for item in element["prompt"]:
                if isinstance(item, list):
                    texts = [p["content"] for p in item if "content" in p]
                    prompts.append(" ".join(texts))
                else:
                    prompts.append(item.get("content", ""))

            outputs = tokenizer(prompts, padding=False, truncation=True, max_length=1024)
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=2,
        )


    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    ################
    # Training
    ################
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()