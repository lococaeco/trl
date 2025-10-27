import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import trl
print(f"TRL import path: {trl.__file__}")

from trl import RewardTrainer, RewardConfig
from datasets import load_dataset

# 1. 데이터셋 준비 (chosen, rejected 컬럼이 있어야 함)
# dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train_prefs")
# (실제로는 train/test로 나누는 것이 좋습니다)
# dataset = dataset.train_test_split(test_size=0.1)
# train_data = dataset["train"]
# eval_data = dataset["test"]


# 1. 원본 데이터셋 로드
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
print(f"Original dataset size: {len(dataset)}")

# 2. 1/20 크기로 줄이기
sample_size = len(dataset) // 3
small_dataset = dataset.shuffle(seed=42).select(range(sample_size))

print(f"Reduced dataset size: {len(small_dataset)}")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"


# RewardConfig 객체를 생성할 때 TrainingArguments의 인자를 바로 사용할 수 있습니다.
training_args = RewardConfig(
    do_train=True,
    output_dir="./reward_model",    
    num_train_epochs=1,                 
    per_device_train_batch_size=4,      
    learning_rate=5e-5,                 
    logging_steps=10,
    gradient_checkpointing=True,
    bf16=None,
    model_init_kwargs=None,
    chat_template_path=None,
    disable_dropout=True,
    dataset_num_proc=None,
    eos_token=None,
    pad_token=None,
    max_length=2048,
    pad_to_multiple_of=None,
    center_rewards_coefficient=None,
    activation_offloading=False,
    report_to="wandb",                  

)

# 4. PEFT 설정 (LoraConfig)
# peft_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.1,
#     bias="none",
#     task_type="SEQ_CLS",
#     modules_to_save=["score"], # RM 학습 시 권장
# )

trainer = RewardTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=small_dataset,
    args=training_args,
)

trainer.train() 