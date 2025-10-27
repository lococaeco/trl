import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 1. Reward Model 로드
# ============================================================
model_path = "/workspace/trl/workdir/reward_model/checkpoint-646"
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# ============================================================
# 2. 데이터셋 로드
# ============================================================
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="test")
print(f"✅ Loaded {len(dataset)} samples from trl-lib/ultrafeedback_binarized/test")

# ============================================================
# 3. Evaluation Loop
# ============================================================
records = []

for sample in tqdm(dataset, desc="Evaluating RM (with length tracking)"):
    try:
        chosen_msgs = sample["chosen"]
        rejected_msgs = sample["rejected"]

        # Chat 템플릿 적용
        text_chosen = tokenizer.apply_chat_template(chosen_msgs, tokenize=False)
        text_rejected = tokenizer.apply_chat_template(rejected_msgs, tokenize=False)

        # Tokenize
        inputs_chosen = tokenizer(text_chosen, return_tensors="pt", truncation=True, max_length=1024)
        inputs_rejected = tokenizer(text_rejected, return_tensors="pt", truncation=True, max_length=1024)

        chosen_len = inputs_chosen["input_ids"].shape[1]
        rejected_len = inputs_rejected["input_ids"].shape[1]

        inputs_chosen = {k: v.to(device) for k, v in inputs_chosen.items()}
        inputs_rejected = {k: v.to(device) for k, v in inputs_rejected.items()}

        with torch.no_grad():
            r_chosen = model(**inputs_chosen).logits.item()
            r_rejected = model(**inputs_rejected).logits.item()

        records.append({
            "chosen_len": chosen_len,
            "rejected_len": rejected_len,
            "r_chosen": r_chosen,
            "r_rejected": r_rejected,
        })

    except Exception as e:
        print(f"⚠️ Skipped sample due to error: {e}")
        continue

# ============================================================
# 4. 길이 구간별 평균 점수 계산
# ============================================================
bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

def compute_bin_means(lengths, rewards, bins):
    bin_means = []
    bin_centers = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (lengths >= lo) & (lengths < hi)
        if np.any(mask):
            mean_val = np.mean(rewards[mask])
        else:
            mean_val = np.nan
        bin_means.append(mean_val)
        bin_centers.append((lo + hi) / 2)
    return np.array(bin_centers), np.array(bin_means)

records_np = {k: np.array([r[k] for r in records]) for k in records[0].keys()}

chosen_centers, chosen_means = compute_bin_means(records_np["chosen_len"], records_np["r_chosen"], bins)
rejected_centers, rejected_means = compute_bin_means(records_np["rejected_len"], records_np["r_rejected"], bins)

# ============================================================
# 5. 시각화 및 저장
# ============================================================
os.makedirs("reward_eval_results", exist_ok=True)
save_path = "reward_eval_results/reward_length_bias.png"

plt.figure(figsize=(8, 5))
plt.plot(chosen_centers, chosen_means, "-o", label="Chosen Mean Reward", color="tab:blue")
plt.plot(rejected_centers, rejected_means, "-o", label="Rejected Mean Reward", color="tab:orange")
plt.xlabel("Sequence Length (tokens)")
plt.ylabel("Average Reward")
plt.title("Reward vs. Sequence Length (Length Bias Analysis)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✅ Length-bias plot saved to: {save_path}")

# ============================================================
# 6. 수치 결과 요약
# ============================================================
import pandas as pd
df = pd.DataFrame({
    "bin_center": chosen_centers,
    "chosen_mean_reward": chosen_means,
    "rejected_mean_reward": rejected_means,
})
csv_path = "reward_eval_results/reward_length_bias.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Saved per-length mean reward data to: {csv_path}")
