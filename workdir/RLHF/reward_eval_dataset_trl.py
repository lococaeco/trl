import os
import torch
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
# 2. 데이터셋 로드 (trl-lib/ultrafeedback_binarized)
# ============================================================
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="test")
print(f"✅ Loaded {len(dataset)} samples from trl-lib/ultrafeedback_binarized/test")

# ============================================================
# 3. Evaluation Loop (no prompt column)
# ============================================================
correct = 0
total = 0
diffs = []

for sample in tqdm(dataset, desc="Evaluating RM"):
    try:
        # chosen / rejected 둘 다 이미 전체 대화 구조임
        chosen_msgs = sample["chosen"]
        rejected_msgs = sample["rejected"]

        # Chat 템플릿 적용
        text_chosen = tokenizer.apply_chat_template(chosen_msgs, tokenize=False)
        text_rejected = tokenizer.apply_chat_template(rejected_msgs, tokenize=False)

        # Tokenize
        inputs_chosen = tokenizer(text_chosen, return_tensors="pt", truncation=True, max_length=1024).to(device)
        inputs_rejected = tokenizer(text_rejected, return_tensors="pt", truncation=True, max_length=1024).to(device)

        # Reward 계산
        with torch.no_grad():
            r_chosen = model(**inputs_chosen).logits.item()
            r_rejected = model(**inputs_rejected).logits.item()

        diffs.append(r_chosen - r_rejected)
        if r_chosen > r_rejected:
            correct += 1
        total += 1
    except Exception as e:
        print(f"⚠️ Skipped sample due to error: {e}")
        continue

accuracy = correct / total if total > 0 else 0
print(f"\n✅ Reward Model Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")

# ============================================================
# 4. 시각화 (서버용 저장)
# ============================================================
os.makedirs("reward_eval_results", exist_ok=True)
save_path = "reward_eval_results/trl_lib_test_reward_diff_hist.png"

plt.figure(figsize=(8, 5))
plt.hist(diffs, bins=50, color="steelblue", edgecolor="black", alpha=0.75)
plt.title("Reward Difference Distribution (r_chosen - r_rejected)")
plt.xlabel("Reward Margin")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✅ Histogram saved to: {save_path}")
