import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# 1. 모델 & 토크나이저 로드
# ------------------------------------------------------------
model_path = "/workspace/trl/workdir/reward_model/checkpoint-646"
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# ------------------------------------------------------------
# 2. 데이터셋 로드
# ------------------------------------------------------------
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
print(f"✅ Loaded {len(dataset)} samples from ultrafeedback_binarized/test_prefs")

# ------------------------------------------------------------
# 3. Helper 함수
# ------------------------------------------------------------
def merge_prompt_and_response(sample, which="chosen"):
    """
    Merge prompt (string or list[dict]) and completion (string, dict, or list[dict])
    into a single chat history for apply_chat_template().
    """

    # --- handle prompt ---
    if isinstance(sample["prompt"], str):
        prompt_messages = [{"role": "user", "content": sample["prompt"]}]
    elif isinstance(sample["prompt"], list):
        prompt_messages = sample["prompt"]
    else:
        raise TypeError(f"Unexpected prompt type: {type(sample['prompt'])}")

    # --- handle completion (chosen/rejected) ---
    completion = sample[which]
    if isinstance(completion, str):
        completion = [{"role": "assistant", "content": completion}]
    elif isinstance(completion, dict):
        completion = [completion]
    elif not isinstance(completion, list):
        raise TypeError(f"Unexpected completion type: {type(completion)}")

    # --- merge into one chat sequence ---
    return prompt_messages + completion


# ------------------------------------------------------------
# 4. Evaluation loop
# ------------------------------------------------------------
correct = 0
total = 0
diffs = []

for sample in tqdm(dataset, desc="Evaluating RM"):
    chosen_msgs = merge_prompt_and_response(sample, "chosen")
    rejected_msgs = merge_prompt_and_response(sample, "rejected")

    # Chat 템플릿 적용
    text_chosen = tokenizer.apply_chat_template(chosen_msgs, tokenize=False)
    text_rejected = tokenizer.apply_chat_template(rejected_msgs, tokenize=False)

    # Tokenize
    inputs_chosen = tokenizer(text_chosen, return_tensors="pt", truncation=True, max_length=1024).to(device)
    inputs_rejected = tokenizer(text_rejected, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        r_chosen = model(**inputs_chosen).logits.item()
        r_rejected = model(**inputs_rejected).logits.item()

    if r_chosen > r_rejected:
        correct += 1
    total += 1
    diffs.append(r_chosen - r_rejected)

accuracy = correct / total
print(f"\n✅ Reward Model Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")

# ------------------------------------------------------------
# 5. (Optional) 분포 시각화
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import os

# 결과 저장 디렉터리 생성
os.makedirs("reward_eval_results", exist_ok=True)

# 파일 경로 지정
save_path = "reward_eval_results/reward_diff_hist.png"

# 그래프 그리기
plt.figure(figsize=(8, 5))
plt.hist(diffs, bins=50, color='steelblue', edgecolor='black', alpha=0.75)
plt.title("Reward Difference Distribution (r_chosen - r_rejected)")
plt.xlabel("Reward Margin")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)

# 이미지 파일로 저장
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✅ Histogram saved to: {save_path}")