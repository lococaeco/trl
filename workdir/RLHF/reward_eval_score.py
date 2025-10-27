import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ============================================================
# 1. RewardTrainer로 학습시킨 모델 경로
# ============================================================
model_path = "/workspace/trl/workdir/reward_model/checkpoint-646"

# ------------------------------------------------------------
# 2. 모델과 토크나이저 로드
# ------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ============================================================
# 3. 평가할 프롬프트 + 응답 정의
# ============================================================
messages = [
{
"content": "You are given a paragraph (Passage), a question (Question) and two answer options (Option1 and Option2). Your task is to choose the correct answer to the question based on the passage. Answer of the question cannot be found directly from the passage, it is indirectly indicated in the passage. Do not answer with words other than the two options, and do not answer with option titles(e.g. \"option1\" or \"1\" are not acceptable answers.)\n\nExample input: Passage: Omelettes make a great breakfast , and I enjoy making them because they are very easy to cook . First , I get the eggs out and begin cracking them on the side of the bowl and dumping them into the bowl . I normally use two eggs whites and one whole egg to cut the calories . I add some salt and pepper to the mix for flavor . I pour the mixture into the hot frying pan . I make sure that I add some vegetable oil to the pan before I add the eggs though because I do n't want them to stick . As this occurs , I add some cheese or vegetable to the center of the egg mixture . Really , almost any type of cheese or vegetable makes a good omelette . I then flip one side of the omelette over on to itself to seal in the contents in the middle . It is then ready to serve ! Question: Where is the omelette being browned? Option1: In the pan Option2: On the street\nExample output: In the pan\nExample explanation: As mentioned in the passage that I pour the mixture into the hot frying pan; it is clear that omelette is being browned in pan only. Also, there is nothing related to street mention in the passage.\nQ: Passage: Last week I posted my first movie review . My friends and I went out to see the movie \" Sorry to bother you \" . We did n't like the movie very much , so afterwards I felt compelled to post a movie review . After doing some research about where I should post something like a movie review where others can read it , I settled on the idea of posting the movie review to a website called Rotten Tomatoes . After I 'd done that , I was given the opportunity write about why I decided to give the movie the score that I did . Afterwards , I spell checked my review and read it through once more . I had my friends read my review to see what they thought of it , and then I decided to publish my review on Rotten Tomatoes . Overall , it was fun to write my own movie review , and I 'll likely be doing more in the future . Question: What was the person asked to assign a score for? Option1: future Option2: movie\nA:",
"role": "user"
},
{
"content": "Of course! I'm here to help you with this task. Please provide me with the passage, question, and answer options, and I will choose the correct answer based on the passage.",
"role": "assistant"
}
]

# ------------------------------------------------------------
# 4. Chat 템플릿 적용 (모델 학습 시 사용된 포맷 복원)
# ------------------------------------------------------------
formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
print("=" * 80)
print("📘 [Formatted Text for Reward Model Input]")
print("=" * 80)
print(formatted_text)
print()

# ------------------------------------------------------------
# 5. 토크나이징 및 토큰 정보 출력
# ------------------------------------------------------------
inputs = tokenizer(
    formatted_text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=1024,
)

num_tokens = len(inputs["input_ids"][0])
print("=" * 80)
print(f"🧩 Total Tokens: {num_tokens}")
print("=" * 80)

# 보기 좋은 문자열 버전 (특수 토큰 정리)
decoded_text = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
)
print("🔤 [Decoded Tokens as String]")
print(decoded_text)
print()

# ------------------------------------------------------------
# 6. 모델 추론 (보상 점수 계산)
# ------------------------------------------------------------
with torch.no_grad():
    model.eval()
    outputs = model(**inputs)
    reward_score = outputs.logits.item()

# ------------------------------------------------------------
# 7. 결과 요약 출력
# ------------------------------------------------------------
print("=" * 80)
print(" [Reward Model Evaluation Result]")
print("=" * 80)
print(f"Prompt: {messages[0]['content']}")
print(f"Response: {messages[1]['content']}\n")
print(f"→ Calculated Reward Score: {reward_score:.4f}")
print("=" * 80)
