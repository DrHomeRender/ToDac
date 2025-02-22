import os
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

#  1. 훈련된 KoGPT2 모델 로드
model_path = os.path.abspath("./trained_kogpt2")  # 절대 경로 설정
tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

#  2. 모델을 평가 모드로 변경
model.eval()

#  3. 뉴스, 연설문 등의 불필요한 단어 차단
bad_words = ["대통령", "기상청", "연설", "공식", "기자", "기념식", "보도", "중국", "일본", "스승의 날"]
bad_words_ids = [tokenizer.encode(word, add_special_tokens=False) for word in bad_words]

#  4. 질문을 입력하면 챗봇이 대답하도록 설정
def chat_with_bot(question):
    #  5. 질문 입력을 KoGPT2가 이해할 수 있도록 변환
    input_text = f"질문: {question}\n대답:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    #  6. attention_mask 추가하여 `generate()` 문제 해결
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()  # pad_token_id가 아닌 부분만 1로 설정

    #  7. KoGPT2가 답변 생성
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,  #  attention_mask 추가
            max_length=50,  #  너무 짧게 잘리는 문제 해결 (기존 30 → 50)
            num_return_sequences=1,  # 생성할 응답 개수
            top_p=0.85,  #  nucleus sampling 범위 조정 (기존 0.8 → 0.85)
            temperature=0.7,  #  조금 더 자연스러운 문장 생성 (기존 0.6 → 0.7)
            repetition_penalty=1.8,  # 반복되는 문장 억제 (기존 1.7 → 1.8)
            no_repeat_ngram_size=2,  # 2개 단어 이상의 n-그램 반복 방지
            do_sample=True,  # 샘플링 적용
            bad_words_ids=bad_words_ids,  #  불필요한 단어 차단
            eos_token_id=tokenizer.eos_token_id  #  문장 완성을 유도
        )

    #  8. KoGPT2의 대답을 디코딩하여 출력
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.replace(input_text, "").strip()  # 입력 질문 제거

    #  9. 특수 문자 필터링 및 응답 정제
    response = response.split("대답:")[0].strip()  # 불필요한 반복 제거
    response = response.replace("하지만 하지만", "하지만")  # 반복된 연결어 제거

    return response


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer = chat_with_bot(question)
        print(answer)
    else:
        print("질문을 입력하세요.")