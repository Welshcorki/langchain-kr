import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

# 문장 데이터 준비 - 이 부분을 새로 제시된 문장들로 교체하면 됩니다.
sentences = [
    # 스포츠
    "손흥민 선수가 환상적인 드리블로 골을 성공시켰다.",
    "김연아 선수의 피겨스케이팅 기술은 예술 그 자체였다.",
    "농구 경기에서 짜릿한 버저비터가 터졌다.",
    "축구 경기를 보며 치킨을 먹는 것은 최고의 즐거움이다.",
    # 역사
    "이순신 장군은 임진왜란 당시 거북선을 이용하여 나라를 지켜냈다.",
    "로마 제국은 유럽 역사에 지대한 영향을 미쳤다.",
    "세종대왕은 백성을 위해 훈민정음을 창제했다.",
    "피라미드는 고대 이집트 문명의 위대한 건축물이다.",
    # 예술 (영어)
    "Vincent van Gogh's 'The Starry Night' is famous for its intense colors.",
    "Monet painted a series of water lilies to capture the changing light.",
    "Classical music has a relaxing effect on the mind.",
    "The movie 'Parasite' won the top prize at the Cannes Film Festival.",
    # 음식
    "이탈리아 피자는 얇고 바삭한 도우가 생명이다.",
    "초콜릿 케이크는 달콤한 맛으로 사람들을 행복하게 만든다.",
    "파스타는 소스에 따라 다양한 맛을 낼 수 있다.",
    "매운 떡볶이는 스트레스를 해소하는 데 최고다.",
    # 동물
    "강아지는 사람에게 가장 친근한 반려동물이다.",
    "고양이는 독립적이고 깔끔한 성격으로 잘 알려져 있다.",
    "코끼리는 거대한 몸집과 긴 코를 가지고 있다.",
    "펭귄은 남극에 사는 귀여운 새다."
]

# 임베딩 모델 로드
def get_embeddings(model_name):
    if model_name == "HuggingFace":
        # 'sentence-transformers/all-MiniLM-L6-v2'는 작고 성능 좋은 모델로 실습에 적합합니다.
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elif model_name == "OpenAI":
        # OpenAI 임베딩은 유료 API 키가 필요합니다.
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API 키가 .env 파일에 설정되지 않았습니다.")
        return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    elif model_name == "Upstage":
        # Upstage 임베딩은 유료 API 키가 필요합니다.
        if not UPSTAGE_API_KEY:
            raise ValueError("Upstage API 키가 .env 파일에 설정되지 않았습니다.")
        return UpstageEmbeddings(model="solar-embedding-1-large", api_key=UPSTAGE_API_KEY)
    else:
        raise ValueError("Invalid model name")

def compare_similarity(base_sentence, model_name):
    """
    기준 문장과 모든 문장의 유사도를 측정하여 결과를 반환합니다.
    """
    try:
        # 1. 선택한 임베딩 모델 로드
        embeddings_model = get_embeddings(model_name)
        
        # 2. 모든 문장을 벡터로 변환 (임베딩)
        sentence_embeddings = embeddings_model.embed_documents(sentences)
        base_embedding = embeddings_model.embed_documents([base_sentence])[0]
        
        # 3. 코사인 유사도 계산
        similarities = cosine_similarity([base_embedding], sentence_embeddings)[0]
        
        # 4. 결과 정렬
        results = []
        for i, (sentence, score) in enumerate(zip(sentences, similarities)):
            results.append({
                "sentence": sentence,
                "similarity": round(score, 4)
            })
        
        # 유사도 점수가 높은 순으로 정렬
        sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        
        # 5. 결과 포맷팅
        output = f"**기준 문장:** {base_sentence}\n\n"
        output += f"**선택한 모델:** {model_name}\n\n"
        output += "---유사도 순위---\n"
        for rank, result in enumerate(sorted_results):
            output += f"{rank + 1}. **유사도: {result['similarity']:.4f}** - {result['sentence']}\n"
        
        return output

    except Exception as e:
        return f"오류 발생: {e}"

# Gradio 인터페이스 정의
iface = gr.Interface(
    fn=compare_similarity,
    inputs=[
        gr.Textbox(lines=2, label="기준 문장을 입력하세요."),
        gr.Dropdown(choices=["HuggingFace", "OpenAI", "Upstage"], label="임베딩 모델 선택")
    ],
    outputs="markdown",
    title="문장 유사도 비교기",
    description="선택한 임베딩 모델을 사용하여 기준 문장과 다른 문장들의 의미적 유사도를 비교합니다."
)

if __name__ == "__main__":
    iface.launch(share=True)