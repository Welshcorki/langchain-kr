import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings, OllamaEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

# 문장 데이터 준비
sentences = [
    "오늘 아침에는 날씨가 맑고 시원했다.",
    "어제는 비가 내려서 도로가 많이 미끄러웠다.",
    "주말에 가족과 함께 공원에 산책을 다녀왔다.",
    "점심으로 김치찌개와 공깃밥을 먹었다.",
    "오늘 저녁은 치킨과 맥주를 먹을 계획이다.",
    "나는 커피보다는 녹차를 더 좋아한다.",
    "최근에 인공지능에 관한 책을 읽고 있다.",
    "데이터베이스는 정보를 효율적으로 저장하고 검색할 수 있게 해준다.",
    "파이썬은 데이터 분석에 자주 사용되는 프로그래밍 언어다.",
    "운동을 꾸준히 하면 건강에 많은 도움이 된다.",
    "여행은 새로운 경험을 하고 시야를 넓히는 좋은 방법이다.",
    "나는 여름보다는 겨울이 더 좋다.",
    "스마트폰은 현대인들의 생활에 없어서는 안 될 필수품이 되었다.",
    "인터넷 쇼핑은 편리하지만 충동구매를 조심해야 한다.",
    "어제 본 영화는 생각보다 재미있었다.",
    "음악을 들으면 집중력이 향상되는 경우가 많다.",
    "고양이는 혼자 있는 것을 잘 견디는 반면, 개는 사람과 함께 있기를 좋아한다.",
    "아침에 가볍게 스트레칭을 하면 하루가 활기차다.",
    "독서는 사고력을 키우고 상상력을 자극한다.",
    "새로운 언어를 배우는 것은 뇌 건강에도 좋다."
]

# 임베딩 모델 로드
def get_embeddings(model_name):
    if model_name == "HuggingFace":
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elif model_name == "OpenAI":
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API 키가 .env 파일에 설정되지 않았습니다.")
        return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    elif model_name == "Upstage":
        if not UPSTAGE_API_KEY:
            raise ValueError("Upstage API 키가 .env 파일에 설정되지 않았습니다.")
        return UpstageEmbeddings(model="solar-embedding-1-large", api_key=UPSTAGE_API_KEY)
    elif model_name == "Ollama":
        # Ollama가 로컬에서 실행 중이어야 합니다.
        return OllamaEmbeddings(model="nomic-embed-text")
    else:
        raise ValueError("Invalid model name")

# VectorStore를 저장할 딕셔너리
vector_stores = {}

def get_vector_store(model_name):
    if model_name not in vector_stores:
        print(f"{model_name} VectorStore(Chroma)를 생성합니다.")
        embeddings_model = get_embeddings(model_name)
        # 각 모델별로 고유한 collection_name을 지정하여 차원 충돌 방지
        collection_name = f"collection_{model_name.lower().replace(' ', '_')}"
        vector_stores[model_name] = Chroma.from_texts(
            sentences, 
            embeddings_model, 
            collection_name=collection_name
        )
    return vector_stores[model_name]

def compare_similarity(base_sentence, model_name):
    """
    유사도 비교 함수 (실행 상태 표시 기능 추가)
    """
    yield "유사도 비교를 실행 중입니다..."
    try:
        vector_store = get_vector_store(model_name)
        results = vector_store.similarity_search_with_relevance_scores(base_sentence, k=len(sentences))

        output = f"**기준 문장:** {base_sentence}\n\n"
        output += f"**선택한 모델:** {model_name}\n\n"
        output += "---코사인 유사도 순위---\n"
        results.sort(key=lambda x: x[1], reverse=True)
        for rank, (doc, score) in enumerate(results):
            output += f"{rank + 1}. **유사도: {score:.4f}** - {doc.page_content}\n"
        
        output += "\n---\n"
        output += "**데이터 저장 위치**\n"
        output += "현재 실행 중인 애플리케이션의 메모리(In-Memory)에 임시 저장되어 있으며, 별도 파일로 저장되지 않습니다."

        yield output

    except Exception as e:
        yield f"오류 발생: {e}"

# Gradio 인터페이스 정의 (gr.Blocks 사용)
with gr.Blocks() as iface:
    gr.Markdown("## VectorStore(Chroma)를 이용한 문장 유사도 비교기")
    gr.Markdown("코사인 유사도를 사용하여 기준 문장과 가장 유사한 문장을 찾습니다.")

    with gr.Row():
        with gr.Column(scale=1):
            model_name_dd = gr.Dropdown(choices=["HuggingFace", "OpenAI", "Upstage", "Ollama"], label="임베딩 모델 선택", value="OpenAI")
            base_sentence_tb = gr.Textbox(lines=2, label="기준 문장을 입력하세요.")
            similarity_btn = gr.Button("유사도 비교")

        with gr.Column(scale=2):
            with gr.Group():
                similarity_output_md = gr.Markdown(label="유사도 비교 결과", value="유사도 비교 결과가 여기에 표시됩니다.")

    # 버튼 클릭 이벤트와 함수 연결
    similarity_btn.click(
        fn=compare_similarity,
        inputs=[base_sentence_tb, model_name_dd],
        outputs=similarity_output_md
    )

if __name__ == "__main__":
    iface.launch(share=True)
