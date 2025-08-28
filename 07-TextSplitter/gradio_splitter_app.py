import gradio as gr
from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

def split_document(file_obj, splitter_type, chunk_size, chunk_overlap):
    """
    사용자로부터 업로드된 파일과 분할 옵션을 받아 문서를 분할합니다.
    """
    if file_obj is None:
        return "파일을 업로드해 주세요.", ""

    file_path = file_obj.name
    file_extension = file_path.split('.')[-1].lower()
    
    # 1. 문서 로드 (DocumentLoader)
    try:
        if file_extension == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == 'txt':
            loader = TextLoader(file_path)
        # Markdown 또는 HTML 문서의 경우, 확장자를 이용해 분할기 선택
        # 여기서는 TextLoader로 로드 후, TextSplitter가 처리하도록 함
        elif file_extension in ['md', 'html']:
             loader = TextLoader(file_path)
        else:
            return "지원되지 않는 파일 형식입니다.", ""
            
        docs = loader.load()
        full_text = docs[0].page_content
        
    except Exception as e:
        return f"문서 로딩 중 오류가 발생했습니다: {e}", ""

    # 2. 문서 분할 (TextSplitter)
    if splitter_type == "CharacterTextSplitter":
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == "RecursiveCharacterTextSplitter":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # 다른 분할기는 여기에서 추가할 수 있습니다.
    else:
        return "지원되지 않는 분할기 유형입니다.", ""
        
    chunks = splitter.split_text(full_text)
    
    # 결과 요약 정보
    summary = f"**원본 문서 길이:** {len(full_text)}자\n"
    summary += f"**분할된 청크 개수:** {len(chunks)}개\n"
    summary += f"**평균 청크 길이:** {sum(len(c) for c in chunks) / len(chunks):.2f}자"
    
    # 청크 내용 출력 (줄바꿈으로 구분)
    chunk_output = "\n---\n".join(chunks)

    return summary, chunk_output

# Gradio 인터페이스 정의
with gr.Blocks(title="Text Splitter Demo") as demo:
    gr.Markdown("# 📄 Text Splitter 시각화 도구")
    gr.Markdown("문서를 업로드하고 Text Splitter의 동작을 비교해 보세요.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="문서 업로드 (PDF, TXT)")
            splitter_dropdown = gr.Dropdown(
                choices=["CharacterTextSplitter", "RecursiveCharacterTextSplitter"],
                value="RecursiveCharacterTextSplitter",
                label="Text Splitter 유형"
            )
            chunk_size_slider = gr.Slider(minimum=100, maximum=2000, value=500, label="Chunk Size")
            chunk_overlap_slider = gr.Slider(minimum=0, maximum=500, value=50, label="Chunk Overlap")
            split_button = gr.Button("문서 분할하기", variant="primary")
        
        with gr.Column(scale=2):
            summary_output = gr.Markdown(label="요약 정보")
            chunk_output = gr.Textbox(label="분할된 청크 결과", lines=20)
            
    # 이벤트 연결
    split_button.click(
        split_document,
        inputs=[file_input, splitter_dropdown, chunk_size_slider, chunk_overlap_slider],
        outputs=[summary_output, chunk_output]
    )

if __name__ == "__main__":
    demo.launch()