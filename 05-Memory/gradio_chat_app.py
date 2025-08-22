import gradio as gr
from operator import itemgetter
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Runnable
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class MyConversationChain(Runnable):
    def __init__(self, llm, prompt, memory, input_key="input"):
        self.prompt = prompt
        self.memory = memory
        self.input_key = input_key
        
        self.chain = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(self.memory.load_memory_variables)
                | itemgetter(memory.memory_key)
            )
            | prompt
            | llm
            | StrOutputParser()
        )
    
    def invoke(self, query, configs=None, **kwargs):
        answer = self.chain.invoke({self.input_key: query})
        self.memory.save_context(inputs={"human": query}, outputs={"ai": answer})
        return answer
    
    def clear_memory(self): # 메모리 초기화
        self.memory.clear()


class GradioChatApp:    # gradio 챗앱이 동작되는 방식
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.conversation_chain = None
        self.initialize_chain()
    
    def initialize_chain(self):
        # ChatOpenAI 모델 초기화
        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        
        # 대화형 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful and friendly AI assistant. Respond in Korean unless otherwise requested."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # 대화 버퍼 메모리 생성
        # memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        
        # ConversationSummaryMemory 사용
        memory = ConversationSummaryMemory(
            llm=llm, # 요약을 위해 LLM 모델 전달
            return_messages=True,
            memory_key="chat_history",
        )

        # 대화 체인 생성
        self.conversation_chain = MyConversationChain(llm, prompt, memory)
    
    def respond(self, message, history):
        """Gradio 채팅 인터페이스를 위한 응답 함수"""
        try:
            # LangChain 대화 체인을 통해 응답 생성
            response = self.conversation_chain.invoke(message)
            return response
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"
    
    def clear_conversation(self):
        """대화 기록 초기화"""
        if self.conversation_chain:
            self.conversation_chain.clear_memory()
        return None, None
    
    def update_model_settings(self, model_name, temperature):
        """모델 설정 업데이트"""
        self.model_name = model_name
        self.temperature = temperature
        self.initialize_chain()
        return "모델 설정이 업데이트되었습니다."


def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    app = GradioChatApp()
    
    with gr.Blocks(title="LangChain Gradio Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🤖 LangChain Gradio Chat
            LangChain과 OpenAI를 활용한 대화형 AI 챗봇입니다.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="대화",
                    height=500,
                    bubble_full_width=False,
                    avatar_images=(None, "🤖")
                )
                msg = gr.Textbox(
                    label="메시지 입력",
                    placeholder="메시지를 입력하세요... (Enter 키로 전송)",
                    lines=2
                )
                with gr.Row():
                    submit = gr.Button("전송", variant="primary")
                    clear = gr.Button("대화 초기화")
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 설정")
                model_dropdown = gr.Dropdown(
                    choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                    value="gpt-4o-mini",
                    label="모델 선택"
                )
                temperature_slider = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=0.7,
                    step=0.1,
                    label="Temperature (창의성 수준)"
                )
                update_btn = gr.Button("설정 적용", variant="secondary")
                settings_output = gr.Textbox(label="설정 상태", interactive=False)
                
                gr.Markdown(
                    """
                    ### 📝 사용 방법
                    1. 메시지를 입력하고 Enter 또는 전송 버튼을 클릭
                    2. AI가 대화 맥락을 기억하며 응답
                    3. 새로운 대화를 시작하려면 '대화 초기화' 클릭
                    4. 모델이나 Temperature 변경 시 '설정 적용' 클릭
                    """
                )
        
        # 이벤트 핸들러 연결
        def user_submit(user_message, history):
            if not user_message:
                return "", history
            history = history or []
            return "", history + [[user_message, None]]
        
        def bot_respond(history):
            if history and history[-1][1] is None:
                user_message = history[-1][0]
                bot_message = app.respond(user_message, history[:-1])
                history[-1][1] = bot_message
            return history
        
        # 메시지 전송 이벤트
        msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(
            bot_respond, chatbot, chatbot
        )
        submit.click(user_submit, [msg, chatbot], [msg, chatbot]).then(
            bot_respond, chatbot, chatbot
        )
        
        # 대화 초기화
        clear.click(
            lambda: (app.clear_conversation(), "", []),
            outputs=[chatbot, msg, chatbot]
        )
        
        # 설정 업데이트
        update_btn.click(
            app.update_model_settings,
            inputs=[model_dropdown, temperature_slider],
            outputs=[settings_output]
        )
        
        # 예제 입력
        gr.Examples(
            examples=[
                "안녕하세요! 자기소개를 해주세요.",
                "파이썬으로 피보나치 수열을 구현하는 방법을 알려주세요.",
                "오늘 날씨가 어떤가요?",
                "인공지능의 미래에 대해 어떻게 생각하시나요?",
                "건강한 생활 습관에 대해 조언해주세요."
            ],
            inputs=msg
        )
    
    return demo


if __name__ == "__main__":
    # Gradio 인터페이스 실행
    demo = create_gradio_interface()
    demo.launch(
        share=False,  # True로 설정하면 공개 URL 생성
        server_name="0.0.0.0",  # 모든 네트워크 인터페이스에서 접근 가능
        server_port=7860,  # 포트 설정
        show_error=True
    )