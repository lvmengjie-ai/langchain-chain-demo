import gradio as gr
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def chat(message, history):
    prompt = PromptTemplate(
        template="你是智能AI助手，请详细回答：{input}",
        input_variables=["input"]
    )
    llm = ChatTongyi(
        model_name="qwen-turbo",
        dashscope_api_key="sk-"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"input": message})

demo = gr.ChatInterface(
    fn=chat,
    title="LangChain多步骤智能助手",
    description="新版LangChain链式调用"
)

if __name__ == "__main__":
    demo.launch()
