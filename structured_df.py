import time
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv('GEMINI_API_KEY')
PROJECT = os.getenv('PROJECT')


class RAG:
    def __init__(self, df):
        self.google_api_key = google_api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key,
                                          convert_system_message_to_human=True)
        self.embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=google_api_key)
        self.df = df
        self.parse_llm_response = lambda res: res.get('answer')
        self.lst_to_str = lambda history: '\n'.join(history) if history else ''

    def rag_prompt(self, chat_history, question):
        qa_system_prompt = """You are a helpful assistant. Try to accurately answer the user's query. Use the below 
        history to respond, but if no context is found in the history, you should use the dataframe with name df with
        columns {columns} to generate python code for answering:

        {chat_history}
        User: {question}
        Assistant: """
        qa_prompt = ChatPromptTemplate.from_template(qa_system_prompt)

        prompt = qa_prompt.format(chat_history=chat_history, question=question, columns=list(self.df.columns))
        return prompt

    def response_generation(self, chat_history, question):
        if not question:
            return {'answer': 'Hello!'}
        payload = {"chat_history": self.lst_to_str(chat_history), "question": question}
        res = self.llm.invoke(self.rag_prompt(**payload))
        return res

    def chat(self):
        st.title("OE RAG")
        try:
            if "messages" not in st.session_state:
                st.session_state.messages = []

            chat_history = []
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    content = message["content"]
                    st.write(content)
                    chat_history.append(f'{message["role"].title()}: {content}')

            if question := st.chat_input(f"message {PROJECT}..."):
                with st.chat_message("user"):
                    st.write(question)
                st.session_state.messages.append({"role": "user", "content": question})
            start = time.time()

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.response_generation(chat_history, question)
                    st.write(response)
            print('time required', time.time() - start)
            # response = self.parse_llm_response(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            print('history', st.session_state.messages, '\n')
        except Exception as e:
            print(f'got an exception: {e}')


if __name__ == '__main__':
    uploaded_file = st.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"])
    df = pd.read_csv(uploaded_file)
    RAG(df).chat()

