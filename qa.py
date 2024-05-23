import os
from dotenv import load_dotenv
import time
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')
PROJECT = os.getenv('PROJECT')
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 100
EMBEDDING_DIRECTORY = 'embeddings'


class RAG:
    def __init__(self):
        self.google_api_key = google_api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key,
                                          convert_system_message_to_human=True)
        self.embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=google_api_key)
        self.rag_chain = self.pdf_retrieval_chain()
        self.format_docs = lambda docs: "\n\n".join(doc.page_content for doc in docs)
        self.parse_llm_response = lambda res: res.get('answer')
        self.lst_to_str = lambda history: '\n'.join(history) if history else ''

    def pdf_ingestion(self):
        docs = [f for f in os.listdir('docs') if f.endswith('.pdf')]
        pdf_files = [os.path.join('docs', pdf) for pdf in docs]
        raw_pdf_docs = []
        for pdf_file in pdf_files:
            raw_pdf_docs.extend(PyPDFLoader(pdf_file).load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        split_documents = text_splitter.split_documents(raw_pdf_docs)
        vector = Chroma.from_documents(split_documents, self.embedding_model, persist_directory=EMBEDDING_DIRECTORY)

        return vector

    def retriever(self):
        ex_vector_retriever = (Chroma(persist_directory=EMBEDDING_DIRECTORY, embedding_function=self.embedding_model)
                               .as_retriever())
        if not ex_vector_retriever.invoke('0'):
            print('No existing embedding found, creating and saving a new one...')
            vector = self.pdf_ingestion()
            return vector.as_retriever()
        return ex_vector_retriever

    def pdf_retrieval_chain(self):
        qa_system_prompt = """You are a helpful assistant. Answer and cite the users' questions with a concise response
        using relevant information from the context given. The context may not be properly formatted, use your best
        judgement. Do not add the words 'User' or 'Assistant' in your response. If you need more context or
        clarification, please ask for it:

        <CONTEXT>
         {context}
        </CONTEXT>

        {chat_history}
        User: {question}
        Assistant: """
        qa_prompt = ChatPromptTemplate.from_template(qa_system_prompt)
        vector_retriever = self.retriever()

        setup_and_retrieval = RunnableParallel({"context": itemgetter("meta") | vector_retriever,
                                                "question": itemgetter("question"),
                                                "chat_history": itemgetter("chat_history")
                                                })
        rag_chain = ((RunnablePassthrough.assign(context=(lambda x: self.format_docs(x["context"]))) | qa_prompt |
                      self.llm) | StrOutputParser())
        rag_chain_with_source = setup_and_retrieval.assign(answer=rag_chain)

        return rag_chain_with_source

    def response_generation(self, chat_history, question):
        if not question:
            return {'answer': 'Hello!'}
        payload = {"chat_history": self.lst_to_str(chat_history), "question": question,
                   "meta": self.lst_to_str(chat_history[-6:]) + question if question else 'Hello!'}
        res = self.rag_chain.invoke(payload)
        return res  # self.parse_llm_response(res)

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
            response = self.parse_llm_response(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            print('history', st.session_state.messages, '\n')
        except Exception as e:
            print(f'got an exception: {e}')


if __name__ == '__main__':
    RAG().chat()
