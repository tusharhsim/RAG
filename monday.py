import os
import time
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import UnstructuredExcelLoader
import streamlit as st

import pandas as pd
from contextlib import redirect_stdout
from io import StringIO
import numpy as np

google_api_key = 'AIzaSyBP40MHpvgCE_ktsiUSf2tEOEyqKeQJ6nI'
PROJECT = 'GARLOCK'
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 100
EMBEDDING_DIRECTORY = 'embeddings'


llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key,
                             convert_system_message_to_human=True)
embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=google_api_key)

# Streamlit App
st.title("Gemini Pro Generator")

# File Upload
uploaded_file = st.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"])
if uploaded_file is not None:
    # Check if the file exists
    if not os.path.isfile(uploaded_file.name):
        st.error(f"Error: File '{uploaded_file.name}' does not exist.")
        st.stop()

    # Read uploaded file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display top 10 rows of the DataFrame
    st.subheader("Top 10 Rows of the DataFrame:")
    st.write(df.head(10))

   # User Input
    variable = st.text_input("Enter the prompt:")

    # Generate Question based on user input and DataFrame
    question = f"Use the dataframe with name df with columns {df.columns} and generate python code for " + variable

    # Generate response using Gemini Pro
    response = llm.invoke(question)
    st.write(response)
    # st.write(response)
    exec_code = response.content[9:-3]
    print('eebu-', exec_code, '  xxx')
    with StringIO() as output_buffer:
        with redirect_stdout(output_buffer):
            exec(exec_code)
        captured_output = output_buffer.getvalue()
    st.subheader("Captured Output:")
    st.code(captured_output, language='python')
