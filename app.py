from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import cassio
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from dotenv import load_dotenv
from langchain import OpenAI
from PyPDF2 import PdfReader
import os
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template
import io
from langchain.chains.question_answering import load_qa_chain

# read from pdf
def read_doc(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with io.BytesIO(pdf.read()) as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

# Into chunks
def into_chunks(documents,chunk_size=800,chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(documents)
    return doc

def get_vectorstore(docs):
    cassio.init(token=os.environ['ASTRA_DB_APPLICATION_TOKEN'], database_id=os.environ['ASTRA_DB_ID'])
    embedding = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    vector_store = Cassandra(
        embedding=embedding,
        table_name="constNepa",
    )
    vector_store.add_texts(docs)
    index = VectorStoreIndexWrapper(vectorstore = vector_store)
    return index


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key = 'chat_history', return_messages=True)
    chain = load_qa_chain(llm, chain_type="stuff", memory = memory)
    return chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    load_dotenv()
    st.set_page_config(page_title = "Chat With you PDF", page_icon = ":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDF using PDF Reader")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_doc = st.file_uploader("Upload your PDFs here", )
        if st.button("Upload"):
            with st.spinner("Uploading"):
                pdf_text = read_doc(pdf_doc)
                chunks = into_chunks(pdf_text)
                vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()

