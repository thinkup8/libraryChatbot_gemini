import os
import streamlit as st
import nest_asyncio

nest_asyncio.apply()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

# sqlite 충돌 해결
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_chroma import Chroma


# Google API Key 읽기
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("Google API Key가 설정되어 있지 않습니다.")
    st.stop()


# ----------------------------
# PDF → Document 변환
# ----------------------------
def load_multiple_pdfs(uploaded_files):
    """
    여러 PDF 파일을 받아 pages(Document 리스트)로 합침.
    """
    all_docs = []
    for file in uploaded_files:
        temp_path = f"./temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.read())
        loader = PyPDFLoader(temp_path)
        docs = loader.load_and_split()
        all_docs.extend(docs)
    return all_docs


# ----------------------------
# Chroma 벡터스토어 생성
# ----------------------------
def create_vector_store_from_docs(_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore


# ----------------------------
# RAG 구성 생성
# ----------------------------
def initialize_rag(selected_model, vectorstore):
    retriever = vectorstore.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Given chat history and a user question, rewrite it as a standalone question."),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             """You are a Korean question-answering assistant.
                Use the following context to answer.
                If you don't know the answer, say you don't know.
                답변은 반드시 한국어 존댓말로 작성하세요.
                {context}"""),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.2,
        convert_system_message_to_human=True
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="PDF 기반 RAG 챗봇", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        background-color: #1E88E5;
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-size: 26px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 20px;
    }
    .chat-container {
        background-color: #F5F5F5;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">PDF 기반 RAG 챗봇</div>', unsafe_allow_html=True)

st.info("PDF 여러 개를 업로드하면 해당 문서 전체를 기반으로 답변합니다.")


# PDF 업로드
uploaded_files = st.file_uploader(
    "PDF 파일 업로드 (여러 개 가능)",
    type=["pdf"],
    accept_multiple_files=True
)

model_name = st.selectbox(
    "Gemini Model 선택",
    ["gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.0-flash-lite"]
)

if uploaded_files:
    with st.spinner("PDF 처리 중…"):
        pages = load_multiple_pdfs(uploaded_files)  # 여러 PDF 로드
        vectorstore = create_vector_store_from_docs(pages)

    with st.spinner("RAG 구성 중…"):
        rag_chain = initialize_rag(model_name, vectorstore)

    st.success("챗봇 준비 완료")

    chat_history = StreamlitChatMessageHistory(key="chat_messages")

    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer"
    )

    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)

    if user_input := st.chat_input("질문을 입력하세요"):
        st.chat_message("human").write(user_input)

        with st.chat_message("ai"):
            with st.spinner("답변 생성 중…"):
                response = conversational_rag.invoke(
                    {"input": user_input},
                    {"configurable": {"session_id": "default"}}
                )
                answer = response["answer"]
                st.write(answer)

else:
    st.warning("PDF 파일을 업로드하세요.")
