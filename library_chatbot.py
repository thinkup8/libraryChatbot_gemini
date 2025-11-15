import os
import streamlit as st
import nest_asyncio
import shutil

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

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma


# --------------------------------------------
#  API key
# --------------------------------------------
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요!")
    st.stop()


# --------------------------------------------
#  PDF 로드
# --------------------------------------------
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()


# --------------------------------------------
#  *새로 생성된 PDF 기준으로 Vector DB 강제 재생성*
# --------------------------------------------
@st.cache_resource
def rebuild_vectorstore(_docs):
    persist_directory = "./chroma_db"

    # 기존 DB 삭제
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    # 임베딩 모델 로드
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(_docs)

    # Chroma DB 새로 생성
    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_directory
    )

    return vectorstore


# --------------------------------------------
#  RAG 컴포넌트 초기화
# --------------------------------------------
@st.cache_resource
def initialize_components(selected_model):
    # PDF 파일 경로 - GitHub 저장된 PDF 불러오는 경로
    file_path = r"/mount/src/librarychatbot_gemini/드론의 해양과학조사 활용 국제동향.pdf"

    pages = load_and_split_pdf(file_path)

    # 기존 DB 무시하고 매번 새로 생성
    vectorstore = rebuild_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # System prompt
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # Q&A prompt
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. please use imogi with the answer.
    대답은 한국어로 하고, 존댓말을 써줘.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.7,
        convert_system_message_to_human=True
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


# --------------------------------------------
#  UI
# --------------------------------------------
st.header("드론의 해양과학조사 활용 동향 Q&A 챗봇 💬 📚")

if not os.path.exists("./chroma_db"):
    st.info("🔄 첫 실행입니다. 임베딩 모델 다운로드 및 PDF 처리 중... (약 5-7분 소요)")

option = st.selectbox(
    "Select Gemini Model",
    ("gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.0-flash-lite"),
    index=0
)

try:
    with st.spinner("🔧 챗봇 초기화 중..."):
        rag_chain = initialize_components(option)
    st.success("✅ 챗봇이 준비되었습니다!")
except Exception as e:
    st.error(f"⚠️ 초기화 중 오류 발생: {str(e)}")
    st.stop()

chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant",
                                     "content": "드론의 해양과학조사 활용 동향에 대해 무엇이든 물어보세요!"}]

for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)


if prompt_message := st.chat_input("Your question"):
    st.chat_message("human").write(prompt_message)

    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke(
                {"input": prompt_message},
                config
            )

            answer = response['answer']
            st.write(answer)

            with st.expander("참고 문서 확인"):
                for doc in response['context']:
                    st.markdown(doc.metadata['source'], help=doc.page_content)
