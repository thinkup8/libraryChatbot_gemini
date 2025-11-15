import os
import streamlit as st
import nest_asyncio

# Streamlit에서 async 관련 오류가 발생하지 않도록 이벤트 루프 패치
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

# SQLite 관련 충돌 해결용 패치
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma


# ---------------------------------------------------------
# 1) Gemini API 키 로드
# ---------------------------------------------------------
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.error("GOOGLE_API_KEY가 Streamlit Secrets에 설정되어 있지 않음")
    st.stop()


# ---------------------------------------------------------
# 2) PDF 로드 & 텍스트 분할
# ---------------------------------------------------------
@st.cache_resource
def load_and_split_pdf(file_path):
    """
    하나의 PDF 파일을 로드하여 페이지 단위로 분할.
    반환값: LangChain Document 객체 리스트
    """
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()


# ---------------------------------------------------------
# 3) 벡터스토어 생성 (PDF → 텍스트 청크 → 임베딩 → Chroma 저장)
# ---------------------------------------------------------
@st.cache_resource
def create_vector_store(_docs):
    """
    Document 리스트를 받아 텍스트 청크 분할 후 Chroma DB를 생성.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(_docs)
    st.info(f"{len(split_docs)}개의 청크 생성됨")

    persist_directory = "./chroma_db"

    # HuggingFace 임베딩 로드
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Chroma DB 생성 및 저장
    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_directory
    )
    return vectorstore


# ---------------------------------------------------------
# 4) 기존 Chroma DB가 있으면 로드, 없으면 새로 생성
# ---------------------------------------------------------
@st.cache_resource
def get_vectorstore(_docs):
    persist_directory = "./chroma_db"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if os.path.exists(persist_directory):
        # 기존 벡터 DB 사용
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        # 새로 생성
        return create_vector_store(_docs)


# ---------------------------------------------------------
# 5) RAG 구성 요소 초기화 (PDF → Vector → Retriever → LLM Chain)
# ---------------------------------------------------------
@st.cache_resource
def initialize_components(selected_model):
    # *** 현재는 PDF 경로가 고정되어 있음 ***
    file_path = r"/mount/src/librarychatbot_gemini/드론의 해양과학조사 활용 국제동향.pdf"

    # pdf 로드
    pages = load_and_split_pdf(file_path)

    # 벡터스토어 생성 또는 로드
    vectorstore = get_vectorstore(pages)

    # retriever 구성
    retriever = vectorstore.as_retriever()

    # --- 대화 맥락 기반 질문 재구성 Prompt ---
    contextualize_q_system_prompt = """Given a chat history ... """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # --- 최종 답변 Prompt ---
    qa_system_prompt = """You are an assistant ... {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # Gemini LLM 로드
    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.7,
        convert_system_message_to_human=True
    )

    # 대화 기반 Retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 문서 기반 QA 체인
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # RAG 구성 (Retriever → LLM)
    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )
    return rag_chain


# ---------------------------------------------------------
# 6) Streamlit UI (챗봇 인터페이스)
# ---------------------------------------------------------
st.header("드론의 해양과학조사 활용 동향 Q&A 챗봇")

# 모델 선택
option = st.selectbox(
    "Select Gemini Model",
    ("gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.0-flash-lite")
)

# RAG 초기화
with st.spinner("초기화 중..."):
    try:
        rag_chain = initialize_components(option)
        st.success("초기화 완료")
    except Exception as e:
        st.error(str(e))
        st.stop()

# 채팅 히스토리 저장
chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

# 기존 메시지 출력
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# 사용자 입력 처리
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

            # 참고 문서 표시
            with st.expander("참고 문서"):
                for doc in response['context']:
                    st.markdown(doc.metadata['source'], help=doc.page_content)

