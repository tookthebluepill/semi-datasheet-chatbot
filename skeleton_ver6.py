import streamlit as st
import os
import hashlib
from dotenv import load_dotenv
import nest_asyncio

# 1. 라이브러리 import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document

# 표준 LangChain 모듈
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# PDF 처리
from llama_parse import LlamaParse
from streamlit_pdf_viewer import pdf_viewer

# 비동기 충돌 방지
nest_asyncio.apply()

# 환경 변수 및 설정
load_dotenv() # .env 파일 로드

# API 키
if not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    st.stop()

if not os.getenv("LLAMA_CLOUD_API_KEY"):
    st.error("❌ LLAMA_CLOUD_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    st.stop()

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

# UI 설정
st.set_page_config(layout="wide", page_title="Semi-Datasheet-Chatbot")
st.title("⚡ 반도체 데이터시트 Chatbot (Pro Ver.)")

# 유틸: 파일 해시 생성 (캐싱 키로 사용)
def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# VectorStore 생성
@st.cache_resource
def get_vectorstore(file_path, file_hash):
    """
    1순위: 로컬에 저장된 FAISS DB가 있으면 즉시 로드 (가장 빠름)
    2순위: 파싱된 Markdown 파일이 있으면 로드 후 임베딩 (LlamaParse 절약)
    3순위: 아무것도 없으면 LlamaParse API 호출 -> Markdown 저장 -> FAISS 저장
    """
    
    # 캐시 폴더 경로 설정
    faiss_cache_dir = os.path.join("faiss_cache", file_hash)
    parsed_cache_path = os.path.join("parsed_cache", f"{file_hash}.md")
    
    # 임베딩 모델 준비 (로딩과 생성 모두에 필요)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 이미 만들어둔 FAISS DB가 있는지 확인
    if os.path.exists(faiss_cache_dir):
        if os.path.exists(os.path.join(faiss_cache_dir, "index.faiss")):
            st.info(f"캐시된 벡터 DB를 로드합니다.")
            # allow_dangerous_deserialization=True는 로컬에서 내가 만든 파일을 믿는다는 뜻
            vectorstore = FAISS.load_local(
                faiss_cache_dir, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            return vectorstore

    # FAISS가 없으면 텍스트(Markdown)를 준비해야 함
    markdown_text = ""
    llama_documents = []

    # 파싱된 텍스트 캐시가 있는지 확인
    if os.path.exists(parsed_cache_path):
        st.info(f"저장된 파싱 텍스트를 불러옵니다.")
        with open(parsed_cache_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()
        llama_documents = [Document(page_content=markdown_text, metadata={"source": file_path})]
    
    else:
        # 캐시가 전혀 없으면 LlamaParse API 실행
        try:
            st.info("LlamaCloud에서 문서를 분석 중입니다... (최초 1회, 토큰 사용)")
            parser = LlamaParse(
                api_key=LLAMA_CLOUD_API_KEY,
                result_type="markdown",
                verbose=True
            )
            parsed_docs = parser.load_data(file_path)
            
            if not parsed_docs:
                st.error("PDF 내용이 비어있습니다.")
                return None
            
            # 텍스트 합치기 및 저장
            markdown_text = "\n\n".join([doc.text for doc in parsed_docs])
            
            if not os.path.exists("parsed_cache"):
                os.makedirs("parsed_cache")
                
            with open(parsed_cache_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
                
            llama_documents = parsed_docs

        except Exception as e:
            st.error(f"LlamaParse 오류: {e}")
            return None

    # 텍스트 -> 청킹 -> 임베딩 -> FAISS
    st.info("텍스트를 벡터로 변환 및 저장 중입니다...")
    
    # LangChain Document 형식 정리
    langchain_documents = []
    if isinstance(llama_documents[0], Document):
         langchain_documents = llama_documents
    else:
        for doc in llama_documents:
            doc_metadata = doc.metadata.copy()
            doc_metadata["source"] = file_path
            langchain_documents.append(
                Document(page_content=doc.text, metadata=doc_metadata)
            )

    # 청킹 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(langchain_documents)
    
# FAISS 생성
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # 완성된 FAISS DB를 통째로 저장 (다음 실행을 위해)
    vectorstore.save_local(faiss_cache_dir)
    st.success("벡터 DB 저장 완료! 다음부터는 즉시 로딩됩니다.")
    
    return vectorstore

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# 사이드바 (파일 업로드)
with st.sidebar:
    st.header("문서 업로드")
    uploaded_file = st.file_uploader("데이터시트 PDF 업로드", type="pdf")

# 메인 어플리케이션 로직
if uploaded_file is not None:
    # 1. 파일 해시 계산 (고유 ID)
    binary_data = uploaded_file.getvalue()
    file_hash = get_file_hash(binary_data)
    
    # 2. 임시 파일 저장
    file_path = f"temp_{file_hash}.pdf"
    with open(file_path, "wb") as f:
        f.write(binary_data)

    # 3. VectorStore 로드 (세션에 없다면 생성)
    # file_hash가 바뀌면(다른 파일) 다시 로드함
    if "vectorstore" not in st.session_state or st.session_state.get("current_file_hash") != file_hash:
        vs = get_vectorstore(file_path, file_hash)
        
        if vs is None:
            st.stop()
            
        st.session_state.vectorstore = vs
        st.session_state.current_file_hash = file_hash # 현재 파일 해시 저장

    # 화면 분할 (왼쪽: 채팅 / 오른쪽: PDF)
    col1, col2 = st.columns([1, 1])

    # [Right] PDF Viewer
    with col2:
        st.info(f"문서 뷰어 (Page: {st.session_state.current_page})")
        
        # key를 "pdf_viewer"로 고정하고, pages_to_render를 세션 상태로 제어
        # Rerun 될 때마다 이 부분이 다시 실행되며 페이지가 갱신됨
        pdf_viewer(
            input=binary_data,
            width=700,
            height=800,
            pages_to_render=[st.session_state.current_page],
            key="pdf_viewer"
        )

    # [Left] Chat Interface
    with col1:
        st.subheader("💬 AI 엔지니어")

        # 채팅 기록 표시
        chat_container = st.container(height=600)
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 사용자 입력 처리
        if prompt := st.chat_input("질문을 입력하세요..."):
            # UI 업데이트
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # 답변 생성
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("분석 중..."):
                        # [안정성] 429 에러 방지용 max_retries 추가
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-1.5-flash",
                            temperature=0,
                            max_retries=2
                        )

                        qa_chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=st.session_state.vectorstore.as_retriever(),
                            memory=st.session_state.memory,
                            return_source_documents=True
                        )

                        result = qa_chain.invoke({"question": prompt})
                        response = result["answer"]
                        source_docs = result["source_documents"]

                        st.markdown(response)

                        # 페이지 이동 로직
                        target_page = st.session_state.current_page
                        
                        if source_docs:
                            # 디버그용 (필요시 주석 해제)
                            # st.toast(f"메타데이터: {source_docs[0].metadata}")
                            try:
                                best_doc = source_docs[0]
                                page_label = best_doc.metadata.get("page_label")
                                page_num = best_doc.metadata.get("page")

                                if page_label:
                                    target_page = int(page_label)
                                elif page_num is not None:
                                    target_page = int(page_num) + 1 # 0-based -> 1-based

                                target_page = max(1, target_page) # 최소 1페이지 보장

                            except Exception as e:
                                print(f"페이지 추출 실패: {e}")

                            # 근거 문서 표시
                            with st.expander("참고한 문서 내용"):
                                for doc in source_docs:
                                    p = doc.metadata.get("page_label") or doc.metadata.get("page")
                                    st.caption(f"[Page {p}] {doc.page_content[:200]}...")

            # AI 응답 저장
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # 페이지가 달라졌으면 Rerun (뷰어 갱신)
            if target_page != st.session_state.current_page:
                st.session_state.current_page = target_page
                st.rerun()