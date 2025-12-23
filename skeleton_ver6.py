import streamlit as st
import os
import hashlib
import json
import re  # [추가] 정규표현식 (페이지 번호 추출용)
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
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ GOOGLE_API_KEY가 설정되지 않았습니다.")
    st.stop()

if not os.getenv("LLAMA_CLOUD_API_KEY"):
    st.error("❌ LLAMA_CLOUD_API_KEY가 설정되지 않았습니다.")
    st.stop()

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

st.set_page_config(layout="wide", page_title="Semi-Datasheet-Chatbot")
st.title("반도체 데이터시트 Chatbot (Pro Ver.)")

# 유틸 1: 파일 해시 (캐싱 키)
def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# 유틸 2: 안전한 페이지 번호 추출 (Regex 적용)
def get_safe_page_number(page_value, default=1):
    """
    'Page 3', '3/10', 'iv' 등 다양한 형식에서 숫자만 추출
    """
    if page_value is None:
        return default
    
    # 문자열로 변환 후 숫자 탐색
    s_val = str(page_value)
    match = re.search(r"(\d+)", s_val)
    
    if match:
        return int(match.group(1))
    return default

# VectorStore 생성 (JSON 캐싱 + 페이지 보존)
@st.cache_resource
def get_vectorstore(file_path, file_hash):
    # 폴더 준비
    faiss_cache_dir = os.path.join("faiss_cache", file_hash)
    # MD 대신 JSON으로 저장하여 메타데이터 보존
    json_cache_path = os.path.join("parsed_cache", f"{file_hash}.json")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # FAISS DB 로드
    if os.path.exists(faiss_cache_dir):
        if os.path.exists(os.path.join(faiss_cache_dir, "index.faiss")):
            st.info(f"캐시된 벡터 DB를 로드합니다. (0.1초 컷)")
            return FAISS.load_local(
                faiss_cache_dir, 
                embeddings, 
                allow_dangerous_deserialization=True
            )

    # 파싱 데이터 준비
    llama_documents = []

    # 2. JSON 캐시 확인
    if os.path.exists(json_cache_path):
        st.info(f"파싱된 데이터(JSON)를 로드합니다. (LlamaParse 절약)")
        with open(json_cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            # JSON -> Document 객체 복원
            for item in cached_data:
                llama_documents.append(
                    Document(
                        page_content=item["text"],
                        metadata=item["metadata"]
                    )
                )
    else:
        # 3. LlamaParse 실행
        try:
            st.info("LlamaCloud에서 문서를 분석 중입니다... (토큰 사용)")
            parser = LlamaParse(
                api_key=LLAMA_CLOUD_API_KEY,
                result_type="markdown",
                verbose=True
            )
            # LlamaParse는 기본적으로 Document 객체 리스트를 반환함
            parsed_docs = parser.load_data(file_path)
            
            if not parsed_docs:
                return None
            
            llama_documents = parsed_docs

            # 결과를 JSON으로 저장 (메타데이터 포함)
            cache_data = []
            for doc in parsed_docs:
                cache_data.append({
                    "text": doc.text,
                    "metadata": doc.metadata # 여기에 page_label이 들어있음
                })
            
            if not os.path.exists("parsed_cache"):
                os.makedirs("parsed_cache")
                
            with open(json_cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            st.error(f"LlamaParse 오류: {e}")
            return None

    # [Vector DB Build] 메타데이터 정리 및 청킹
    st.info("데이터를 벡터화 중입니다...")
    
    langchain_documents = []
    
    for doc in llama_documents:
        # 1. 텍스트 추출
        content = doc.text if hasattr(doc, 'text') else doc.page_content
        
        # 2. 메타데이터 안전 추출
        original_meta = doc.metadata if hasattr(doc, 'metadata') else {}
        
        # [핵심] 안전한 페이지 번호 추출 함수 사용
        raw_page_label = original_meta.get("page_label")
        raw_page_index = original_meta.get("page")
        
        # page_label이 있으면 우선 쓰고, 없으면 인덱스 사용
        final_page_num = get_safe_page_number(raw_page_label, default=None)
        
        if final_page_num is None and raw_page_index is not None:
             final_page_num = int(raw_page_index) + 1 # 0부터 시작하므로 +1
        
        if final_page_num is None:
            final_page_num = 1 # 최후의 보루

        new_metadata = {
            "source": file_path,
            "page": final_page_num,     # 이제 'page'는 무조건 깨끗한 정수(int)
            "original_label": str(raw_page_label) # 참고용 원본
        }

        langchain_documents.append(
            Document(page_content=content, metadata=new_metadata)
        )

    # 청킹 (Chunking) - 이제 쪼개져도 'page' 메타데이터는 유지됨!
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(langchain_documents)
    
    # FAISS 생성 및 저장
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(faiss_cache_dir)
    
    st.success("DB 생성 완료!")
    return vectorstore

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# [메모리 최적화] ConversationBufferMemory는 체인 내부에서만 쓰고,
# UI 표시는 st.session_state.chat_history로 관리하여 이중 저장을 방지하는 패턴 권장
# 하지만 코드 수정을 최소화하기 위해 기존 구조 유지하되, 메모리 키를 명확히 함
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# 사이드바
with st.sidebar:
    st.header("문서 업로드")
    uploaded_file = st.file_uploader("데이터시트 PDF", type="pdf")

# 메인 로직
if uploaded_file is not None:
    # 1. 파일 처리
    binary_data = uploaded_file.getvalue()
    file_hash = get_file_hash(binary_data)
    file_path = f"temp_{file_hash}.pdf"
    
    with open(file_path, "wb") as f:
        f.write(binary_data)

    # 2. 로딩
    if "vectorstore" not in st.session_state or st.session_state.get("current_file_hash") != file_hash:
        vs = get_vectorstore(file_path, file_hash)
        if vs is None: st.stop()
        st.session_state.vectorstore = vs
        st.session_state.current_file_hash = file_hash

    # 화면 분할
    col1, col2 = st.columns([1, 1])

    # [Right] PDF Viewer
    with col2:
        st.info(f"문서 뷰어 (Page: {st.session_state.current_page})")
        pdf_viewer(
            input=binary_data,
            width=700,
            height=800,
            pages_to_render=[st.session_state.current_page],
            key="pdf_viewer"
        )

    # [Left] Chat
    with col1:
        st.subheader("💬 AI 엔지니어")
        chat_container = st.container(height=600)

        # 기록 출력
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 질문 입력
        if prompt := st.chat_input("질문을 입력하세요..."):
            # UI 즉시 표시
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # 답변 생성
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("분석 중..."):
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

                        # [페이지 점프 로직 개선]
                        target_page = st.session_state.current_page
                        if source_docs:
                            # 가장 유사도가 높은 첫 번째 문서의 'page' 메타데이터 사용
                            # (위에서 이미 int로 정제해둠)
                            doc_page = source_docs[0].metadata.get("page")
                            if doc_page:
                                target_page = int(doc_page)

                            # 근거 표시
                            with st.expander("참고한 문서 내용"):
                                for doc in source_docs:
                                    p = doc.metadata.get("page")
                                    st.caption(f"[Page {p}] {doc.page_content[:200]}...")

            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Rerun
            if target_page != st.session_state.current_page:
                st.session_state.current_page = target_page
                st.rerun()