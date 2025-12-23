import streamlit as st
import os
from dotenv import load_dotenv
import nest_asyncio

# 1. 라이브러리 import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document

# 표준 LangChain 모듈 (메모리 & 체인)
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# [핵심] LlamaParse & PDF Viewer
from llama_parse import LlamaParse
from streamlit_pdf_viewer import pdf_viewer

# 비동기 충돌 방지 (LlamaParse 필수)
nest_asyncio.apply()

# API 키 설정
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY가 설정되지 않았습니다.")

if not os.getenv("LLAMA_CLOUD_API_KEY"):
    st.error("LLAMA_CLOUD_API_KEY가 설정되지 않았습니다.")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

# [UI 설정] 와이드 모드 (화면 분할을 위해 필수)
st.set_page_config(layout="wide", page_title="반도체 데이터시트 Chatbot")

st.title("반도체 데이터시트 Chatbot")

# 2. 벡터 DB 생성 함수 (캐싱 적용)
@st.cache_resource
def get_vectorstore(file_path):
    try:
        # [1] LlamaParse로 PDF 읽기 (Markdown 변환)
        parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            verbose=True
        )
        llama_documents = parser.load_data(file_path)
        
        if not llama_documents:
            st.error("PDF를 읽었으나 내용이 비어있습니다.")
            return None

        # [2] LangChain 포맷으로 변환
        langchain_documents = []
        for doc in llama_documents:
            # LlamaParse의 메타데이터(페이지 번호 등)를 복사
            doc_metadata = doc.metadata.copy()
            doc_metadata["source"] = file_path
            
            langchain_documents.append(
                Document(page_content=doc.text, metadata=doc_metadata)
            )
        
        # [3] 청킹 (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(langchain_documents)
        
        if not texts:
            st.error("텍스트 변환 결과가 없습니다.")
            return None

        # [4] 임베딩 & 벡터 저장
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(texts, embeddings)
        return vectorstore
        
    except Exception as e:
        st.error(f"문서 분석 중 오류 발생: {e}")
        return None

# 3. 세션 스테이트 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# 현재 페이지 상태 (PDF 뷰어 제어용)
if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# 사이드바 & 파일 업로드
with st.sidebar:
    st.header("📂 문서 업로드")
    uploaded_file = st.file_uploader("데이터시트 PDF 업로드", type="pdf")

# 메인 로직 시작
if uploaded_file is not None:
    file_path = "temp.pdf"
    binary_data = uploaded_file.getvalue()
    
    # 임시 파일 저장
    with open(file_path, "wb") as f:
        f.write(binary_data)
    
    # VectorStore 로딩 (세션에 저장하여 Rerun 시 재분석 방지)
    if "vectorstore" not in st.session_state:
        with st.spinner("LlamaParse가 표를 분석 중입니다... (최초 1회, 시간 소요됨)"):
            vs = get_vectorstore(file_path)
            if vs:
                st.session_state.vectorstore = vs
                st.success("분석 완료! 질문을 입력하세요.")
            else:
                st.stop() # 분석 실패 시 중단

    # 화면 50:50 분할
    col1, col2 = st.columns([1, 1])

    # [Right] PDF Viewer (먼저 배치)
    with col2:
        st.info(f"📄 문서 뷰어 (현재 페이지: {st.session_state.current_page})")
        
        # [핵심] key를 고정("pdf_viewer")하고, pages_to_render를 세션 변수로 제어
        # 페이지가 바뀌면 st.rerun()이 발생하여 이 코드가 다시 실행되고, 뷰어가 갱신됨
        pdf_viewer(
            input=binary_data,
            width=700,
            height=800,
            pages_to_render=[st.session_state.current_page],
            key=f"pdf_viewer_page_{st.session_state.current_page}"
        )

    # [Left] Chat Interface
    with col1:
        st.subheader("💬 AI 엔지니어")
        
        # 채팅 기록 컨테이너
        chat_container = st.container(height=600)
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 질문 입력 처리
        if prompt := st.chat_input("질문을 입력하세요..."):
            # 1. 사용자 질문 표시
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # 2. AI 답변 생성
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("문서 검색 및 답변 생성 중..."):
                        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
                        
                        # 체인 생성
                        qa_chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=st.session_state.vectorstore.as_retriever(),
                            memory=st.session_state.memory,
                            return_source_documents=True
                        )
                        
                        # 실행
                        result = qa_chain.invoke({"question": prompt})
                        response = result["answer"]
                        source_docs = result['source_documents']
                        
                        st.markdown(response)
                        
                        # 페이지 정보 추출 및 이동 대상 결정
                        target_page = st.session_state.current_page # 기본값: 유지
                        
                        if source_docs:
                            # 디버깅용: AI가 찾은 메타데이터 확인 (개발 완료 후 주석 처리 가능)
                            # st.toast(f"메타데이터: {source_docs[0].metadata}", icon="🔍")
                            
                            try:
                                best_doc = source_docs[0]
                                # LlamaParse 우선순위: page_label(문서 번호) -> page(인덱스)
                                page_label = best_doc.metadata.get("page_label")
                                page_num = best_doc.metadata.get("page")
                                
                                if page_label:
                                    target_page = int(page_label)
                                elif page_num is not None:
                                    target_page = int(page_num) + 1 # 0-based index 보정
                                
                                # 안전장치: 1페이지 미만 방지
                                target_page = max(1, target_page)
                                
                            except Exception as e:
                                print(f"페이지 추출 에러: {e}")
                                # 에러 시 페이지 이동 안 함

                            # 근거 문서 아코디언 표시
                            with st.expander("참고한 문서 내용"):
                                for doc in source_docs:
                                    p_info = doc.metadata.get("page_label") or doc.metadata.get("page")
                                    st.caption(f"[Page {p_info}] {doc.page_content[:200]}...")

            # 3. 대화 기록 저장
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # 4. [Rerun 트리거] 페이지가 변경되어야 한다면, 상태 업데이트 후 즉시 Rerun!
            if target_page != st.session_state.current_page:
                st.session_state.current_page = target_page
                st.rerun()