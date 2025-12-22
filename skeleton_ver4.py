import streamlit as st
import os
import nest_asyncio  # 비동기 처리를 위해 필요할 수 있음

# 1. 라이브러리 임포트
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document  # LlamaParse 결과를 변환하기 위해 필요

# LangChain 표준 모듈
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# LlamaParse 임포트
from llama_parse import LlamaParse

# 비동기 루프 패치 (Streamlit과 충돌 방지)
nest_asyncio.apply()

# API 키 설정
os.environ["GOOGLE_API_KEY"] = ""
LLAMA_CLOUD_API_KEY = ""

st.set_page_config(page_title="반도체 챗봇 (LlamaParse)", page_icon="⚡")
st.title("반도체 데이터시트 Chatbot (v4. LlamaParse 탑재)")

# 2. 캐싱된 벡터 DB 생성 (LlamaParse 적용)
@st.cache_resource
def get_vectorstore(file_path):
    # [1] LlamaParse로 PDF 읽기 (Markdown으로 변환)
    # result_type="markdown"이 핵심입니다. 표를 마크다운 표로 바꿔줍니다.
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        verbose=True
    )
    
    # LlamaParse는 LlamaIndex 문서 객체를 반환하므로...
    llama_documents = parser.load_data(file_path)
    
    # [2] LangChain용 Document 객체로 변환
    # (LlamaIndex -> LangChain 호환 작업)
    langchain_documents = []
    for doc in llama_documents:
        langchain_documents.append(
            Document(
                page_content=doc.text,
                metadata={"source": file_path}
            )
        )
    
    # [3] 청킹 (Chunking)
    # 마크다운 헤더를 기준으로 자르면 더 좋지만, 일단 기본적으로 처리합니다.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(langchain_documents)
    
    # [4] 임베딩 & 벡터 저장소
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# 3. 세션 스테이트 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

uploaded_file = st.file_uploader("데이터시트 PDF를 업로드하세요", type="pdf")

if uploaded_file is not None:
    file_path = "temp.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # LlamaParse는 클라우드를 갔다 오느라 시간이 조금 더 걸립니다.
    with st.spinner("LlamaParse가 표를 분석 중입니다... (약간의 시간 소요)"):
        vectorstore = get_vectorstore(file_path)
        st.success("분석 완료! 표 데이터까지 완벽하게 읽었습니다.")

    # 4. 대화 기록 출력
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 5. 질문 처리
    if prompt := st.chat_input("질문을 입력하세요..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("표 데이터를 검색 중..."):
                llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
                
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(),
                    memory=st.session_state.memory,
                    return_source_documents=True
                )
                
                result = qa_chain.invoke({"question": prompt})
                response = result["answer"]
                source_docs = result['source_documents']
                
                st.markdown(response)
                
                with st.expander("참고한 문서 (Markdown 변환됨)"):
                    for doc in source_docs:
                        st.caption(doc.page_content[:300] + "...") # 마크다운으로 변한 내용 확인
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})