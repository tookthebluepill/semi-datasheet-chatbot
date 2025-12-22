import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_classic.chains import RetrievalQA
except ImportError:
    from langchain_community.chains import RetrievalQA

# API 키 설정
os.environ["GOOGLE_API_KEY"] = ""

st.title("반도체 데이터시트 챗봇")

# 캐싱 함수: PDF 분석은 '파일이 바뀔 때만' 한 번 실행하고 결과를 저장해둡니다.
@st.cache_resource
def get_vectorstore(file_path):
    # 1. 문서 로드
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    
    # 2. 청킹 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)
    
    # 3. 임베딩 (로컬 CPU 사용)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 4. 벡터 DB 생성
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# 파일 업로드 UI
uploaded_file = st.file_uploader("데이터시트 PDF를 업로드하세요", type="pdf")

if uploaded_file is not None:
    # 파일을 임시 저장
    file_path = "temp.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # [핵심] 스피너가 첫 실행 때만 돌고, 두 번째부터는 즉시 통과합니다.
    with st.spinner("문서 분석 중... (처음에만 오래 걸립니다)"):
        vectorstore = get_vectorstore(file_path)
        st.success("분석 완료! 데이터시트에 관련 질문을 하시오.")

    # 질문 입력
    query = st.text_input("질문:")
    
    if query:
        # 모델: Gemini Flash Latest (빠르고 무료)
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        
        result = qa_chain.invoke({"query": query})
        
        st.write("### 🤖 답변:")
        st.write(result['result'])
        
        st.write("---")
        st.write("### 📄 참고한 페이지:")
        for doc in result['source_documents']:
            st.caption(f"Page {doc.metadata.get('page', '?')}: {doc.page_content[:150]}...")