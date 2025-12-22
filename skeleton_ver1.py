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

st.title("반도체 데이터시트 AI 챗봇")

uploaded_file = st.file_uploader("데이터시트 PDF를 업로드하세요", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("문서 분석 중... (로컬 임베딩 사용)"):
        # 문서 로드
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load_and_split()
        
        # 청킹 (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(pages)
        
        # [핵심 수정] GoogleEmbeddings -> HuggingFaceEmbeddings
        # 이 모델(all-MiniLM-L6-v2)은 작고 빨라서 노트북 CPU로도 충분합니다.
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 벡터 DB 생성
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        st.success("분석 완료! 질문해주세요.")

    query = st.text_input("질문:")
    
    if query:
        # 답변 생성은 여전히 똑똑한 Gemini가 담당합니다
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