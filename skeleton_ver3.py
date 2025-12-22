import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_classic.memory import ConversationBufferMemory
    from langchain_classic.chains import ConversationalRetrievalChain
except ImportError:
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain

# API 키 설정
os.environ["GOOGLE_API_KEY"] = ""

st.title("반도체 데이터시티 Chatbot(v3:대화 기억)")

# 2. 캐싱된 벡터 DB 생성
@st.cache_resource
def get_vectorstore(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# 3. 세션 스테이트 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 4. 메모리 초기화 (라이브러리 사용)
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
    
    with st.spinner("문서 분석 중..."):
        vectorstore = get_vectorstore(file_path)
        st.success("분석 완료! 질문을 해주세요")

    # 5. 대화 기록 출력
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 6. 질문 입력 및 처리
    if prompt := st.chat_input("질문을 입력하세요..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                # 모델: 최신 버전 리스트에 있던 'gemini-flash-latest' 사용
                llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
                
                # Chain 생성 (Classic 패키지 사용)
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
                
                with st.expander("참고한 문서 페이지"):
                    for doc in source_docs:
                        st.caption(f"Page {doc.metadata.get('page', '?')}: {doc.page_content[:150]}...")
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})