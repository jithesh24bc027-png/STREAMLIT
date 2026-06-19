import streamlit as st
import os
import tempfile

st.set_page_config(page_title="LegalDoc AI", page_icon="\u2696\ufe0f", layout="wide")
st.title("\u2696\ufe0f Legal Document Assistant")
st.caption("Upload agreements & contracts to extract clauses, spot risks, and get plain-language explanations.")
st.divider()

with st.sidebar:
    st.header("\u2699\ufe0f Setup")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    uploaded_files = st.file_uploader(
        "Upload Agreement(s)/Contract(s) (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )
    process_btn = st.button("\u26a1 Analyse Document(s)", use_container_width=True)

    st.divider()
    st.subheader("\U0001f50d Quick Actions")
    quick_summary_btn = st.button("\U0001f4c4 Contract Summary", use_container_width=True)
    quick_clauses_btn = st.button("\U0001f4cc Extract Key Clauses", use_container_width=True)
    quick_risks_btn = st.button("\u26a0\ufe0f Identify Risks", use_container_width=True)

    st.warning("\u26a0\ufe0f For informational purposes only. This is not legal advice \u2014 consult a qualified lawyer for binding decisions.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if process_btn:
    if not groq_key or not uploaded_files:
        st.sidebar.error("Please provide both API key and at least one PDF.")
    else:
        with st.spinner("Reading document(s)..."):
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import PromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_core.output_parsers import StrOutputParser

            all_documents = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                all_documents.extend(PyPDFLoader(tmp_path).load())
                os.unlink(tmp_path)

            documents = all_documents
            docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

            llm = ChatOpenAI(
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1"
            )

            prompt = PromptTemplate(
                template="""You are a legal document assistant. Use the context to explain the contract in simple language, extract key clauses (termination, payment, liability, confidentiality, indemnity, etc.), and flag risky or one-sided terms with \u26a0\ufe0f.
Context: {context}
Question: {question}
Answer (plain language, not formal legal advice):""",
                input_variables=["context", "question"]
            )

            def fmt(docs):
                return "\n\n".join(d.page_content for d in docs)

            st.session_state.qa_chain = (
                {"context": retriever | fmt, "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )
            st.session_state.messages = []
        st.sidebar.success(f"\u2705 Done! {len(documents)} pages loaded from {len(uploaded_files)} file(s).")

quick_prompts = {
    "summary": "Provide a concise summary of this contract, including the parties involved, purpose, and key obligations.",
    "clauses": "Extract and list the key clauses in this document (e.g. termination, payment terms, liability, confidentiality, indemnity, dispute resolution).",
    "risks": "Identify any risky, unusual, or one-sided clauses in this document and explain why they could be a concern.",
}

triggered_query = None
if quick_summary_btn:
    triggered_query = quick_prompts["summary"]
elif quick_clauses_btn:
    triggered_query = quick_prompts["clauses"]
elif quick_risks_btn:
    triggered_query = quick_prompts["risks"]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.qa_chain:
    user_input = st.chat_input("Ask about your contract/agreement...")
    final_query = user_input or triggered_query

    if final_query:
        st.session_state.messages.append({"role": "user", "content": final_query})
        with st.chat_message("user"):
            st.write(final_query)
        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                answer = st.session_state.qa_chain.invoke(final_query)
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("\U0001f448 Upload your PDF(s) and click Analyse Document(s) to start.")
