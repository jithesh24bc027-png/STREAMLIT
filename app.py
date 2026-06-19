import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Page Configuration ---
st.set_page_config(
    page_title="Team 4: Legal Document Assistant",
    page_icon="⚖️",
    layout="wide"
)

# --- 1. Model & Session Initialization ---
@st.cache_resource
def initialize_models():
    """Cache models so they don't reload on every user interaction."""
    # Initialize the ultra-fast Groq LLM
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.2  # Low temperature for strict legal accuracy
    )
    # Load local open-source embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

try:
    llm, embeddings = initialize_models()
except Exception as e:
    st.error("Error initializing models. Please make sure GROQ_API_KEY is set correctly.")

# Helper function to format documents for the LLM
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 2. RAG Chain Generator Engine ---
def create_rag_chain(prompt_template, retriever):
    return (
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt_template
        | llm
        | StrOutputParser()
    )

# --- 3. Legal Prompt Templates ---
qa_prompt = ChatPromptTemplate.from_template(
    "You are a helpful, precise legal AI assistant. Use the following context from the contract to answer the question.\n"
    "If you cannot find the answer, say 'I cannot find that information in the provided document.'\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

summary_prompt = ChatPromptTemplate.from_template(
    "You are an expert corporate legal counsel. Provide a clear, highly structured executive summary of the contract below. "
    "Highlight key dates, parties, and core arrangements.\n\nContext:\n{context}\n\nExecutive Summary:"
)

clause_prompt = ChatPromptTemplate.from_template(
    "You are an expert legal document analyst. Extract and list the core legal sections and clauses present in this context "
    "(e.g., Indemnification, Governing Law, Force Majeure, etc.).\n\nContext:\n{context}\n\nExtracted Clauses:"
)

risk_prompt = ChatPromptTemplate.from_template(
    "You are a critical legal risk assessor. Analyze the text and explicitly identify any hidden liabilities, high-risk elements, "
    "harsh penalty metrics, automatic renewals, or lopsided termination constraints.\n\nContext:\n{context}\n\nIdentified Risks:"
)

terms_prompt = ChatPromptTemplate.from_template(
    "Extract the most important defined legal terms (e.g., Disclosing Party, Indemnitee) along with their specific concise "
    "definitions from the text. Format as 'Term: Definition'.\n\nContext:\n{context}\n\nCore Defined Terms:"
)

simplification_prompt = ChatPromptTemplate.from_template(
    "You are a legal educator. Take the complex legal jargon (legalese) from the contract context and translate it "
    "into plain, straightforward English that an ordinary business owner would understand instantly.\n\nContext:\n{context}\n\nPlain English Breakdown:"
)

# --- 4. Streamlit UI Layout ---
st.title("⚖️ Team 4: Legal Document Assistant")
st.write("An AI-powered RAG application to analyze, parse, simplify, and audit complex legal contracts.")
st.write("---")

# Sidebar Configuration
st.sidebar.header("🔑 Authentication & Setup")
api_key_input = st.sidebar.text_input("Enter Groq API Key:", type="password", value=os.getenv("GROQ_API_KEY", ""))

if api_key_input:
    os.environ["GROQ_API_KEY"] = api_key_input
else:
    st.sidebar.warning("Please enter your Groq API Key to run the assistant.")

st.sidebar.write("---")
st.sidebar.header("📂 Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload a Contract (PDF)", type=["pdf"])

# Processing logic runs when a file is dropped in
if uploaded_file and api_key_input:
    # Save the file temporarily to pass to PyPDFLoader
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.sidebar.spinner("Parsing & Indexing Agreement..."):
        # Load and Chunk
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
        splits = text_splitter.split_documents(docs)
        
        # Build vector store
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        st.sidebar.success("Contract Fully Indexed!")

    # Clean up temporary file
    if os.path.exists(temp_pdf_path):
        os.remove(temp_pdf_path)

    # --- Feature Navigation Tabs ---
    st.write("### 🛠️ Select Analysis Feature")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 Interactive Chat", 
        "📝 Contract Summary", 
        "🔍 Clause Extraction", 
        "⚠️ Risk Assessment", 
        "📖 Defined Terms", 
        "💡 Plain English"
    ])

    # Feature 1: Interactive QA
    with tab1:
        st.subheader("💬 Ask the Contract Anything")
        user_query = st.text_input("Enter your question regarding specific clauses or parameters:")
        if user_query:
            qa_chain = create_rag_chain(qa_prompt, retriever)
            with st.spinner("Searching document background..."):
                answer = qa_chain.invoke(user_query)
            st.markdown("#### **Answer:**")
            st.write(answer)

    # Feature 2: Executive Summary
    with tab2:
        st.subheader("📝 Contract Summary & Executive Overview")
        if st.button("Generate Summary"):
            summary_chain = create_rag_chain(summary_prompt, retriever)
            with st.spinner("Synthesizing executive summary..."):
                summary = summary_chain.invoke("Provide an overview summary highlighting parties, dates, and terms.")
            st.markdown(summary)

    # Feature 3: Clause Extraction
    with tab3:
        st.subheader("🔍 Key Clause & Compliance Structural Extraction")
        if st.button("Extract Core Clauses"):
            clause_chain = create_rag_chain(clause_prompt, retriever)
            with st.spinner("Analyzing contract structure..."):
                clauses = clause_chain.invoke("Extract key operational boilerplate or core compliance clauses.")
            st.markdown(clauses)

    # Feature 4: Risk Identification
    with tab4:
        st.subheader("⚠️ Hidden Liabilities & Financial Risk Audit")
        if st.button("Run Risk Assessment"):
            risk_chain = create_rag_chain(risk_prompt, retriever)
            with st.spinner("Auditing contract text for hidden liabilities..."):
                risks = risk_chain.invoke("Scan for indemnity, high damages, automatic renewals, and severe penalty risks.")
            st.markdown(risks)

    # Feature 5: Defined Terms Extraction
    with tab5:
        st.subheader("📖 Core Defined Contract Entities")
        if st.button("Extract Core Definitions"):
            terms_chain = create_rag_chain(terms_prompt, retriever)
            with st.spinner("Compiling legal glossary definitions..."):
                terms = terms_chain.invoke("Extract definitions, key naming terms, or capital defined phrases.")
            st.markdown(terms)

    # Feature 6: Simplified Explanations
    with tab6:
        st.subheader("💡 Legalese to Plain English Translation")
        if st.button("Simplify Legal Jargon"):
            simplification_chain = create_rag_chain(simplification_prompt, retriever)
            with st.spinner("Translating legal phrases into standard English..."):
                simplifications = simplification_chain.invoke("Simplify complex legal jargon into straightforward summaries.")
            st.markdown(simplifications)

elif not uploaded_file:
    st.info("👈 Please upload a legal agreement PDF in the sidebar to initiate analysis features.")
