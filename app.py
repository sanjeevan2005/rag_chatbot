import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings  # Optional alternative embeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()  # Use OpenAI embeddings
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Uncomment if needed
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key="answer"  # still required here too
    )

    return conversation_chain



def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})

    answer = response.get("answer", "")
    source_docs = response.get("source_documents", [])
    st.session_state.chat_history = response.get('chat_history', [])

    for message in st.session_state.chat_history:
        if message.type == "human":
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        elif message.type == "ai":
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    # Remove duplicate rendering of the same bot answer
    # Now just display the source documents
    if source_docs:
        with st.expander("Show source text used for the answer"):
            for i, doc in enumerate(source_docs):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)  # Load custom CSS for styling

    # Maintain state across Streamlit reruns
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")

    user_question = st.text_input("Enter your question here:")
    if user_question and st.session_state.conversation:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process PDFs") and pdf_docs:
            with st.spinner("Processing PDFs..."):
                # Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                # Chunk the text
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success("PDFs processed successfully!")


if __name__ == "__main__":
    main()
