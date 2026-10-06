#*************rag question answer*******
# from langchain_ollama import OllamaEmbeddings, OllamaLLM
# from langchain_chroma import Chroma

# # Embeddings
# embeddings = OllamaEmbeddings(model="nomic-embed-text")

# # Load Chroma database
# vectorstore = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=embeddings
# )

# # LLM
# llm = OllamaLLM(model="llama3.2:latest")

# # User question
# query = "What is the required attendance?"

# # Retrieve relevant information
# results = vectorstore.similarity_search(query, k=2)

# context = "\n\n".join(doc.page_content for doc in results)

# # Create prompt
# prompt = f"""
# Answer the question using only the information given below.

# College Information:
# {context}

# Question:
# {query}

# Answer:
# """

# # Generate answer
# answer = llm.invoke(prompt)

# print("\nAnswer:")
# print(answer)

#*******************
import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma

st.set_page_config(
    page_title="College Knowledge Navigator",
    page_icon="🎓"
)

st.title("🎓 College Knowledge Navigator")
st.write("Ask questions about your college information.")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Clear history button
if st.button("🗑️ Clear History"):
    st.session_state.chat_history = []
    st.rerun()

# Load embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Load Chroma database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Load Llama
llm = OllamaLLM(
    model="llama3.2:latest"
)

# Question box
query = st.text_input("Ask your question:")

if query:
    # Retrieve relevant information
    results = vectorstore.similarity_search(query, k=2)

    context = "\n\n".join(
        doc.page_content for doc in results
    )

    # Prompt
    prompt = f"""
    Answer the question using only the college information below.

    College Information:
    {context}

    Question:
    {query}

    If the answer is not available in the college information,
    say "I could not find this information in the college knowledge base."

    Answer:
    """

    answer = llm.invoke(prompt)

     # Save conversation
    st.session_state.chat_history.append(
        {
            "question": query,
            "answer": answer
        }
    )

    st.subheader("Answer")
    st.write(answer)

# Display chat history
if st.session_state.chat_history:
    st.subheader("📜 Chat History")

    for i, chat in enumerate(
        st.session_state.chat_history, 1
    ):
        st.markdown(f"**Q{i}: {chat['question']}**")
        st.write(chat["answer"])
        st.divider()