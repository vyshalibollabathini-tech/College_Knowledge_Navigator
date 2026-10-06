#------Testing the Working of Python----------
from pathlib import Path

file_path = Path("data/college_info.txt")

text = file_path.read_text(encoding="utf-8")

print(text)


#---------------------------Document Loaded Successfully-------------------------------
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/college_info.txt", encoding="utf-8")

documents = loader.load()

print("Document loaded successfully!")
print("Number of documents:", len(documents))
print(documents[0].page_content)


#***********Creating Chunks*********************************
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("data/college_info.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Document loaded successfully!")
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)

#**************************Creating Embedding and vector database******************
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Load document
loader = TextLoader("data/college_info.txt", encoding="utf-8")
documents = loader.load()

# Split document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Document loaded successfully!")
print("Number of chunks:", len(chunks))

# Create embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Store embeddings in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Embeddings created successfully!")
print("Chroma database created successfully!")