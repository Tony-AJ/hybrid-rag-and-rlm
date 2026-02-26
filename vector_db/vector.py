import os
import logging

# Suppress ChromaDB telemetry and logging
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('posthog').setLevel(logging.ERROR)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# ChromaDB batch size limit (using 5000 to be safe, max is ~5461)
BATCH_SIZE = 400

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    # List all text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path, encoding='utf-8')
        book_docs = loader.load()
        for doc in book_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            documents.append(doc)

    # Split the documents into chunks using RecursiveCharacterTextSplitter
    # for better handling of chunk boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, 
        chunk_overlap=40,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and add documents in batches
    print("\n--- Creating and persisting vector store ---")
    
    # Create an empty vector store first
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings
    )
    
    # Add documents in batches to avoid exceeding ChromaDB's batch size limit
    total_docs = len(docs)
    num_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    
    print(f"Processing {total_docs} documents in {num_batches} batch(es) of max {BATCH_SIZE} documents each")
    
    for i in range(0, total_docs, BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"Adding batch {batch_num}/{num_batches} ({len(batch)} documents)...")
        db.add_documents(batch)
        print(f"Batch {batch_num}/{num_batches} completed.")
    
    print("\n--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
