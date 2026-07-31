import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import json
from langchain_community.document_loaders import PyPDFLoader
import faiss
import uuid

#LOADING DOCUMENTS

def loader(folder_path):
    docs=[]
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            pages=loader.load()
            for page in pages:
                docs.append({"content":page.page_content,"source":file,"page":page.metadata["page"]})
    return docs

#CHUNKING

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(docs,chunk_size=500,overlap=50):
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks=[]
    for doc in docs:
        doc_chunks=splitter.split_text(doc["content"])
        for chunk in doc_chunks:
            chunks.append({"content":chunk,"source":doc["source"],"page":doc["page"],"chunk_id":str(uuid.uuid4())})
    return chunks

#EMBEDDING

from sentence_transformers import SentenceTransformer


def embed_chunks(chunks, embedding_model):
    embeddings=[]
    texts=[c["content"] for c in chunks]
    vectors=embedding_model.encode(texts, batch_size=32, device="cuda",show_progress_bar=True, convert_to_numpy=True)
    return vectors

#VECTOR INDEXING

def build_indexing(chunks, embeddings):
    embeddings=embeddings.astype('float32')
    embedding_dim=embeddings.shape[1]
    index=faiss.IndexFlatL2(embedding_dim)
    metadata_store={str(i):{"content":chunks[i]["content"],"source":chunks[i]["source"],"page":chunks[i]["page"]} for i in range(len(chunks))}
    return index, metadata_store

#PERSIST TO DISK

def save_to_disk(index, metadata_store, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    faiss.write_index(index, os.path.join(output_folder, "vector_index.faiss"))
    with open(os.path.join(output_folder, "metadata_store.json"), "w") as f:
        json.dump(metadata_store, f)
        print(f"Saved metadata store to {os.path.join(output_folder, 'metadata_store.json')}")
        print(index.ntotal, "vectors saved to", os.path.join(output_folder, "vector_index.faiss"))

def main():
    folder_path = "C:\\Users\\abhin\\OneDrive\\Desktop\\Enterprise knowledge agents\\docs"   # Replace with your PDF folder path
    output_folder = "C:\\Users\\abhin\\OneDrive\\Desktop\\Enterprise knowledge agents\\Output"  # Replace with your desired output folder path  
    embedding_model_name = "all-MiniLM-L6-v2"  # You can change this to any other model

    print("Loading documents...")
    docs = loader(folder_path)
    print(f"Loaded {len(docs)} documents.")

    print("Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(embedding_model_name)

    print("Embedding chunks...")
    embeddings = embed_chunks(chunks, embedding_model)
    print(f"Generated embeddings for {len(embeddings)} chunks.")

    print("Building FAISS index...")
    index, metadata_store = build_indexing(chunks, embeddings)
    print("FAISS index built.")

    print("Saving to disk...")
    save_to_disk(index, metadata_store, output_folder)
    print("Process completed.")