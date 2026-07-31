import numpy as np
import os
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import ollama

index_path=r"C:\Users\abhin\OneDrive\Desktop\Enterprise_knowledge_agents\Output\vector_index.faiss"
metadata_path=r"C:\Users\abhin\OneDrive\Desktop\Enterprise_knowledge_agents\Output\metadata_store.json"

def load_index(index_path, metadata_path):
    print(f"Loading index from {index_path} and metadata from {metadata_path}...")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file {index_path} does not exist.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file {metadata_path} does not exist.")
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_store = json.load(f)
    if index and metadata_store:
        print("Index and metadata loaded successfully.")
    return index, metadata_store
    


def embed_query(query, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])
    if query_embedding is not None:
        print("Query embedding generated successfully.")
    return query_embedding

def search_index(index, query_embedding, metadata_store, k=3):
    print("Searching the index for relevant documents...")
    query_embedding = np.array(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding, k)
    results = []
    for i in range(len(indices[0])):
        doc_id = str(indices[0][i])
        distance = distances[0][i]
        if doc_id in metadata_store:
            results.append((distance, metadata_store[doc_id]))
    if results:
        print(f"Found {len(results)} relevant documents.")
    print("Distance and indices:", distances, indices)
    return results

def rewrite_query(query):
    print("Rewriting the query for better context...")
    response = ollama.chat(model="mistral", messages=[{"role": "system", "content": """You are a search query optimizer for technical documentation retrieval.
Rewrite the user's question as a keyword-rich search phrase.
Include relevant technical terms, synonyms, and related concepts.
Output ONLY the rewritten query. No explanation. No punctuation. Maximum 12 words."""},{"role": "user", "content": query}])
    rewritten_query = response.message.content
    if rewritten_query:
        print("Query rewritten successfully.")
        print("Rewritten query:", rewritten_query)
    return rewritten_query

def build_prompt_and_query(results, rewritten_query):
    print("Building prompt for the language model...")
    prompt= """You are an enterprise knowledge assistant. 
Answer the user's question using ONLY the context provided below.
Do not add information from outside the provided context.
Do not start your response with phrases like 'You've provided' or 'As you mentioned'.
If the context does not contain enough information to answer, say: 
'I could not find sufficient information in the knowledge base.'"""
    prompt += f"Query: {rewritten_query}\n\n"
    prompt += "Top relevant documents:\n"
    for i, (distance, metadata) in enumerate(results):
        prompt += f"{i+1}. Source: {metadata['source']}, Page: {metadata['page']}\nContent: {metadata['content']}\nDistance: {distance}\n\n"
    response= ollama.chat(model="mistral", messages=[{"role": "system", "content": prompt},{"role": "user", "content": rewritten_query}])
    answer= response.message.content
    if answer:
        print("Answer generated successfully.")
    return answer

def cited_answer(answer, results):
    print("Compiling cited answer with sources...")
    cited_answer = answer + "\n\nCited Sources:\n"
    for i, (distance, metadata) in enumerate(results):
        cited_answer += f"{i+1}. Source: {metadata['source']}, Page: {metadata['page']}\n"
    if cited_answer:
        print("Cited answer compiled successfully.")
    return cited_answer

def main():
    index, metadata_store = load_index(index_path, metadata_path)
    query = input("Enter your query: ")
    rewritten_query = rewrite_query(query)
    query_embedding = embed_query(rewritten_query)
    results = search_index(index, query_embedding, metadata_store)
    if all(distance > 0.6 for distance, _ in results):
        print("Low confidence retrieval. Using original query instead.")
        query_embedding = embed_query(query)
        results = search_index(index, query_embedding, metadata_store)
    if not results:
        print("No relevant documents found.")
        return
    answer = build_prompt_and_query(results, rewritten_query)
    final_answer = cited_answer(answer, results)
    print(final_answer)

if __name__ == "__main__":
    main()