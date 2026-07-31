import torch
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import csv
import json
import os
import ollama

vector_index_path = r"C:\Users\abhin\OneDrive\Desktop\Enterprise_knowledge_agents\Output\vector_index.faiss"
metadata_store_path = r"C:\Users\abhin\OneDrive\Desktop\Enterprise_knowledge_agents\Output\metadata_store.json"
embedder_model_name = 'all-MiniLM-L6-v2'

sample_queries=["what is kubernetes","how do pods communicate","explain kubernetes namespaces","what is deployment","how does kubect1 work"]

#Benchmark-1
#Embedding Speed: CPU vs GPU    

def benchmark_embedding(texts,devices,batch_sizes):
    results=[]
    for device in devices:
        if device=="cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Skipping GPU benchmark.")
            continue
        for batch_size in batch_sizes:
            model=SentenceTransformer(embedder_model_name,device=device)
            model.encode(texts[:5],batch_size=batch_size)
            torch.cuda.synchronize() if device=="cuda" else None
            start=time.perf_counter()
            embeddings=model.encode(texts,batch_size=batch_size,convert_to_numpy=True,show_progress_bar=False)
            torch.cuda.synchronize() if device=="cuda" else None
            elapsed=time.perf_counter()-start

            if device=="cuda":
                memory_used_mb=torch.cuda.memory_allocated()/1e6
                memory_reserved_mb=torch.cuda.memory_reserved()/1e6
            else:
                memory_used_mb=0
                memory_reserved_mb=0

            chunks_per_sec=len(texts)/elapsed
            results.append({
                "benchmark":"embedding",
                "device":device,
                "batch_size":batch_size,
                "total chunks":len(texts),
                "elapsed sec":round(elapsed,3),
                "chunks/sec":round(chunks_per_sec,1),
                'gpu_memory_used_mb':round(memory_used_mb,1),
                'gpu_memory_reserved_mb':round(memory_reserved_mb,1)
            })

            print(f"[Embedding] Device:{device}, Batch size={batch_size}: {chunks_per_sec:.1f} chunks/sec | elapsed: {elapsed:.3f}s | GPU Memory Used: {memory_used_mb:.1f} MB")

            if device=="cuda":
                torch.cuda.empty_cache()

    return results

#Benchmark-2
#FAISS search latency

def benchmark_faiss(index, metadata_store, queries,embedding_model):


    results=[]
    for query in queries:
        query_vec=embedding_model.encode([query],convert_to_numpy=True).astype("float32").reshape(1,-1)

        start=time.perf_counter()
        distances, indices=index.search(query_vec,k=3)
        elapsed_ms=(time.perf_counter()-start)*1000

        results.append({"benchmark":"faiss_search",
            "query":query,
            "elapsed_ms":round(elapsed_ms,3),
            "top distance":round(float(distances[0][0]),4),
            "retrieed indices":indices[0].tolist()
        })

        print(f"[FAISS] '{query}':{elapsed_ms:.2f}ms | top distance={distances[0][0]:.4f}")

    return results

#Benchmark-3
# Ollama LLM Inference Latency

def benchmark_llm(queries):
    results=[]
    for query in queries:
        messages=[{"role":"system","content":"Answer in one sentence."},{"role":"user","content":query}]
        #cold start
        #Ollama first loads model on VRAM
        start_cold=time.perf_counter()
        response=ollama.chat(model="mistral",messages=messages)
        cold_start_sec=time.perf_counter()-start_cold
        output_text=response.message.content
        token_estimate=len(output_text.split())

        #warm call
        #Model loaded in VRAM
        start_warm=time.perf_counter()
        response=ollama.chat(model="mistral",messages=messages)
        warm_sec=(time.perf_counter())-start_warm
        output_text=response.message.content
        token_estimate_warm=len(output_text.split())

        results.append({
            "benchmark":"llm_interference",
            "query":query,
            "cold_start_sec":round(cold_start_sec,2),
            "warm_sec":round(warm_sec,2),
            "token_estimate_warm":token_estimate_warm,
            "tokens_per_sec_warm":round(token_estimate_warm/warm_sec,1)
        })

        print(f"[LLm] '{query}':cold={cold_start_sec:.2f}s | warm={warm_sec:.2f}s | ~{token_estimate_warm/warm_sec:.1f} tokens/sec")

    return results

#Benchmark-4
#Pipeline latency breakdown per stage

def benchmark_pipeline(index, metadata_store, queries, embedding_model):
    results=[]
    for query in queries:
        stages={}

        #query embedding

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start=time.perf_counter()
        query_vector=embedding_model.encode([query],convert_to_numpy=True).astype("float32").reshape(1,-1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stages["query_embedding_ms"]=(time.perf_counter()-t_start)*1000

        #FAISS embedding

        t_start=time.perf_counter()
        distances, indices=index.search(query_vector,k=3)
        stages["faiss_embedding_ms"]=(time.perf_counter()-t_start)*1000

        #metadata lookup

        t_start=time.perf_counter()
        retrieved_chunks=[]
        for idx in indices:
            chunk=metadata_store.get(str(idx),None)
            if chunk is not None:
                retrieved_chunks.append(chunk)
        stages["metadata_lookup_ms"]=(time.perf_counter()-t_start)*1000

        #LLM Generation

        content = "\n".join([c["content"] for c in retrieved_chunks if c is not None])
        messages=[{"role":"system", "content": f"Answer using only: {content}"},{"role":"user","content":query}]
        t_start=time.perf_counter()
        response=ollama.chat(model="mistral",messages=messages)
        stages["llm_generation_ms"]=(time.perf_counter()-t_start)*1000

        stages["total_ms"]=stages["query_embedding_ms"]+stages["faiss_embedding_ms"]+stages["metadata_lookup_ms"]+stages["llm_generation_ms"]
        stages["query"]=query

        for stage, values in stages.items():
            if stage!="query":
                print(f"{stage}: {values:.1f}ms")
        print(f"TOTAL: {stages["total_ms"]:.1f}ms")

        results.append(stages)

    return results

#Saving Results

def save_results(all_results, output_path=r"C:\Users\abhin\OneDrive\Desktop\Enterprise_knowledge_agents\Output\benchmark_results.csv"):
    flat_results=[]
    for result_grp in all_results:
        for result in result_grp:
            flat_results.append(result)

    if flat_results:
        fieldnames=[]
        for result in flat_results:
            fieldnames+=list(result.keys())
        with open(output_path,"w") as f:
            writer=csv.DictWriter(f,fieldnames=fieldnames)
            writer.writeheader()
            for row in flat_results:
                writer.writerow(row)

    print(f"Results saved to output file:{output_path}")

def main():
    print(f"Version of torch: {torch.__version__}")
    print(f"Check if CUDA is available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Capability: {props.major}.{props.minor}")
        print(f"Multiprocessors: {props.multi_processor_count}")

    index=faiss.read_index(vector_index_path)
    metadata_store=json.load(open(metadata_store_path,"r",encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(embedder_model_name, device=device)

    texts = [metadata_store[k]["content"] for k in metadata_store.keys()]   

    print("\n---Benchmark 1: Embedding---")
    embedding_results=benchmark_embedding(texts, devices=["cpu","cuda"],batch_sizes=[1,16,32,64])

    print("\n---Benchmark 2: FAISS---")
    faiss_results=benchmark_faiss(index, metadata_store, sample_queries, embedding_model)

    print("\n---Benchmark 3: LLM inference---")
    llm_results=benchmark_llm(sample_queries)

    print("\n---Benchmark 4: End-to-End Pipeline---")
    pipeline_results=benchmark_pipeline(index, metadata_store, sample_queries, embedding_model)

    save_results([embedding_results, faiss_results, llm_results, pipeline_results])

if __name__=="__main__":
    main()
