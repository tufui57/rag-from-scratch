import os
import torch
from typing import List
from transformers import AutoModel, AutoTokenizer
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from credentials import *

def set_envvar():
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY
    os.environ['OPENAI_API_VERSION']=OPENAI_API_VERSION
    os.environ['AZURE_OPENAI_ENDPOINT']=AZURE_OPENAI_ENDPOINT
    os.environ['AZURE_OPENAI_API_KEY']=AZURE_OPENAI_API_KEY 

def get_vectorstore(splits, embed_model_id = "cl-nagoya/ruri-v3-310m"):
    embd= HuggingFaceEmbeddings(model_name=embed_model_id)
    return Chroma.from_documents(documents=splits, 
                                    embedding=embd,)

set_envvar()
llm=AzureChatOpenAI(model_name="gpt-4o-mini", temperature=0)

class CustomEmbeddings:
    def __init__(self,):
    
        # You can download models from the Hugging Face Hub 🤗 as follows:
        self.tokenizer = AutoTokenizer.from_pretrained("pfnet/plamo-embedding-1b", trust_remote_code=True)
        model = AutoModel.from_pretrained("pfnet/plamo-embedding-1b", trust_remote_code=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with torch.inference_mode():
            document_embeddings = self.model.encode_document(texts, self.tokenizer)
        return document_embeddings
            
    def embed_query(self, query: str) -> List[float]:
        with torch.inference_mode():
            # For embedding query texts in information retrieval, please use the `encode_query` method.
            # You also need to pass the `tokenizer`.
            query_embedding = self.model.encode_query(query, self.tokenizer)
        return query_embedding

from sentence_transformers import SentenceTransformer
class MyEmbeddings:
    def __init__(self, model:str):
        self.model = SentenceTransformer(model, trust_remote_code=True)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]
    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query]).tolist()
    
# embed_model_id = "cl-nagoya/ruri-v3-310m"
# embeddings = HuggingFaceEmbeddings(model_name=embed_model_id)