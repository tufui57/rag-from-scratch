import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# You can download models from the Hugging Face Hub 🤗 as follows:
tokenizer = AutoTokenizer.from_pretrained("pfnet/plamo-embedding-1b", trust_remote_code=True)
model = AutoModel.from_pretrained("pfnet/plamo-embedding-1b", trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

query = "PLaMo-Embedding-1Bとは何ですか？"
documents = [
    "PLaMo-Embedding-1Bは、Preferred Networks, Inc. によって開発された日本語テキスト埋め込みモデルです。",
    "最近は随分と暖かくなりましたね。"
]

with torch.inference_mode():
    # For embedding query texts in information retrieval, please use the `encode_query` method.
    # You also need to pass the `tokenizer`.
    query_embedding = model.encode_query(query, tokenizer)
    # For other texts/sentences, please use the `encode_document` method.
    # Also, for applications other than information retrieval, please use the `encode_document` method.
    document_embeddings = model.encode_document(documents, tokenizer)

# The similarity between vectors obtained by inputting sentences into the model is high for similar sentences and low for dissimilar sentences.
# This feature can be utilized for applications such as information retrieval.
similarities = F.cosine_similarity(query_embedding, document_embeddings)
print(similarities)