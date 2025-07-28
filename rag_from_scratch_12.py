from prepare import llm
from argparse import ArgumentParser
import uuid
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.storage import InMemoryByteStore
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_huggingface import HuggingFaceEmbeddings


### Part 12: Multi-representation Indexing
def main(query):
    print("query:", query)
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    docs = loader.load()

    loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
    docs.extend(loader.load())


    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | llm
        | StrOutputParser()
    )

    summaries = chain.batch(docs, {"max_concurrency": 5})

    # The vectorstore to use to index the child chunks
    embed_model_id = "cl-nagoya/ruri-v3-310m"
    embd= HuggingFaceEmbeddings(model_name=embed_model_id)
    vectorstore = Chroma(collection_name="summaries",
                        embedding_function=embd)

    # The storage layer for the parent documents
    store = InMemoryByteStore()
    id_key = "doc_id"

    # The retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )
    doc_ids = [str(uuid.uuid4()) for _ in docs]

    # Docs linked to summaries
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    # Add
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))
    sub_docs = vectorstore.similarity_search(query,k=1)

    retrieved_docs = retriever.get_relevant_documents(query,n_results=1)
    
    return retrieved_docs, sub_docs


if __name__ == "__main__":
    parser=ArgumentParser()
    parser.add_argument("--query", type=str, default= "Memory in agents")
    args=parser.parse_args()
    
    res1, res2=main(**vars(args))
    print("sub_docs:", res2[0].page_content[0:500])
    print("retrieved_docs:", res1[0].page_content[0:500])