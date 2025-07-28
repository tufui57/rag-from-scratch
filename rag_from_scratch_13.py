from argparse import ArgumentParser
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough

from prepare import llm
from raptor_util import *

def get_docs():
    # LCEL docs
    url = "https://python.langchain.com/docs/expression_language/"
    loader = RecursiveUrlLoader(
        url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs = loader.load()

    # LCEL w/ PydanticOutputParser (outside the primary LCEL docs)
    url = "https://python.langchain.com/docs/how_to/output_parser_structured/"
    loader = RecursiveUrlLoader(
        url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs_pydantic = loader.load()

    # LCEL w/ Self Query (outside the primary LCEL docs)
    url = "https://python.langchain.com/docs/how_to/self_query/"
    loader = RecursiveUrlLoader(
        url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs_sq = loader.load()

    # Doc texts
    docs.extend([*docs_pydantic, *docs_sq])
    docs_texts = [d.page_content for d in docs]
    return docs, docs_texts

def split_contents(concatenated_content):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    # Doc texts split
    chunk_size_tok = 2000
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tok, chunk_overlap=0
    )
    texts_split = text_splitter.split_text(concatenated_content)
    return texts_split

### Part 13: https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb
def main(question):
    print("question:", question)

    docs, leaf_texts=get_docs()
    # Doc texts concat
    d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [doc.page_content for doc in d_reversed]
    )
    print(
        "Num tokens in all context: %s"
        % num_tokens_from_string(concatenated_content, "cl100k_base")
    )

    embed_model_id = "cl-nagoya/ruri-v3-310m"
    embd= HuggingFaceEmbeddings(model_name=embed_model_id)

    # Build tree
    # leaf_texts = docs_texts
    results = recursive_embed_cluster_summarize(leaf_texts, llm, embd, level=1, n_levels=3)

    # Initialize all_texts with leaf_texts
    all_texts = leaf_texts.copy()

    # Iterate through the results to extract summaries from each level and add them to all_texts
    for level in sorted(results.keys()):
        # Extract summaries from the current level's DataFrame
        summaries = results[level][1]["summaries"].tolist()
        # Extend all_texts with the summaries from the current level
        all_texts.extend(summaries)

    # Now, use all_texts to build the vectorstore with Chroma
    vectorstore = Chroma.from_texts(texts=all_texts, embedding=embd)
    retriever = vectorstore.as_retriever()

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")


    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Question
    return rag_chain.invoke(question)


if __name__ == "__main__":
    parser=ArgumentParser()
    parser.add_argument("--question", type=str, default="How to define a RAG chain? Give me a specific code example.")
    args=parser.parse_args()
    
    res =main(**vars(args))
    print(res)
