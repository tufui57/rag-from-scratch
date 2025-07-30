from typing import Any
import glob
import os
import uuid
from pathlib import Path
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from ollama import Client
from prepare import *


### Install tesseract to run OCR on PDF files before running this script ###

def generate_image_summary(client, image_path):
    response = client.chat(
    model='gemma3',
    messages=[
        {
        'role': 'user',
        'content': 'Summarize this image',
        'images': [image_path],
        }
    ],
    )

    print(response.message.content)
    return response


def add_to_retriever(retriever, item_list: dict):
    id_key= retriever.id_key
    for k, i in item_list.items():
        if k=='texts' and len(i['summary'])>0:
            texts=i
        
            # Add texts
            doc_ids = [str(uuid.uuid4()) for _ in texts['texts']]
            summary_texts = [
                Document(page_content=s, metadata={id_key: doc_ids[i]})
                for i, s in enumerate(texts['summary'])
            ]
            retriever.vectorstore.add_documents(summary_texts)
            retriever.docstore.mset(list(zip(doc_ids, texts)))

        elif k=='tables' and len(i['summary'])>0:
            tables=i
            # Add tables
            table_ids = [str(uuid.uuid4()) for _ in tables['tables']]
            summary_tables = [
                Document(page_content=s, metadata={id_key: table_ids[i]})
                for i, s in enumerate(tables['summary'])
            ]
            retriever.vectorstore.add_documents(summary_tables)
            retriever.docstore.mset(list(zip(table_ids, tables)))

        elif k=='img' and len(i['summary'])>0:
            img=i
            # Add images
            img_ids = [str(uuid.uuid4()) for _ in img['summary']]
            summary_img = [
                Document(page_content=s, metadata={id_key: img_ids[i]})
                for i, s in enumerate(img['summary'])
            ]
            retriever.vectorstore.add_documents(summary_img)
            retriever.docstore.mset(
                list(zip(img_ids, img['summary']))
            )
        else:
            print(f'the item type {k} is not supported')
            print(f'{k}: {i}')

    return retriever


class Element(BaseModel):
    type: str
    text: Any

def main(
    query: str,
    input_pdf: str,
    # pdf_languages: list,
    pdf_language: str,
    ollama_endpoint: str,
    multimodal_llm_name: str,
    ):
    # Path to save images
    input_pdf_name=Path(input_pdf).stem
    path = f"./{input_pdf_name}_{multimodal_llm_name}"

    # Store the image summary as the raw document
    # Get elements
    raw_pdf_elements = partition_pdf(
        filename=input_pdf,
        # Using pdf format to find embedded image blocks
        extract_images_in_pdf=True,
        # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
        # Titles are any sub-section of the document
        infer_table_structure=True,
        # Post processing to aggregate text once we have the title
        chunking_strategy="by_title",
        # Chunking params to aggregate text blocks
        # Attempt to create a new chunk 3800 chars
        # Attempt to keep chunks > 2000 chars
        # Hard max on chunks
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        # image_output_dir_path=path,
        extract_image_block_output_dir=path,
        languages=[pdf_language]
    )

    # Create a dictionary to store counts of each type
    category_counts = {}

    for element in raw_pdf_elements:
        category = str(type(element))
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1


    # Categorize by type
    categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))

    # Tables
    table_elements = [e for e in categorized_elements if e.type == "table"]

    # Text
    text_elements = [e for e in categorized_elements if e.type == "text"]

    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and text. \
    Give a concise summary of the table or text. Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model=llm
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Apply to text
    texts = [i.text for i in text_elements if i.text != ""]
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    # Apply to tables
    tables = [i.text for i in table_elements]
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})


    # img_dir="/home/nomuram/rag-from-scratch/figures"
    img_dir=path

    if multimodal_llm_name=='LLaVA':
        client = Client(
            host=ollama_endpoint,
        )
    elif multimodal_llm_name=='azure-gpt':
        from azure_multimodal_model_local_img import generate_summary

    # Loop through each image in the directory
    for img in Path(img_dir).glob("*.jpg"):
        # Extract the base name of the image without extension
        base_name=Path(img).stem

        # Define the output file name based on the image name
        output_file=f"{img_dir}/{base_name}.txt"

        # Execute the command and save the output to the defined output file\

        if multimodal_llm_name=='LLaVA':
            
            if not Path(output_file).exists():
                response=generate_image_summary(client, img)
                with open(output_file, 'w') as f:
                    f.writelines(response.message.content)
            else:
                print(f'{output_file} exists. Skip generating image summary')
        elif multimodal_llm_name=='azure-gpt':
                        
            if not Path(output_file).exists():
                response=generate_summary(img, language=pdf_language[0])
                with open(output_file, 'w') as f:
                    f.writelines(response['choices'][0]['message']['content'])
            else:
                print(f'{output_file} exists. Skip generating image summary')

    # Get all .txt files in the directory
    file_paths = glob.glob(os.path.expanduser(os.path.join(img_dir, "*.txt")))

    # Read each file and store its content in a list
    img_summaries = []
    for file_path in file_paths:
        with open(file_path, "r") as file:
            img_summaries.append(file.read())

    # Clean up residual logging
    cleaned_img_summary = []
    for s in img_summaries:
        for preface in ["Here's a summary of the image you sent:\n\n", "Here\'s a summary of the image:\n\n"]:
            if preface in s:
                cleaned_img_summary.append(
                    s.split(preface, 1)[1].strip())
            else:
                cleaned_img_summary.append(s)

    # The vectorstore to use to index the child chunks
    if 'English' in pdf_language:
        vectorstore = Chroma(
            collection_name="summaries", embedding_function=GPT4AllEmbeddings()
        )
    if 'Japanese' in pdf_language:
        embed_model_id = "cl-nagoya/ruri-v3-310m"
        embd= HuggingFaceEmbeddings(model_name=embed_model_id)
        vectorstore = Chroma(collection_name="summaries",
                            embedding_function=embd)
    
    # The storage layer for the parent documents
    store = InMemoryStore()  # <- Can we extend this to images
    id_key = "doc_id"

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    item_list= {
        'texts': {
            'texts': texts,
            'summary': text_summaries
            },
        'tables':{
            'tables': tables,
            'summary': table_summaries
            },
        'img':{'img': [], 'summary': cleaned_img_summary}
    }

    retriever = add_to_retriever(retriever, item_list)



    # Prompt template
    template = """Answer the question based only on the following context, which can include text and tables:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Option 1: LLM
    # model = ChatOllama(model="llama2:13b-chat")
    model=llm
    # Option 2: Multi-modal LLM
    # model = LLaVA

    # RAG pipeline
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    import IPython
    IPython.embed()
    return chain.invoke(query)


if __name__== "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--query', type=str, default="Explain any images / figures in the paper with playful and creative examples.", help='Query string')
    parser.add_argument('--input_pdf', type=str, default="/home/nomuram/rag-from-scratch/LLaVA/2304.08485v2.pdf", help='Input PDF file path')
    # parser.add_argument('--pdf_languages', type=list, default=['English'], help='List of languages in the PDF')
    parser.add_argument('--pdf_language', type=str, default='English', help='language in the PDF')
    parser.add_argument('--ollama_endpoint', type=str, default='http://192.168.151.48:11434', help='Ollama endpoint URL')
    parser.add_argument('--multimodal_llm_name', type=str, default='LLaVA', 
                        help='Name of the multimodal LLM. Choose one of the followings; azure-gpt or LLaVA')

    
    # Update the main function call to pass the parsed arguments
    args = parser.parse_args()
    res = main(**vars(args))
    print(res)