from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_openai import AzureChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from prepare import set_envvar
set_envvar()

# loader = PyPDFLoader("salt.pdf")
loader = PyPDFLoader("~/azure-chat-search-for-tmi/data/【走行画像_青緑看板】アノテーション仕様書.pdf")
# file_paths=[
    # "~/azure-chat-search-for-tmi/data/202311_オルソ画像アノテーション仕様書.pdf",
            # "~/azure-chat-search-for-tmi/data/【走行画像_青緑看板】アノテーション仕様書.pdf"]
# pages = []
# for file_path in file_paths:
#     loader = PyPDFLoader(file_path)
#     async for page in loader.alazy_load():
#         pages.append(page)

embed_model_id = "cl-nagoya/ruri-v3-310m"
embeddings = HuggingFaceEmbeddings(model_name=embed_model_id)
# embeddings = OllamaEmbeddings(base_url='http://192.168.151.48:11434',
#                               model="mxbai-embed-large")
# embeddings =OllamaEmbeddings(base_url="http://192.168.151.48:11434/api/embed",
#                         model="kun432/cl-nagoya-ruri-large",)
# llm = OllamaLLM(model="gemma2:27b")
# llm = OllamaLLM(base_url='http://192.168.151.48:11434', model="gemma3")

llm = AzureChatOpenAI(model_name="gpt-4o-mini", temperature=0)
index = VectorstoreIndexCreator(vectorstore_cls=Chroma,
                                embedding=embeddings).from_loaders([loader])
import IPython
IPython.embed()

query = '標識のアノテーションで囲う矩形サイズの条件はありますか？'#"しぼルトメニューとは何ですか?日本語で回答して下さい。"

answer = index.query(query, llm=llm)
print(answer)
