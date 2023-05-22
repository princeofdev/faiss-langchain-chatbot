import os, dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate

# loads .env file with your OPENAI_API_KEY
dotenv.load_dotenv()

# CSV from https://gist.github.com/IvanCampos/94576c9746be280cf5b64083c8ea5b4d
loader = CSVLoader("midjourney-20230505.csv", csv_args = {"delimiter": ','})
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

faissIndex = FAISS.from_documents(docs, OpenAIEmbeddings())
faissIndex.save_local("faiss_midjourney_docs")

chatbot = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0, model_name="gpt-3.5-turbo", max_tokens=50
    ), 
    chain_type="stuff", 
    retriever=FAISS.load_local("faiss_midjourney_docs", OpenAIEmbeddings())
        .as_retriever(search_type="similarity", search_kwargs={"k":1})
)

template = """
respond as succinctly as possible. {query}?
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template=template,
)

print(chatbot.run(
    prompt.format(query="what is --v")
))
# --v is a parameter used to specify a specific model version in Midjourney's AI image generation tool.