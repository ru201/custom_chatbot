# python -m pip install
import os
import gradio

from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
import constants

# Enter APIKEY in constants.py file
os.environ["OPENAI_API_KEY"] = constants.APIKEY

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def customChatBot(query):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

    loader = TextLoader('data.txt')
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )   

    response = ""
    for chunk in rag_chain.stream(query):
        response += chunk
    
    return response

demo = gradio.Interface(fn=customChatBot, inputs="text", outputs = "text", title = "Digital Chatbot")

demo.launch(share=True)
