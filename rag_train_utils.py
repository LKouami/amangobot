
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from operator import itemgetter


# PDF_FILE = "basedata.pdf"
# MODEL = "llama3.1"



def load_model_and_chunkPdf(PDF_FILE):
    # Loading the PDF document
    loader = PyPDFLoader(PDF_FILE)
    pages = loader.load()

    print(f"Number of pages: {len(pages)}")
    print(f"Length of a page: {len(pages[1].page_content)}")
    print("Content of a page:", pages[1].page_content)
    
    # Splitting the pages in chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)

    chunks = splitter.split_documents(pages)
    print(f"Number of chunks: {len(chunks)}")
    print(f"Length of a chunk: {len(chunks[1].page_content)}")
    print("Content of a chunk:", chunks[1].page_content)
    return chunks

def store_chunks_in_vector_store(MODEL, chunks):

    embeddings = OllamaEmbeddings(model=MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def setting_up_retriever(vectorstore):
    retriever = vectorstore.as_retriever()
    # retriever.invoke(question)
    return retriever

def configure_the_model(MODEL):
    model = ChatOllama(model=MODEL, temperature=0)
    return model

def parsing_models_response(model, question):
    parser = StrOutputParser()
    chain = model | parser 
    print(chain.invoke(question))

def setting_up_prompt(template):
    prompt = PromptTemplate.from_template(template)
    return prompt

def add_retriever_to_the_chain(retriever, prompt, model):
    parser = StrOutputParser()
    chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | parser
    )
    return chain

def answer_question(question, chain):
    try:
        answer = chain.invoke({'question': question})
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("*************************\n")
        return answer
    except AttributeError as e:
        print(f"Erreur : {e}")
        print(f"Vérifie que le 'chain' est bien un objet et non une chaîne : {chain}")