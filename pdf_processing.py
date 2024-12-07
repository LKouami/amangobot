from rag_train_utils import *

PDF_FILE = "documents/basedata.pdf"
MODEL = "llama3.1"

def process_pdf():
    print("Chargement du PDF et création des chunks...")
    chunks = load_model_and_chunkPdf(PDF_FILE)
    
    print(f"Nombre de chunks: {len(chunks)}")
    
    print("Stockage des chunks dans la base vectorielle...")
    vectorstore = store_chunks_in_vector_store(MODEL, chunks)
    
    return vectorstore
