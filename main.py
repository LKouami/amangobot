import os
import pickle
from pdf_processing import process_pdf
from chain_setup import setup_chain

VECTORSTORE_FILE = "vectorstore.pkl"

def get_vectorstore():
    """Retourne la base vectorielle, en la chargeant ou en la créant si nécessaire."""
    if not os.path.exists(VECTORSTORE_FILE):
        # Si la base n'existe pas, traiter le PDF et créer la base vectorielle
        vectorstore = process_pdf()
        # Sauvegarder la base vectorielle
        with open(VECTORSTORE_FILE, "wb") as f:
            pickle.dump(vectorstore, f)
    else:
        # Si la base existe, la charger depuis le fichier
        with open(VECTORSTORE_FILE, "rb") as f:
            vectorstore = pickle.load(f)
    
    # Configurer et retourner la chaîne
    chain = setup_chain(vectorstore)
    return chain
