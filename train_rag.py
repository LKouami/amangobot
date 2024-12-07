from rag_train_utils import *

# Initialisation des variables globales
PDF_FILE = "documents/basedata.pdf"
MODEL = "llama3.1"
test_question = "Hello, fait un résumé en 10 mots de ce que tu sais"



# Chargement initial (ne sera fait qu'une seule fois)
print("Chargement du PDF et création des chunks...")
chunks = load_model_and_chunkPdf(PDF_FILE)

print("Stockage des chunks dans la base vectorielle...")
vectorstore = store_chunks_in_vector_store(MODEL, chunks)

print("Configuration du retriever...")
retriever = setting_up_retriever(vectorstore, test_question)

print("Configuration du modèle personnel...")
personal_model = configure_the_model(MODEL, test_question)

print("Préparation du prompt...")
# Template pour le prompt
my_bot_prompt_template = """
You are TogoBot, an assistant that provides answers to questions based on a given context. 
Introduce yourself in less than 5 words.
Answer the question in French based on the context. 
If you don't know the answer, say something cool like 
"Uhmm.. Interesting ! I don't know I will search".
Be as concise as possible and go straight to the point 
in French.

Context: {context}

Question: {question}
"""
prompt = setting_up_prompt(my_bot_prompt_template)

def create_chain():
    """Crée et retourne la chaîne avec les objets déjà chargés en mémoire."""
    print("Création du chain avec les objets déjà en mémoire...")
    chain = add_retriever_to_the_chain(retriever, prompt, personal_model)
    return chain

# Cette partie s'exécutera uniquement si le fichier est exécuté directement
if __name__ == "__main__":
    # Ne fait que créer le chain une fois pour vérifier que tout fonctionne
    chain = create_chain()
    print("Le chain a été créé avec succès.")
