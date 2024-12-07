from rag_train_utils import *

def setup_chain(vectorstore):
    MODEL = "llama3.1"
    my_bot_prompt_template = """
    You are TogoBot, an assistant that provides answers to questions based on a given context. Introduce yourself in less than 5 words.

    Answer the question in French based on the context. If you don't know the answer, say something cool like "Uhmm.. Interesting ! I don't know I will search".

    Be as concise as possible and go straight to the point in French.

    Context: {context}

    Question: {question}
    """

    retriever = setting_up_retriever(vectorstore)
    personal_model = configure_the_model(MODEL)
    prompt = setting_up_prompt(my_bot_prompt_template)
    
    chain = add_retriever_to_the_chain(retriever, prompt, personal_model)
    return chain
