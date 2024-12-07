from fastapi import FastAPI, Request
import requests
import os
from requests.auth import HTTPBasicAuth
import whisper
from gtts import gTTS
from twilio.rest import Client
from utils import *
from main import get_vectorstore
from rag_train_utils import answer_question
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
app = FastAPI()

# Les informations du compte personnel Twilio
TWILIO_ACCOUNT_SID = 'ACa0c72dd7cab1dee91533b0c3522cb9eb'
TWILIO_AUTH_TOKEN = '7e0e1d25500144a05c6be7f505c7feff'
TWILIO_PHONE_NUMBER = 'whatsapp:+14155238886'

# Créer un client Twilio
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    print("Requête reçue sur /webhook")
    
    try:
        data = await request.form()
        print(f"Données reçues : {data}")
    except Exception as e:
        print(f"Erreur lors de l'extraction des données : {e}")
        return {"message": "Erreur lors de l'extraction des données"}

    media_url = data.get('MediaUrl0')
    media_type = data.get('MediaContentType0')
    
    print(f"media_url: {media_url}, media_type: {media_type}")
    
    if media_url and media_type.startswith('audio'):
        if 'ogg' in media_type:
            file_extension = 'ogg'
        elif 'wav' in media_type:
            file_extension = 'wav'
        elif 'amr' in media_type:
            file_extension = 'amr'
        else:
            file_extension = 'audio'
        
        # Générer un nom unique pour le fichier audio reçu
        audio_file = generate_unique_filename("received_voice_note", file_extension)
        
        print(f"Téléchargement de la note vocale depuis : {media_url}")
        response = requests.get(media_url, auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        
        if response.status_code == 200:
            print(f"Fichier téléchargé avec succès, taille : {len(response.content)} octets")
            
            try:
                with open(audio_file, "wb") as f:
                    f.write(response.content)
                print(f"Fichier sauvegardé sous : {audio_file}")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde du fichier : {e}")
                return {"message": "Erreur lors de la sauvegarde du fichier"}
            
            file_size = os.path.getsize(audio_file)
            if file_size > 0:
                print(f"Le fichier téléchargé est valide, taille : {file_size} octets")
                
                # Générer un nom unique pour le fichier MP3 converti
                mp3_file = generate_unique_filename("converted_voice_note", "mp3")
                convert_to_mp3(audio_file, mp3_file)

                # Transcription avec Whisper
                model = whisper.load_model("base")
                result = model.transcribe(mp3_file, fp16=False)
                print("VOCI EN TEXTE CE QUE TU AS DIT")
                
                print(result['text'])
                


                print("J'AI COMMENCER PAR GENERER LA REPONSE")

              
                chain = get_vectorstore()

                # Répondre à la question
                answer = answer_question(result['text'], chain)



                print("J'AI FINI")






                # Générer un nom unique pour la réponse vocale MP3
                response_mp3_file = generate_unique_filename("response", "mp3")
                # Transcription du Text en audio
                tts = gTTS(answer, lang="fr")
                tts.save(response_mp3_file)

                # Télécharger le fichier MP3 sur S3
                s3_url = upload_to_s3(response_mp3_file, BUCKET_NAME, response_mp3_file)
                if not s3_url:
                    return {"message": "Erreur lors du téléchargement sur S3"}
                
                print(f"L'URL du fichier téléchargé sur S3 est : {s3_url}")

                # Envoyer la réponse vocale via WhatsApp
                message = twilio_client.messages.create(
                    body="Voici votre réponse vocale.",
                    from_=TWILIO_PHONE_NUMBER,
                    to=data.get('From'),
                    media_url=[s3_url]
                )

                print(f"Message envoyé avec succès, SID : {message.sid}")

                return {"message": "Note vocale reçue, sauvegardée, convertie et renvoyée."}
            else:
                print("Le fichier téléchargé est vide")
                return {"message": "Le fichier téléchargé est vide"}
        else:
            print(f"Échec du téléchargement du fichier audio, code : {response.status_code}")
            return {"message": f"Échec du téléchargement du fichier audio : {response.status_code}"}

    return {"message": "Pas de note vocale trouvée"}
