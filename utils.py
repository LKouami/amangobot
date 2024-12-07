import requests
import os
import subprocess
import uuid
import boto3



# Crée un client S3
s3_client = boto3.client('s3')
BUCKET_NAME = 'amangobotbucket'


def generate_unique_filename(base_name, extension):
    unique_id = uuid.uuid4().hex  # Générer un identifiant unique
    return f"{base_name}_{unique_id}.{extension}"

def convert_to_mp3(input_file, output_file):
    command = [
        'ffmpeg', '-i', input_file, 
        '-codec:a', 'libmp3lame', 
        '-q:a', '2', output_file
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Conversion réussie : {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de la conversion : {e}")
        print(f"FFmpeg output: {e.stdout}")
        print(f"FFmpeg error: {e.stderr}")

def upload_to_s3(file_path, bucket_name, object_name):
    try:
        s3_client.upload_file(file_path, bucket_name, object_name,  ExtraArgs={'ContentType': 'audio/mpeg'})
        print(f"Fichier {file_path} téléchargé avec succès sur S3")
        return f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
    except Exception as e:
        print(f"Erreur lors du téléchargement sur S3 : {e}")
        return None

