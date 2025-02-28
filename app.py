import os
import gradio as gr
import requests
import io
import re
from PIL import Image
from groq import Groq

# Set Your API Keys
#  Use environment variables securely
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

if not GROQ_API_KEY or not HF_API_KEY:
    raise ValueError("GROQ_API_KEY and HF_TOKEN must be set in the environment variables.")
# Initialize Groq API client
client = Groq(api_key=GROQ_API_KEY)

# Use a Public Hugging Face Image Model
HF_IMAGE_MODEL = "stabilityai/stable-diffusion-2-1"


# Function 1: Tamil Audio to Tamil Text (Transcription)
def transcribe_audio(audio_path):
    if not audio_path:
        return "Error: Please upload an audio file."

    try:
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), file.read()),
                model="whisper-large-v3",
                language="ta",  # Tamil
                response_format="verbose_json",
            )
        return transcription.text.strip()

    except Exception as e:
        return f"Error in transcription: {str(e)}"


# Function 2: Tamil Text to English Translation
def translate_tamil_to_english(tamil_text):
    if not tamil_text:
        return "Error: Please enter Tamil text for translation."

    prompt = f"Translate this Tamil text to English: {tamil_text}\nGive only the translated text as output."

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Groq-supported model
            messages=[{"role": "user", "content": prompt}],
        )
        translated_text = response.choices[0].message.content.strip()

        #  Fix: Remove unwanted XML tags like <think></think>
        translated_text = re.sub(r"</?think>", "", translated_text).strip()
        return translated_text

    except Exception as e:
        return f"Error in translation: {str(e)}"


# Function 3: English Text to Image Generation (Hugging Face)
def generate_image(english_text):
    if not english_text:
        return "Error: Please enter a description for image generation."

    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": english_text}

        response = requests.post(f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}",
                                 headers=headers, json=payload)
        response.raise_for_status()
        image_bytes = response.content

        #  Check if the response is a valid image
        if not image_bytes:
            return "Error: Received empty response from API."

        return Image.open(io.BytesIO(image_bytes))

    except Exception as e:
        return f"Error in image generation: {str(e)}"


# Function 4: English Text to AI-Generated Text

def generate_text(english_text):
    if not english_text:
        return "Please enter a prompt."

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  #  Ensure you're using a valid model
            messages=[{"role": "user", "content": english_text}],
        )

        # Extract the response content
        generated_text = response.choices[0].message.content.strip()

        # Remove unwanted XML-like tags
        cleaned_text = re.sub(r"</?think>", "", generated_text).strip()

        return cleaned_text

    except Exception as e:
        return f"Error in text generation: {str(e)}"


# Combined Function to Process All Steps
def process_audio(audio_path):
    # Step 1: Tamil Audio → Tamil Text
    tamil_text = transcribe_audio(audio_path)
    if "Error" in tamil_text:
        return tamil_text, None, None, None

    # Step 2: Tamil Text → English Text
    english_text = translate_tamil_to_english(tamil_text)
    if "Error" in english_text:
        return tamil_text, english_text, None, None

    # Step 3: English Text → Image
    image = generate_image(english_text)
    if isinstance(image, str) and "Error" in image:
        return tamil_text, english_text, None, None

    # Step 4: English Text → AI-Generated Text
    generated_text = generate_text(english_text)
    return tamil_text, english_text, image, generated_text


# Create Gradio Interface
iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Upload Tamil Audio"),
    outputs=[
        gr.Textbox(label="Transcribed Tamil Text"),
        gr.Textbox(label="Translated English Text"),
        gr.Image(label="Generated Image"),
        gr.Textbox(label="Generated Text from English Prompt"),
    ],
    title="Tamil Audio to AI Processing Pipeline",
    description="Upload a Tamil audio file and get transcription, translation, image generation, and further text generation.",
)

# Launch Gradio App
iface.launch()
