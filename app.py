import streamlit as st
import zipfile
import tempfile
import requests
import pdfplumber
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import warnings
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
whisper_model_id = "openai/whisper-medium"

# Load Whisper model and processor
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_id)
whisper_processor = AutoProcessor.from_pretrained(whisper_model_id)

# Create Whisper pipeline
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    device=device
)

# IBM Granite API URL and Headers
granite_url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
granite_headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer eyJraWQiOiIyMDI0MDgwMzA4NDEiLCJhbGciOiJSUzI1NiJ9.eyJpYW1faWQiOiJJQk1pZC02OTQwMDBJTlNIIiwiaWQiOiJJQk1pZC02OTQwMDBJTlNIIiwicmVhbG1pZCI6IklCTWlkIiwianRpIjoiMTEyYTZjZTUtZDk1NS00OWI5LTlmYTgtZjk5OTRhYzdlNDkxIiwiaWRlbnRpZmllciI6IjY5NDAwMElOU0giLCJnaXZlbl9uYW1lIjoiVW1hciIsImZhbWlseV9uYW1lIjoiTWFqZWVkIiwibmFtZSI6IlVtYXIgTWFqZWVkIiwiZW1haWwiOiJ1bWFybWFqZWVkb2ZmaWNpYWxAZ21haWwuY29tIiwic3ViIjoidW1hcm1hamVlZG9mZmljaWFsQGdtYWlsLmNvbSIsImF1dGhuIjp7InN1YiI6InVtYXJtYWplZWRvZmZpY2lhbEBnbWFpbC5jb20iLCJpYW1faWQiOiJJQk1pZC02OTQwMDBJTlNIIiwibmFtZSI6IlVtYXIgTWFqZWVkIiwiZ2l2ZW5fbmFtZSI6IlVtYXIiLCJmYW1pbHlfbmFtZSI6Ik1hamVlZCIsImVtYWlsIjoidW1hcm1hamVlZG9mZmljaWFsQGdtYWlsLmNvbSJ9LCJhY2NvdW50Ijp7InZhbGlkIjp0cnVlLCJic3MiOiIyZTY5MjI1ZjNmMjc0Nzc2ODkwMGE2MGQ5MDBkM2UzNyIsImltc191c2VyX2lkIjoiMTI2MjI5MTciLCJmcm96ZW4iOnRydWUsImltcyI6IjI3NDQzNDQifSwiaWF0IjoxNzI0NjcyMDA5LCJleHAiOjE3MjQ2NzU2MDksImlzcyI6Imh0dHBzOi8vaWFtLmNsb3VkLmlibS5jb20vaWRlbnRpdHkiLCJncmFudF90eXBlIjoidXJuOmlibTpwYXJhbXM6b2F1dGg6Z3JhbnQtdHlwZTphcGlrZXkiLCJzY29wZSI6ImlibSBvcGVuaWQiLCJjbGllbnRfaWQiOiJkZWZhdWx0IiwiYWNyIjoxLCJhbXIiOlsicHdkIl19.XD3Pi8o1y-DoQt4ZvwD-8eXTVKIWxVT69LhayfosgtE9qsUce5NTI2NQusoamonUCgf-h2eYM48-Wa9SD2QKb3SUkcxrzm6Lsq2sRhcg1NroioGwCODdv1CFXUGaFB9TSTHQf1qEE3qXwaanT94b5aW1SXngHuh3gUwlenYeQU1SwZz3Pu9xMIC_vk-Ds3v8hD_---PkHo9W7VZtCMxPDDERLM3IN_O8OjUcP2jzaA3HeRzVDf_QC-w-7KZ05hRRv_u_fbmE-49k3o1tbkaT4FOovYJ7gEQ61iDcy1ojCbBZhB9QUdl99ip2tt6Gj0YqYfDBWfGi4BJ3HtcEA0N9_w"  # Replace with your actual API key
}

# Function to transcribe audio files
def transcribe_audio(file_path):
    result = whisper_pipe(file_path)
    return result['text']

# Function to extract text and questions from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    questions = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
                questions += [line.strip() for line in page_text.split("\n") if line.strip()]
    return text, questions

# Function to generate form data with Granite
def generate_form_data(text, questions):
    question_list = "\n".join(f"- {question}" for question in questions)
    body = {
        "input": f"""The following text is a transcript from an audio recording. Read the text and extract the information needed to fill out the following form.\n\nText: {text}\n\nForm Questions:\n{question_list}\n\nExtracted Form Data:""",
        "parameters": {
            "decoding_method": "sample",
            "max_new_tokens": 900,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 1,
            "repetition_penalty": 1.05
        },
        "model_id": "ibm/granite-13b-chat-v2",
        "project_id": "698f0da7-6b34-4642-8540-978e70e85c8e",  # Replace with your actual project ID
        "moderations": {
            "hap": {
                "input": {
                    "enabled": True,
                    "threshold": 0.5,
                    "mask": {"remove_entity_value": True}
                },
                "output": {
                    "enabled": True,
                    "threshold": 0.5,
                    "mask": {"remove_entity_value": True}
                }
            }
        }
    }
    response = requests.post(granite_url, headers=granite_headers, json=body)
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))
    data = response.json()
    return data['results'][0]['generated_text'].strip()

# Function to save responses to PDF
def save_responses_to_pdf(responses, output_pdf_path):
    document = SimpleDocTemplate(output_pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom style for numbered responses
    number_style = ParagraphStyle(
        name='NumberedStyle',
        parent=styles['BodyText'],
        fontSize=10,
        spaceAfter=12
    )
    
    content = []
    
    for index, response in enumerate(responses, start=1):
        # Add the response number and content
        heading = Paragraph(f"<b>File {index}:</b>", styles['Heading2'])
        response_text = Paragraph(response.replace("\n", "<br/>"), number_style)
        
        content.append(heading)
        content.append(Spacer(1, 6))  # Space between heading and response
        content.append(response_text)
        content.append(Spacer(1, 18))  # Space between responses
    
    document.build(content)

# Streamlit Interface
st.title("Audio to Form Filling")
st.write("Upload a ZIP file containing audio files and a PDF form. The audio will be transcribed, and the form will be filled out based on the content.")

# File uploader widgets
zip_file = st.file_uploader("Upload ZIP File with Audio Files", type="zip")
pdf_file = st.file_uploader("Upload PDF Form", type="pdf")

if st.button("Process"):
    if zip_file and pdf_file:
        # Extract the ZIP file
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)

            responses = []
            # Process each audio file in the extracted directory
            for filename in os.listdir(tmp_dir):
                if filename.endswith((".wav", ".mp3")):
                    file_path = os.path.join(tmp_dir, filename)
                    # Transcribe audio
                    transcribed_text = transcribe_audio(file_path)
                    # Extract text and form fields from PDF
                    pdf_text, pdf_questions = extract_text_from_pdf(pdf_file)
                    # Generate form data
                    form_data = generate_form_data(transcribed_text, pdf_questions)
                    responses.append(form_data)
                    st.write(f"File {len(responses)}:\n{form_data}\n")  # Print the extracted form data with numbering

            # Save all responses to a PDF
            output_pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
            save_responses_to_pdf(responses, output_pdf_path)

            # Display download link for the generated PDF
            with open(output_pdf_path, "rb") as file:
                st.download_button("Download Processed PDF", file, file_name="processed_output.pdf")
    else:
        st.error("Please upload both a ZIP file and a PDF form.")
