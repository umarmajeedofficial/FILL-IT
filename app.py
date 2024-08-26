import io
import os
import requests
import pdfplumber
import torch
import ffmpeg
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Define paths for temporary files
temp_audio_folder = "/tmp/audios/"
temp_pdf_path = "/tmp/uploaded_pdf.pdf"
temp_output_pdf_path = "/tmp/response_output.pdf"

# Ensure temporary directories exist
os.makedirs(temp_audio_folder, exist_ok=True)

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

# Granite model URL and headers
granite_url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
granite_headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer eyJraWQiOiIyMDI0MDgwMzA4NDEiLCJhbGciOiJSUzI1NiJ9.eyJpYW1faWQiOiJJQk1pZC02OTQwMDBJTlNIIiwiaWQiOiJJQk1pZC02OTQwMDBJTlNIIiwicmVhbG1pZCI6IklCTWlkIiwianRpIjoiNzIxMTJlNWUtOTRhNC00MTY1LTk2ZDgtMTAxYTg0YjhlNmQxIiwiaWRlbnRpZmllciI6IjY5NDAwMElOU0giLCJnaXZlbl9uYW1lIjoiVW1hciIsImZhbWlseV9uYW1lIjoiTWFqZWVkIiwibmFtZSI6IlVtYXIgTWFqZWVkIiwiZW1haWwiOiJ1bWFybWFqZWVkb2ZmaWNpYWxAZ21haWwuY29tIiwic3ViIjoidW1hcm1hamVlZG9mZmljaWFsQGdtYWlsLmNvbSIsImF1dGhuIjp7InN1YiI6InVtYXJtYWplZWRvZmZpY2lhbEBnbWFpbC5jb20iLCJpYW1faWQiOiJJQk1pZC02OTQwMDBJTlNIIiwibmFtZSI6IlVtYXIgTWFqZWVkIiwiZ2l2ZW5fbmFtZSI6IlVtYXIiLCJmYW1pbHlfbmFtZSI6Ik1hamVlZCIsImVtYWlsIjoidW1hcm1hamVlZG9mZmljaWFsQGdtYWlsLmNvbSJ9LCJhY2NvdW50Ijp7InZhbGlkIjp0cnVlLCJic3MiOiIyZTY5MjI1ZjNmMjc0Nzc2ODkwMGE2MGQ5MDBkM2UzNyIsImltc191c2VyX2lkIjoiMTI2MjI5MTciLCJmcm96ZW4iOnRydWUsImltcyI6IjI3NDQzNDQifSwiaWF0IjoxNzI0NjM3ODUyLCJleHAiOjE3MjQ2NDE0NTIsImlzcyI6Imh0dHBzOi8vaWFtLmNsb3VkLmlibS5jb20vaWRlbnRpdHkiLCJncmFudF90eXBlIjoidXJuOmlibTpwYXJhbXM6b2F1dGg6Z3JhbnQtdHlwZTphcGlrZXkiLCJzY29wZSI6ImlibSBvcGVuaWQiLCJjbGllbnRfaWQiOiJkZWZhdWx0IiwiYWNyIjoxLCJhbXIiOlsicHdkIl19.ZKnoQjFyXxXRtsP5cMfv0H1Measiz3Wd5D1srfV4i4QLRwHy6rR6X8up-xNT-O9tccWNo2z5fhPaihz-5n_qPbGnM3-CfZemTr0d9PnbmgKLejsUy3EywPu3Q87J1bjeE2XY0Zm7Sjf9w-TCyUHeFmbBGruv60rzQXXuUd802YInpAcvKaD3_QzVGHtZQTqGmohSWTF8y879B0TfDFD3R3g8GSUchl5ith3qqUGms3IWy8-DRNdkn53M9qMeRrOLAI36v8J-kZdNXbPoG86DiFThvHTNSZj_Sbc6Iiu2N-J9T6ygKNVDH_1tcPJckfAoStVstGugm0i3spun5HsE6w"  # Replace with your actual API key
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

# Set up the Streamlit app
st.title("FILL IT")

# Upload multiple audio files
uploaded_audios = st.file_uploader("Upload audio files", type=["wav", "mp3"], accept_multiple_files=True)

# Upload PDF file
uploaded_pdf = st.file_uploader("Upload a PDF file with questions", type=["pdf"])

# Output box to display responses
output_box = st.empty()

# Button to start processing
if st.button("Start Processing"):
    if uploaded_audios and uploaded_pdf:
        responses = []
        
        # Read uploaded PDF file
        pdf_bytes = uploaded_pdf.read()
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_bytes)

        # Process each uploaded audio file
        for audio_file in uploaded_audios:
            audio_bytes = audio_file.read()
            audio_path = os.path.join(temp_audio_folder, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            
            # Transcribe audio
            transcription = transcribe_audio(audio_path)
            
            # Extract text and questions from PDF
            pdf_text, questions = extract_text_from_pdf(temp_pdf_path)
            
            # Generate form data with Granite
            form_data = generate_form_data(transcription, questions)
            responses.append(form_data)
        
        # Display responses in output box
        output_box.write("Processing completed. Here are the results:")
        for index, response in enumerate(responses, start=1):
            output_box.write(f"File {index}:\n{response}\n")
        
        # Save responses to PDF
        save_responses_to_pdf(responses, temp_output_pdf_path)
        
        # Button to download the PDF with responses
        with open(temp_output_pdf_path, "rb") as f:
            st.download_button(
                label="Download Responses as PDF",
                data=f,
                file_name="response_output.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("Please upload both audio files and a PDF file.")
