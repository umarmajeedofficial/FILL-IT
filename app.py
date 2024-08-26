import streamlit as st
import os
import requests
import pdfplumber
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Setup models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
whisper_model_id = "openai/whisper-medium"

# Load Whisper model and processor
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_id)
whisper_processor = AutoProcessor.from_pretrained(whisper_model_id)
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    device=device
)

# Retrieve secrets
granite_url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
granite_headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer eyJraWQiOiIyMDI0MDgwMzA4NDEiLCJhbGciOiJSUzI1NiJ9.eyJpYW1faWQiOiJJQk1pZC02OTQwMDBJTlNIIiwiaWQiOiJJQk1pZC02OTQwMDBJTlNIIiwicmVhbG1pZCI6IklCTWlkIiwianRpIjoiNzIxMTJlNWUtOTRhNC00MTY1LTk2ZDgtMTAxYTg0YjhlNmQxIiwiaWRlbnRpZmllciI6IjY5NDAwMElOU0giLCJnaXZlbl9uYW1lIjoiVW1hciIsImZhbWlseV9uYW1lIjoiTWFqZWVkIiwibmFtZSI6IlVtYXIgTWFqZWVkIiwiZW1haWwiOiJ1bWFybWFqZWVkb2ZmaWNpYWxAZ21haWwuY29tIiwic3ViIjoidW1hcm1hamVlZG9mZmljaWFsQGdtYWlsLmNvbSIsImF1dGhuIjp7InN1YiI6InVtYXJtYWplZWRvZmZpY2lhbEBnbWFpbC5jb20iLCJpYW1faWQiOiJJQk1pZC02OTQwMDBJTlNIIiwibmFtZSI6IlVtYXIgTWFqZWVkIiwiZ2l2ZW5fbmFtZSI6IlVtYXIiLCJmYW1pbHlfbmFtZSI6Ik1hamVlZCIsImVtYWlsIjoidW1hcm1hamVlZG9mZmljaWFsQGdtYWlsLmNvbSJ9LCJhY2NvdW50Ijp7InZhbGlkIjp0cnVlLCJic3MiOiIyZTY5MjI1ZjNmMjc0Nzc2ODkwMGE2MGQ5MDBkM2UzNyIsImltc191c2VyX2lkIjoiMTI2MjI5MTciLCJmcm96ZW4iOnRydWUsImltcyI6IjI3NDQzNDQifSwiaWF0IjoxNzI0NjM3ODUyLCJleHAiOjE3MjQ2NDE0NTIsImlzcyI6Imh0dHBzOi8vaWFtLmNsb3VkLmlibS5jb20vaWRlbnRpdHkiLCJncmFudF90eXBlIjoidXJuOmlibTpwYXJhbXM6b2F1dGg6Z3JhbnQtdHlwZTphcGlrZXkiLCJzY29wZSI6ImlibSBvcGVuaWQiLCJjbGllbnRfaWQiOiJkZWZhdWx0IiwiYWNyIjoxLCJhbXIiOlsicHdkIl19.ZKnoQjFyXxXRtsP5cMfv0H1Measiz3Wd5D1srfV4i4QLRwHy6rR6X8up-xNT-O9tccWNo2z5fhPaihz-5n_qPbGnM3-CfZemTr0d9PnbmgKLejsUy3EywPu3Q87J1bjeE2XY0Zm7Sjf9w-TCyUHeFmbBGruv60rzQXXuUd802YInpAcvKaD3_QzVGHtZQTqGmohSWTF8y879B0TfDFD3R3g8GSUchl5ith3qqUGms3IWy8-DRNdkn53M9qMeRrOLAI36v8J-kZdNXbPoG86DiFThvHTNSZj_Sbc6Iiu2N-J9T6ygKNVDH_1tcPJckfAoStVstGugm0i3spun5HsE6w",  # Replace with your secret key name
}

def transcribe_audio(file):
    result = whisper_pipe(file)
    return result['text']

def extract_text_from_pdf(pdf_file):
    text = ""
    questions = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
                questions += [line.strip() for line in page_text.split("\n") if line.strip()]
    return text, questions

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
        "project_id": st.secrets['project_id'],  # Replace with your secret key name
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

def save_responses_to_pdf(responses, output_pdf_path):
    document = SimpleDocTemplate(output_pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    number_style = ParagraphStyle(
        name='NumberedStyle',
        parent=styles['BodyText'],
        fontSize=10,
        spaceAfter=12
    )
    
    content = []
    
    for index, response in enumerate(responses, start=1):
        heading = Paragraph(f"<b>File {index}:</b>", styles['Heading2'])
        response_text = Paragraph(response.replace("\n", "<br/>"), number_style)
        
        content.append(heading)
        content.append(Spacer(1, 6))
        content.append(response_text)
        content.append(Spacer(1, 18))
    
    document.build(content)

# Streamlit UI
st.title("FILL IT")
st.markdown("Developed by Umar Majeed, Team Mixed Intelligence")

uploaded_audio_folder = st.file_uploader("Upload Audio Files", type=["wav", "mp3"], accept_multiple_files=True)
uploaded_pdf = st.file_uploader("Upload PDF with Questions", type=["pdf"])

if st.button("Process"):
    if uploaded_audio_folder and uploaded_pdf:
        responses = []
        pdf_text, pdf_questions = extract_text_from_pdf(uploaded_pdf)
        
        for audio_file in uploaded_audio_folder:
            transcribed_text = transcribe_audio(audio_file)
            form_data = generate_form_data(transcribed_text, pdf_questions)
            responses.append(form_data)
            st.write(f"File {len(responses)}:\n{form_data}\n")
        
        output_pdf_path = "/kaggle/working/response_output.pdf"
        save_responses_to_pdf(responses, output_pdf_path)
        
        st.markdown(f"Responses have been saved to [response_output.pdf]({output_pdf_path})")
        st.download_button(label="Download PDF", data=open(output_pdf_path, "rb").read(), file_name="response_output.pdf", mime="application/pdf")
    else:
        st.error("Please upload both audio files and the PDF.")
