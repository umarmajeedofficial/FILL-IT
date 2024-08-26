import streamlit as st
from pydub import AudioSegment
import numpy as np
from transformers import pipeline
from PyPDF2 import PdfReader
import io
from fpdf import FPDF

# Initialize Whisper pipeline
whisper_pipe = pipeline(model="openai/whisper-large")

# Function to convert uploaded file to numpy array
def uploaded_file_to_numpy(uploaded_file):
    # Load the audio file with pydub
    audio = AudioSegment.from_file(uploaded_file)
    # Ensure audio is mono and at 16kHz sampling rate
    audio = audio.set_channels(1).set_frame_rate(16000)
    # Convert audio to numpy array
    samples = np.array(audio.get_array_of_samples())
    return samples.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

# Function to transcribe audio files
def transcribe_audio(uploaded_file):
    audio_np = uploaded_file_to_numpy(uploaded_file)
    # Process audio with whisper_pipe
    result = whisper_pipe(audio_np)
    return result['text']

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    pdf_text = ""
    pdf_questions = []
    
    # Extract text from each page
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    
    # For simplicity, assume questions are lines starting with a question mark
    pdf_questions = [line.strip() for line in pdf_text.split('\n') if '?' in line]
    
    return pdf_text, pdf_questions

# Function to generate form data (placeholder example)
def generate_form_data(transcribed_text, pdf_questions):
    # Simple example of filling out form with transcribed text
    form_data = {}
    for question in pdf_questions:
        form_data[question] = transcribed_text  # Placeholder: Fill all questions with the same text
    return form_data

# Function to save responses to PDF
def save_responses_to_pdf(responses, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size = 12)
    
    for index, response in enumerate(responses, start=1):
        pdf.cell(200, 10, txt = f"File {index}:", ln = True)
        for question, answer in response.items():
            pdf.cell(200, 10, txt = f"{question}: {answer}", ln = True)
        pdf.ln()
    
    pdf.output(filename)

# Streamlit app code
st.title("FILL IT: By Umar Majeed")

audio_files = st.file_uploader("Upload audio files", type=["wav", "mp3"], accept_multiple_files=True)
pdf_file = st.file_uploader("Upload PDF form", type="pdf")

if audio_files and pdf_file:
    st.write("Processing...")
    
    responses = []
    for audio_file in audio_files:
        # Transcribe audio file
        transcribed_text = transcribe_audio(audio_file)
        # Process PDF and generate form data
        pdf_text, pdf_questions = extract_text_from_pdf(pdf_file)
        form_data = generate_form_data(transcribed_text, pdf_questions)
        responses.append(form_data)
    
    # Display results
    st.write("Extracted Form Data:")
    for index, response in enumerate(responses, start=1):
        st.write(f"File {index}:")
        st.write(response)
    
    # Save responses to PDF and provide download link
    output_pdf_filename = "response_output.pdf"
    save_responses_to_pdf(responses, output_pdf_filename)
    with open(output_pdf_filename, "rb") as file:
        st.download_button("Download Responses PDF", data=file.read(), file_name=output_pdf_filename)
