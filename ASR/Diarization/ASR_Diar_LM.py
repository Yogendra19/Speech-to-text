from pyannote.audio import Pipeline
from pydub import AudioSegment
from transformers import pipeline
import os
import torch
from flask import Flask, request, jsonify
import time
import datetime, pytz
import warnings
from language_tool_python import LanguageTool
warnings.filterwarnings(action="ignore")


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def correct_grammar(text):
    tool = LanguageTool('en-US')
    corrected_text = tool.correct(text)
    return corrected_text


asr_pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-medium",
  tokenizer="openai/whisper-medium",
  chunk_length_s=30,
  device=device,
)


pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="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")

def diarization_rttm_prep(file_path):
    # apply the pipeline to an audio file
    diarization = pipeline(file_path)
    # dump the diarization output to disk using RTTM format
    with open("audio.rttm", "w") as rttm:
        diarization.write_rttm(rttm)

def split_and_transcribe(input_audio, rttm_file):
    def split_audio(input_audio, rttm_file):
        audio = AudioSegment.from_file(input_audio)

        segments = []

        with open(rttm_file, 'r') as rttm:
            for line in rttm: 
                parts = line.strip().split()
                if len(parts) >= 5:
                    speaker = parts[7]
                    start_time = float(parts[3]) * 1000  # Convert to milliseconds
                    duration = float(parts[4]) * 1000     # Convert to milliseconds

                    segment = audio[start_time:start_time + duration]
                    segments.append((segment, speaker))

        return segments

    audio_segments = split_audio(input_audio, rttm_file)

    transcriptions = []

    for segment, speaker in audio_segments:
        segment.export('temp.wav', format="wav")
        result = asr_pipe('temp.wav')['text']
        result = correct_grammar(result)
        transcriptions.append((speaker, result))
    os.remove('temp.wav')
    return transcriptions





app = Flask(__name__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


@app.route('/convert', methods=['POST'])
def predict():
    try:
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        
        # Retrieve additional metadata from the client
        metadata = request.form.get('metadata', None)

        # Check if the file is empty
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Check if the file has a valid extension (e.g., .wav)
        if not file.filename.endswith('.wav'):
            return jsonify({"error": "Invalid file format, only .wav files are allowed"}), 400
        
        # Get current date
        current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))

        # Save the uploaded file to a temporary directory
        temp_file_path = "incoming.wav"
        file.save(temp_file_path)

        # To check processing time
        start = time.time()

        # Get transcriptions 
        diarization_rttm_prep(temp_file_path)
        transcription = split_and_transcribe(temp_file_path, 'audio.rttm')

        end = time.time()
        processing_time = end-start

        # Clean up the temporary file
        os.remove(temp_file_path)
        response_data = {
            "Transcription": transcription,
            "Metadata": metadata,  # Include the retrieved metadata in the response
            "Processing Time (in Seconds)": processing_time,
            "Time-stamp": current_time
        }

        return jsonify(response_data)
    except Exception as e:
        print("HERE")
        os.remove(temp_file_path)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
