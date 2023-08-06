from flask import Flask, request, jsonify
from transformers import Wav2Vec2ForCTC,Wav2Vec2Processor
import torch
import torchaudio
import os



app = Flask(__name__)

# Get Gpu/Cpu
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load pretrained model
wav2vec2_model_name = "facebook/wav2vec2-large-960h-lv60-self" # pretrained 1.26GB

# Load processor and model
wav2vec2_processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_name)
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model_name).to(device)


def load_audio(audio_path):
    """Load the audio file & convert to 16,000 sampling rate

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        torch.Tensor: The mono audio if the input audio is mono. If the input audio is stereo,
        returns a list containing two mono audio tensors (left channel and right channel).
    """
    speech, sr = torchaudio.load(audio_path)
    resampler = torchaudio.transforms.Resample(sr, 16000)

    # Check if the audio is stereo or mono
    if len(speech.shape) > 1 and speech.shape[0] == 2:
        # If stereo, split into left and right channels
        left_channel = speech[0]
        right_channel = speech[1]
        left_channel = resampler(left_channel)
        right_channel = resampler(right_channel)
        return [left_channel.squeeze(), right_channel.squeeze()]
    else:
        # If mono, resample and return the mono audio
        speech = resampler(speech)
        return [speech.squeeze()]


# Get the transcription of audio file
def get_transcription_wav2vec2(audio_path, model, processor, chunk_size=10):
    """
    Args:
        audio_path (str): Path to the audio file.
        model: Pretrained Wav2Vec2 model.
        processor: Wav2Vec2 processor.
        chunk_size (int): Size of audio chunks in seconds for processing.

    Returns:
        str or dict: The transcription of the audio. If the input audio is stereo, it returns a dictionary with
        transcriptions for left and right channels. If the input audio is mono, it returns the transcription as a string.
    """
    def process_chunk(chunk):
        # Check if the chunk is long enough for processing (avoiding kernel size > chunk size)
        if len(chunk) < chunk_size * 16000:
            return ""

        input_features = processor(chunk, return_tensors="pt", sampling_rate=16000, padding=True, max_length=chunk_size*16000)["input_values"].to(device)
        logits = model(input_features)["logits"]
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription.lower()

    # Loading audio
    speech = load_audio(audio_path)
    if len(speech) != 2:
        # If mono
        speech = speech[0]
        return process_chunk(speech)

    else:
        # If stereo, process each channel separately and return transcriptions as a dictionary
        split_transcription = {}
        c = 0
        for channel in speech:
            channel_length = len(channel) / 16000  # Calculate channel length in seconds
            num_chunks = int(channel_length / chunk_size) + 1  # Calculate number of chunks needed
            channel_transcription = ""
            for i in range(num_chunks):
                start = int(i * chunk_size * 16000)
                end = int(min((i + 1) * chunk_size * 16000, len(channel)))

                chunk = channel[start:end]
                channel_transcription += process_chunk(chunk)
            split_transcription['left' if c == 0 else 'right'] = channel_transcription
            c += 1

        return split_transcription


@app.route('/convert', methods=['POST'])
def predict():
    try:
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Check if the file has a valid extension (e.g., .wav)
        if not file.filename.endswith('.wav'):
            return jsonify({"error": "Invalid file format, only .wav files are allowed"}), 400

        # Save the uploaded file to a temporary directory
        temp_file_path = "incoming.wav"
        file.save(temp_file_path)

        transcript_list = get_transcription_wav2vec2(temp_file_path,wav2vec2_model, wav2vec2_processor)

        # Clean up the temporary file
        os.remove(temp_file_path)

        return jsonify({"Transcription":transcript_list})

    except Exception as e:
        print("HERE")
        os.remove(temp_file_path)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)