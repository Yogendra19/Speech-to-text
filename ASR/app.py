from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="http://127.0.0.1:5500")
# Load your trained model and any other required functions here
# Replace 'your_model_path' with the actual path to your saved model
# model = tf.keras.models.load_model("saved_model/my_model")
model = 5

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

frame_length = 256
frame_step = 160
fft_length = 384


def encode_single_sample(wave_file):
    file = tf.io.read_file(wave_file)
    audio, _ = tf.audio.decode_wav(file)
    print(audio)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)


    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    return spectrogram




def decode_batch_predictions(pred):

    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 401

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 402

        # Check if the file has a valid extension (e.g., .wav)
        if not file.filename.endswith('.wav'):
            return jsonify({"error": "Invalid file format, only .wav files are allowed"}), 403

        # Save the uploaded file to a temporary directory
        temp_file_path = "/tmp/temp_file.wav"
        file.save(temp_file_path)

        # Preprocess input
        X = encode_single_sample(temp_file_path)
        print("Insert model now, everthing else seems to be working......Maybe")

        
        # Get Predictions 
        predictions = model.predict(X)

        # Decode the predictions
        batch_predictions = decode_batch_predictions(predictions)

        # Clean up the temporary file
        os.remove(temp_file_path)

        # Return the predictions as a JSON response
        # return jsonify({"predictions": batch_predictions})

        return jsonify({"predictions": batch_predictions})

    except Exception as e:
        print("HERE")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
