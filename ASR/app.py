from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="http://127.0.0.1:5500")


# model = tf.keras.models.load_model("saved_model/my_model")

def CTCLoss (y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


with tf.keras.utils.custom_object_scope({'CTCLoss': CTCLoss}):
    # Load the model and optimizer state from the checkpoint file
    model = tf.keras.models.load_model("saved_model/my_model")
# opt = keras.optimizers.Adam(learning_rate=1e-4)
# loss=CTCLoss
# model.compile(optimizer=opt, loss=loss)


characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

frame_length = 256
frame_step = 160
fft_length = 384



def encode_single_sample(wave_file):
    req_file = tf.io.read_file(wave_file)
    audio, _ = tf.audio.decode_wav(req_file, desired_channels=1)
    print(audio,_)
    print(tf.math.count_nonzero(audio))
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)

    print(audio)
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

        # Preprocess input
        X = encode_single_sample(temp_file_path)

        # Get Predictions 
        batch_predictions = model.predict(X)

        # Decode the predictions
        batch_predictions = decode_batch_predictions(batch_predictions)

        transcript_list = []
        transcript_list.extend(batch_predictions)

        # Clean up the temporary file
        os.remove(temp_file_path)

        # Return the predictions as a JSON response
        text = ' '.join(transcript_list)
        return jsonify({"predictions": text, "list":transcript_list})

    except Exception as e:
        print("HERE")
        os.remove(temp_file_path)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
