# -*- coding: utf-8 -*-
"""Speech_to_text.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dhc1A6PLpC6GT0sBF1_x2x7gaoOYB5g3
"""

# Importing all required libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from IPython import display
import pandas as pd
import numpy as np
from jiwer import wer

# Getting data
data_url = 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
data_path= keras.utils.get_file("LJSpeech-1.1", data_url, untar=True)
print(data_path)

wavs_path = data_path + "/wavs/"
metadata_path = data_path + "/metadata.csv"

# Making Pandas Dataframe
metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
print(metadata_df.head())

# Naming coloumns
metadata_df.columns=['file_name', 'normal', 'transcription']
metadata_df.drop(columns='normal', inplace=True)
print(metadata_df.head())

# Splitting into train and validation dataframes
split = int(len(metadata_df) * 0.90)
df_train = metadata_df[:split]
df_val = metadata_df[split:]
print(f"Size of the training set: {len(df_train)}")
print(f"Size of the training set: {len(df_val)}")

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

print(f"The vocabulary is: {char_to_num.get_vocabulary()}")
print(f"size ={char_to_num.vocabulary_size()}")


# Defining parameters for input processing
frame_length = 256
frame_step = 160
fft_length = 384

# Preprocessing a single sample
def encode_single_sample(wave_file, label):

    # Get .wav file and convert to tensor
    file = tf.io.read_file(wavs_path + wave_file + ".wav")
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)

    # Perform short term fourier transform
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    # Encode labels (Transcriptions)
    label = tf.strings.lower(label)
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    label = char_to_num(label)

    return spectrogram, label

# Definig batch size and training and validation datasets
batch_size=32
train_dataset = tf.data.Dataset.from_tensor_slices(
(list(df_train["file_name"]), list(df_train["transcription"]))
)
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


validation_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_val["file_name"]), list(df_val["transcription"]))
)
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Plotting samples
fig = plt.figure(figsize= (8, 5))
for batch in train_dataset.take(1):
    spectrogram = batch[0][0].numpy()
    spectrogram = np.array( [np.trim_zeros(x) for x in np. transpose(spectrogram)])
    label = batch [1][0]
    # Spectrogram
    label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
    ax = plt.subplot(2, 1, 1)
    ax.imshow(spectrogram, vmax=1)
    ax.set_title(label)
    ax.axis("off")
    file = tf.io.read_file(wavs_path + list(df_train["file_name"])[0] + ".wav")
    audio,_ = tf.audio.decode_wav(file)
    audio = audio.numpy()
    ax = plt.subplot (2, 1, 2)
    plt.plot (audio)
    ax.set_title("Signal Wave")
    ax.set_xlim(0, len(audio))
    display.display(display.Audio(np.transpose (audio), rate=16000))
plt.show()


# Defining loss function
def CTCLoss (y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape (y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones (shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# Building the model
def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    input_spectrogram = keras.layers.Input((None, input_dim), name="input")
    x = keras.layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)

    # First Conv layer
    x = keras.layers.Conv2D(32,kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False,name="conv_1")(x)
    x = keras.layers.BatchNormalization(name="conv_1_bn")(x)
    x = keras.layers.ReLU(name="conv_1_relu")(x)

    # Second Conv layer
    x = keras.layers.Conv2D(32,kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False,name="conv_2")(x)
    x = keras.layers.BatchNormalization(name="conv_2_bn")(x)
    x = keras.layers.ReLU(name="conv_2_relu")(x)

    # Reshape for RNN layer
    x = keras.layers.Reshape((-1,x.shape[-2] * x.shape[-1]))(x)

    # RNN Layers
    for i in range(1, rnn_layers + 1):
        recurrent = keras.layers.GRU(
            units = rnn_units,
            activation = 'tanh',
            recurrent_activation = "sigmoid",
            use_bias = True,
            return_sequences = True,
            reset_after = True,
            name=f"gru{i}"
        )
        x = keras.layers.Bidirectional(recurrent, name=f"bidirectional_{i}", merge_mode="concat")(x)
        if i < rnn_layers:
            x = keras.layers.Dropout(rate=0.5)(x)
    # Dense Layer
    x = keras.layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x = keras.layers.ReLU(name = "dense_1_relu")(x)
    x = keras.layers.Dropout(rate=0.5)(x)

    # Classification layer
    output = keras.layers.Dense(units=output_dim + 1, activation="softmax" )(x)

    # Model
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")

    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=1e-4)

    # Compiling Model
    model.compile(optimizer=opt, loss=CTCLoss)
    return model

model = build_model(
input_dim = fft_length// 2 + 1, output_dim=char_to_num.vocabulary_size(),
rnn_units=512,)

print(model.summary(line_length=110))


# Decoding Predictions
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

# Definig callback eval function
class CallbackEval (keras.callbacks.Callback):
    def __init__ (self, dataset):
        super().__init__()
        self.dataset = dataset
    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8"))
                targets.append(label)

        wer_score = wer(targets, predictions)
        print("."* 100)
        print(f"Word Error Rate: {wer_score: .4}")
        print("." * 100)
        for i in np.random.randint(0, len(predictions), 5):
            print(f"Target :{targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-"* 100)


epochs = 50

validation_callback = CallbackEval(validation_dataset)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoint.h5',
    save_weights_only=False,
    save_freq='epoch'
)
# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[validation_callback]
)
# Save model
model.save('saved_model/my_model')

# Some testing predictions done on Validation Dataset
predictions = []
targets = []

for batch in validation_dataset:
  X,y = batch
  batch_predictions = model.predict(X)
  batch_predictions = decode_batch_predictions(batch_predictions)
  predictions.extend(batch_predictions)
  for label in y:
    label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
    targets.append(label)
  wer_score = wer(targets, predictions)
  print(predictions)
  print("."* 100)
  print(f"Word Error Rate: {wer_score: .4}")
  print("." * 100)
  for i in np.random.randint(0, len(predictions), 5):
            print(f"Target :{targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-"* 100)