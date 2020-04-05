import requests
import re
import numpy as np
import tensorflow as tf
import os


# import Meditations by Marcus Aurelius
response = requests.get('http://classics.mit.edu/Antoninus/meditations.mb.txt')
data = response.text
del response


# settings
SEQ_LEN = 200  # maximum sequence length we want for training
BUFFER_SIZE = 10000  # buffer size for shuffling the dataset
EPOCHS = 128  # number of times we iterate over the full dataset during training
RNN = 'LSTM'  # whether we use LSTM or GRU RNN units
UNITS = 1024  # how many units we use
BATCH_SIZE = 64  # no. sequences of SEQ_LEN we train on before updating weights
EMBED_DIM = 256  # vector dimension of character vector embeddings
PRINT = 100000  # how many characters we print during text generation

# remove everything before and including "Translated by George Long"
data = data.split('Translated by George Long')[1]

# remove "----" lines, as "-" is not a useful character we will remove it completely
data = data.replace('-', '')

# remove "BOOK ..." lines, for this we use regular expressions
data = re.sub('BOOK [A-Z]+\n', '', data)

# remove "THE END" and all that follows it
data = data.split("THE END")[0]

vocab_char = sorted(set(data))  # character level vocab
print(f'{len(vocab_char)} unique characters found')

# splitting by newline characters
data = data.split('\n\n')

# remove empty samples
empty = lambda x: x.replace('\s+', '') != ''
data = list(filter(empty, data))

# remove final '\n' characters
data = list(map(lambda x: x.replace('\n', ' '), data))

print(f"We have {len(data)} stoic lessons from Marcus Aurelius")

# now join back together in full text
data = '\n'.join(map(lambda x: x.strip(), data))  # we also use map to strip each paragraph

char2idx = {u:i for i, u in enumerate(vocab_char)}
idx2char = np.array(vocab_char)

data_idx = np.array([char2idx[c] for c in data])

# max length sentence we want
seq_length = SEQ_LEN
examples_per_epoch = len(data) // (seq_length+1)

# create training data
char_dataset = tf.data.Dataset.from_tensor_slices(data_idx)

# batch method allows conversion of individual characters to sequences of a desired size
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# for each sequence, we duplicate and shift it to form the input and target text
def split_input_output(chunk):
    input_data = chunk[:-1]
    target_data = chunk[1:]
    return input_data, target_data

dataset = sequences.map(split_input_output)

# shuffle the dataset
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# model build function
def build_model(vocab_size, embed_dim, rnn_units, batch_size):
    if RNN == 'GRU':
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embed_dim,
                                    batch_input_shape=[batch_size, None]),
            tf.keras.layers.GRU(rnn_units, return_sequences=True,
                                stateful=True,
                                dropout=0.1),
            tf.keras.layers.Dense(vocab_size)
        ])  # TODO try dropout=0.1
    elif RNN == 'LSTM':
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embed_dim,
                                    batch_input_shape=[batch_size, None]),
            tf.keras.layers.LSTM(rnn_units, return_sequences=True,
                                 stateful=True,
                                 dropout=0.1),
            tf.keras.layers.Dense(vocab_size)
        ])  # TODO try dropout=0.1
    return model


# build model
model = build_model(
    vocab_size=len(vocab_char),
    embed_dim=EMBED_DIM,
    rnn_units=UNITS,
    batch_size=BATCH_SIZE)

# print model summary
print(model.summary())

# define the loss function
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# compile the model
model.compile(optimizer='adam', loss=loss)

# configure checkpoints
checkpoint_dir = './training_checkpoints'
# name of files
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

# execute training
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# restore final checkpoint
tf.train.latest_checkpoint(checkpoint_dir)

# rebuild model but with batch size of 1
model = build_model(len(vocab_char), EMBED_DIM, UNITS, 1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()


# defining text generation function
def generate_text(model, start_string, length=3000, temp=1.0):
    # low temp = more predictable text, high temp = more surprising text

    # converting start string to numbers (vectorisation)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # initialise empty string to store results
    meditation = ""

    # batch size is 1
    model.reset_states()
    for i in range(length):
        predictions = model(input_eval)
        # remove batch dimension
        predictions = tf.squeeze(predictions, 0)

        # use categorical distribution to predict character returned by model
        predictions = predictions / temp
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # we pass the predicted character as the next input to the model
        # along with previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        # append predicted text
        meditation += idx2char[predicted_id]

    return (start_string + meditation)


# print our generated meditation
print(generate_text(model, start_string=u'From ', length=PRINT))