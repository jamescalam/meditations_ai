import numpy as np
import tensorflow as tf
import os
import shutil
import sys
sys.path.insert(0, './')
import data_writer as dw
import data as d

# settings
SEQ_LEN = 100  # maximum sequence length we want for training
BUFFER_SIZE = 10000  # buffer size for shuffling the dataset
EPOCHS = 32  # number of times we iterate over the full dataset during training
RNN = 'LSTM'  # whether we use LSTM or GRU RNN units
UNITS = 1024  # how many units we use
BATCH_SIZE = 64  # no. sequences of SEQ_LEN we train on before updating weights
EMBED_DIM = 256  # vector dimension of character vector embeddings
PRINT = 10000  # how many characters we print during text generation

# import data
txt_meditations = d.meditations()
txt_letters = d.hello_lucilius()

# format letters into text from {letter: [address, text]}
txt_letters = "\n".join([txt_letters[key][1] for key in txt_letters])

# join data
data = "\n".join([txt_meditations, txt_letters])

# create vocab
vocab = sorted(set(data))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

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
    # "hello"
    input_data = chunk[:-1]  # "hell"
    target_data = chunk[1:]  # "ello"
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
    vocab_size=len(vocab),
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
model = build_model(len(vocab), EMBED_DIM, UNITS, 1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

# empty the checkpoint directory
for filename in os.listdir(checkpoint_dir):
    file_path = os.path.join(checkpoint_dir, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# saving model and char2idx dictionary
MODEL_NAME = dw.model_name(RNN, EPOCHS, SEQ_LEN, 'meditations')
dw.save_model(model, MODEL_NAME)
dw.save_char2idx(char2idx, MODEL_NAME)

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


# output our generated meditation
text = generate_text(model, start_string=u'From ', length=PRINT)
summary = f"""SEQ_LEN = {SEQ_LEN}
BUFFER_SIZE = {BUFFER_SIZE}
EPOCHS = {EPOCHS}
RNN = {RNN}
UNITS = {UNITS}
BATCH_SIZE = {BATCH_SIZE}
EMBED_DIM = {EMBED_DIM}"""
dw.save_text(text, MODEL_NAME, summary)
