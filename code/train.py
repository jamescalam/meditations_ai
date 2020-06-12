import numpy as np
import tensorflow as tf
import os
import shutil
from datetime import datetime
import sys
sys.path.insert(0, './')
import data_writer as dw
import data as d


# for each sequence, we duplicate and shift it to form the input and target text
def split_xy(chunk):
    # "hello"
    input_data = chunk[:-1]  # "hell"
    target_data = chunk[1:]  # "ello"
    return input_data, target_data

def build_model(batchsize, vocab_size, rnn='LSTM', rnn_units=1024, embed_dim=256):
    if rnn == 'GRU':
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embed_dim,
                                      batch_input_shape=[batchsize, None]),
            tf.keras.layers.GRU(rnn_units, return_sequences=True,
                                stateful=True,
                                dropout=0.1),
            tf.keras.layers.Dense(vocab_size)
        ])
    elif rnn == 'LSTM':
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embed_dim,
                                      batch_input_shape=[batchsize, None]),
            tf.keras.layers.LSTM(rnn_units, return_sequences=True,
                                 stateful=True,
                                 dropout=0.1),
            tf.keras.layers.Dense(vocab_size)
        ])
    # print a model summary
    print(model.summary())
    return model

class Model:
    def __init__(self, seqlen, buffer, batchsize, embed_dim, name=""):
        self.seqlen = seqlen
        self.buffer = buffer
        self.batchsize = batchsize
        self.embed_dim = embed_dim

        self.model = {}  # initialise model dictionary

    def format_data(self, txt):
        self.vocab = sorted(set(txt))  # create vocab from text string (txt)

        self.char2idx = {c: i for i, c in enumerate(self.vocab)}  # create char2idx mappings
        # TODO in vide, char2idx -> self.char2idx

        data_idx = np.array([self.char2idx[c] for c in txt])  # converting data from characters to indexes

        # max length sentence we want
        self.max_seqlen = len(txt) // (self.seqlen + 1)
        # TODO, we don't actually need this!

        # create training data
        dataset = tf.data.Dataset.from_tensor_slices(data_idx)

        # batch method allows conversion of individual characters to sequences of a desired size
        sequences = dataset.batch(self.seqlen + 1, drop_remainder=True)

        dataset = sequences.map(split_xy)  # mapping dataset sequences to input-target segments

        # shuffle the dataset AND batch into batches of 64 sequences
        self.dataset = dataset.shuffle(self.buffer).batch(self.batchsize, drop_remainder=True)
        # TODO in video, dataset -> self.dataset
    # model build function
    def build_model(self, epochs, rnn, units, name):
        # build modelname for save/load
        modelname = dw.model_name(rnn, epochs, self.seqlen, preappend=name)
        print(f"Building model '{modelname}'.")

        # build model and print summary with build_model function, store model in model dict
        model = build_model(self.batchsize, len(self.vocab), rnn, units, self.embed_dim)

        # define the loss function
        def loss(labels, logits):
            return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

        # compile the model
        model.compile(optimizer='adam', loss=loss)

        # execute training
        history = model.fit(self.dataset, epochs=epochs)
        # TODO in video, this line is completely different

        # save model weights
        model.save_weights('../models/tmp/weights')
        del model  # remove from memory

        # rebuild model but with batch size of 1 (for text generation)
        model = build_model(1, len(self.vocab), rnn, units, self.embed_dim)
        model.load_weights('../models/tmp/weights')
        model.build(tf.TensorShape([1, None]))

        # saving model and char2idx dictionary
        dw.save_model(model, modelname)
        dw.save_char2idx(self.char2idx, modelname)
