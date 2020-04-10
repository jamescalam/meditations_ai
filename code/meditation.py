"""
Top-level Python file for Meditations project.
"""

import numpy as np
import tensorflow as tf
import os
import sys
import re
from collections import Counter

sys.path.insert(0, os.getcwd())
import data as d
import data_writer as dw


# text generation function
def generate(model, char2idx, start):
    # create idx2char dictionary
    idx2char = {char2idx[key]: key for key in char2idx}
    # converting start string to numbers (vectorisation)
    input_eval = [char2idx[s] for s in start]
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
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # we pass the predicted character as the next input to the model
        # along with previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        # append predicted text
        meditation += idx2char[predicted_id]

    return (start + meditation)


# function for rating outputs
def rate(text):
    # intialise rating
    rating = 0.
    # get normalised text (no punc, all lowercase)
    norm = re.sub(r'[^\w\s]', '', text).lower()

    # checking for correct punctuation
    if text[-1:] in ['.', '!', '?']:
        rating += 20.
    else:
        rating -= 20.
    print(f"punc checked, score: {rating}")
    # checking for too much repetition
    repetition_score = 20.  # initialise score
    count = Counter([word.strip() for word in norm.split(" ")])  # get count dictionary
    count = [(key, count[key]) for key in count]  # convert dictionary to list of tuples
    count = sorted(count, key=lambda x: x[1])  # sorting tuple list
    # now calculate the repetition score
    for x in count:
        if x[1] == 1:
            repetition_score *= 0.5  # if just one word, reduce the repetition_score
        else:
            repetition_score *= x[1]  # else, the repetition acts as a multiplier
        print(repetition_score)
    # now we subtract the repetition score from our rating
    rating -= repetition_score
    print(f"repetition checked, score: {rating}")
    return rating


# class for ensemble learning
class ensemble:
    def __init__(self):
        # initialise prediction dictionary
        self.meditations = {}

    def predict(self, model_list, start):
        # initialise prediction dictionary
        self.meditations = {}
        # loop through each model
        for modelname in model_list:
            # load the model
            model = dw.load_model(modelname)
            # load the char2idx
            char2idx = dw.load_char2idx(modelname)
            # make prediction
            self.meditations[modelname] = generate(model, char2idx, start)

"""
# import texts
# The Meditations by Marcus Aurelius
txt_meditations = d.meditations()

# Epistulae Morales ad Lucilium by Seneca
txt_letters = d.hello_lucilius()

# join together
txt = "\n".join([txt_meditations, txt_letters])
"""