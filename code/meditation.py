"""
Top-level Python file for Meditations project.
"""

import tensorflow as tf
from datetime import datetime
import os
import sys
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import cuda

sys.path.insert(0, os.getcwd())
import data as d
import data_writer as dw

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# ensemble function visualisation
def visualise(df):
    dir = f"../models/ensemble_rnn_{datetime.now().strftime('%m%d')}"
    # if directory does not exist make it
    if not os.path.isdir(dir):
        os.makedirs(dir)
    sns.set(style="darkgrid")  # setting style
    sns.set_palette(['#212B38', '#08C6AB',
     '#726EFF', '#37465B',
     '#5AFFE7'])  # setting palette colours
    plt.figure(figsize=(14, 6))
    # df should be dataframe with 'model', 'score', 'iteration' columns
    print(len(df))
    sns.lineplot(
        data=df,
        x='iteration', y='score',
        hue='model',
        linewidth=3.
    )
    plt.tight_layout()
    plt.savefig(f"{dir}/gladiator.jpg")

# text generation function
def generate(model, char2idx, start, counter=1, maxlen=1000, end='\n', keep_start=True):
    # create idx2char dictionary
    idx2char = {char2idx[key]: key for key in char2idx}
    # converting start string to numbers (vectorisation)
    input_eval = [char2idx[s] for s in start]
    input_eval = tf.expand_dims(input_eval, 0)

    # initialise empty string to store results
    meditation = ""
    text = ""

    # batch size is 1
    model.reset_states()
    while counter > 0 and len(meditation) < maxlen:
        predictions = model(input_eval)
        # remove batch dimension
        predictions = tf.squeeze(predictions, 0)

        # use categorical distribution to predict character returned by model
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # we pass the predicted character as the next input to the model
        # along with previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        # append predicted text
        text += idx2char[predicted_id]
        # check if we got an end character
        if end in text or len(text) > maxlen:
            meditation += text  # add text to meditation variable
            text = ""  # now reset text
            # this is a text segment, so reduce counter by 1
            counter -= 1
    if keep_start:
        return (start + meditation)
    return meditation


# function to find how much a particular piece of text matches to another (the original)
def match_score(text, original):
    # start with full text, then split more and more per level
    # remove punctuation and lowercase text and original
    text = re.sub(r'[^\w\s]', '', text).lower()
    original = re.sub(r'[^\w\s]', '', original).lower()
    text = text.split()  # split by whitespace to create list of words

    seq = 3  # we start by attempting to match just 3 words
    matches = 0  # initialise total matches

    # matching function
    def match(text, original):
        if " ".join(text) in original:
            return True

    # loop through the whole text sequence from start to end
    for i in range(len(text) - seq - 1):
        # loop through increasing number of joined words
        for j in range(i + seq, len(text) - 1):
            if not " ".join(text[i:j]) in original:
                break
        matches.append(j - i - seq)  # append highest match score (take seq from this as this is the min value


# function for rating outputs
def rate(text):
    # if text is empty, super low rating and return
    if len(text.strip()) == 0:
        return -500.00
    # intialise rating
    rating = 0.
    # get normalised text (no punc, all lowercase)
    norm = re.sub(r'[^\w\s]', '', text).lower()

    # plenty of punctuation is often a good thing
    rating += ((len(re.sub(r'[\w\s]', '', text))+1)/len(text)) * 20

    # check that we have an equal number of speech marks
    if len(re.sub(r'[^"]', '', text)) % 2 == 0:
        rating += 5
    else:
        rating -= 5

    # checking for correct ending punctuation - may include newline too, eg '.' and '.\n' are both good
    if text[-1:] in ['.', '!', '?'] or text[-2:-1] in ['.', '!', '?']:
        rating += 20.
    else:
        rating -= 20.

    # checking for too much repetition
    repetition_score = 20.  # initialise score
    count = Counter([word.strip() for word in norm.split()])  # get count dictionary
    count = [(key, count[key]) for key in count]  # convert dictionary to list of tuples
    count = sorted(count, key=lambda x: x[1])  # sorting tuple list
    # now calculate the repetition score
    for x in count:
        repetition_score *= x[1] * .5  # high repetition acts as a multiplier
    # now we subtract the repetition score from our rating
    rating -= repetition_score

    # checking all words are actual words
    vocab = re.compile(r"\b" + r"\b|\b".join(dw.read_vocab()) + r"\b")
    not_real = re.sub(vocab, "", norm)  # remove real word
    rating -= len(not_real.split()) * 10  # not real words
    rating += ((len(norm.split())+1)/(len(not_real.split())+1)) * 10.  # % * 20 of real words to non-real words
    # !!! normalise the score to length of text?
    return round(rating, 2)


# class for ensemble learning
class ensemble:
    def __init__(self):
        # initialise prediction dictionary
        self.meditations = {}

    def gladiator_predict(self, model_list, start, end='.', sequences=10, vis=False):
        text = ""
        meditations = {}  # initialise generated text dictionary
        models = {}  # initialise models dictionary
        for modelname in model_list:
            models[modelname] = {}
            models[modelname]['model'] = dw.load_model(modelname)
            models[modelname]['char2idx'] = dw.load_char2idx(modelname)
        # if visualising performance over time, initialise performance tracking dataframe
        if vis:
            performance = pd.DataFrame({
                'model': [],
                'score': [],
                'iteration': []
            })
        keep_start = True
        for i in range(sequences):
            # loop through each model and make prediction
            for modelname in model_list:
                pred = generate(models[modelname]['model'],
                                models[modelname]['char2idx'],
                                start=start, end=end,
                                keep_start=keep_start)
                # get score
                score = rate(pred)
                # add score and generated text to the meditations dictionary entry for that model
                meditations[modelname] = [score, pred]
                print(f"[{score}] ({modelname}): {pred[:70]}...")

            # find highest scoring sentence
            # convert to list and sort (high -> low)
            scores = sorted([meditations[key] for key in meditations], key=lambda x: x[0], reverse=True)

            print(f"#1: {scores[0][0]}, #2: {scores[1][0]}, #3: {scores[2][0]}")

            text += scores[0][1]  # getting the highest rated sentence
            start = scores[0][1]  # new start is the winner
            keep_start = False  # turn off as we only want to keep the starting string on the very first iteration

            # if we want to visualise, and performance metrics to performance list
            if vis:
                for modelname in meditations:
                    new_row = pd.DataFrame({
                        'model': [modelname],
                        'score': [meditations[modelname][0]],
                        'iteration': [i]
                    })
                    performance = pd.concat([performance, new_row], ignore_index=True)
        if vis:
            visualise(performance)
        return text
