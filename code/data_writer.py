import tensorflow as tf
import pickle
import os
from datetime import datetime

# setup model naming
def model_name(rnn, epochs, seq_len, preappend=""):
    return f"{preappend}{rnn}_e{epochs}_sl{seq_len}_{datetime.now().strftime('%m%d')}"

# save and load char2idx dictionaries
def save_char2idx(char2idx, modelname):
    # if directory does not exist make it
    if not os.path.isdir(f'../models/{modelname}'):
        os.makedirs(f'../models/{modelname}')
    # save the char2idx dictionary
    with open(f"../models/{modelname}/char2idx.json", 'wb') as f:
        pickle.dump(char2idx, f, pickle.HIGHEST_PROTOCOL)

def load_char2idx(modelname):
    with open(f"../models/{modelname}/char2idx.json", 'rb') as f:
        return pickle.load(f)

# save and load models
def save_model(model, modelname):
    # if directory does not exist make it
    if not os.path.isdir(f'../models/{modelname}'):
        os.makedirs(f'../models/{modelname}')
    # save model
    # !!! use this code when TF 2.2 is released
    #model.save(f'../models/{modelname}/model')
    model.save(f'../models/{modelname}/model.h5')

def load(modelname):
    # load the model
    # !!! use this code when TF 2.2 is released
    #return tf.keras.models.load_model(f"../models/{model}/model")
    return tf.keras.models.load_model(f'../models/{modelname}/model.h5',
                                      compile=False)

# output text
def save_text(text, modelname, summary):
    # if directory does not exist make it
    if not os.path.isdir(f'../models/{modelname}'):
        os.makedirs(f'../models/{modelname}')
    # add extra newline characters for HTML formatting
    text = text.replace("\n", "\n\n")
    summary = summary.replace("\n", "\n\n")
    # format the text into easier to view html format
    html = f"""<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
</head>

<body>
<div class="container">
<h1>{modelname}</h1>
<p>Output at {datetime.now().strftime('%d/%m/%Y at %h:%m')}.</p>
<p>""" + summary.replace("\n", "\n\n") + """</p>
<p>
""" + text.replace("\n", "</p>\n<p>") + """
</p>
</body>

</html>
"""
    # now save to file
    with open(f"../models/{modelname}/meditation.html", "w") as f:
        f.write(html)
