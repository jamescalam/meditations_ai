import tensorflow as tf
import pickle
import os
from datetime import date

# setup model naming
def model_name(rnn, epochs, seq_len, preappend=""):
    return f"{preappend}{rnn}_e{epochs}_sl{seq_len}_{date.today().strftime('%m%d')}"

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
