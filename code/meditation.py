"""
Top-level Python file for Meditations project.
"""

import numpy as np
import tensorflow as tf
import os
import sys

sys.path.insert(0, os.getcwd())
import data as d
import datawriter as dw


# class for ensemble learning
class ensemble:
    def __init__(self):
        pass

    def predict(self, model_list):
        # loop through each model
        for modelname in model_list:
            # load the model
            model = dw.load_model(modelname)
            # load the char2idx
            char2idx = dw.load_char2idx(modelname)
# import texts
# The Meditations by Marcus Aurelius
txt_meditations = d.meditations()

# Epistulae Morales ad Lucilium by Seneca
txt_letters = d.hello_lucilius()

# join together
txt = "\n".join([txt_meditations, txt_letters])