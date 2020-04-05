"""
Top-level Python file for Meditations project.
"""

import numpy as np
import tensorflow as tf
import os
import sys

sys.path.insert(0, os.getcwd())
import data as d

# import texts
# The Meditations by Marcus Aurelius
txt_meditations = d.meditations()

# Epistulae Morales ad Lucilium by Seneca
txt_letters = d.hello_lucilius()

# join together
txt = "\n".join([txt_meditations, txt_letters])