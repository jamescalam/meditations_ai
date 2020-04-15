import sys
from datetime import datetime
sys.path.insert(0, './')
import meditation as m
import data as d
import data_writer as dw
import train as t

PREDICT = True
SEQ = 50
# settings for building new model
NEW = False

MODEL_PARAMS = []
MODEL_PARAMS.append({
    'SEQLEN': 250, 'BUFFER': 10000, 'BATCHSIZE': 64, 'EMBED_DIM': 256,
    'EPOCHS': 2, 'RNN': 'LSTM', 'RNN_UNITS': 1024, 'NAME': ''
})
MODEL_PARAMS.append({
    'SEQLEN': 250, 'BUFFER': 10000, 'BATCHSIZE': 64, 'EMBED_DIM': 256,
    'EPOCHS': 64, 'RNN': 'LSTM', 'RNN_UNITS': 2048, 'NAME': ''
})
"""
MODEL_PARAMS.append({
    'SEQLEN': 250, 'BUFFER': 10000, 'BATCHSIZE': 64, 'EMBED_DIM': 256,
    'EPOCHS': 512, 'RNN': 'LSTM', 'RNN_UNITS': 1024, 'NAME': ''
})
MODEL_PARAMS.append({
    'SEQLEN': 250, 'BUFFER': 10000, 'BATCHSIZE': 64, 'EMBED_DIM': 256,
    'EPOCHS': 512, 'RNN': 'LSTM', 'RNN_UNITS': 2048, 'NAME': ''
})"""


if NEW:
    # import data
    txt_meditations = d.meditations()
    txt_letters = d.hello_lucilius()
    # format letters into text from {letter: [address, text]}
    txt_letters = "\n".join([txt_letters[key][1] for key in txt_letters])
    # join data
    TXT = "\n".join([txt_meditations, txt_letters])
    # clear
    del txt_meditations, txt_letters

    for model in MODEL_PARAMS:
        # new model build
        NEW_MODEL = t.Model(model['SEQLEN'],
                            model['BUFFER'],
                            model['BATCHSIZE'],
                            model['EMBED_DIM'])

        # formatting input data
        NEW_MODEL.format_data(TXT)
        # building new model
        NEW_MODEL.build_model(model['EPOCHS'],
                              model['RNN'],
                              model['RNN_UNITS'],
                              model['NAME'])

    dw.notify_me(f"Model training complete for {len(MODEL_PARAMS)} model(s).")

if PREDICT:
    MODEL_LIST = [
        'LSTM_e64_sl100_0412',
        'LSTM_e128_sl100_0413',
        'LSTM_e256_sl100_0414'
        #'LSTM_e64_sl250_0415'
    ]

    model = m.ensemble()

    text = model.gladiator_predict(MODEL_LIST, 'From ', end='.', sequences=SEQ, vis=True)

    dw.save_text(text, f"ensemble_rnn_{datetime.now().strftime('%m%d')}", "Summary exempt.")

    dw.notify_me(f"Ensemble model text generation complete, {len(text)} characters generated.\n"+
                 f"Text saved to local ensemble_rnn_{datetime.now().strftime('%m%d')} directory.\n\n"+
                 text,
                 img=f"../models/ensemble_rnn_{datetime.now().strftime('%m%d')}/gladiator.jpg")

"""__________BUILDING MEDITATIONS VOCAB__________
# import texts
# The Meditations by Marcus Aurelius
txt_meditations = d.meditations()

# Epistulae Morales ad Lucilium by Seneca
txt_letters = d.hello_lucilius()

# convert letters into string
txt_letters = "\n".join([txt_letters[key][1] for key in txt_letters])
# join together
txt = "\n".join([txt_meditations, txt_letters])

# save and load vocab
dw.create_vocab(txt)
vocab = dw.read_vocab()
"""