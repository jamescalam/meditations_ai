import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')
import data as d

# settings
BATCHSIZE = 64
SEQLEN = 100
BUFFER = 10000
EMBED_DIM = 50
EPOCHS = 2000
LATENT_UNITS = 50

# data preprocessing
# import The Meditations by Marcus Aurelius
txt = d.meditations()
print(txt[:200])


def format_data(txt):
    vocab = sorted(set(txt))  # create vocab from text string (txt)
    char2idx = {c: i for i, c in enumerate(vocab)}

    data_idx = np.array([char2idx[c] for c in txt])

    # one hot encode
    #hot1 = np.zeros((data_idx.size, data_idx.max() + 1))
    #hot1[np.arange(data_idx.size), data_idx] = 1
    #print(f"hot1 = {hot1.shape}")
    # and flatten
    #hot1 = hot1.flatten()
    #print(f"hot1 flatten = {hot1.shape}")

    # max length sequence we can have
    max_seqlen = len(txt) // SEQLEN

    # normalise data
    data_idx = data_idx / SEQLEN

    dataset = tf.data.Dataset.from_tensor_slices(data_idx)

    sequences = dataset.batch(SEQLEN, drop_remainder=True)

    dataset = sequences.shuffle(BUFFER).batch(BATCHSIZE, drop_remainder=True)

    return dataset, char2idx

data, char2idx = format_data(txt)

def format_fake_data(array, char2idx):
    array *= len(char2idx)
    return tf.math.round(array)


def build_generator(latent_units, vocab_size, seqlen, batchsize):
    # dense -> leakyrelu ->  batchnorm * 3 -> reshape to seqlen
    # final dense must be len(vocab_size) which we will then map to chars
    model = tf.keras.Sequential()

    # dense -> leakyrelu -> batchnorm
    model.add(tf.keras.layers.Dense(128, input_shape=(latent_units,)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    # dense -> leakyrelu -> batchnorm
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    # dense -> leakyrelu -> batchnorm
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    # reshape output to seqlen
    model.add(tf.keras.layers.Dense(seqlen, activation='softmax'))

    print(model.summary())

    return model

generator = build_generator(LATENT_UNITS, len(char2idx), SEQLEN, BATCHSIZE)
generator_optimiser = tf.optimizers.Adam(1e-3)

def generator_loss(fake_preds):
    # take sigmoid of output FAKE predictions only
    fake_preds = tf.sigmoid(fake_preds)
    # calculate the loss with binary cross-entropy
    fake_loss = tf.losses.binary_crossentropy(tf.ones_like(fake_preds), fake_preds)
    # return the fake predictions loss
    return fake_loss


def build_lstm_discriminator(vocab_size, embed_dim, batchsize, units):
    model = tf.keras.Sequential()
    # add our embedding layer
    model.add(tf.keras.layers.Embedding(vocab_size, embed_dim,
                                        batch_input_shape=[batchsize, None]))
    # the LSTM layer
    model.add(tf.keras.layers.LSTM(units, return_sequences=True,
                                   stateful=True, dropout=.1))
    # a DNN layer
    model.add(tf.keras.layers.Dense(vocab_size))
    # leaky ReLU activation layer
    model.add(tf.keras.layers.LeakyReLU())

    # final binary classifier layer, 1 or 0
    model.add(tf.keras.layers.Dense(1))

    print(model.summary())

    return model

def build_discriminator(vocab_size, batchsize, units):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(units, input_shape=(SEQLEN,),
                                    activation='sigmoid'))

    model.add(tf.keras.layers.Dense(32,
                                    activation='sigmoid'))

    # final binary classifier layer, 1 or 0
    model.add(tf.keras.layers.Dense(1))

    print(model.summary())

    return model

discriminator = build_discriminator(len(char2idx), BATCHSIZE, 128)

discriminator_optimiser = tf.optimizers.Adam(1e-6)

def discriminator_loss(real_preds, fake_preds):
    # take sigmoid of out output predictions
    real_preds = tf.sigmoid(real_preds)
    fake_preds = tf.sigmoid(fake_preds)
    # calculate the loss with binary cross-entropy
    real_loss = tf.losses.binary_crossentropy(tf.ones_like(real_preds), real_preds)
    fake_loss = tf.losses.binary_crossentropy(tf.zeros_like(fake_preds), fake_preds)
    # return the total loss from both real and fake predictions
    return real_loss + fake_loss

def train_step(sequences):
    fake_sequence_noise = np.random.randn(BATCHSIZE, LATENT_UNITS)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # generate fake sequences with the generator model
        generated_sequences = generator(fake_sequence_noise)
        # convert to correct data format
        #generated_sequences = format_fake_data(generated_sequences, char2idx)

        # get real and fake output predictions from the discriminator model
        real_output = discriminator(sequences)
        fake_output = discriminator(generated_sequences)

        # get the loss for both the generator and discriminator models
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        # get the gradients for both the generator and discriminator models
        grads_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables
        )
        grads_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )

        generator_optimiser.apply_gradients(
            zip(
                grads_generator, generator.trainable_variables
            )
        )

        discriminator_optimiser.apply_gradients(
            zip(
                grads_discriminator, discriminator.trainable_variables
            )
        )

        return (np.mean(gen_loss), np.mean(disc_loss), generated_sequences)

def train(dataset, epochs):
    loss = []
    for i in range(epochs):
        print(f"Epoch {i}")
        for j, sequences in enumerate(dataset):
            loss.append(train_step(sequences))
        print(f"Gen loss: {loss[-1][0]}\nDisc loss: {loss[-1][1]}")

    loss = pd.DataFrame({
        'epoch': range(len(loss)),
        'generator_loss': map(lambda x: x[0], loss),
        'discriminator_loss': map(lambda x: x[1], loss),
        'generated_sequence': map(lambda x: x[2], loss)
    })

    plt.figure(figsize=(14, 8))
    sns.lineplot(data=loss, x='epoch', y='generator_loss')
    sns.lineplot(data=loss, x='epoch', y='discriminator_loss')
    plt.show()

train(data, EPOCHS)
