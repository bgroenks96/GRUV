from keras.models import *
from keras.layers import *
from keras import initializers
from keras import optimizers
import numpy as np

# Prototype gneerator network that deconvolves a frequency domain signal with 'num_frequency_dimensions' and 'num_timesteps'
# from a random seed of size 'seed_size'.
def create_deconvolutional_generator_network(seed_size, batch_size, num_frequency_dimensions, num_timesteps, config, stateful=True):
    num_hidden_dimensions = config['generator_hidden_dims']
    inputs = Input(batch_shape=(batch_size, seed_size))
    dense_in = Dense(num_hidden_dimensions*num_timesteps, activation='relu')(inputs)
    conv_h1 = Conv1D(num_hidden_dimensions, kernel_size=2, activation='relu', padding='causal')(Reshape((num_timesteps, num_hidden_dimensions))(dense_in))
    lstm = LSTM(num_hidden_dimensions, return_sequences=True, activation='relu', stateful=stateful)(conv_h1)
    conv_out = Conv1D(num_frequency_dimensions, kernel_size=2, padding='causal')(lstm)
    model = Model(inputs=inputs, outputs=conv_out)
    model.compile(loss='logcosh', optimizer=config['generator_optimizer'])
    return model

# GAN friendly adaptation of Nayebi et Vitelli's autoencoder concept; replaces TDD input/output layers with
# 1D convolutional layers and uses two hidden LSTM layers.
def create_autoencoding_generator_network(num_frequency_dimensions, config, num_timesteps=None, batch_size=None, stateful=False):
    inputs = Input(batch_shape=(batch_size, num_timesteps, num_frequency_dimensions))
    num_hidden_dimensions = config['generator_hidden_dims']
    dropout = GaussianDropout(config['generator_dropout'])
    conv_in = Conv1D(num_hidden_dimensions, kernel_size=2, padding='causal')(inputs)
    lstm_1 = LSTM(num_hidden_dimensions, return_sequences=True, stateful=stateful)(dropout(conv_in))
    lstm_2 = LSTM(num_hidden_dimensions, return_sequences=True, stateful=stateful)(dropout(lstm_1))
    td_dense = TimeDistributed(Dense(num_hidden_dimensions))(lstm_2)
    conv_out = Conv1D(num_frequency_dimensions, kernel_size=2, padding='causal')(dropout(td_dense))
    model = Model(inputs=inputs, outputs=conv_out)
    model.compile(loss='mean_squared_error', optimizer=config['generator_optimizer'])
    return model

# Simple decoder that applies TDD and then flattens the time dimensions before computing the probability
# of the sample being real or fake.
def create_decoder_network(num_frequency_dimensions, num_timesteps, config, batch_size=None):
    num_hidden_dimensions = config['decoder_hidden_dims']
    dropout = GaussianDropout(['decoder_dropout'])
    inputs = Input(batch_shape=(batch_size, num_timesteps, num_frequency_dimensions))
    conv_in = Conv1D(num_hidden_dimensions, kernel_size=2, activation='relu', padding='causal')(inputs)
    conv_h1 = Conv1D(num_hidden_dimensions, kernel_size=2, activation='relu', padding='causal')(dropout(conv_in))
    td_dense = TimeDistributed(Dense(num_hidden_dimensions, activation='relu'))(dropout(conv_h1))
    lstm = LSTM(1, activation='sigmoid', return_sequences=False)(dropout(td_dense))
    #dense_out = Dense(1, activation='sigmoid')(dropout(lstm))
    decoder = Model(inputs=inputs, outputs=lstm)
    decoder.compile(loss='binary_crossentropy', optimizer=config['decoder_optimizer'], metrics=['accuracy'])
    return decoder

# Constructs a GAN using the autoencoder generator.
def create_autoencoder_gan(num_frequency_dimensions, config, num_timesteps=None, batch_size=None, stateful=False):
    # Create generator network
    generator = create_autoencoding_generator_network(num_frequency_dimensions, config, num_timesteps=num_timesteps, batch_size=batch_size, stateful=stateful)
    # Create decoder (or "discriminator") network
    decoder = create_decoder_network(num_frequency_dimensions, num_timesteps, config)
    # Create GAN (combined model)
    return GAN(generator, decoder, config)

def create_deconvolutional_gan(seed_size, batch_size, num_frequency_dimensions, num_timesteps, config, stateful=True):
    # Create generator network
    generator = create_deconvolutional_generator_network(seed_size, batch_size, num_frequency_dimensions, num_timesteps, config, stateful)
    # Create decoder (or "discriminator") network
    decoder = create_decoder_network(num_frequency_dimensions, num_timesteps, config, batch_size=batch_size)
    # Create GAN (combined model)
    return GAN(generator, decoder, config)

# Original single layer, LSTM autoencoder network proposed by Nayebi et Vitelli (2015)
def create_lstm_network(num_frequency_dimensions, num_hidden_dimensions):
    model = Sequential()
    #This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(None, num_frequency_dimensions)))
    model.add(LSTM(units=num_hidden_dimensions, return_sequences=True))
    #This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

# Original single layer, GRU autoencoder network proposed by Nayebi et Vitelli (2015)
def create_gru_network(num_frequency_dimensions, num_hidden_dimensions):
    model = Sequential()
    #This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(None, num_frequency_dimensions)))
    model.add(GRU(units=num_hidden_dimensions, return_sequences=True))
    #This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

# Simple time-distributed network that just adds Gaussian noise to the input signal.
def create_noise_network(num_frequency_dimensions):
    model = Sequential()
    model.add(TimeDistributed(Lambda(lambda x: np.random.choice((0,1), p=[0.1,0.9])*np.random.normal(1,0.3)*x*0.5), input_shape=(None, num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

# Simple implementation of a Generative Adverserial Network taking an arbitrary generator and decoder network.
class GAN:
    def __init__(self, generator, decoder, config):
        model = Sequential()
        model.add(generator)
        decoder.trainable = False
        model.add(decoder)
        model.compile(loss='binary_crossentropy', optimizer=config['combined_optimizer'], metrics=['accuracy'])
        decoder.trainable = True
        self.generator = generator
        self.decoder = decoder
        self.model = model
    
    # Fit the generator network against the given training data
    def fit_generator(self, X_train, y_train, batch_size=None, epochs=10, shuffle=False, verbose=1, callbacks=None, validation_data=None):
        return self.generator.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose, callbacks=callbacks, validation_data=validation_data)

    # Fit the decoder network against the given real and fake X examples. An example output of '1' will be generated for each real example, and '0' for each fake one.
    def fit_decoder(self, X_real, X_fake, batch_size=None, epochs=10, shuffle=False, verbose=1, callbacks=None, validation_data=[]):
        num_real = X_real.shape[0]
        num_fake = X_fake.shape[0]
        X_train = np.concatenate((X_real, X_fake), axis=0)
        y_train = np.concatenate((np.ones(num_real), np.zeros(num_fake)))
        val_data = None
        if len(validation_data) > 0:
            val_real = validation_data[0]
            val_fake = validation_data[1]
            X_val = np.concatenate((val_real, val_fake))
            y_val = np.concatenate((np.ones(val_real.shape[0]), np.zeros(val_fake.shape[0])))
            val_data = (X_val, y_val)
        return self.decoder.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose, callbacks=callbacks, validation_data=val_data)

    def fit(self, X_train, batch_size=None, epochs=10, shuffle=False, verbose=1, callbacks=None, validation_x=[]):
        num_examples = X_train.shape[0]
        num_timesteps = X_train.shape[1]
        y_train = np.ones(num_examples)
        val_data = None
        if len(validation_x) > 0:
            num_val = validation_x.shape[0]
            y_val = np.ones(num_val)
            val_data = (validation_x, y_val)
        self.decoder.trainable = False
        hist = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose, callbacks=callbacks, validation_data=val_data)
        self.decoder.trainable = True
        return hist
    
    def summary(self):
        print('==== Generator ====')
        self.generator.summary()
        print('==== Decoder ====')
        self.decoder.summary()
        print('==== Combined (GAN) ====')
        self.decoder.trainable = False
        self.model.summary()
        self.decoder.trainable = True
        
