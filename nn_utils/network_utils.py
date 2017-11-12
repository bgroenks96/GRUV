from keras.models import Sequential
from keras.layers import *
import numpy as np

def create_lstm_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1, optimizer='rmsprop', dropout_rate=0.3):
    model = Sequential()
    #This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(None, num_frequency_dimensions)))
    model.add(GaussianDropout(dropout_rate))
    for cur_unit in xrange(num_recurrent_units):
        model.add(LSTM(units=num_hidden_dimensions, return_sequences=True))
    #This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def create_gru_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1, optimizer='rmsprop', dropout_rate=0.3):
    model = Sequential()
    #This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(None, num_frequency_dimensions)))
    model.add(GaussianDropout(dropout_rate))
    for cur_unit in xrange(num_recurrent_units):
        model.add(GRU(units=num_hidden_dimensions, return_sequences=True))
    #This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
    
def create_noise_network(num_frequency_dimensions, num_hidden_dimensions):
    model = Sequential()
    model.add(TimeDistributed(Lambda(lambda x: np.random.choice((0,1), p=[0.1,0.9])*np.random.normal(1,0.3)*x*0.5), input_shape=(None, num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

def create_gan(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1, optimizer='adam', dropout_rate=0.3):
    # Create generator network
    generator = create_lstm_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=num_recurrent_units, optimizer=optimizer, dropout_rate=dropout_rate)
    # Create discriminator network
    discriminator = Sequential()
    discriminator.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(None, num_frequency_dimensions)))
    discriminator.add(GaussianDropout(0.1))
    discriminator.add(Dense(num_hidden_dimensions, activation='tanh'))
    discriminator.add(Dense(num_hidden_dimensions, activation='tanh'))
    discriminator.add(TimeDistributed(Dense(num_frequency_dimensions)))
    discriminator.compile(loss='mean_squared_error', optimizer=optimizer)
    # Create GAN (combined model)
    return GAN(generator, discriminator)

class GAN:
    def __init__(self, generator, discriminator):
        model = Sequential()
        model.add(generator)
        discriminator.trainable = False
        model.add(discriminator)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        self.generator = generator
        self.discriminator = discriminator
        self.model = model
        
    def fit_generator(self, X_train, y_train, batch_size, epochs, shuffle=False, verbose=1, val_data=None):
        return self.generator.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose, validation_data=val_data)
    
    def fit_discriminator(self, X_train, y_train, batch_size, epochs, shuffle=False, verbose=1, val_data=None):
        return self.discriminator.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose, validation_data=val_data)
    
    def fit(self, X_train, y_train, batch_size, epochs, shuffle=False, verbose=1, val_data=None):
        return self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose, validation_data=val_data)
    
