from keras.models import Sequential
from keras.layers import *
import numpy as np

def create_lstm_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1):
    model = Sequential()
    #This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(None, num_frequency_dimensions)))
    model.add(GaussianDropout(0.3))
    for cur_unit in xrange(num_recurrent_units):
        model.add(LSTM(units=num_hidden_dimensions, return_sequences=True))
    #This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

#def create_gru_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1):
    #model = Sequential()
    ##This layer converts frequency space to hidden space
    #model.add(TimeDistributedDense(input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions))
    #for cur_unit in xrange(num_recurrent_units):
        #model.add(GRU(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))
    ##This layer converts hidden space back to frequency space
    #model.add(TimeDistributedDense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions))
    #model.compile(loss='mean_squared_error', optimizer='rmsprop')
    #return model
    
def create_noise_network(num_frequency_dimensions, num_hidden_dimensions):
    model = Sequential()
    model.add(TimeDistributed(Lambda(lambda x: np.random.choice((0,1), p=[0.1,0.9])*np.random.normal(1,0.3)*x*0.5), input_shape=(None, num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model
