from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, LSTM, GRU

def create_lstm_network(num_timesteps, num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1):
	model = Sequential()
	#This layer converts frequency space to hidden space
	model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(None, num_frequency_dimensions)))
	for cur_unit in xrange(num_recurrent_units):
		model.add(LSTM(units=num_hidden_dimensions, input_shape=(None, num_hidden_dimensions), return_sequences=True))
	#This layer converts hidden space back to frequency space
	model.add(TimeDistributed(Dense(num_frequency_dimensions), input_shape=(num_timesteps, num_frequency_dimensions)))
	model.compile(loss='mean_squared_error', optimizer='adadelta')
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
