from keras import optimizers

def get_neural_net_configuration():
	nn_params = {}
	nn_params['sampling_frequency'] = 44100
	nn_params['generator_hidden_dims'] = 256
	nn_params['generator_dropout'] = 0.2
	nn_params['generator_optimizer'] = 'adam'
	nn_params['decoder_dropout'] = 0.3
	nn_params['decoder_optimizer'] = optimizers.SGD(0.01, momentum=0.1, decay=0.01)
	nn_params['decoder_hidden_dims'] = 128
	nn_params['combined_optimizer'] = optimizers.SGD(0.01, momentum=0.1, decay=0.01)
	#The weights filename for saving/loading trained models
	nn_params['model_basename'] = './NuGruvModelWeights'
	#The training data directory
	nn_params['dataset_directory'] = './datasets/'
	return nn_params
