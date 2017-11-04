def get_neural_net_configuration():
	nn_params = {}
	nn_params['sampling_frequency'] = 44100
	#Number of hidden dimensions.
	#For best results, this should be >= freq_space_dims, but most consumer GPUs can't handle large sizes
	nn_params['hidden_dimension_size'] = 1024
	nn_params['hidden_recurrent_layers'] = 1
	#The weights filename for saving/loading trained models
	nn_params['model_basename'] = './YourMusicLibraryNPWeights'
	#The model filename for the training data
	nn_params['model_file'] = './datasets/train/YourMusicLibraryNP'
	#The training data directory
	nn_params['dataset_directory'] = './datasets/train/YourMusicLibrary/'
	#The model filename of the generation data
	nn_params['gen_file'] = './datasets/gen/YourMusicLibraryNP'
	#The generation data directory
	nn_params['gen_directory'] = './datasets/gen/YourMusicLibrary/'
	return nn_params
