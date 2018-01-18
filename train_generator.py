from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
import nn_utils.network_utils as network_utils
import config.nn_config as nn_config
import tensorflow as tf
import argparse

config = nn_config.get_default_configuration()

parser = argparse.ArgumentParser(description="Train the NuGRUV generator network against the current dataset.")
parser.add_argument("dataset_name", help="Name of the dataset to use for training.")
parser.add_argument("-m", "--model", default='gruv', type=str, help="Generator model to use. Valid values are 'gruv' and 'rgan' (generator only). Defaults to 'gruv'")
parser.add_argument("-s", "--start-iter", default=0, type=int, help="Current training iteration to start from.")
parser.add_argument("-n", "--num-iters", default=10, type=int, help='Number of training iterations to run.')
parser.add_argument("-e", "--epochs", default=25, type=int, help="Number of epochs per iteration.")
parser.add_argument("-b", "--max-batch", default=500, type=int, help="Maximum number of training examples to batch per gradient update.")
parser.add_argument("-v", "--validation", default=True, type=bool, help="Use cross validation data.")
parser.add_argument("-i", "--interval", default=5, type=int, help="Number of iterations to run in between retaining saved weights.")
parser.add_argument("-r", "--run", default=0, type=int, help="Run id for this training session. Defaults to 0")
parser.add_argument("--skip-validation", action="store_true", help="Do not use cross validation data.")
args = parser.parse_args()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

input_file = config['dataset_directory'] + args.dataset_name + '/' + args.dataset_name
cur_iter = args.start_iter
model_basename = config['model_basename'] + str(args.run)
model_filename = '{0}_{1}'.format(model_basename, str(cur_iter))
skip_validation = args.skip_validation

#Load up the training data
print('Loading training data')
#X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
#y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
X_train = np.load(input_file + '_x.npy')
y_train = np.load(input_file + '_y.npy')
if not skip_validation:
    X_val = np.load(input_file + '_val_x.npy')
    y_val = np.load(input_file + '_val_y.npy')
print('Finished loading training data')

#Figure out how many frequencies we have in the data
num_timesteps = X_train.shape[1]
freq_space_dims = X_train.shape[2]
hidden_dims = config['generator_hidden_dims']

num_iters = cur_iter + args.num_iters 		#Number of iterations for training
epochs_per_iter = args.epochs	                #Number of iterations before we save our model
batch_size = X_train.shape[0]
while batch_size > args.max_batch:
    batch_size = int(np.ceil(batch_size / 2.0))

#Creates a lstm network
print('Initializing network...')
if args.model == 'gruv':
    model = network_utils.create_lstm_network(num_frequency_dimensions=freq_space_dims, num_hidden_dimensions=hidden_dims)
elif args.model == 'rgan':
    model = network_utils.create_regression_generator_network(freq_space_dims, config, num_timesteps=None)

print('Model summary:')
model.summary()

#Load existing weights if available
if os.path.isfile(model_filename):
    print ('Loading existing weight data from {}'.format(model_filename))
    model.load_weights(model_filename)

print('Training set shape: {0}'.format(X_train.shape))
val_data = None
if not skip_validation:
    print('Validation set shape: {0}'.format(X_val.shape))
    val_data = (X_val, y_val)
    
def print_hist_stats(h):
    loss = h.history['loss']
    print('loss min: {0}  loss max: {1}'.format(min(loss), max(loss)))
    if 'acc' in h.history:
        acc = h.history['acc']
        print('acc min: {0}  acc max: {1}'.format(min(acc), max(acc)))

print('Starting training...')
# If we're not starting at zero, then bump current iteration up one, assuming we've loaded weights for the starting iteration
if cur_iter > 0:
    cur_iter += 1
num_iters = cur_iter + args.num_iters
hist = {}
while cur_iter < num_iters:
    # Start training iteration for each model
    print('Iteration: {0}'.format(cur_iter))
    print('Training for {0} epochs (batch size: {1})'.format(args.epochs, batch_size))
    cur_hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=args.epochs, shuffle=True, verbose=1, validation_data=val_data)
    print_hist_stats(cur_hist)
    print('Saving weights for iteration {0} ...'.format(cur_iter))
    model.save_weights(model_basename + str(cur_iter))
    
    hist[cur_iter] = {'gen' : cur_hist.history}
    np.save('metrics-train-{0}.npy'.format(args.run), hist)

    # Clean weights from last iteration, if between persistent save intervals
    last_iter = cur_iter - 1
    if last_iter >= 0 and last_iter % args.interval != 0:
        if os.path.isfile(model_basename + str(last_iter)):
            os.remove(model_basename + str(last_iter))

    cur_iter += 1

print ('Training complete!')

# Deprecated, dynamic validation splitting. Incurs a lot of additional computational overhead per training recurrent_units
# ...
#def prepare_validation_split(x_train, y_train, x_val, y_val, validate_size):
    #if validate_size == 0:
        #return x_train, y_train, x_val, y_val
    #sample_max = x_train.shape[0] - validate_size
    #stride = x_train.shape[0] / validate_size
    #for i in xrange(0, sample_max, stride):
        #x_val = np.concatenate((x_val, np.reshape(x_train[i], (1, x_train.shape[1], x_train.shape[2]))), axis=0)
        #y_val = np.concatenate((y_val, np.reshape(y_train[i], (1, y_train.shape[1], y_train.shape[2]))), axis=0)
        #x_train = np.delete(x_train, i, axis=0)
        #y_train = np.delete(y_train, i, axis=0)
    #return x_train, y_train, x_val, y_val

## Set up validation split
#validate_size = int(round(X_train.shape[0] * validation_split))
#assert X_train.shape[0] == y_train.shape[0]
#X_val = np.zeros((0, X_train.shape[1], X_train.shape[2]))
#y_val = np.zeros((0, y_train.shape[1], y_train.shape[2]))
#print('Splitting data for validation...')
#data_parts = prepare_validation_split(X_train, y_train, X_val, y_val, validate_size)
#X_train = data_parts[0]
#y_train = data_parts[1]
#X_val = data_parts[2]
#y_val = data_parts[3]
