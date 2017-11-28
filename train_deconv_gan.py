from __future__ import absolute_import
from __future__ import print_function
from generate import generate_from_seeds
from keras.callbacks import EarlyStopping, LambdaCallback
import numpy as np
import os
import nn_utils.network_utils as network_utils
import config.nn_config as nn_config
import gen_utils.seed_generator as seed_generator
import gen_utils.sequence_generator as sequence_generator
import tensorflow as tf
import argparse

config = nn_config.get_deconv_gan_configuration()

parser = argparse.ArgumentParser(description="Train the NuGRUV GAN against the current dataset.")
parser.add_argument("dataset_name", help="Name of the dataset to use for training.")
parser.add_argument("-s", "--start-iter", default=0, type=int, help="Current training iteration to start from.")
parser.add_argument("-n", "--num-iters", default=10, type=int, help='Number of training iterations to run.')
parser.add_argument("--dec-epochs", default=5, type=int, help="Number of epochs per iteration to train the decoder.")
parser.add_argument("--gen-epochs", default=2, type=int, help="Number of epochs per iteration of the generator.")
#parser.add_argument("--com-epochs", default=1, type=int, help="Number of epochs per iteration to train the combined GAN model.")
parser.add_argument("--dec-samples", default=10, type=int, help="Number of samples to generate for the decoder to train against on each iteration.")
parser.add_argument("-b", "--batch-size", default=8, type=int, help="Number of seeds to generate per epoch.")
parser.add_argument("--seed-dims", default=256, type=int, help="Dimensions of the generator's seed samples.")
parser.add_argument("--skip-validation", action="store_true", help="Do not use cross validation data.")
parser.add_argument("-i", "--interval", default=10, type=int, help="Number of iterations to run in between retaining saved weights.")
parser.add_argument("-r", "--run", default=0, type=int, help="Integer id for this run (used for weight files). Defaults to zero.")
args = parser.parse_args()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

input_file = config['dataset_directory'] + args.dataset_name + '/' + args.dataset_name
cur_iter = args.start_iter
dec_basename = config['model_basename'] + str(args.run) + '-Dec_'
dec_filename = dec_basename + str(cur_iter)
gen_basename = config['model_basename'] + str(args.run) + '_'
gen_filename = gen_basename + str(cur_iter)
skip_validation = args.skip_validation
batch_size = args.batch_size

# Load up the training data
print('Loading training data')
# X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
# y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
X_train = np.load(input_file + '_x.npy')
y_train = np.load(input_file + '_y.npy')
if not skip_validation:
    X_val = np.load(input_file + '_val_x.npy')
    y_val = np.load(input_file + '_val_y.npy')
print('Finished loading training data')

# Figure out how many frequencies and timesteps we have in the data
num_timesteps = X_train.shape[1]
freq_space_dims = X_train.shape[2]

# Creates a Genearative Adverserial Network (GAN)
print('Initializing network...')
gan = network_utils.create_deconvolutional_gan(args.seed_dims, batch_size, freq_space_dims, num_timesteps, config, stateful=True)

print('Model summary:')
gan.summary()

# Load existing weights if available
if os.path.isfile(dec_filename):
    print('Loading existing weight data (Decoder) from {}'.format(dec_filename))
    gan.decoder.load_weights(dec_filename)
if os.path.isfile(gen_filename):
    print('Loading existing weight data (Generator) from {}'.format(gen_filename))
    gan.generator.load_weights(gen_filename)

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

print('Training set shape: {0}'.format(X_train.shape))
val_data = None
if not skip_validation:
    print('Validation set shape: {0}'.format(X_val.shape))
    val_data = (X_val, y_val)
    
def generate_random_seeds(seed_dims, repeat_count=0, mean=0, std=1):
    seeds = np.random.uniform(low=-1, high=1, size=seed_dims) #np.random.normal(loc=mean, scale=std, size=seed_dims)
    #copies = seeds.copy()
    #for i in xrange(repeat_count - 1):
    #    copies = np.concatenate((copies, seeds), axis=0)
    return seeds

def random_training_examples(X_train, X_val=[], seed_len=1, train_count=1, val_count=0):
    X_val = np.array(X_val)
    num_train_examples = X_train.shape[0]
    num_val_examples = X_val.shape[0]
    train_examples = np.zeros((0,X_train.shape[1]*seed_len,X_train.shape[2]))
    for i in xrange(train_count):
        next_example = seed_generator.generate_copy_seed_sequence(seed_length=seed_len, training_data=X_train)
        train_examples = np.concatenate((train_examples, next_example), axis=0)
    val_examples = np.zeros((0,X_val.shape[1]*seed_len,X_val.shape[2]))
    for i in xrange(val_count):
        next_example = seed_generator.generate_copy_seed_sequence(seed_length=seed_len, training_data=X_val)
        val_examples = np.concatenate((val_examples, next_example), axis=0)
    return (train_examples, val_examples)

def train_decoder(X_train, X_val, callbacks=[]):
    print('Training decoder...')
    train_seeds = generate_random_seeds(seed_dims=(batch_size, 1, args.seed_dims))
    val_seeds = generate_random_seeds(seed_dims=(batch_size, 1, args.seed_dims))
    print(train_seeds.shape)
    print(val_seeds.shape)
    X_train_real, X_val_real = random_training_examples(X_train, X_val, seed_len=1, train_count=batch_size, val_count=batch_size)
    X_train_fake = generate_from_seeds(gan.generator, train_seeds, max_seq_len=num_timesteps, batch_size=batch_size, uncenter_data=False)
    X_val_fake = generate_from_seeds(gan.generator, val_seeds, max_seq_len=num_timesteps, batch_size=batch_size, uncenter_data=False)
    print(X_train_real.shape)
    print(X_train_fake.shape)
    dec_hist = gan.fit_decoder(X_train_real, X_train_fake, batch_size=batch_size, epochs=args.dec_epochs, shuffle=False, verbose=1, callbacks=callbacks, validation_data=(X_val_real, X_val_fake))
    
reset_states = LambdaCallback(on_epoch_end=lambda epoch,logs: gan.generator.reset_states())
early_stop = EarlyStopping(monitor='acc', min_delta=0.01, patience=2, verbose=1, mode='auto')
    
# Training phase 1: Generator pre-training
print('Starting training...')
# If we're not starting at zero, then bump current iteration up one, assuming we've loaded weights for the starting iteration
if cur_iter > 0:
    cur_iter += 1
num_iters = cur_iter + args.num_iters
while cur_iter < num_iters:
    # Start training iteration for each model
    print('Iteration: {0}'.format(cur_iter))
    train_seeds = generate_random_seeds(seed_dims=(batch_size, 1, args.seed_dims))
    val_seeds = None
    if not args.skip_validation:
        val_seeds = generate_random_seeds(seed_dims=(batch_size, 1, args.seed_dims))
    #print('Training generator for {0} epochs (batch size: {1})'.format(args.gen_epochs, batch_size))
    #gen_hist = gan.fit_generator(X_train, y_train, batch_size=batch_size, epochs=args.gen_epochs, shuffle=True, verbose=1, validation_data=val_data)
    #print('Saving generator weights (pre-train) for iteration {0} ...'.format(cur_iter))
    #gan.generator.save_weights(gen_basename + str(cur_iter))
    print('Training decoder for {0} epochs'.format(args.dec_epochs))
    dec_hist = train_decoder(X_train, X_val, callbacks=[])
    print('Saving decoder weights for iteration {0} ...'.format(cur_iter))
    gan.decoder.save_weights(dec_basename + str(cur_iter))
    print('Training combined model for {0} epochs'.format(args.gen_epochs))
    gan.fit(train_seeds, batch_size=batch_size, epochs=args.gen_epochs, shuffle=False, verbose=1, validation_x=val_seeds)
    print('Saving generator weights (post-train) for iteration {0} ...'.format(cur_iter))
    gan.generator.save_weights(gen_basename + str(cur_iter))
    gan.generator.reset_states()
    
    # Clean weights from last iteration, if between persistent save intervals
    last_iter = cur_iter - 1
    if last_iter >= 0 and last_iter % args.interval != 0:
        if os.path.isfile(gen_basename + str(last_iter)):
            os.remove(gen_basename + str(last_iter))
        if os.path.isfile(dec_basename + str(last_iter)):
            os.remove(dec_basename + str(last_iter))
            
    cur_iter += 1
    
print('Training complete!')

