from data_utils.parse_files import *
import config.nn_config as nn_config

import argparse

parser = argparse.ArgumentParser(description="Convert MP3 files to WAV and prepare training data. Audio data should be placed in ./datasets/<set-name>/audio/")
parser.add_argument("name", type=str, help="Name of the dataset; should match the name of the directory in './datasets'")
parser.add_argument("-v", "--validation", default=0.2, type=float, help="Validation split. Defaults to 0.2")
parser.add_argument("--skip-conv", action='store_true', default=False, help="Skip conversion to WAV and just generate data. This assumes the WAV files are already present.")
parser.add_argument("--sample-freq", default=44100, type=int, help="Sampling frequency in Hz")
args = parser.parse_args()

input_directory = '{0}/{1}/audio/'.format('datasets', args.name)

freq = args.sample_freq #sample frequency in Hz
clip_len = 10 		#length of clips for training. Defined in seconds
block_size = freq / 4 #block sizes used for training - this defines the size of our input state
max_seq_len = int(round((freq * clip_len) / block_size)) #Used later for zero-padding song sequences
#Step 1 - convert MP3s to WAVs
if not args.skip_conv:
    new_directory = convert_folder_to_wav(input_directory, freq)
else:
    new_directory = input_directory + 'wave' + '/'

output_filename = args.name

#Step 2 - convert WAVs to frequency domain with mean 0 and standard deviation of 1
convert_wav_files_to_nptensor(new_directory, block_size, max_seq_len, output_filename, validation_split=args.validation)
