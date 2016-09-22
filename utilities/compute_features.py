"""
Computes and saves the features from the file containing the dataset of spoken digit recordings.
"""

import os
import sys
import h5py
import numpy as np

sys.path.append(os.getcwd())
from source.filter_bank import *

input_path  = 'output/tidigits_small.hdf5'
output_path = 'output/tidigits_small_features.hdf5'

input_file        = h5py.File(input_path, 'r')
input_train_group = input_file['train']
input_test_group  = input_file['test']

output_file        = h5py.File(output_path, 'w')
output_train_group = output_file.create_group('train')
output_test_group  = output_file.create_group('test')

sample_freq   = 20000
nyquist_freq  = sample_freq / 2
window_length = round(sample_freq * 0.025)
window_stride = round(sample_freq * 0.010)
filter_count  = 40

filter_bank = make_mel_triangular_filter_bank(0, nyquist_freq, sample_freq, filter_count,
	window_length)

def compute_features(input_group, output_group):
	print("Computing features for '{}'.".format(input_group))

	for i in range(10):
		print("Working on digit {} / {}.".format(i + 1, 10))

		digit_group_name   = 'digit_{}'.format(i)
		input_digit_group  = input_group[digit_group_name]
		output_digit_group = output_group.create_group(digit_group_name)

		for i, instance_name in enumerate(input_digit_group):
			data = np.array(input_digit_group[instance_name], dtype=np.float64)
			f = apply_filter_bank(filter_bank, data, window_stride)
			g = output_digit_group.create_dataset(instance_name, data=f)

compute_features(input_train_group, output_train_group)
compute_features(input_test_group, output_test_group)
