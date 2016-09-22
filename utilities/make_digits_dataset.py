"""
Creates a small digits dataset using a subset of the TIDIGITS dataset.
"""

import re
import os
import h5py
import numpy as np
import soundfile as sf

gender      = 'man'
source_dir  = 'data/tidigits_flac/data/adults'
output_path = 'output/tidigits_small.hdf5'
digit_pat   = re.compile(r'([z1-9])[ab].flac')

train_inst_per_digit = 60
test_inst_per_digit  = 10

def get_utterance_paths(source_path):
	paths = [[] for _ in range(10)]

	for speaker in os.listdir(source_path):
		if speaker == '.DS_Store':
			continue

		speaker_path = os.path.join(source_path, speaker)

		for utterance in os.listdir(speaker_path):
			if utterance == '.DS_Store':
				continue

			m = digit_pat.match(utterance)

			if not m:
				continue

			utterance_path = os.path.join(speaker_path, utterance)

			char = m.groups()[0]
			digit = 0 if char == 'z' else int(char)
			paths[digit].append(utterance_path)

	return paths

train_source_path = os.path.join(source_dir, 'train', gender)
test_source_path  = os.path.join(source_dir, 'test', gender)

train_digit_paths = get_utterance_paths(train_source_path)
test_digit_paths  = get_utterance_paths(test_source_path)

def print_digit_counts(paths, dataset):
	total = 0
	print("Total number of {} instances for:".format(dataset))

	for i, digit_paths in enumerate(paths):
		total += len(digit_paths)
		print("* Digit {}: {}.".format(i, len(digit_paths)))

	print("Total: {}.\n".format(total))

print_digit_counts(train_digit_paths, 'train')
print_digit_counts(test_digit_paths, 'test')

def sample_digit_paths(paths, inst_per_digit):
	assert len(paths) == 10
	new_paths = [[] for _ in range(10)]

	for i, digit_paths in enumerate(paths):
		indices = np.random.permutation(len(digit_paths))[:inst_per_digit]

		for index in indices:
			new_paths[i].append(digit_paths[index])

	return new_paths

train_set_paths = sample_digit_paths(train_digit_paths, train_inst_per_digit)
test_set_paths = sample_digit_paths(test_digit_paths, test_inst_per_digit)

output_file = h5py.File(output_path, 'w')
train_group = output_file.create_group('train')
test_group  = output_file.create_group('test')

def write_audio(paths, dst_group):
	for i in range(10):
		g = dst_group.create_group('digit_{}'.format(i))

		for j, path in enumerate(paths[i]):
			audio, _ = sf.read(path)
			g.create_dataset(str(j), data=audio)

write_audio(train_set_paths, train_group)
write_audio(test_set_paths, test_group)
