"""
Evaluates the performance of DTW using a file containing the precomputed features. Metrics reported:
* Top-1 accuracy.
* Top-3 accuracy.
"""

import os
import sys
import h5py
import numpy as np

from queue import Queue
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.getcwd())
from source.warp import *

input_file  = h5py.File('output/tidigits_small_features.hdf5', 'r')
train_group = input_file['train']
test_group  = input_file['test']

train_utt   = []
test_utt    = []
train_dists = []

def read_features(group, dest):
	for i in range(10):
		digit_group = group['digit_{}'.format(i)]
		
		for instance_name in digit_group:
			features = digit_group[instance_name][:]
			dest.append((i, features))

read_features(train_group, train_utt)
read_features(test_group, test_utt)
print(len(train_utt), len(test_utt))

def compute_distance(input_index, input_count, input_, templates, queue):
	input_digit, input_features = input_
	assert 0 <= input_digit <= 9

	print("Working on utterance {} / {}.".format(input_index, input_count))
	dists = []

	for j, template in enumerate(templates):
		template_digit, template_features = template
		assert 0 <= template_digit <= 9

		dist_func = lambda u, v: np.linalg.norm(input_features[u] -
			template_features[v])
		_, dist = shortest_path(width=input_features.shape[0],
			height=template_features.shape[0], dist=dist_func)

		assert dist >= 0
		dists.append((j, dist))

	return queue.put((input_index, dists))

def compute_distances(inputs, templates, dists):
	queue = Queue()

	with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
		for i, input_ in enumerate(inputs):
                        executor.submit(compute_distance, i, len(inputs), input_, templates, queue)

	results = []

	for _ in range(len(inputs)):
		results.append(queue.get())

	results.sort(key=lambda x: x[0])
	return [x[0] for x in results]

compute_distances(train_utt, test_utt, train_dists)

def digit_in_top_k(digit, k, dists):
	for i in range(k):
		template_id, _ = dists[i]
		template_digit, _ = test_utt[template_id]
		assert 0 <= template_digit <= 9

		if digit == template_digit:
			return True

	return False

def print_scores(group, group_dists):
	assert len(group) == len(group_dists)
	top_1_count = 0
	top_3_count = 0

	for i in range(len(group_dists)):
		digit, _ = group[i]
		assert 0 <= digit <= 9
		dists = sorted(group_dists[i], key=lambda x: x[1])

		if digit_in_top_k(digit, 1, dists):
			top_1_count += 1
			top_3_count += 1
		elif digit_in_top_k(digit, 3, dists):
			top_3_count += 1

	print("Top-1 accuracy: {}.".format(top_1_count / len(group_dists)))
	print("Top-3 accuracy: {}.".format(top_3_count / len(group_dists)))

print_scores(train_utt, train_dists)
