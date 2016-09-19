"""
Implementation of dynamic time warping.
"""

import math
import numpy as np

"""
Implementation of the symmetric variant of dynamic time warping, in which upward, downward, and
diagonal moves are permitted.

Parameters:
- ``dist``: Should return a non-negative real number given a pair of non-negative integers.
"""
def shortest_path(width, height, dist, h_penalty=0, v_penalty=0):
	assert width >= 1
	assert height >= 1

	dist_map = np.full(shape=(width, height), dtype='float', fill_value=float('nan'))
	index_map = np.full(shape=(width, height, 2), dtype=np.int64, fill_value=-1)
	stack = []

	dist_map[0][0] = dist(0, 0)
	assert dist_map[0][0] >= 0
	stack.append((width - 1, height - 1))

	while len(stack) != 0:
		i, j = stack[-1]
		c_left, c_down, c_diag = float('inf'), float('inf'), float('inf')

		d = dist(i, j)
		assert d >= 0
		
		if i != 0:
			if math.isnan(dist_map[i - 1][j]):
				stack.append((i - 1, j))
				continue
			else:
				c_left = dist_map[i - 1][j] + d + h_penalty

		if j != 0:
			if math.isnan(dist_map[i][j - 1]):
				stack.append((i, j - 1))
				continue
			else:
				c_down = dist_map[i][j - 1] + d + v_penalty

		if i != 0 and j != 0:
			if math.isnan(dist_map[i - 1][j - 1]):
				stack.append((i - 1, j - 1))
				continue
			else:
				c_diag = dist_map[i - 1][j - 1] + 2 * d

		assert math.isnan(dist_map[i][j])
		assert np.all(index_map[i][j] == (-1, -1))

		loc, dist_map[i][j] = min(enumerate([c_left, c_down, c_diag]), key=lambda x: x[1])
		assert dist_map[i][j] >= 0 and dist_map[i][j] != float('inf')
		index_map[i][j] = [(i - 1, j), (i, j - 1), (i - 1, j - 1)][loc]

		stack.pop()

	min_dist = dist_map[width - 1, height - 1]
	assert min_dist != float('inf')

	cur_i, cur_j = width - 1, height - 1
	best_path = [(cur_i, cur_j)]

	while (cur_i, cur_j) != (0, 0):
		cur_i, cur_j = index_map[cur_i][cur_j]
		assert cur_i != -1 and cur_j != -1
		best_path.insert(0, (cur_i, cur_j))
	
	return best_path, min_dist
