"""
Converts the TI46 dataset from SPH to WAV format.
"""

import os
import subprocess

dataset_path = 'data/ti46'
audio_dirs = ['ti20/train', 'ti20/test', 'ti_alpha/train', 'ti_alpha/test']

for data_dir in audio_dirs:
	data_path = os.path.join(dataset_path, data_dir)
	assert os.path.isdir(data_path)

	print("Working on directory '{}'.".format(data_dir))
	count = len([d for d in os.listdir(data_path)])

	for i, speaker_dir in enumerate(os.listdir(data_path)):
		print("Working on speaker {} / {}.".format(i + 1, count))

		if speaker_dir == '.DS_Store':
			continue

		speaker_path = os.path.join(data_path, speaker_dir)
		
		for sph_file in os.listdir(speaker_path):
			if sph_file == '.DS_Store':
				continue

			basename, ext = os.path.splitext(sph_file)
			sph_path = os.path.join(speaker_path, sph_file)

			if ext == '.wav':
				continue
			elif ext != '.sph':
				print("Warning: file {} does not have extension 'sph'; skipping.".
					format(sph_path))
				continue

			wav_path = os.path.join(speaker_path, basename + '.wav')
			#print("{} => {}".format(sph_path, wav_path))
			subprocess.call(['utilities/sph2pipe_v2.5/sph2pipe', '-f', 'rif', sph_path, wav_path])
