import h5py

output_file = h5py.File('output/tidigits_small.hdf5', 'r')

for digit_group in output_file['train']:
	print("********************")
	print(digit_group)
	print("********************")

	for instance in output_file['train'][digit_group]:
		print(instance)
