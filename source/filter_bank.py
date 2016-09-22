"""
Utilities for creating filter bank configurations.
"""

import math
import numpy as np

from enum import Enum
from scipy import fftpack

def _freq_to_mel(f):
	return 700 * (10 ** (f / 2595) - 1)

def _mel_to_freq(m):
	return 2595 * math.log10(1 + m / 700)

def _generate_sample_times(sample_count, sample_freq):
	"""
	Generates a list of uniformly-spaced points along the time axis centered at the origin. Note
	that if ``sample_count`` is even, then the origin will not actually be in the list.
	"""

	sample_period = 1 / sample_freq
	extent = sample_period * (sample_count - 1) / 2
	t = np.arange(-extent, extent + sample_period, sample_period)

	"""
	Due to rounding issues, `arange` may "undershoot" the right endpoint using the given stride,
	leading to an extra sample point. Is there a more elegant way to deal with this?
	"""
	if len(t) > sample_count:
		t = t[:-1]
		
	return t

def _make_rabiner_band_pass_filter(bin_freqs, sample_freq, sample_count, beta=4.864):
	"""
	Creates a band-pass filter of the kind described in the paper "On the Effects of Varying
	Filter Bank Parameters on Isolated Word Recognition", by Dautrich, Rabiner, and Martin.
	"""

	f_start, f_center, f_stop = bin_freqs
	filter_width = f_stop - f_start
	nyquist_freq = sample_freq / 2

	assert nyquist_freq >= f_stop > f_center > f_start >= 0
	assert filter_width < nyquist_freq / 2
	assert sample_count > 0
	assert beta > 0

	"""
	If we create the filter directly at the requested frequency, then numerical inaccuracies
	will lead to the filters in a uniform filter bank being slightly asymmetrical. As a
	consequence, the overall frequency response will be bumpier. To work around this, we center
	all filters at half the nyquist frequency, and modulate them to their desired locations.
	"""
	base_center = nyquist_freq / 2
	base_start = base_center - filter_width / 2
	base_stop = base_center + filter_width / 2

	t = _generate_sample_times(sample_count, sample_freq / 2)
	f_1, f_2 = (f / math.pi * np.sinc(f * t) for f in [base_start, base_stop])

	w = np.kaiser(sample_count, beta)
	F = np.abs(np.fft.fft(w * (f_2 - f_1) * np.exp(math.pi * (f_start - base_start) * 1j * t)))

	# We zero out the response for any frequency above the Nyquist frequency.
	F[math.ceil((sample_count - 1) / 2):] = 0
	return F

def _make_triangular_band_pass_filter(bin_freqs, sample_freq, sample_count):
	f_start, f_center, f_stop = bin_freqs
	nyquist_freq = sample_freq / 2

	assert nyquist_freq >= f_stop > f_center > f_start >= 0
	assert sample_count > 0

	sample_freqs = np.linspace(0, sample_freq, sample_count)
	step = sample_freq / (sample_count - 1)

	"""
	Since `f_start`, `f_center`, and `f_stop` likely to do not coincide with grid points, the
	filter's response will never reach one or decay to zero symmetrically on both sides. To
	circumvent this issue, we parameterize the triangular curve using the sample frequencies
	closest to the specified values, rather than the specified values themselves.
	"""
	f_l, f_c, f_r = (step * round(f / step) for f in [f_start, f_center, f_stop])

	for f in [f_l, f_c, f_r]:
		assert f in sample_freqs

	a_1, b_1 =  1 / (f_c - f_l), -f_l / (f_c - f_l)
	a_2, b_2 = -1 / (f_r - f_c),  f_r / (f_r - f_c)

	def tent(f):
		if f <= f_l or f >= f_r:
			return 0
		elif f > f_l and f <= f_c:
			return a_1 * f + b_1
		else:
			return a_2 * f + b_2

	return np.array([tent(f) for f in sample_freqs])

"""
Filter sequence definitions.
"""

class LinearFilterSequence:
	"""
	Generates a list of filter locations, where the centers of two consecutive are ``stride``
	units in frequency apart.
	"""

	def __init__(self, f_start, stride, count, width=None):
		assert f_start >= 0
		assert stride > 0
		assert count >= 1
		
		self.f_start = f_start
		self.stride = stride
		self.count = count
		
		if width is not None:
			self.width = width
		else:
			self.width = 2 * stride
		
	@classmethod
	def from_range(class_, f_start, f_stop, count, width=None):
		assert f_start < f_stop
		return class_(f_start, (f_stop - f_start) / (count + 1), count, width=width)
	
	def __iter__(self):
		self.k = 0
		return self
		
	def __next__(self):
		if self.k == self.count:
			raise StopIteration()

		assert self.k < self.count
		center = self.f_start + self.width / 2 + self.k * self.stride
		
		self.k += 1
		return center - self.width / 2, center, center + self.width / 2

class RabinerFilterSequence:
	"""
	A variant of ``LinearFilterSequence`` that is more appropriate for Rabiner-type band-pass
	filters.
	"""

	def __init__(self, f_start, stride, count, width=None):
		assert f_start >= 0
		assert stride > 0
		assert count >= 1
		
		self.f_start = f_start
		self.stride = stride
		self.count = count
		
		if width is not None:
			self.width = width
		else:
			self.width = stride
			
	@classmethod
	def from_range(class_, f_start, f_stop, count, width=None, delta_l=0, delta_r=0):
		assert f_start < f_stop
		return class_(f_start + delta_l, (f_stop - f_start - (delta_l + delta_r)) / (count + 1),
			count, width=width)
	
	def __iter__(self):
		self.k = 1
		return self
		
	def __next__(self):
		if self.k == self.count + 1:
			raise StopIteration()

		assert self.k < self.count + 1
		center = self.f_start + self.k * self.stride
		
		self.k += 1
		return center - self.width / 2, center, center + self.width / 2

class MelFilterSequence:
	"""
	Adapts a existing filter sequence to output frequencies along the mel scale.
	"""

	def __init__(self, source):
		self.source = source
		
	def __iter__(self):
		self.source.__iter__()
		return self
	
	def __next__(self):
		return (_freq_to_mel(f) for f in self.source.__next__())

"""
Filter type definitions.
"""

def _fix_stop_freq(f_stop, nyquist_freq):
	if f_stop > nyquist_freq and f_stop - nyquist_freq < 1e-10:
		print("Warning: f_stop ({}) > nyquist_freq ({}). The difference is small ({}), "
			"so f_stop will be clamped to nyquist_freq.".format(f_stop, nyquist_freq,
			f_stop - nyquist_freq))
		return nyquist_freq

	return f_stop

class RabinerBandPass:
	def __init__(self, size, sample_freq, beta=4.864):
		self.size = size
		self.sample_freq = sample_freq
		self.beta = beta
	
	def __call__(self, f_start, f_center, f_stop):
		f_stop = _fix_stop_freq(f_stop, self.sample_freq / 2)
		return _make_rabiner_band_pass_filter((f_start, f_center, f_stop),
			self.sample_freq, self.size, self.beta)
		
class TriangularBandPass:
	def __init__(self, size, sample_freq):
		self.size = size
		self.sample_freq = sample_freq
	
	def __call__(self, f_start, f_center, f_stop):
		f_stop = _fix_stop_freq(f_stop, self.sample_freq / 2)
		return _make_triangular_band_pass_filter((f_start, f_center, f_stop), self.sample_freq,
			self.size)

"""
API functions.
"""

def make_filter_bank(filter_seq, filter_type):
	return [filter_type(f_start, f_center, f_stop) for (f_start, f_center, f_stop) in filter_seq]

"""
Note: due to the unpredictable response decay of filters created using the window design method, the
resulting filter bank will typically have the following problems:

  1. There will typically be gaps between the left and right edges of the overall frequency response
     and ``f_start`` and ``f_stop``, respectively. That is, the filter bank will not span the
     entirety of the requested frequency range.
  2. There will typically be gaps between the edges of adjacent filters in the filter bank.

I have found that the following procedure works well to address these problems:

  - Create two plots: one with the responses of the individual filters superimposed, and another
    with the overall frequency response.
  - Increase ``beta`` until there is no gap between the edges of adjacent filters.
  - Increase ``delta_l`` and ``delta_r`` until there are no gaps between the edges overall frequency
    response and ``f_start`` and ``f_stop``.
"""
def make_uniform_rabiner_filter_bank(f_start, f_stop, sample_freq, filter_count, window_length,
	beta=4.864, delta_l=0, delta_r=0):
	
	responses = make_filter_bank(
		RabinerFilterSequence.from_range(f_start, f_stop, filter_count, delta_l=delta_l,
			delta_r=delta_r),
		RabinerBandPass(window_length, sample_freq, beta=beta)
	)
	
	mag_max = max(np.max(np.abs(resp)) for resp in responses)
	
	for resp in responses:
		resp /= mag_max
		
	return responses

def make_uniform_triangular_filter_bank(f_start, f_stop, sample_freq, filter_count, window_length):
	return make_filter_bank(
		LinearFilterSequence.from_range(f_start, f_stop, filter_count),
		TriangularBandPass(window_length, sample_freq)
	)

def make_mel_triangular_filter_bank(f_start, f_stop, sample_freq, filter_count, window_length):
	return make_filter_bank(
		MelFilterSequence(LinearFilterSequence.from_range(_mel_to_freq(f_start),
			_mel_to_freq(f_stop), count=filter_count)),
		TriangularBandPass(window_length, sample_freq)
	)

"""
See the notes for ``make_uniform_rabiner_filter_bank`` for information on how to adjust the
parameters.
"""
def make_mel_rabiner_filter_bank(f_start, f_stop, sample_freq, filter_count, window_length,
	beta=4.864, delta_l=0, delta_r=0):
	
	responses = make_filter_bank(
		MelFilterSequence(RabinerFilterSequence.from_range(_mel_to_freq(f_start),
			_mel_to_freq(f_stop), filter_count, delta_l=delta_l, delta_r=delta_r)),
		RabinerBandPass(window_length, sample_freq, beta=beta)
	)
	
	mag_max = max(np.max(np.abs(resp)) for resp in responses)
	
	for resp in responses:
		resp /= mag_max
		
	return responses

class WindowType(Enum):
	rectangular = 0
	hamming     = 1

"""
Computes the cepstral coefficients from the given signal by using the following procedure:
  - Split the signal into chunks separated by the given stride. The chunk size is inferred from the
    length of the filters in the filter bank.
  - Apply the specified window to each chunk, and compute its power spectrum.
  - For each chunk and each filter, compute the log of the dot product of the filter's response with
    the power spectrum of the chunk. This yields a set of coefficients for each chunk, one for each
    each filter.
  - For each chunk, compute the DCT-II of the coefficients obtained from the previous step. This
    gives us the cepstral coefficients.
"""
def apply_filter_bank(filter_resps, signal, stride, window_type=WindowType.hamming):
	channels = len(filter_resps)
	assert channels > 0
	assert stride > 0
	assert type(window_type) is WindowType
	
	chunk_size = len(filter_resps[0])

	for resp in filter_resps[1:]:
		assert len(resp) == chunk_size
	
	magnitudes   = [np.abs(resp) for resp in filter_resps]
	chunk_buffer = np.empty((chunk_size))
	chunk_count  = math.ceil(len(signal) / stride)
	features     = np.empty((chunk_count, channels))
	
	window = None
	
	if window_type == WindowType.hamming:
		window = np.hamming(chunk_size)
	
	for i, a in enumerate(range(0, len(signal), stride)):
		b = min(a + chunk_size, len(signal))
		np.copyto(dst=chunk_buffer[:b - a], src=signal[a:b])

		# We choose to zero-pad the last window if necessary, rather than discard it.
		chunk_buffer[b:] = 0

		if window is not None:
			chunk_buffer = chunk_buffer * window

		# TODO: we could optimize this by using ``rfft`` and expanding the result.
		S = np.abs(np.fft.fft(chunk_buffer)) ** 2

		for j, resp in enumerate(filter_resps):
			features[i][j] = np.log(np.sum((S * resp)))

		features[i] = fftpack.dct(x=features[i], overwrite_x=True)

	return features
