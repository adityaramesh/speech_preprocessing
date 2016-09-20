# Overview

An evaluation of the following preprocessing methods for speech using dynamic time warping:
- Cepstral coefficients using various filter bank coefficients.
- LPC.
- PLP.

The preprocessing schemes are evaluated based on recognition accuracy of spoken digits, using the
TIDIGITS data set.

# Agenda

- Implement and test DTW
  - [x] Implement DTW.
  - [x] Test on toy example (make a separate notebook called `warp.py`).
  - [ ] Test with MFCC on TI46, using both two-norm and infinity-norm. Compare performance to that
	of librosa's MFCC implementation. **Ensure that we divide the distance by the length of the
        pattern that we are attempting to recognize.**
    - [x] Script to create small version of the dataset.
    - [ ] Script to compute MFCCs and save as another hdf5 file.
    - [ ] Script to perform the recognition using DTW.
    - [ ] First experiment.

- Find best filter bank for cepstral coefficient representation
  - [ ] How to determine horizontal and vertical penalties?
    - Unclear how to do this, but they should be nonzero. They should certainly be greater than zero
      and less than the returned minimum distance values, so try a grid search between these two
      numbers.
  - [ ] Which norm works the best?
    - One more experiment.

  - [ ] Does using the first DCT coefficient help?
    - One experiment.
  - [ ] Is it better to use only the DCT-II coefficients up to the 13th?
    - One experiment.
  - [ ] Is it better to use a small or large number of filters?
    - Try 26 and 40; one more experiment.
  - [ ] Which filter bank is best?
    - Three more experiments, after designing the filter banks using either 26 or 40 filters,
      whichever was found to be more effective.
  - [ ] Does using rectangular vs Hamming window when computing the STFT matter?
    - One more experiment.

- [ ] Function to actually warp one utterance to another given the output of DTW. This requires a
      function to supersample audio.

- [ ] Implement deltas and double deltas.

- Later:
  - [ ] Implement LPC.
  - [ ] Implement PLP.
