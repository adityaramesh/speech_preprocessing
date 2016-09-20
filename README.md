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

- Find best filter bank for cepstral coefficient representation
  - [ ] What is the best number of filters to use? Is it better to keep or discard the first DCT-II
        coefficient? What is the best number of coefficients to keep?
  - [ ] Does using rectangular vs Hamming window when computing the STFT matter?
  - [ ] Which norm works the best?
  - [ ] How to determine horizontal and vertical penalties?

- [ ] Function to actually warp one utterance to another given the output of DTW. This requires a
      function to supersample audio.

- [ ] Implement deltas and double deltas.

- Later:
  - [ ] Implement LPC.
  - [ ] Implement PLP.
