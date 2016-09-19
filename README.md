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
  - [ ] Test on toy grid example (make a separate notebook called `warp.py`).
  - [ ] Test with MFCC, using both two-norm and infinity-norm.
  - [ ] Function to actually warp one utterance to another given the output of DTW.

- Find best filter bank for cepstral coefficient representation
  - [ ] Find best number of filters to use.
  - [ ] Which is the best norm to use?
  - [ ] How to determine horizontal and vertical penalties?

- Implement deltas and double deltas
- Implement LPC
- Implement PLP
