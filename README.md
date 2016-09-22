# Overview

An evaluation of the following preprocessing methods for speech using dynamic time warping:
- Cepstral coefficients using various filter bank coefficients.
- LPC.
- PLP.

The preprocessing schemes are evaluated based on recognition accuracy of spoken digits, using the
TIDIGITS data set.

# Notes on Microsoft's Switchboard Paper

- Ensemble of convolutional and recurrent architectures.
  - Three types of convolutional architectures: VGG, ResNet, and LACE.
  - RNNs used: BLSTMs with at most six layers, without frame skipping.
  - Acoustic features are log-filterbank values (e.g. MFCCs) computed every 10 ms using a 25 ms
    window.
- Use of i-vectors:
  - For RNNs, the i-vector is appended to the features for each frame.
  - For CNNs, the i-vector is projected and added to the bias of each CNN layer, before the
    activation function.
- Use of lattice-free MMI training.
- Decoding is initially done using WFSTs with an n-gram language model.
- The $N$ best hypotheses are then rescored using RNN language models. There are many variants:
  - Use of both forward-predicting and backward-predicting RNNs; the log probabilities from both are
    added. (Why not predict the RNN to fill in missing words using beam search instead? This
    generalizes both schemes.)
  - Interpolation of RNN LMs with n-gram LMs (the forward- and backward-predicting RNNs are
    interpolated separately).
  - Pre-training of RNNs on out-of-domain data, and fine-tuning on in-domain data. (Use of
    multi-column networks like in the Deep Mind paper may be helpful here.)
  - Best results with RNN that has second, non-recurrent layer.
  - Training uses NCE.
  - For rescoring purposes, an empirically determined penalty is applied to out-of-domain words.
  - Fisher and Switchboard transcripts were used as in-domain training data.
- Use of 1-bit SGD to efficiently parallelize training.

# Agenda

- Find best filter bank for cepstral coefficient representation
  - [ ] How to determine horizontal and vertical penalties?
    - Unclear how to do this, but they should be nonzero. They should certainly be greater than zero
      and less than the returned minimum distance values, so try a grid search between these two
      numbers.
  - [ ] Does using the first DCT coefficient help?
    - One experiment.
  - [ ] Which norm works the best?
    - One more experiment.

  - [ ] Is it better to use only the DCT-II coefficients up to the 13th?
    - One experiment.
  - [ ] Is it better to use a small or large number of filters?
    - Try 26 and 40; one more experiment. Recompute horizontal and vertical penalties.
  - [ ] Which filter bank is best?
    - Three more experiments, after designing the filter banks using either 26 or 40 filters,
      whichever was found to be more effective.
  - [ ] Does using rectangular vs Hamming window when computing the STFT matter?
    - One more experiment.

- [ ] Function to actually warp one utterance to another given the output of DTW. This requires a
      function to supersample audio.

- [ ] Implement deltas and double deltas.

- Later (if necessary):
  - [ ] Implement and evaluate LPC.
  - [ ] Implement and evaluate PLP.
