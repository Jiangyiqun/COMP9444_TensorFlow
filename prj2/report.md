# COMP9444 Assignment 2 Report

### learning rate

learning rate | accuracy
--------------|---------
0.0001        | 79% (continue grow)
0.001         | 81%
0.01          | 81%
0.1           | could not converge in training

I think a good learning rate should between 0.0001 and 0.1. I pick 0.01, because it trains faster than 0.001.

### number of hidden unit

No. of unit | accuracy
------------|---------
16          | 79%
32          | 82.6%
64          | 82.3%
128         | 81%

I pick the highest accuracy one, which is 32 hidden units.

### Model

Model       | accuracy
------------|---------
LSTM        | 82.6% (13s / 100 iteration)
GRU         | 82.8% (20s / 100 iteration)
BIO-LSTM    | 81.0% (40s / 100 iteration)

since the performance is similar, I pick LSTM because it trains faster.

### Other
I have tried to increase MAX_WORDS_IN_REVIEW to 200, or increase BATCH_SIZE from 128 to 1250. Training become much slower, but accuracy does not improve much.