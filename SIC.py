import numpy as np

np.random.seed(1)

# Number of bits
N = 1000

# Binary signals (BPSK)
s_strong = np.random.choice([-1, 1], N)
s_weak   = np.random.choice([-1, 1], N)

# Power-domain NOMA, u can change them 
P_strong = 0.2
P_weak   = 0.8

# Noise
sigma = 0.1
noise = np.random.normal(0, sigma, N)

# Hard decision (avoid zeros from sign)
def hard_decision(x):
    return np.where(x >= 0, 1, -1)

# Superposed received signal
y = np.sqrt(P_strong)*s_strong + np.sqrt(P_weak)*s_weak + noise

# SIC decoding
s_weak_hat = hard_decision(y)                 # decode high-power user first
y_residual = y - np.sqrt(P_weak)*s_weak_hat   # subtract weak user
s_strong_hat = hard_decision(y_residual)      # decode strong user

# BER calculation
ber_weak = np.mean(s_weak != s_weak_hat)
ber_strong = np.mean(s_strong != s_strong_hat)
