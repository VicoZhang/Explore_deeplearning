import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
import scipy.signal

N = 4000  # 共采样1024个点
sample_freq = 1000  # 采样频率[Hz]
sample_t = 1 / sample_freq  # 采样间隔[s]
signal_len = N * sample_t  # 信号长度[s]
t = np.linspace(0, signal_len, N)  # 信号时域序列[s]

ft = np.zeros_like(t)
ft[:1000] = 10 * np.cos(2 * np.pi * 10 * t[:1000])
ft[1000:2000] = 20 * np.cos(2 * np.pi * 20 * t[1000:2000])
ft[2000:3000] = 30 * np.cos(2 * np.pi * 30 * t[2000:3000])
ft[3000:4000] = 40 * np.cos(2 * np.pi * 40 * t[3000:4000])

fw = scipy.signal.stft(ft, sample_freq)

# plt.pcolormesh(fw[1], fw[0], np.abs(fw[2]))
plt.pcolor(fw[1], fw[0], np.abs(fw[2]))
plt.show()
