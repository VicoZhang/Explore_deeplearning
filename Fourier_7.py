import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
from scipy import fft

N = 4000  # 共采样1024个点
sample_freq = 100  # 采样频率[Hz]
sample_t = 1 / sample_freq  # 采样间隔[s]
signal_len = N * sample_t  # 信号长度[s]
t = np.arange(0, signal_len, sample_t)  # 信号时域序列[s]

ft = np.zeros_like(t)
ft[:1000] = 10 * np.cos(2 * np.pi * 10 * t[:1000])
ft[1000:2000] = 20 * np.cos(2 * np.pi * 20 * t[1000:2000])
ft[2000:3000] = 30 * np.cos(2 * np.pi * 30 * t[2000:3000])
ft[3000:4000] = 40 * np.cos(2 * np.pi * 40 * t[3000:4000])

fw = np.abs(scipy.fft.fft(ft[::-1], norm="forward"))
fe = scipy.fft.fftfreq(fw.size, sample_t)

fw_positive = fw[fe > 0]
fw_positive[1:] *= 2
fe_positive = fe[fe > 0]

plt.figure()
plt.plot(fe_positive, fw_positive, linewidth=0.3)
plt.show()
