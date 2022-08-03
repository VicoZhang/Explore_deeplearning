import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
from scipy import fft

N = 2000  # 共采样1024个点
sample_freq = 100  # 采样频率[Hz]
sample_t = 1 / sample_freq  # 采样间隔[s]
signal_len = N * sample_t  # 信号长度[s]
t = np.linspace(0, signal_len, N)  # 信号时域序列[s]

ft = np.zeros_like(t)
ft[:500] = 10 * np.cos(2 * np.pi * 10 * t[:500])
ft[500:1000] = 20 * np.cos(2 * np.pi * 20 * t[500:1000])
ft[1000:1500] = 30 * np.cos(2 * np.pi * 30 * t[1000:1500])
ft[1500:2000] = 40 * np.cos(2 * np.pi * 40 * t[1500:2000])

fw = np.abs(scipy.fft.fft(ft[::-1], norm="forward"))
fw_1 = np.abs(scipy.fft.fft(ft, norm="forward"))
fe = scipy.fft.fftfreq(fw.size, sample_t)

fw_positive = fw[fe > 0]
fw_positive[1:] *= 2
fe_positive = fe[fe > 0]

plt.figure()
plt.plot(fe_positive, fw_positive, linewidth=0.5)
plt.show()

print(np.sum(fw_1 - fw))
