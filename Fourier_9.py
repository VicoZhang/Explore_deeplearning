import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.signal
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

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

win = 128
fw = scipy.signal.stft(ft, sample_freq, window='blackman', nperseg=win, noverlap=win//8, return_onesided=True)

plt.pcolormesh(fw[1], fw[0], np.abs(fw[2]), vmin=0, cmap='jet')
plt.colorbar()
# x, y = fw[0], fw[1]
# X, Y = np.meshgrid(y, x)
# z = np.abs(fw[2])
# fig = plt.figure()
# ax3d = Axes3D(fig)
# ax3d.plot_surface(X, Y, z, cmap="cool")

plt.show()
