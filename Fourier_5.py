import numpy as np
import matplotlib.pyplot as plt

sample_frequency = 100E06  # Hz
sample_time = 1/sample_frequency  # 采样间隔 s
N = 1000  # 采样1000个点
T = sample_time * N  # 采样周期 s

t = np.linspace(0, T, N)  # 时域序列
w1, w2 = 1E06, 1.05E06
ft = np.cos(2*np.pi*w1*t)+np.cos(2*np.pi*w2*t)

fw = np.abs(np.fft.fft(ft, n=8000, norm="forward"))
fe = np.fft.fftfreq(fw.size, sample_time)

fw_positive = fw[fe >= 0]*2
fw_positive[0] /= 2
fe_positive = fe[fe >= 0]

plt.xlim(0, 0.5E7)
plt.plot(fe_positive, fw_positive)
plt.show()
