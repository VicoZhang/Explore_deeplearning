import numpy as np
import matplotlib.pyplot as plt
import scipy.fft

N = 1000
sample_frequency = 1000
sample_time = 1/sample_frequency
signal_len = N * sample_time

t = np.linspace(0, signal_len, N)

ft = 2 * np.cos(2 * 10 * t) + 4 * np.sin(2 * 30 * t)

ft[500:510] += 5 * np.cos(2 * np.pi * 100 * np.linspace(0, 1, 10))

# plt.plot(t, ft)
# plt.show()

fw = np.abs(scipy.fft.fft(ft, norm="forward"))
fe = scipy.fft.fftfreq(fw.size, sample_time)

fw_positive = fw[fe >= 0] * 2
fe_positive = fe[fe >= 0] * 2
fw_positive[0] /= 2

plt.xlim(0, 500)
plt.plot(fe_positive, fw_positive)
plt.show()
