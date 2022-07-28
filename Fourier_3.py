import numpy as np
from matplotlib import pyplot as plt

# 产生两个工频周波
sample_f = 1000  # Hz
T_s = 1 / 1000  # 1ms
T = 0.02  # 20ms
w = 2 * np.pi / T
t = np.arange(0, T, 0.001)  # 选择一个周波，在一个周波上采样 0.2/T_s 个点
ft = np.sin(3 * w * t)
# plt.grid()
# plt.plot(t, ft)
# plt.show()

fw = np.abs(np.fft.fft(ft, n=(np.ceil(np.log2(np.abs(int(T / T_s)))).astype("int"))**2, norm="forward"))
fe = np.fft.fftfreq(fw.size, t[1] - t[0])

fw_positive = fw[fe >= 0]
fw_positive[0] /= 2
fe_positive = fe[fe >= 0]

print(np.fft.fftshift(fw_positive))
print(fe_positive)
plt.grid()
plt.plot(fe_positive, fw_positive)
plt.show()
