import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

# 产生两个工频周波
sample_f = 1000  # Hz
T_s = 1 / 1000  # 1ms
T = 0.02  # 20ms
w = 2 * np.pi / T
t = np.arange(0, T, 0.001)  # 选择一个周波，在一个周波上采样 0.2/T_s 个点
ft = np.sin(2 * w * t)
# plt.grid()
# plt.plot(t, ft)
# plt.show()

fw = fftpack.fft(ft)
fe = fftpack.fftfreq(int(0.02 / T_s), t[1] - t[0])
plt.grid()
# print(ft)
# print(np.abs(fw))
# print(fe)
# plt.plot(fe[fe > 0], np.abs(fw[fe > 0]))
plt.plot(fe, np.abs(fw))
plt.show()
# print("f_s = 1/T = {}Hz".format(fe[1]-fe[0]))
# print("T_s = {}s".format(t[1]-t[0]))

# 自行设计离散傅里叶变换：
# fk = np.zeros_like(fw)
# for k in range(20):
#     for n in range(len(ft)):
#         fk[k] += ft[n]*np.exp(-1j*2*np.pi*k*n/len(ft))
# print(np.abs(fk))
