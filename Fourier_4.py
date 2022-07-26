import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

N = 1024  # 共采样1024个点
sample_freq = 120  # 采样频率[Hz]
sample_t = 1 / sample_freq  # 采样间隔[s]
signal_len = N * sample_t  # 信号长度[s]
t = np.arange(0, signal_len, sample_t)  # 信号时域序列[s]

ft = 5 + 2 * np.sin(2 * np.pi * 20 * t) + 3 * np.sin(2 * np.pi * 30 * t) \
     + 4 * np.sin(2 * np.pi * 40 * t)  # 信号，有三个频率分量，20，30，40Hz，一个直流分量

fw_original = fftpack.fft(ft)  # 原始的快速傅里叶变换数据

# fw_temp = fftpack.fftshift(fw_original)
fw = 2 / N * np.abs(fw_original)
fw[0] = fw[0] / 2  # 对原始快速傅里叶变换的处理，解决幅值问题和直流分量的问题

fe = fftpack.fftfreq(N, sample_t)  # 计算频域分量

fw_positive = fw[fe >= 0]
fe_positive = fe[fe >= 0]  # 单边频域图


# 绘图
# plt.figure()
# plt.title('ft-t')
# plt.xlabel('t/s')
# plt.ylabel('ft')
# plt.plot(t, ft)

# plt.title('fw-fe')
# plt.plot(fftpack.fftshift(fe), fftpack.fftshift(fw))
# 关于底部的直线，其实是画图过程中，由于先从0开始向正半轴画，最后返回的负半轴，所以多出来一条线，这是由fftfreq导致的,必须搭配fftshift函数
# plt.xlabel('fe/Hz')
# plt.ylabel('fw')

# plt.title('fw_positive-fe_positive')
# plt.plot(fe_positive, fw_positive)
# plt.xlabel('fe/Hz')
# plt.ylabel('fw')

# plt.show()

print("最小时间刻度Ts={}".format(t[1]-t[0]))
print("采样间隔={}".format(sample_t))
print("最小频域刻度fs={}".format(fe[1]-fe[0]))
print("采样周期的倒数={}".format(1/signal_len))
print("最大频域={}".format(sample_freq/2))
print("最大频域={}".format(fftpack.fftshift(fe)[-1]))


