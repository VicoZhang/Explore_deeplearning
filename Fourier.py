import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

# case1:
# f(t) = \begin{cases}
# 0 \quad -5 \le t < -2 \\
# 3 \quad -2 \le t \le 2 \\
# 0 \quad 2 \le t < 5
# \end{cases}
# 做出其函数图像，利用fft变换得到的傅里叶指数形式，以及利用公式得到的图像
# 经计算：其傅里叶指数为：
# f(t) = \frac{6}{5}+\sum_{-\infty, n\ne0}^{\infty} \frac{3}{5} \times \frac{1}{n\omega} \times \sin(2n\omega)

# t = np.linspace(start=-5, stop=5, num=500)
# ft = [(3 if np.abs(_) <= 2 else 0) for _ in t]
# omega = 2 * np.pi / 10

# ft_1 = np.zeros_like(t, dtype=complex)
# for i in range(-100, 100):
#     if i:
#         ft_1 += 3 / (np.pi * i) * np.sin((2*i*np.pi)/5) * np.exp(1j * i * np.pi * t / 5)
#     else:
#         ft_1 += 6/5
# plt.plot(t, np.abs(ft_1))
# plt.show()


# ft_2 = np.fft.fft(ft)
# plt.plot(ft_2)
# plt.show()


# case2:
# f(x) = \begin{cases}
# \quad 0    &\quad x<0 \\
# \quad e^{-t} &\quad x \ge 0\end{cases}
# 单边衰减指数函数

x = np.linspace(-5, 5, 1000)
fx = np.array([np.exp(-i) if i > 0 else 0 for i in x])
# plt.grid()
# plt.plot(x, fx)
# plt.show()

w = np.linspace(-10, 10, 1000)
fw = -1 / (1j * w + 1) * (np.exp(-(1j * w + 1) * 5) - 1)

# plt.grid()
# plt.plot(w, np.abs(fw))
# plt.show()

ft = np.fft.fft(fx)
ft_1 = ft[ft >= 0]
fe = np.fft.fftfreq(x.size, x[1] - x[0])
fe_1 = fe[ft >= 0]
A = 2/len(x) * np.abs(ft_1)
plt.xlim(-10, 10)
plt.plot(fe_1, A)
plt.show()
