import numpy as np
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

t = np.linspace(start=-5, stop=5, num=500)
ft = [(3 if np.abs(_) <= 2 else 0) for _ in t]
omega = 2 * np.pi / 10

ft_1 = np.zeros_like(t, dtype=complex)
for i in range(-100, 100):
    if i:
        ft_1 += 3 / (np.pi * i) * np.sin((2*i*np.pi)/5) * np.exp(1j * i * np.pi * t / 5)
    else:
        ft_1 += 6/5
plt.plot(t, np.abs(ft_1))
plt.show()
