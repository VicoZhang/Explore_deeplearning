import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 500)
y = np.sin(2 * np.pi * x)
fw = np.fft.fft(y)
freq = np.fft.fftfreq(x.size, x[1] - x[0])
fe = freq
A = 2 / len(x) * np.abs(fw)
A[0] = A[0]/2
plt.xlim(-5, 5)
plt.plot(A)
plt.show()
print(len(fe))
