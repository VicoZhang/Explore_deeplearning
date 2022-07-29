import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

# window = scipy.signal.windows.hann(51, True)
# window = scipy.signal.windows.hamming(51, True)
# window = scipy.signal.windows.flattop(51, True)
# window = scipy.signal.windows.blackman(51, True)
window = scipy.signal.windows.kaiser(51, True)


A = np.abs(np.fft.fft(window, n=2048, norm="forward")) * 2
freq = np.linspace(-0.5, 0.5, len(A))
response = np.fft.fftshift(20*np.log10(np.maximum(A/A.max(), 1e-10)))

plt.axis([-0.5, 0.5, -120, 0])
plt.plot(freq, response)
plt.show()

