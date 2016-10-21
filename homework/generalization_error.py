import numpy as np
import matplotlib.pyplot as plt

Ns = np.arange(1.,100.)
delta = 0.05
d = 50

vc = [np.sqrt( (8/N) * np.log((4*(2*N**d) / delta)) ) for N in Ns]

plt.plot(Ns, vc)
plt.show()
