from __future__ import division, print_function
import control.matlab as cm
import control as cp
import matplotlib.pyplot as plt
from IPython.display import Image
import numpy as np

end = 8
s = cm.tf([1, 0], [1])
A = np.array([[0, 1, 0, 0],
     [-1, -1, 1, 0],
     [0, 0, 0, 1],
     [1, 0, -1, 1]])

b = np.array([[0], [0], [0], [1]])
c = np.array([1, 0, 0, 0])
d = [0]

SS = cm.feedback(cm.ss(A, b, c, d), 1)

TF_conv = cm.ss2tf(SS)
TF = cm.feedback((1) / (s**4 + s**2))


out_SS, t = cm.step(1 * SS, np.linspace(0, end, 200))
out_TF, t = cm.step(1 * TF, np.linspace(0, end, 200))
out_TF_conv, t = cm.step(1 * TF_conv, np.linspace(0, end, 200))


plt.plot(t, out_SS, lw=2, label="ss")
plt.plot(t, out_TF, lw=1, label="tf")
plt.plot(t, out_TF_conv, lw=0.5, label="tf_conv")
plt.legend()
plt.show()

# https://www.wolframalpha.com/input/?i=(%7B%7Bs,+0,+0,+0%7D,+%7B0,+s,+0,+0%7D,+%7B0,+0,+s,+0%7D,+%7B0,+0,+0,+s%7D%7D+-+%7B%7B0,+1,+0,+0%7D,+%7B-1,+-1,+1,+0%7D,+%7B0,+0,+0,+1%7D,+%7B1,+0,+-1,+1%7D%7D)
# https://www.wolframalpha.com/input/?i=%7B1,0,0,0%7D+*+%7B%7Bs,+-1,+0,+0%7D,+%7B1,+1+%2B+s,+-1,+0%7D,+%7B0,+0,+s,+-1%7D,+%7B-1,+0,+1,+-1+%2B+s%7D%7D%5E-1*%7B%7B0%7D,%7B0%7D,%7B0%7D,%7B1%7D%7D
