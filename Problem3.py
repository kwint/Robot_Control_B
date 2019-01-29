from __future__ import division, print_function
import numpy as np
import control.matlab as cm
import control as cp
import matplotlib.pyplot as plt
from IPython.display import Image

s = cm.tf([1, 0], [1])
G = 1/(s*(3*s + 3))

KPKD = [(16, 7), (9, 9), (1200, 117)] # The pairs of PD gains

plt.figure(figsize=(10, 8))

for K in KPKD:
    KP, KD = K
    KI = 4
    # To construct the closed-loop transfer function Gcl from theta^d to theta ,
    # we have a number of options:
    # (1) we can use the command "feedback" as introduced earlier.


    ######
    Gcl = cm.feedback((KP + KD * s) * G, 1) # PD controller
    # Gcl = cm.feedback((KP + KD * s + KI/s) * G, 1) # PID controller
    print(Gcl)
    print(cm.pole(Gcl))
    cm.damp(Gcl, True)
    ######

    # (2) If you are interested to know how Eq. (6.18) is derived,
    # you may want to use the following general rule:
    # +++++++++++++++++++
    # Gcl = <Transfer function of the open loop between the input and output>/(1
    # + <Transfer function of the closed loop>)
    # +++++++++++++++++++
    # Accordingly:

    ######
    # Gcl = (KP*G + KD*s*G)/(1 + KP*G + KD*s*G)
    ######
    # Gd = (G) / (1 + (KP + KD * s) * (G))
    Gd = (G) / (1 + (KP + KD * s + KI/s) * (G))


    # Now, according to the Example, we calculate the response of
    # the closed loop system to a step input of size 10:
    theta, t = cm.step(10 * Gcl, np.linspace(0, 6, 200))
    theta_d, t = cm.step(-20 * Gd, np.linspace(0, 6, 200))
    theta_new = theta + theta_d
    theta_new = theta


    plt.plot(t, theta_new, lw=2)
    plt.legend(['($K_\mathrm{p}$, $K_\mathrm{d}$) = ' + str(KPKD[0]), '($K_\mathrm{p}$, $K_\mathrm{d}$) = ' + str(KPKD[1]),
                '($K_\mathrm{p}$, $K_\mathrm{d}$) = ' + str(KPKD[2])])
    plt.xlabel('Time')
    plt.ylabel('Position')
plt.show()