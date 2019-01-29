from __future__ import division, print_function
import numpy as np
import control.matlab as cm
import control as cp
import matplotlib.pyplot as plt
from IPython.display import Image


def control(s, Y, D):
    end = 10
    G = 1/(3*s*(s + 1))
    F = (3*s*(s + 1))

    KP, KD, KI = 16, 7, 4

    H_pd = (KP + KD * s)
    H_pid = (KP + KD * s + KI / s)

    PD = cm.feedback((KP + KD * s) * G, 1) # PD controller
    PID = cm.feedback((KP + KD * s + KI / s) * G, 1)  # PID controller
    PD_FF = cm.tf([1], [1])
    PID_FF = cm.tf([1], [1])  # PID controller

    D_PD = (G) / (1 + (KP + KD * s) * (G))
    D_PID = (G) / (1 + (KP + KD * s + KI / s) * (G))

    # Now, according to the Example, we calculate the response of
    # the closed loop system to a step input of size 10:
    out_PD, t = cm.step(Y * PD, np.linspace(0, end, 200))
    out_PID, t = cm.step(Y * PID, np.linspace(0, end, 200))
    out_PD_FF, t = cm.step(Y * PD_FF, np.linspace(0, end, 200))
    out_PID_FF, t = cm.step(Y * PID_FF, np.linspace(0, end, 200))

    out_PD_D, t = cm.step(-D * D_PD, np.linspace(0, end, 200))
    out_PID_D, t = cm.step(-D * D_PID, np.linspace(0, end, 200))

    theta_PD = out_PD + out_PD_D
    theta_PID = out_PID + out_PID_D
    theta_PD_FF = out_PD_FF + out_PD_D
    theta_PID_FF = out_PID_FF + out_PID_D

    y_out, t = cm.step(Y, np.linspace(0, end, 200))

    plt.plot(t, theta_PD, lw=2, label="PD")
    plt.plot(t, theta_PID, lw=2, label="PID")
    if D != 0:
        plt.plot(t, theta_PD_FF, lw=2, label="PD_FF")
        plt.plot(t, theta_PID_FF, lw=2, label="PID_FF")

    plt.plot(t, y_out, lw=1, label="Reference")
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()


s = cm.tf([1, 0], [1])
y = 5
d = 10

Y = [y,   y,  y,    y/s,  y/s,  (y*2)/(s**2),  (y*2)/(s**2)]
D = [0,   d,  d/s,  0,    d,    0,             d]

title = ["constant reference, no disturbance",
         "constant reference, constant disturbance",
         "constant reference, ramp disturbance",
         "ramp reference, no disturbance",
         "ramp reference, constant disturbance",
         "second-order polynomial reference signal, no disturbance",
         "second-order polynomial reference signal, constant disturbance"]

# plt.figure(figsize=(12, 4*3))

for i in range(len(Y)):
    control(s, Y[i], D[i])
    plt.title(title[i])
    plt.show()


print("lata")