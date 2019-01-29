

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import sin, cos
from IPython.display import HTML


def forward(t, q1, q2):
    toanimate = np.hstack((q1.reshape(len(q1), 1), q2.reshape(len(q2), 1)))

    l = np.ones((2, 1))

    x1 = l[0] * cos(toanimate[:, 0])
    y1 = l[0] * sin(toanimate[:, 0])

    x2 = l[1] * cos(toanimate[:, 0] + toanimate[:, 1]) + x1
    y2 = l[1] * sin(toanimate[:, 0] + toanimate[:, 1]) + y1

    return toanimate, x1, y1, x2, y2


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (t[i]))
    return line, time_text
# Code above gotten from practical assignments

def LSPBTraj(q0, qf, t0, tf, V):
    """
    Compute a LSPB reference trajectory
    q0 = initial position
    qf = final position
    tb = blend time
    tf = final time
    """
    from numpy import matrix, array, linspace
    import numpy as np
    tb = -1* ((qf / V) - tf)
    a = V / tb

    # define the time sequence
    t = linspace(0, tf, 200, endpoint=True)
    # append tb into t
    t = np.append(t, [tb], axis=0)
    t = np.append(t, [tf - tb], axis=0)
    t = np.sort(t)
    #     V =  qf / ( (tf- 2*tb) + tf)
    # V = qf / (tf - tb)

    angle, velocity, acceleration = [], [], []

    for i, time in enumerate(t):
        if time <= tb:
            q = q0 + V / (2 * tb) * time ** 2
            dq = V / (tb) * time
            ddq = V / (tb)
        elif time > tb and time <= tf - tb:
            q = (qf + q0 - V * tf) / 2 + V * time
            dq = V
            ddq = 0
        elif time > tf - tb:
            q = qf - (a * tf ** 2) / 2 + a * tf * time - a / 2 * time ** 2
            dq = a * tf - a * time
            ddq = -a

        angle.append(q)
        velocity.append(dq)
        acceleration.append(ddq)

    angle = np.array(angle).reshape(len(t))
    velocity = np.array(velocity).reshape(len(t))
    acceleration = np.array(acceleration).reshape(len(t))

    return [t, angle, velocity, acceleration]


def MinTim(q0, qf, t0, tf):
    """
    Compute a LSPB reference trajectory
    q0 = initial position
    qf = final position
    t0 = start time
    tf = final time
    """
    from numpy import matrix, array, linspace
    import numpy as np

    ts = (tf - t0) / 2 # halfway point
    a = (qf - q0)/(ts**2) # max acc
    print("a", a)
    V = (qf - q0) / ts

    # define the time sequence
    t = linspace(0, ts * 2, 200, endpoint=True)
    # append ts into t
    t = np.append(t, [ts], axis=0)
    angle, velocity, acceleration = [], [], []
    t = np.sort(t)
    tf = ts * 2

    for i, time in enumerate(t):
        if time <= ts:
            q = 1 / 2 * a * time ** 2
            dq = a * time
            ddq = a
        elif time > ts:
            q = qf - (a * tf ** 2) / 2 + a * tf * time - a / 2 * time ** 2
            dq = -a * (time - ts) + a * ts
            ddq = -a

        angle.append(q)
        velocity.append(dq)
        acceleration.append(ddq)

    angle = np.array(angle).reshape(len(t))
    velocity = np.array(velocity).reshape(len(t))
    acceleration = np.array(acceleration).reshape(len(t))

    return [t, angle, velocity, acceleration]

t, q2, dq2, ddq2 = LSPBTraj(0, np.pi, 0, 2, (10*np.pi) / 17)

plt.figure(figsize=(12,4))
plt.subplot(131)
plt.plot(t, q2);plt.xlabel('time ($s$)');plt.ylabel('Angle ($rad$)')

plt.subplot(132)
plt.plot(t, dq2);plt.xlabel('time ($s$)');plt.ylabel('Velocity ($rad/s$)')

plt.subplot(133)
plt.plot(t, ddq2);plt.xlabel('time ($s$)');plt.ylabel('Acceleration ($rad/s^2$)')
plt.tight_layout()
plt.show()



t, q2, dq2, ddq2 = MinTim(0, np.pi/2, 0, 2)

# plot results
plt.figure(figsize=(12,4))
plt.subplot(131)
plt.plot(t, q2);plt.xlabel('time ($s$)');plt.ylabel('Angle ($rad$)')

plt.subplot(132)
plt.plot(t, dq2);plt.xlabel('time ($s$)');plt.ylabel('Velocity ($rad/s$)')

plt.subplot(133)
plt.plot(t, ddq2);plt.xlabel('time ($s$)');plt.ylabel('Acceleration ($rad/s^2$)')
plt.tight_layout()
plt.show()
