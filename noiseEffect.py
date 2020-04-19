import numpy as np
import matplotlib.pyplot as plt


B = 1.5
BPositive = 2.5
BNegative = 0.5
gyroRatio = 30
w = gyroRatio * B
wPositive = gyroRatio * BPositive
wNegative = gyroRatio * BNegative
T1 = 490/1000
T2 = 43/1000
t = np.arange(start=0, stop=10, step=0.001)

omega = 2*np.pi*w*t
omegaPositive = 2*np.pi*wPositive*t + np.pi/8
omegaNegative = 2*np.pi*wNegative*t - np.pi/8


Mx = np.exp(-1*t/T2)*np.sin(omega)
MxPositive = np.exp(-1*t/T2)*np.sin(omegaPositive)
MxNegative = np.exp(-1*t/T2)*np.sin(omegaNegative)


My = np.exp(-1*t/T2)*np.cos(omega)
MyPositive = np.exp(-1*t/T2)*np.cos(omegaPositive)
MyNegative = np.exp(-1*t/T2)*np.cos(omegaNegative)


Mxy = np.sqrt(Mx**2 + My**2)
MxyPositive = np.sqrt(MxPositive**2 + MyPositive**2)
MxyNegative = np.sqrt(MxNegative**2 + MyNegative**2)

plt.figure(1)
plt.plot(t[:250], Mx[:250], 'r', label="No Noise")
plt.plot(t[:250], MxPositive[:250], 'b', label="Positive Noise")
plt.plot(t[:250], MxNegative[:250], 'y', label="Negative Noise")
plt.title("$M_x/M_o$ vs time")
plt.xlabel("time")
plt.ylabel("$M_x/M_o$")
plt.legend()


plt.figure(2)
plt.plot(t[:250], My[:250], 'r', label="No Noise")
plt.plot(t[:250], MyPositive[:250], 'b', label="Positive Noise")
plt.plot(t[:250], MyNegative[:250], 'y', label="Negative Noise")
plt.title("$M_y/M_o$ vs time")
plt.xlabel("time")
plt.ylabel("$M_y/M_o$")
plt.legend()

plt.figure(3)
plt.plot(Mx, My, 'r', label="No Noise")
plt.plot(MxPositive, MyPositive, 'b', label="Positive Noise")
plt.plot(MxNegative, MyNegative, 'y', label="Negative Noise")
plt.title("$M_xy$ in X-Y Plane")
plt.xlabel("$M_y$")
plt.ylabel("$M_x$")
plt.legend()

plt.show()