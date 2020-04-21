import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# required for animation
from datetime import datetime, timedelta
plt.ion()

def precess(phi: float) -> np.array:
    """
    This function returns the precessed matrix.
    preccession is rotation around the Z-axis.

    Parameters
    ----------
    phi : float
        the angle of precession in degrees.

    Returns
    -------
    rZ: np.array
        the precessed matrix.

    """
    rZ = np.array(
     [[np.cos(phi), -np.sin(phi), 0],
      [np.sin(phi), np.cos(phi), 0],
      [0, 0, 1]]
    )
    return rZ


def excite(phi: float) -> np.array:
    """
    This function returns the excited matrix.
    excitation is rotation around the X-axis

    Parameters
    ----------
    phi : float
        the angle of excitation in degrees.

    Returns
    -------
    rX : np.array
        the excited matrix.
    """

    rX = np.array(
        [
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ]
    )
    return rX


def relax(T: int, T1: int, T2: int, F: int):
    """
    This function relaxes the input array and returns list ready for plotting

    Parameters
    ----------
    T : int
         Step time for plotting
    T1 : int
         Longitudinal Relaxation time constant (T1 of tissue)
    T2 : int
         Transverse Relaxation time constant (T2 of tissue)
    F: int
        off-resonance frequency, used for angle calculation

    Returns
    -------
    Af, B: the matrices in the equation M1 = M0 * Af + B
    
    """

    phi = 2*np.pi*F*T/1000
    E1 = np.exp(-T/T1)
    E2 = np.exp(-T/T2)
    z = precess(phi)
    A = np.array(
        [
            [E2, 0, 0],
            [0, E2, 0],
            [0, 0, E1]
        ]
    )
    Af = np.matmul(A, z)
    B = np.array([[0], [0], [1-E1]])
    return Af, B

def getLine(point):
    """
    This function returns a line between the origin and a point

    Parameters
    ----------
    point : np.array
        A 3-D vector represents the point on the curve to match a line of

    Returns
    -------
    xCoordinate : np.array
        X-coordinates of the 2 points (O + P)
    yCoordinate : np.array
        Y-coordinates of the 2 points (O + P)
    zCorrdinate : np.array
        Z-coordinates of the 2 points (O + P)

    """
    
    xCoordinate = np.array([0, point[0, :]])
    yCoordinate = np.array([0, point[1, :]])
    zCorrdinate = np.array([0, point[2, :]])
    return xCoordinate, yCoordinate, zCorrdinate

def fourier(input_array: np.array) -> np.array:
    """
    This function takes an image as a numpy array and returns its fourier transform

    Parameters
    ----------
    input_array : np.array
        Grayscale image in numpy array format

    Returns
    -------
    trans : np.array
        The fourier transform of the input image

    """
    four = np.fft.fft2(input_array)
    four[four == 0] = 0.001
    trans = 20*np.log(np.abs(four))
    trans = np.uint8(trans)
    return trans


def make_constant() -> np.array:
    time = np.linspace(0, 1000,100)
    Bz = np.empty(100)
    Bz.fill(500)
    return time,Bz

def make_uniform() -> np.array:
    """
    This function makes uniform time and Bz arrays for plotting

    Returns
    -------
    time : TYPE
        DESCRIPTION.
    Bz : TYPE
        DESCRIPTION.

    """
    time = np.linspace(0, 1000,100)
    Bz = np.linspace(0, 1000,100)
    return time,Bz

def add_noise(input_array: np.array) -> np.array:
    """
    Superimpose noise on the input array

    Parameters
    ----------
    input_array : np.array
        The input array to have added noise

    Returns
    -------
    output : np.array
        The array after noise addition

    """
    # make a copy to avoid compromising the original array
    output = np.copy(input_array)
    rand_array = np.empty(output.size)
    for i in range (0, output.size):
        rand_array[i] = (2 * np.random.random_sample() - 1)* 100
        
    print (rand_array)
    output += rand_array
    return output


def plot_Bz(ax):
    time_axis, Bz_constant = make_constant()
    _, Bz_uniform = make_uniform()
    Bz_const_noisy = add_noise(Bz_constant)
    Bz_uni_noisy = add_noise(Bz_uniform)
    
    ax[0].plot(time_axis, Bz_constant, label = "Constant magnetic field")
    ax[0].plot(time_axis, Bz_uniform, label = "Uniform magnetic field")
    ax[0].plot(time_axis, Bz_const_noisy, label = "Constant MF with noise")
    ax[0].plot(time_axis, Bz_uni_noisy, label = "Uniform MF with noise")
    ax[0].legend(fontsize=7)
    ax[0].figure.canvas.draw()
    return None

def plot_magnetization():
    # make initial parameters
    dt = 1
    t = 1000
    n = int(np.ceil(t/dt)+1)
    df = 10
    t1 = 600
    t2 = 100
    
    # a and b are functions of magnetization matrix
    a, b = relax(dt, t1, t2, df)
    
    # Init magnetization decay array
    M = np.zeros([3, n])
    M[:, [0]] = np.array([[4], [0], [0]])
    
    # Actually make decaying points
    for i in range(1, n):
        y = np.matmul(a, M[:, [i-1]]) + b
        M[:, [i]] = y
        
    # Make initial line
    line1 = getLine(M[:, [0]])
    zRotated = np.matmul(excite(np.pi*0.5), M[:, [0]])
    line2 = getLine(zRotated)
    
    
    # Change current figure to figure 1
    plt.figure(1)
    
    # Make time array for plotting
    time_axis = np.linspace(0, t, n)
    plt.plot(time_axis, M[0, :], 'r')
    plt.plot(time_axis, M[1, :], 'y')
    plt.plot(time_axis, M[2, :], 'b')
    plt.plot(time_axis, np.sqrt(M[0, :]**2+M[1, :]**2), 'k')
    
    # Change current figure to figure 1
    plt.figure(2)
    ax = plt.subplot(111,projection="3d")
    line, = ax.plot3D(line1[0], line1[1], line1[2], 'r')

    # Required for real-time drawing
    period = timedelta(seconds = dt/100)
    timer = datetime.now() + period
    
    # draw the helix shape of magnetization
    ax.plot3D(M[0, :], M[1, :], M[2, :])
    
    # make the first line of animation
    line3 = getLine(M[:, [-1]])
    ax.plot3D(line3[0], line3[1], line3[2])
    
    # index used in loop
    i = 0
    
    # animation-specific variables
    canvas = ax.figure.canvas
    background = canvas.copy_from_bbox(ax.bbox)
    canvas.draw()
    
    # The actual animation loop
    while i < 500:
        # update every period
        if datetime.now() >= timer:
            timer += period
            i += 1
            # The following lines are GPU animation functions
            # restore the clean slate background
            canvas.restore_region(background)
            # get the next line for drawing
            line3 = getLine(M[:, [i*2]])
            # set the color of previous line to yellow
            line.set_color("y")
            # draw the line on canvas
            ax.draw_artist(line)
            # update canvas
            canvas.blit(ax.bbox)
            # make the new line in red color
            line, = ax.plot3D(line3[0], line3[1], line3[2], 'r')
            # draw the new line on canvas
            ax.draw_artist(line)
            # update canvas
            canvas.blit(ax.bbox)

    #plt.show(block=True)
    return None


def plot_helix(ax):
    # make initial parameters
    dt = 1
    t = 1000
    n = int(np.ceil(t/dt)+1)
    df = 10
    t1 = 600
    t2 = 100
    
    # a and b are functions of magnetization matrix
    a, b = relax(dt, t1, t2, df)
    
    # Init magnetization decay array
    M = np.zeros([3, n])
    M[:, [0]] = np.array([[4], [0], [0]])
    
    # Actually make decaying points
    for i in range(1, n):
        y = np.matmul(a, M[:, [i-1]]) + b
        M[:, [i]] = y
        
    # Make initial line
    line1 = getLine(M[:, [0]])
    zRotated = np.matmul(excite(np.pi*0.5), M[:, [0]])
    line2 = getLine(zRotated)
    
    # draw the helix shape of magnetization
    ax[0].plot3D(M[0, :], M[1, :], M[2, :])
    
    canvas = ax[0].figure.canvas
    #ax[0].figure.canvas.draw()
    canvas.draw()
    
    #ax[0] = plt.subplot(111,projection="3d")
    line, = ax[0].plot3D(line1[0], line1[1], line1[2], 'r')

    # Required for real-time drawing
    period = timedelta(seconds = dt/50)
    timer = datetime.now() + period
    
    
    # make the first line of animation
    line3 = getLine(M[:, [-1]])
    ax[0].plot3D(line3[0], line3[1], line3[2])
    
    # index used in loop
    i = 0
    
    # animation-specific variables
    background = canvas.copy_from_bbox(ax[0].bbox)
    canvas.draw()
    
    # The actual animation loop
    while i < 500:
        # update every period
        if datetime.now() >= timer:
            timer += period
            i += 1
            # The following lines are GPU animation functions
            # restore the clean slate background
            canvas.restore_region(background)
            # get the next line for drawing
            line3 = getLine(M[:, [i*2]])
            # set the color of previous line to yellow
            line.set_color("y")
            # draw the line on canvas
            ax[0].draw_artist(line)
            # update canvas
            canvas.blit(ax[0].bbox)
            
            # make the new line in red color
            line, = ax[0].plot3D(line3[0], line3[1], line3[2], 'r')
            # draw the new line on canvas
            ax[0].draw_artist(line)
            # update canvas
            canvas.blit(ax[0].bbox)
            canvas.flush_events()
    
    canvas.draw()
    #plt.show(block=True)
    return None


def plot_curve(ax):
    # make initial parameters
    dt = 1
    t = 1000
    n = int(np.ceil(t/dt)+1)
    df = 10
    t1 = 600
    t2 = 100
    
    # a and b are functions of magnetization matrix
    a, b = relax(dt, t1, t2, df)
    
    # Init magnetization decay array
    M = np.zeros([3, n])
    M[:, [0]] = np.array([[4], [0], [0]])
    
    # Actually make decaying points
    for i in range(1, n):
        y = np.matmul(a, M[:, [i-1]]) + b
        M[:, [i]] = y
        
    # Make time array for plotting
    time_axis = np.linspace(0, t, n)
    ax[0].plot(time_axis, M[0, :], 'r',label="Mx")
    ax[0].plot(time_axis, M[1, :], 'y',label="My")
    ax[0].plot(time_axis, M[2, :], 'b',label="Mz")
    ax[0].plot(time_axis, np.sqrt(M[0, :]**2+M[1, :]**2), 'k',label="Mxy")
    ax[0].legend()

    ax[0].figure.canvas.draw()
    return None
