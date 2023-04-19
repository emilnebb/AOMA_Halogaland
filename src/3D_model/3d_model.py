import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
g = 9.81  # acceleration due to gravity
L = 50.0  # length of the bridge (m)
m = 1000.0  # mass of the bridge (kg)
c = 10.0  # damping coefficient (N s/m)
k = 1000.0  # spring constant (N/m)
dt = 0.01  # time step (s)
t = np.arange(0, 20, dt)  # time array

# Initial conditions
x0 = np.zeros(2)  # initial displacement and velocity
x0[0] = 1.0  # initial displacement of the bridge

# Function to calculate the acceleration of the bridge
def acceleration(x, t):
    return np.array([x[1], -g - (c/m)*x[1] - (k/m)*x[0]])

# Function to perform numerical integration using the Euler method
def integrate(x0, t):
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = x[i-1] + dt*acceleration(x[i-1], t[i-1])
    return x

# Calculate the motion of the bridge
x = integrate(x0, t)

# Create a figure and axes for the animation
fig, ax = plt.subplots()

# Plot the initial position of the bridge
line, = ax.plot([0, L], [0, -x[0,0]])

# Function to update the position of the bridge in the animation
def update(i):
    line.set_ydata(-x[i,0])
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(t), interval=50, blit=True, repeat=True)

# Set the axis limits and labels
ax.set_xlim(0, L)
ax.set_ylim(-2, 2)
ax.set_xlabel('Position (m)')
ax.set_ylabel('Displacement (m)')

# Display the animation
plt.show()