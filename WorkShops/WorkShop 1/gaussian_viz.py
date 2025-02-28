#
# Advanced Robotics, 2024-2025
# Paul Baxter
# Workshop Week 1
#
# code based on matplotlib tutorial: https://python4mpia.github.io/plotting/matplotlib.html
#
# Human assumed to be at point (0,0) - black arrow facing +x-axis
#
# Run with:     python gaussian_viz.py
#
# May need following depedencies, e.g.:
#   python -mpip install numpy
#                        scipy
#                        matplotlib
#
# See comments on l69 for use of two gaussians
# See l84 for saving plots
#

import numpy as np
import matplotlib.pyplot as plt

###################
# params to alter #
###################
x1_mean = 0.0
y1_mean = 0.0
x1_sig = 1.0
y1_sig = 1.0

x2_mean = 0.0
y2_mean = 0.0
x2_sig = 3
y2_sig = 1.0
###################

def gaussian(x, y, x0, y0, xsig, ysig):
    return np.exp(-0.5*(((x-x0) / xsig)**2 + ((y-y0) / ysig)**2))

delta = 0.025

###################
# single gaussian #
###################
x = np.arange(-6.0, 6.0, delta)
y = np.arange(-6.0, 6.0, delta)
X, Y1 = np.meshgrid(x, y)
z = gaussian(X, Y1, x1_mean, y1_mean, x1_sig, y1_sig)

######################
# multiple gaussians #
######################
# just examples shown, change as desired

# gaussian 1 - negative x-axis
x1 = np.arange(-6.0, 0.0, delta)
X1, Y = np.meshgrid(x1, y)
z1 = gaussian(X1, Y, x1_mean, y1_mean, x1_sig, y1_sig)

# gaussian 2 - positive x-axis
x2 = np.arange(0.0, 6.0, delta)
X2, Y = np.meshgrid(x2, y)
z2 = gaussian(X2, Y, x2_mean, y2_mean, x2_sig, y2_sig)

################
# create plots #
################
# Comment/uncomment as required for gaussians used above
# Uncomment:
# - l74 for single gaussian version (keep l75&l76 commented)
# - l75&l76 for two-gaussian version (keep l74 commented)
plt.figure(figsize=(8,8))
#CS = plt.contour(X, Y1, z)
CS = plt.contour(X1, Y, z1)
CS = plt.contour(X2, Y, z2)

plt.clabel(CS, inline=1, fontsize=10)
plt.grid()
plt.plot(0,0,marker=">",markersize=20,color="black")


# uncomment the following line if you want to save the plot as a pdf:
plt.savefig('gaussian2d.pdf')

plt.show()
