#!/usr/bin/python

# This script reads in a single csv file handed over as an argument
# and plots it. In addition, it computes a fourier transform of
# the field given in the input (which is intended to be the 
# porosity) and the band angle distribution for the given input,
# and plots both.

import numpy as np
import numpy.fft as fft
import scipy.interpolate
import matplotlib.pyplot as plt
import csv as csv
import math
import sys
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
#plt.style.use('classic')

if len(sys.argv)!=2:
	print "usage: shear_bands.csv"
	sys.exit(0)
filename = sys.argv[1]

nplots=3

data = []

data.append(np.genfromtxt(filename, dtype = float, names = True))

coordsX=[]
coordsY=[]
porosity=[]

# data to plot
for i in range(0,len(data)):
	coordsX=np.append(coordsX, data[i]['x'])
	coordsY=np.append(coordsY, data[i]['y'])
	porosity=np.append(porosity, data[i]['porosity'])

x = np.asarray(coordsX)
y = np.asarray(coordsY)
p = np.asarray(porosity)
p = p - 0.03

print len(x), len(y), len(p)

xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()

dx = 1e300
sortedx = np.sort(x)
for i in range(1,len(sortedx)):
	d = sortedx[i]-sortedx[i-1]
	if d>1e-10:
		dx = min(dx, d)
dy = 1e300
sortedy = np.sort(y)
for i in range(1,len(sortedy)):
	d = sortedy[i]-sortedy[i-1]
	if d>1e-10:
		dy = min(dy, d)

print "dx=",dx,"dy=",dy

nrows = (np.round((ymax - ymin) / dy)+1).astype(np.int)
ncols = (np.round((xmax - xmin) / dx)+1).astype(np.int)
aspect_ratio = (np.round((ymax - ymin) / dy)+1)/(np.round((xmax - xmin) / dx)+1)

# Then we make an empty 2D grid...
grid = np.zeros((nrows, ncols), dtype=np.float)
xg = np.arange(xmin, xmax, dx)
yg = np.arange(ymin, ymax, dy)

xy = np.transpose(np.array([x,y]))
xyg = np.meshgrid(xg, yg)
grid = scipy.interpolate.griddata(xy, p, tuple(xyg), method='nearest')

# count the number of points above and below the background porosity to compute band width and spacing.
# we want to go through all rows, starting from the top. 
previous_entry = 0
average_width = 0
average_spacing = 0
current_width = 0
current_spacing = 0
nbands = 0
nspaces = 0

width_list=[]
spacing_list=[]

for j in range(0,int(nrows)):
	for i in range(0,int(ncols)):
		if (grid[j][i] > 0 and previous_entry==1):
			current_width += 1
		elif (grid[j][i] <= 0 and previous_entry==0):
			current_spacing += 1
		elif (grid[j][i] > 0 and previous_entry==0):
			average_spacing += current_spacing
			current_width = 1
			previous_entry = 1
			nspaces += 1
			spacing_list.append(current_spacing * dx)
		elif (grid[j][i] <= 0 and previous_entry==1):
			average_width += current_width
			current_spacing = 1
			previous_entry = 0
			nbands += 1
			width_list.append(current_width * dx)

average_width *= dx / float(nbands)
average_spacing *= dx / float(nspaces)

# compute the standard deviation.
width_standard_deviation = 0
for i in range(0,len(width_list)):
	width_standard_deviation += (width_list[i] - average_width)**2 / float(len(width_list))
width_standard_deviation = width_standard_deviation**0.5

spacing_standard_deviation = 0
for i in range(0,len(spacing_list)):
	spacing_standard_deviation += (spacing_list[i] - average_spacing)**2 / float(len(spacing_list))
spacing_standard_deviation = spacing_standard_deviation**0.5

# apply a hamming window function to make porosity periodic. 
hx = np.ones(len(xg))
hy = np.hanning(len(yg))
ham2d = np.outer(hx,hy)
ham2d_trans = np.transpose(ham2d)
grid_window = np.multiply(grid,ham2d_trans)

# do the fourier transformation. 
fourier = fft.fft2(grid_window)
fourier_centered = fft.fftshift(fourier)
frequencies = fft.fftfreq(ncols,dx)*2

fig, ax = plt.subplots(nplots)
ax[0] = plt.subplot2grid((2,2), (0, 0), colspan=2)
ax[1] = plt.subplot2grid((2,2), (1, 0))
ax[2] = plt.subplot2grid((2,2), (1, 1))

fouriermax = np.absolute(fourier_centered).max()
fourierabs = np.absolute(fourier_centered)/fouriermax

intensity_integral=[]
intensity_points=[]
angle=np.copy(grid)

# calculate the angle for each point
for i in range(0,int(ncols)):
	for j in range(0,int(nrows)):
		a = math.atan2((j-(nrows)/2),(i-(ncols)/2)*aspect_ratio)
		if (a<0):
			a+=np.pi
		angle[j][i] = 90.0 - a/np.pi*180.0

# And now we plot it:
# input data
im0 = ax[0].imshow(grid, interpolation='nearest', 
        extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='RdBu_r')

# fourier transform
window_fraction = 2/10.
fourier2plot = fourierabs[(np.round(len(yg)*(0.5-window_fraction))).astype(np.int):(np.round(len(yg)*(0.5+window_fraction))).astype(np.int), (np.round(len(xg)*(0.5-window_fraction))).astype(np.int):(np.round(len(xg)*(0.5+window_fraction))).astype(np.int)]
#im1 = ax[1].imshow(fourier2plot, interpolation='nearest',
#        extent=(frequencies.min()*window_fraction, frequencies.max()*window_fraction, frequencies.min()*window_fraction, frequencies.max()*window_fraction), origin='lower', aspect=1.0)
#plt.colorbar(im1)

# band angle histogram. 
angle_flat = angle.flatten()
fourier_flat = fourierabs.flatten()
bins=np.linspace(0.1,90,18)

# fit lognormal to get the dominant angle. 
bins_fit=np.linspace(0.1,90,100)
y_hist, bin_edges = np.histogram(angle_flat, bins_fit, normed=True, weights=fourier_flat)
x_hist=bins_fit[1:]

(shape_out, scale_out), pcov = curve_fit(lambda xdata, shape, scale: stats.lognorm.pdf(xdata, shape, loc=0, scale=scale), x_hist, y_hist, p0=[0.25, 20])

# print the results for the width and spacing.
sinphi = math.sin(scale_out * np.pi / 180.0)
print "width =", average_width * sinphi, "standard deviation =", width_standard_deviation * sinphi, "relative =", width_standard_deviation/average_width
print "spacing =", average_spacing * sinphi, "standard deviation =", spacing_standard_deviation * sinphi, "relative =", spacing_standard_deviation/average_spacing
print "ratio =", average_spacing/average_width

# rescale width and spacing with the band angle. 
for i in range(0,len(width_list)):
	width_list[i] *= sinphi
for i in range(0,len(spacing_list)):
	spacing_list[i] *= sinphi

rotated_k=np.copy(grid)
# rotate the coordinate system by th dominant angle, and stack the wave number. 
# x' = x cos (90-theta) + yy sin(90-theta)
for i in range(0,int(ncols)):
	for j in range(0,int(nrows)):
		rotated_k[j][i] = math.cos(np.pi*(90.0 - scale_out)/180.0) * (i-(ncols)/2) * frequencies.max() / ncols + math.sin(np.pi*(90.0 - scale_out)/180.0) * (j-(nrows)/2) * frequencies.max() / nrows

# plot the width and spacing histograms. 
kbins=np.linspace(0,2.8e-4,30)
ax[1].hist(width_list, kbins, normed=True, rwidth=0.75)
ax[2].hist(spacing_list, kbins, normed=True, rwidth=0.75)
ax[1].set_xlabel('Band width in m')
ax[2].set_xlabel('Band spacing in m')
ax[1].set_ylim(0,40000)
ax[2].set_ylim(0,40000)

ax[2].get_yaxis().set_visible(False)

print shape_out, scale_out

fig.savefig('width_vs_spacing', dpi=200)


