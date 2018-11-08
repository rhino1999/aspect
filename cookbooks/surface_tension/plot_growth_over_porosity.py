#!/usr/bin/python

# This script can be used to plot data for the magmatic shear bands testcase
# for Newtonian rheology and analyze it using the relations given in Spiegelman (2003):
# Linear analysis of melt band formation by simple shear, Geochemistry, Geophysics,
# Geosystems, 4(9), 8615.

import numpy as np
import matplotlib.pyplot as plt
import colors
import math

figsize=(7,5)
prop={'size':12}
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = '\usepackage{relsize}'
plt.rc('font', family='sanserif')
figure=plt.figure(dpi=100,figsize=figsize)

file_name="plane_wave_melt_bands_phi"

data = []
data.append(np.genfromtxt(file_name,delimiter=' ', dtype = float))

phi=[]
analytical=[]
numerical=[]

end=len(data[0])

# data to plot
for j in range(0,end):
	phi.append(data[0][j][0])
	analytical.append(data[0][j][1]/0.0003)
	numerical.append(data[0][j][2]/0.0003)

plt.loglog(phi[0:end],numerical[0:end]," ",color=colors.color(3), marker=colors.marker(1), label='numerical')
plt.loglog(phi[0:end],analytical[0:end],"--", color="black", marker="x", mew=1.5, ms=4, label='analytical')

#plt.xlim([-5, 185])
#plt.ylim([4e-3,0.4])
plt.xlabel("Porosity $\phi$")
plt.ylabel("Nondimensionalized melt band growth rate $\dot s$")
#plt.xticks([0,45,90,135,180])
plt.grid(True)
 

plt.legend(loc = "upper left",prop=prop)
plt.savefig('growth_rate_porosity.pdf', #bbox_extra_artists=(legend,), 
            bbox_inches='tight',dpi=200)

