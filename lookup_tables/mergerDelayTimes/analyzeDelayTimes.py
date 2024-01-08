import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('./BH_pair_times_for_angelo_epsConst.txt')
redshift = data[:,2]
'''
delayTime = np.log10(data[:,4]/data[:,5])
clim=(-0.7,1.2)
ylabel = r'$\log(\Delta t/ \Delta t_\mathrm{BK})$'
'''
delayTime = data[:,6]
clim=(-2,2)
ylabel = r'$\Delta t - \Delta t_\mathrm{BK}$'
Mstar = np.maximum(data[:,-4], data[:,-3])
stellarMassRatio = data[:,-3]/data[:,-4]
stellarMassRatio[stellarMassRatio>1] = 1.0/stellarMassRatio[stellarMassRatio>1]
Mvir = np.maximum(data[:,-2], data[:,-1])
virialMassRatio = data[:,-1]/data[:,-2]
virialMassRatio[virialMassRatio>1] = 1.0/virialMassRatio[virialMassRatio>1]

generalFilter = redshift>0
redshift = redshift[generalFilter]
delayTime = delayTime[generalFilter]
Mstar = Mstar[generalFilter]
stellarMassRatio = stellarMassRatio[generalFilter]
Mvir = Mvir[generalFilter]
virialMassRatio = virialMassRatio[generalFilter]

#PDF
plt.hist(delayTime[(redshift>3) & (redshift<10)], normed=True, label='z>3', alpha=0.7)
plt.hist(delayTime[(redshift>0) & (redshift<3)], normed=True, label='z<3', alpha=0.7)
plt.hist(delayTime[(redshift>0) & (redshift<10)], normed=True, label='z>0', alpha=0.7)
plt.hist(delayTime[(redshift>0) & (redshift<1)], normed=True, label='z<1', alpha=0.7)
plt.xlabel(ylabel, fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.legend()
plt.show()

#Redshift Dependence
binedges = np.linspace(0,5,24)
bincenters = 0.5 * (binedges[1:] + binedges[:-1])
indices = np.digitize(redshift, binedges)-1
averageDelayTimes = [np.mean(delayTime[indices==i]) for i in range(len(bincenters))]
plt.hexbin(redshift, delayTime, gridsize=15, zorder=0, cmap='Greys')
plt.scatter(redshift, delayTime, color='r', marker='o')
plt.errorbar(bincenters, averageDelayTimes, marker='s', color='navy')
plt.xlabel('z', fontsize=12)
plt.ylabel(ylabel, fontsize=12)
plt.show()

#Mass dependence
binedges = np.logspace(9,14,24)
bincenters = 10**(0.5 * (np.log10(binedges[1:]) + np.log10(binedges[:-1])))
indices = np.digitize(Mvir, binedges)-1
averageDelayTimes = [np.mean(delayTime[indices==i]) for i in range(len(bincenters))]
plt.hexbin(Mvir, delayTime, gridsize=15, xscale='log', zorder=0, cmap='Greys')
plt.scatter(Mvir, delayTime, color='r', marker='o')
plt.errorbar(bincenters, averageDelayTimes, marker='s', color='navy')
plt.xscale('log')
plt.xlabel('M_vir', fontsize=12)
plt.ylabel(ylabel, fontsize=12)
plt.show()

#Mass dependence
binedges = np.logspace(7,12,24)
bincenters = 10**(0.5 * (np.log10(binedges[1:]) + np.log10(binedges[:-1])))
indices = np.digitize(Mstar, binedges)-1
averageDelayTimes = [np.mean(delayTime[indices==i]) for i in range(len(bincenters))]
plt.hexbin(Mstar, delayTime, gridsize=15, xscale='log', zorder=0, cmap='Greys')
plt.scatter(Mstar, delayTime, color='r', marker='o')
plt.errorbar(bincenters, averageDelayTimes, marker='s', color='navy')
plt.xscale('log')
plt.xlabel('M_star', fontsize=12)
plt.ylabel(ylabel, fontsize=12)
plt.show()

#Ratio dependence
binedges = np.logspace(-2,0,12)
bincenters = 10**(0.5 * (np.log10(binedges[1:]) + np.log10(binedges[:-1])))
indices = np.digitize(virialMassRatio, binedges)-1
averageDelayTimes = [np.mean(delayTime[indices==i]) for i in range(len(bincenters))]
plt.hexbin(virialMassRatio, delayTime, gridsize=15, xscale='log', zorder=0, cmap='Greys')
plt.scatter(virialMassRatio, delayTime, color='r', marker='o')
plt.errorbar(bincenters, averageDelayTimes, marker='s', color='navy')
plt.xscale('log')
plt.xlabel('q_vir', fontsize=12)
plt.ylabel(ylabel, fontsize=12)
plt.show()

#Ratio dependence
binedges = np.logspace(-2,0,12)
bincenters = 10**(0.5 * (np.log10(binedges[1:]) + np.log10(binedges[:-1])))
indices = np.digitize(stellarMassRatio, binedges)-1
averageDelayTimes = [np.mean(delayTime[indices==i]) for i in range(len(bincenters))]
plt.hexbin(stellarMassRatio, delayTime, gridsize=15, xscale='log', zorder=0, cmap='Greys')
plt.scatter(stellarMassRatio, delayTime, color='r', marker='o')
plt.errorbar(bincenters, averageDelayTimes, marker='s', color='navy')
plt.xscale('log')
plt.xlabel('q_star', fontsize=12)
plt.ylabel(ylabel, fontsize=12)
plt.show()

#Color coded t delay
plt.hexbin(stellarMassRatio, Mstar, gridsize=15, xscale='log', yscale='log', zorder=0, cmap='Greys')
colors = plt.cm.jet((delayTime-clim[0])/(clim[1]-clim[0]))
plt.scatter(stellarMassRatio, Mstar, marker='o', color=colors)
sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('jet'), norm=plt.Normalize(vmin=clim[0], vmax=clim[1]))
sm._A = []
cb = plt.colorbar(sm, ticks=np.linspace(clim[0],clim[1],5))
cb.set_label(ylabel, fontsize=12)
plt.xlabel('q_star', fontsize=12)
plt.ylabel('M_star', fontsize=12)
plt.show()

plt.hexbin(redshift, Mstar, gridsize=15, yscale='log', zorder=0, cmap='Greys')
colors = plt.cm.jet((delayTime-clim[0])/(clim[1]-clim[0]))
plt.scatter(redshift, Mstar, marker='o', color=colors)
sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('jet'), norm=plt.Normalize(vmin=clim[0], vmax=clim[1]))
sm._A = []
cb = plt.colorbar(sm, ticks=np.linspace(clim[0],clim[1],5))
cb.set_label(ylabel, fontsize=12)
plt.xlabel('redshift', fontsize=12)
plt.ylabel('M_star', fontsize=12)
plt.show()

plt.hexbin(redshift, stellarMassRatio, gridsize=15, yscale='log', zorder=0, cmap='Greys')
colors = plt.cm.jet((delayTime-clim[0])/(clim[1]-clim[0]))
plt.scatter(redshift, stellarMassRatio, marker='o', color=colors)
sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('jet'), norm=plt.Normalize(vmin=clim[0], vmax=clim[1]))
sm._A = []
cb = plt.colorbar(sm, ticks=np.linspace(clim[0],clim[1],5))
cb.set_label(ylabel, fontsize=12)
plt.xlabel('redshift', fontsize=12)
plt.ylabel('q_star', fontsize=12)
plt.show()

