from __future__ import division,print_function

#import inclination as inc
from scipy.optimize import newton
from scipy.interpolate import UnivariateSpline as interpolate
import pkg_resources

import numpy as np
from scipy.interpolate import LinearNDInterpolator as interpnd

from astropy.coordinates import SkyCoord,Angle

import numpy.random as rand
import sys,re,os
import numpy as np
import matplotlib.pyplot as plt
import plotutils as plu

from astropy import units as u

from astropy import constants as const
MSUN = const.M_sun.cgs.value
AU = const.au.cgs.value
DAY = 86400
G = const.G.cgs.value

from .kepler import Efn

def semimajor(P,mstar=1):
    return ((P*DAY/2/np.pi)**2*G*mstar*MSUN)**(1./3)/AU

def random_spherepos(n):
    """returns SkyCoord object
    """
    signs = np.sign(rand.uniform(-1,1,size=n))
    thetas = Angle(np.arccos(rand.uniform(size=n)*signs),unit=u.rad) #random b/w 0 and 180
    phis = Angle(rand.uniform(0,2*np.pi,size=n),unit=u.rad)
    c = SkyCoord(phis,thetas,1,representation='physicsspherical')
    return c

def orbitproject(x,y,inc,phi=0,psi=0):
    """

    x,y are coordinates in z=0 plane (plane of the orbit)

    observer is at (inc, phi) on celestial sphere; psi is orientation of
    final x-y axes
    """

    x2 = x*np.cos(phi) + y*np.sin(phi)
    y2 = -x*np.sin(phi) + y*np.cos(phi)
    z2 = y2*np.sin(inc)
    y2 = y2*np.cos(inc)

    xf = x2*np.cos(psi) - y2*np.sin(psi)
    yf = x2*np.sin(psi) + y2*np.cos(psi)

    return (xf,yf,z2)

def orbit_posvel(Ms,eccs,semimajors,mreds,obspos=None):
    """returns positions in projected AU and velocities in km/s for given mean anomalies

    returns positions and velocities as SkyCoord objects
    """

    Es = Efn(Ms,eccs)
    
    rs = semimajors*(1-eccs*np.cos(Es))
    nus = 2 * np.arctan2(np.sqrt(1+eccs)*np.sin(Es/2),np.sqrt(1-eccs)*np.cos(Es/2))

    xs = semimajors*(np.cos(Es) - eccs)         #AU
    ys = semimajors*np.sqrt(1-eccs**2)*np.sin(Es)  #AU

    Edots = np.sqrt(G*mreds*MSUN/(semimajors*AU)**3)/(1-eccs*np.cos(Es))
    xdots = -semimajors*AU*np.sin(Es)*Edots/1e5  #km/s
    ydots = semimajors*AU*np.sqrt(1-eccs**2)*np.cos(Es)*Edots/1e5 # km/s
        
    n = np.size(xs)

    #orbpos = inc.spherepos((xs,ys,zeros(n)),normed=False)
    orbpos = SkyCoord(xs,ys,0,representation='cartesian')
    #orbvel = inc.spherepos((xdots,ydots,zeros(n)),normed=False)
    orbvel = SkyCoord(xdots,ydots,0,representation='cartesian')
    if obspos is None:
        #obspos = inc.rand_spherepos(n) #observer position
        obspos = random_spherepos(n) #observer position
    if type(obspos) == type((1,2,3)):
        #obspos = inc.spherepos((obspos[0],obspos[1],obspos[2]))
        obspos = SkyCoord(obspos[0],obspos[1],obspos[2],
                          representation='cartesian').represent_as('physicsspherical')
    if not hasattr(obspos,'theta'):
        obspos = obspos.represent_as('physicsspherical')
        
    
    #random orientation of the sky 'x-y' coordinates
    psi = rand.random(n)*2*np.pi  

    x,y,z = orbitproject(orbpos.x,orbpos.y,obspos.theta,obspos.phi,psi)
    vx,vy,vz = orbitproject(orbvel.x,orbvel.y,obspos.theta,obspos.phi,psi)

    
    return (SkyCoord(x,y,z,representation='cartesian'),
            SkyCoord(vx,vy,vz,representation='cartesian')) #z is line of sight

class TripleOrbitPopulation(object):
    def __init__(self,M1s,M2s,M3s,Plong,Pshort,ecclong=0,eccshort=0,n=None,
                 mean_anomalies_long=None,obsx_long=None,obsy_long=None,obsz_long=None,
                 mean_anomalies_short=None,obsx_short=None,obsy_short=None,obsz_short=None):                 
        """Stars 2 and 3 orbit each other close (short orbit), far from star 1 (long orbit)
        """
        self.orbpop_long = OrbitPopulation(M1s,M2s+M3s,Plong,eccs=ecclong,n=n,
                                           mean_anomalies=mean_anomalies_long,
                                           obsx=obsx_long,obsy=obsy_long,obsz=obsz_long)

        self.orbpop_short = OrbitPopulation(M2s,M3s,Pshort,eccs=eccshort,n=n,
                                           mean_anomalies=mean_anomalies_short,
                                           obsx=obsx_short,obsy=obsy_short,obsz=obsz_short)

        #define Rsky to be the large separation
        self.Rsky = self.orbpop_long.Rsky

        #define instantaneous RV_1, RV_2 and RV_3 relative to COM reference frame
        self.RV_1 = self.orbpop_long.RVs * (self.orbpop_long.M2s / (self.orbpop_long.M1s + self.orbpop_long.M2s))
        self.RV_2 = -self.orbpop_long.RVs * (self.orbpop_long.M1s / (self.orbpop_long.M1s + self.orbpop_long.M2s)) +\
            self.orbpop_short.RVs_com1
        self.RV_3 = -self.orbpop_long.RVs * (self.orbpop_long.M1s / (self.orbpop_long.M1s + self.orbpop_long.M2s)) +\
            self.orbpop_short.RVs_com2

    def dRV_1(self,dt):
        return self.orbpop_long.dRV(dt,com=True)

    def dRV_2(self,dt):
        return -self.orbpop_long.dRV(dt) * (self.orbpop_long.M1s/(self.orbpop_long.M1s + self.orbpop_long.M2s)) +\
            self.orbpop_short.dRV(dt,com=True)

    def dRV_3(self,dt):
        return -self.orbpop_long.dRV(dt) * (self.orbpop_long.M1s/(self.orbpop_long.M1s + self.orbpop_long.M2s)) -\
            self.orbpop_short.dRV(dt) * (self.orbpop_short.M1s/(self.orbpop_short.M1s + self.orbpop_short.M2s))
        

class OrbitPopulation(object):
    def __init__(self,M1s,M2s,Ps,eccs=0,n=None,
                 mean_anomalies=None,obsx=None,obsy=None,obsz=None):
        M1s = np.atleast_1d(M1s)
        M2s = np.atleast_1d(M2s)
        if len(M1s)==1 and len(M2s)==1:
            M1s = np.ones(n)*M1s
            M2s = np.ones(n)*M2s

        self.M1s = M1s
        self.M2s = M2s

        if n is None:
            n = len(M2s)

        self.N = n

        if np.size(Ps)==1:
            Ps = Ps*np.ones(n)

        self.Ps = Ps

        if np.size(eccs) == 1:
            eccs = np.ones(n)*eccs

        self.eccs = eccs

        mred = M1s*M2s/(M1s+M2s)
        semimajors = semimajor(Ps,mred)   #AU
        self.semimajors = semimajors
        self.mreds = mred

        if mean_anomalies is None:
            Ms = rand.uniform(0,2*np.pi,size=n)
        else:
            Ms = mean_anomalies

        self.Ms = Ms

        #coordinate system: all orbits here simulated in x-y plane.
        Es = Efn(Ms,eccs)

        rs = semimajors*(1-eccs*np.cos(Es))
        nus = 2 * np.arctan2(np.sqrt(1+eccs)*np.sin(Es/2),np.sqrt(1-eccs)*np.cos(Es/2))

        xs = semimajors*(np.cos(Es) - eccs)         #AU
        ys = semimajors*np.sqrt(1-eccs**2)*np.sin(Es)  #AU

        Edots = np.sqrt(G*mred*MSUN/(semimajors*AU)**3)/(1-eccs*np.cos(Es))
        xdots = -semimajors*AU*np.sin(Es)*Edots/1e5  #km/s
        ydots = semimajors*AU*np.sqrt(1-eccs**2)*np.cos(Es)*Edots/1e5 # km/s
        
        #coordinates of random observers
        if obsx is None:
            self.obspos = random_spherepos(n)
            #self.obspos = inc.rand_spherepos(n) #observer position
        else:
            #self.obspos = inc.spherepos((obsx,obsy,obsz))
            self.obspos = SkyCoord(obsx,obsy,obsz,representation='cartesian')


        #get positions, velocities relative to M1
        positions,velocities = orbit_posvel(self.Ms,self.eccs,self.semimajors,self.mreds,
                                            self.obspos)

        self.positions = positions
        self.velocities = velocities
        
        #self.x,self.y,self.z = positions
        #self.vx,self.vy,self.vz = velocities

        self.Rsky = np.sqrt(self.positions.x**2 +
                         self.positions.y**2) # on-sky separation, in projected AU
        self.RVs = self.velocities.z  #relative radial velocities

        #velocities relative to center of mass
        self.RVs_com1 = self.RVs * (self.M2s / (self.M1s + self.M2s))
        self.RVs_com2 = -self.RVs * (self.M1s / (self.M1s + self.M2s))

    def dRV(self,dt,com=False):
        """dt in days; if com, then returns the change in RV of component 1 in COM frame
        """
        dt *= DAY

        mean_motions = np.sqrt(G*(self.mreds)*MSUN/(self.semimajors*AU)**3)
        #print mean_motions * dt / (2*pi)

        newMs = self.Ms + mean_motions * dt
        pos,vel = orbit_posvel(newMs,self.eccs,self.semimajors,self.mreds,
                               self.obspos)
        if com:
            return (vel[2] - self.RVs) * (self.M2s / (self.M1s + self.M2s))
        else:
            return vel[2]-self.RVs

    def RV_timeseries(self,ts,recalc=False):
        if not recalc and hasattr(self,'RV_measurements'):
            if ts == self.ts:
                return self.RV_measurements
            else:
                pass
        
        RVs = np.zeros((len(ts),self.N))
        for i,t in enumerate(ts):
            RVs[i,:] = self.dRV(t,com=True)
        self.RV_measurements = RVs
        self.ts = ts
        return RVs

class BinaryGrid(OrbitPopulation):
    def __init__(self, M1, qmin=0.1, qmax=1, Pmin=0.5, Pmax=365, N=1e5, logP=True, eccfn=None):
        M1s = np.ones(N)*M1
        M2s = (rand.random(size=N)*(qmax-qmin) + qmin)*M1s
        if logP:
            Ps = 10**(rand.random(size=N)*((np.log10(Pmax) - np.log10(Pmin))) + np.log10(Pmin))
        else:
            Ps = rand.random(size=N)*(Pmax - Pmin) + Pmin

        if eccfn is None:
            eccs = 0
        else:
            eccs = eccfn(Ps)

        self.eccfn = eccfn

        OrbitPopulation.__init__(self,M1s,M2s,Ps,eccs=eccs)

    def RV_RMSgrid(self,ts,res=20,mres=None,Pres=None,conf=0.95,measured_rms=None,drv=0,
                   plot=True,fig=None,contour=True,sigma=1):
        RVs = self.RV_timeseries(ts)
        RVs += rand.normal(size=np.size(RVs)).reshape(RVs.shape)*drv
        rms = RVs.std(axis=0)

        if mres is None:
            mres = res
        if Pres is None:
            Pres = res

        mbins = np.linspace(self.M2s.min(),self.M2s.max(),mres+1)
        Pbins = np.logspace(np.log10(self.Ps.min()),np.log10(self.Ps.max()),Pres+1)
        logPbins = np.log10(Pbins)

        mbin_centers = (mbins[:-1] + mbins[1:])/2.
        logPbin_centers = (logPbins[:-1] + logPbins[1:])/2.

        #print mbins
        #print Pbins

        minds = np.digitize(self.M2s,mbins)
        Pinds = np.digitize(self.Ps,Pbins)

        #means = np.zeros((mres,Pres))
        #stds = np.zeros((mres,Pres))
        pctiles = np.zeros((mres,Pres))
        ns = np.zeros((mres,Pres))

        for i in np.arange(mres):
            for j in np.arange(Pres):
                w = np.where((minds==i+1) & (Pinds==j+1))
                these = rms[w]
                #means[i,j] = these.mean() 
                #stds[i,j] = these.std()
                n = size(these)
                ns[i,j] = n
                if measured_rms is not None:
                    pctiles[i,j] = (these > sigma*measured_rms).sum()/float(n)
                else:
                    inds = np.argsort(these)
                    pctiles[i,j] = these[inds][int((1-conf)*n)]

        Ms,logPs = np.meshgrid(mbin_centers,logPbin_centers)
        #pts = np.array([Ms.ravel(),logPs.ravel()]).T
        #interp = interpnd(pts,pctiles.ravel())

        #interp = interp2d(Ms,logPs,pctiles.ravel(),kind='linear')

        if plot:
            plu.setfig(fig)

            if contour:
                mbin_centers = (mbins[:-1] + mbins[1:])/2.
                logPbins = np.log10(Pbins)
                logPbin_centers = (logPbins[:-1] + logPbins[1:])/2.
                if measured_rms is not None:
                    levels = [0.68,0.95,0.99]
                else:
                    levels = np.arange(0,20,2)
                c = plt.contour(logPbin_centers,mbin_centers,pctiles,levels=levels,colors='k')
                plt.clabel(c, fontsize=10, inline=1)
                
            else:
                extent = [np.log10(self.Ps.min()),np.log10(self.Ps.max()),self.M2s.min(),self.M2s.max()]
                im = plt.imshow(pctiles,cmap='Greys',extent=extent,aspect='auto')

                fig = plt.gcf()
                ax = plt.gca()


                if measured_rms is None:
                    cbarticks = np.arange(0,21,2)
                else:
                    cbarticks = np.arange(0,1.01,0.1)
                cbar = fig.colorbar(im, ticks=cbarticks)

            plt.xlabel('Log P')
            plt.ylabel('M2')

        #return interp
        return mbins,Pbins,pctiles,ns
            
