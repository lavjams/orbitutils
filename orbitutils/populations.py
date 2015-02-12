from __future__ import division,print_function

import numpy as np
from astropy.coordinates import SkyCoord,Angle
import numpy.random as rand
import sys,re,os
import matplotlib.pyplot as plt

import pandas as pd

from plotutils import setfig
from hashutils import hashcombine, hashdf

from astropy import units as u
from astropy.units.quantity import Quantity
from astropy import constants as const
MSUN = const.M_sun.cgs.value
AU = const.au.cgs.value
DAY = 86400
G = const.G.cgs.value

from .utils import semimajor,random_spherepos,orbitproject,orbit_posvel


class TripleOrbitPopulation(object):
    def __init__(self,M1,M2,M3,Plong,Pshort,ecclong=0,eccshort=0,n=None,
                 mean_anomaly_long=None,obsx_long=None,obsy_long=None,obsz_long=None,
                 obspos_long=None,
                 mean_anomaly_short=None,obsx_short=None,obsy_short=None,obsz_short=None,
                 obspos_short=None):                 
        """Stars 2 and 3 orbit each other close (short orbit), far from star 1 (long orbit)

        Object that defines a triple star system, with orbits calculated approximating
        Pshort << Plong.

        Parameters
        ----------
        M1, M2, M3 : float, array-like, or ``Quantity``
            Masses of stars.  Stars 2 and 3 are in a short orbit, far away from star 1.
            If not ``Quantity`` objects, then assumed to be in solar mass units.

        Plong, Pshort : float or array-like, or ``Quantity``
            Orbital Periods.  Plong is orbital period of 2+3 and 1; Pshort is orbital
            period of 2 and 3.  If not ``Quantity`` objects, assumed to be in days.
            N.B. If any item in Pshort happens to be longer than the corresponding item
            in Plong, they will be switched. 

        ecclong, eccshort : float or array-like, optional
            Eccentricities.  Same story (long vs. short).  Default=0 (circular).

        n : int, optional
            Number of systems to simulate (if M1s, M2s, M3s aren't arrays of size > 1
            already)

        mean_anomaly_short, mean_anomaly_long : float or array_like, optional
            Mean anomalies.  This is only passed if you need to "restore" a
            particular specific configuration (i.e., a particular saved simulation).
            If not provided, then randomized on (0, 2pi).

        obsx_short, obsy_short, obsz_short : float or array_like, optional
            "Observer" positions for the short orbit.

        obsx_long, obsy_long, obsz_long : flot or array_like, optional
            "Observer" positions for long orbit.

        obspos_short, obspos_long : None or ``SkyCoord``
            "Observer" positions for short and long, provided as ``SkyCoord`` objects
            (replaces obsx_short/long, obsy_short/long, obsz_short/long)
        """
        Pshort, Plong = (np.minimum(Pshort,Plong), np.maximum(Pshort,Plong))
        #if Plong < Pshort:
        #    Pshort,Plong = (Plong, Pshort)
        
        self.orbpop_long = OrbitPopulation(M1,M2+M3,Plong,ecc=ecclong,n=n,
                                           mean_anomaly=mean_anomaly_long,
                                           obsx=obsx_long,obsy=obsy_long,obsz=obsz_long)

        self.orbpop_short = OrbitPopulation(M2,M3,Pshort,ecc=eccshort,n=n,
                                           mean_anomaly=mean_anomaly_short,
                                           obsx=obsx_short,obsy=obsy_short,obsz=obsz_short)

    def __hash__(self):
        return hashcombine(self.orbpop_long, self.orbpop_short)
        
    @property
    def RV(self):
        """Instantaneous RV of star 1 with respect to system center-of-mass
        """
        return self.RV_1
        
    @property
    def RV_1(self):
        """Instantaneous RV of star 1 with respect to system center-of-mass
        """
        return self.orbpop_long.RV * (self.orbpop_long.M2 / (self.orbpop_long.M1 + self.orbpop_long.M2))

    @property
    def RV_2(self):
        """Instantaneous RV of star 2 with respect to system center-of-mass
        """
        return -self.orbpop_long.RV * (self.orbpop_long.M1 /
                                        (self.orbpop_long.M1 + self.orbpop_long.M2)) +\
                self.orbpop_short.RV_com1
                
    @property
    def RV_3(self):
        """Instantaneous RV of star 3 with respect to system center-of-mass
        """
        return -self.orbpop_long.RV * (self.orbpop_long.M1 / (self.orbpop_long.M1 + self.orbpop_long.M2)) +\
            self.orbpop_short.RV_com2

    @property
    def Rsky(self):
        """Projected separation of star 2+3 pair from star 1 
        """
        return self.orbpop_long.Rsky

    def dRV(self,dt):
        """Returns difference in RVs (separated by time dt) of star 1.
        """
        return self.dRV_1(dt)
    
    def dRV_1(self,dt):
        """Returns difference in RVs (separated by time dt) of star 1.
        """
        return self.orbpop_long.dRV(dt,com=True)

    def dRV_2(self,dt):
        """Returns difference in RVs (separated by time dt) of star 2.
        """
        return -self.orbpop_long.dRV(dt) * (self.orbpop_long.M1/(self.orbpop_long.M1 + self.orbpop_long.M2)) +\
            self.orbpop_short.dRV(dt,com=True)

    def dRV_3(self,dt):
        """Returns difference in RVs (separated by time dt) of star 3.
        """
        return -self.orbpop_long.dRV(dt) * (self.orbpop_long.M1/(self.orbpop_long.M1 + self.orbpop_long.M2)) -\
            self.orbpop_short.dRV(dt) * (self.orbpop_short.M1/(self.orbpop_short.M1 + self.orbpop_short.M2))

    def save_hdf(self,filename,path=''):
        """Save to .h5 file.
        """
        self.orbpop_long.save_hdf(filename,'{}/long'.format(path))
        self.orbpop_short.save_hdf(filename,'{}/short'.format(path))
        

    def __add__(self, other):
        if type(self) != type(other):
            raise TypeError('Can only add like types of TripleOrbitPopulation')

        newdf_long = pd.concat((self.orbpop_long.dataframe, other.orbpop_long.dataframe))
        newdf_short = pd.concat((self.orbpop_short.dataframe, other.orbpop_short.dataframe))

        return TripleOrbitPopulation_FromDF(newdf_long, newdf_short)

class TripleOrbitPopulation_FromDF(TripleOrbitPopulation):
    def __init__(self, df_long, df_short):
        """Initializes ``TripleOrbitPopulation`` from two DataFrame objects
        """
        self.orbpop_long = OrbitPopulation_FromDF(df_long)
        self.orbpop_short = OrbitPopulation_FromDF(df_short)
            
class TripleOrbitPopulation_FromH5(TripleOrbitPopulation_FromDF):
    def __init__(self,filename,path=''):
        """Restore ``TripleOrbitPopulation`` from saved .h5 file.
        """
        df_long = pd.read_hdf(filename,'{}/long/df'.format(path), autoclose=True)
        df_short = pd.read_hdf(filename,'{}/short/df'.format(path), autoclose=True)
        TripleOrbitPopulation_FromDF.__init__(self, df_long, df_short)
        #self.orbpop_long = OrbitPopulation_FromH5(filename,'{}/long'.format(path))
        #self.orbpop_short = OrbitPopulation_FromH5(filename,'{}/short'.format(path))
        
            
class OrbitPopulation(object):
    def __init__(self,M1,M2,P,ecc=0,n=None,
                 mean_anomaly=None,obsx=None,obsy=None,obsz=None,
                 obspos=None):
        """Population of orbits.

        Parameters
        ----------
        M1, M2 : float or array_like, or ``Quantity``
            Primary and secondary masses (if not ``Quantity``, assumed to be in solar masses)

        P : float or array_like, or ``Quantity``
            Orbital periods (if not ``Quantity``, assumed to be in days)

        ecc : float or array_like, optional
            Eccentricities.

        n : int, optional
            Number of instances to simulate.  If not provided, then this number
            will be the length of M2s (or Ps) provided.

        mean_anomaly : float or array_like, optional
            Mean anomalies of orbits.  Usually this will just be set randomly,
            but can be provided to initialize a particular state (e.g., when
            restoring an ``OrbitPopulation`` object from a saved state).

        obsx, obsy, obsz : float or array_like, optional
            "Observer" positions to define coordinates.  Will be set randomly
            if not provided.

        obspos : ``SkyCoord`` object, optional
            "Observer" positions may be set with a ``SkyCoord`` object (replaces
            obsx, obsy, obsz)
        """
        if type(M1) != Quantity:
            M1 = Quantity(M1, unit='M_sun')
        if type(M2) != Quantity:
            M2 = Quantity(M2, unit='M_sun')
        if type(P) != Quantity:
            P = Quantity(P, unit='day')

        if n is None:
            if M2.size==1:
                n = P.size
            else:
                n = M2.size

        # Below now unnecessary...
        #if len(M1s)==1 and len(M2s)==1:
        #    M1s = np.ones(n)*M1s
        #    M2s = np.ones(n)*M2s

        self.M1 = M1
        self.M2 = M2

        self.N = n

        #if np.size(Ps)==1:
        #    Ps = Ps*np.ones(n)

        self.P = P

        if np.size(ecc) == 1:
            ecc = np.ones(n)*ecc

        self.ecc = ecc

        mred = M1*M2/(M1+M2)
        self.semimajor = semimajor(P,mred)   #AU
        self.mred = mred

        if mean_anomaly is None:
            M = rand.uniform(0,2*np.pi,size=n)
        else:
            M = mean_anomaly

        self.M = M

        #coordinates of random observers
        if obspos is None:
            if obsx is None:
                self.obspos = random_spherepos(n)
            else:
                self.obspos = SkyCoord(obsx,obsy,obsz,representation='cartesian')
        else:
            self.obspos = obspos

        #get positions, velocities relative to M1
        position,velocity = orbit_posvel(self.M,self.ecc,self.semimajor.value,
                                            self.mred.value,
                                            self.obspos)

        self.position = position
        self.velocity = velocity

    def __add__(self, other):
        if type(self) != type(other):
            raise TypeError('Can only add like types of OrbitPopulation')

        newdf = pd.concat((self.dataframe, other.dataframe))

        return OrbitPopulation_FromDF(newdf)

    def __hash__(self):
        return hashdf(self.dataframe)
    
    @property
    def Rsky(self):
        """Projected sky separation of stars
        """
        return np.sqrt(self.position.x**2 + self.position.y**2)

    @property
    def RV(self):
        """Relative radial velocities of two stars
        """
        return self.velocity.z

    @property
    def RV_com1(self):
        """RVs of star 1 relative to center-of-mass
        """
        return self.RV * (self.M2 / (self.M1 + self.M2))

    @property
    def RV_com2(self):
        """RVs of star 2 relative to center-of-mass
        """
        return -self.RV * (self.M1 / (self.M1 + self.M2))
    
    def dRV(self,dt,com=False):
        """Change in RV of star 1 for time separation dt (default=days)

        Parameters
        ----------
        dt : float, array_like, or ``Quantity``
            Time separation for which to compute RV change.  If not a ``Quantity``,
            then assumed to be in days. 

        com : bool, optional
            If ``True``, then return dRV of star 1 in center-of-mass frame.

        Returns
        -------
        dRV : ``Quantity``
            Change in radial velocity over time dt.
        """
        if type(dt) != Quantity:
            dt *= u.day

        mean_motions = np.sqrt(G*(self.mred)*MSUN/(self.semimajor*AU)**3)
        mean_motions = np.sqrt(const.G*(self.mred)/(self.semimajor)**3)
        #print mean_motions * dt / (2*pi)

        newM = self.M + mean_motions * dt
        pos,vel = orbit_posvel(newM,self.ecc,self.semimajor.value,
                               self.mred.value,
                               self.obspos)
        
        if com:
            return (vel.z - self.RV) * (self.M2 / (self.M1 + self.M2))
        else:
            return vel.z-self.RV

    def RV_timeseries(self,ts,recalc=False):
        """Radial Velocity time series for star 1 at given times ts.
        """
        if type(ts) != Quantity:
            ts *= u.day

        if not recalc and hasattr(self,'RV_measurements'):
            if (ts == self.ts).all():
                return self.RV_measurements
            else:
                pass
            
        RVs = Quantity(np.zeros((len(ts),self.N)),unit='km/s')
        for i,t in enumerate(ts):
            RVs[i,:] = self.dRV(t,com=True)
        self.RV_measurements = RVs
        self.ts = ts
        return RVs

    @property
    def dataframe(self):
        if not hasattr(self,'_dataframe'):
            obspos = self.obspos.represent_as('cartesian')
            obsx, obsy, obsz = (obspos.x,obspos.y,obspos.z)
            df = pd.DataFrame({'M1':self.M1,
                               'M2':self.M2,
                               'P':self.P,
                               'ecc':self.ecc,
                               'mean_anomaly':self.M,
                               'obsx':obsx,
                               'obsy':obsy,
                               'obsz':obsz})
            self._dataframe = df
        return self._dataframe


    def scatterplot(self,fig=None,figsize=(7,7),ms=0.5,
                    rmax=None,log=False,**kwargs):
        setfig(fig,figsize=figsize)
        plt.plot(self.position.x.value,self.position.y.value,'o',ms=ms,**kwargs)
        plt.xlabel('projected separation [AU]')
        plt.ylabel('projected separation [AU]')
        if rmax is not None:
            plt.xlim((-rmax,rmax))
            plt.ylim((-rmax,rmax))
        if log:
            plt.xscale('log')
            plt.yscale('log')
    
    def save_hdf(self,filename,path=''):
        """Saves all relevant data to .h5 file; so state can be restored.
        """
        self.dataframe.to_hdf(filename,'{}/df'.format(path))

class OrbitPopulation_FromDF(OrbitPopulation):
    def __init__(self, df):
        """Creates ``OrbitPopulation`` from DataFrame
        """
        OrbitPopulation.__init__(self, df['M1'], df['M2'], df['P'],
                                 ecc=df['ecc'], mean_anomaly=df['mean_anomaly'],
                                 obsx=df['obsx'], obsy=df['obsy'], obsz=df['obsz']) 

class OrbitPopulation_FromH5(OrbitPopulation_FromDF):
    def __init__(self,filename,path=''):
        """Restores ``OrbitPopulation`` from saved .h5 file.
        """
        df = pd.read_hdf(filename,'{}/df'.format(path), autoclose=True)
        OrbitPopulation_FromDF.__init__(self, df)

        
class BinaryGrid(OrbitPopulation):
    def __init__(self, M1, qmin=0.1, qmax=1, Pmin=0.5, Pmax=365, N=1e5, logP=True, eccfn=None):
        """A grid of companions to primary, in mass ratio and period space.

        Parameters
        ----------
        M1 : float
            Primary mass (solar masses)

        qmin,qmax : float, optional
            Minimum and maximum mass ratios.

        Pmin,Pmax : float, optional
            Min/max periods in days.

        N : int, optional
            Total number of simulations.

        logP : bool, optional
            Whether to grid in log-period.  If ``False``, then linear.

        eccfn : callable, or ``None``, optional
            Function that returns eccentricity as a function of period.
        """
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

        OrbitPopulation.__init__(self,M1s,M2s,Ps,ecc=eccs)

    def RV_RMSgrid(self,ts,res=20,mres=None,Pres=None,conf=0.95,measured_rms=None,drv=0,
                   plot=True,fig=None,contour=True,sigma=1):
        """Writes a grid of RV RMS values, assuming observations at given times.

        Hasn't really been tested 
        """
        RVs = self.RV_timeseries(ts)
        RVs += rand.normal(size=np.size(RVs)).reshape(RVs.shape)*drv
        rms = RVs.std(axis=0)

        if mres is None:
            mres = res
        if Pres is None:
            Pres = res

        mbins = np.linspace(self.M2.min(),self.M2.max(),mres+1)
        Pbins = np.logspace(np.log10(self.P.min()),np.log10(self.P.max()),Pres+1)
        logPbins = np.log10(Pbins)

        mbin_centers = (mbins[:-1] + mbins[1:])/2.
        logPbin_centers = (logPbins[:-1] + logPbins[1:])/2.

        #print mbins
        #print Pbins

        minds = np.digitize(self.M2,mbins)
        Pinds = np.digitize(self.P,Pbins)

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
            setfig(fig)

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
                extent = [np.log10(self.P.min()),np.log10(self.P.max()),self.M2.min(),self.M2.max()]
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
            
