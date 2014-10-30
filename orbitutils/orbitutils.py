from __future__ import division,print_function

import numpy as np
from astropy.coordinates import SkyCoord,Angle
import numpy.random as rand
import sys,re,os
import matplotlib.pyplot as plt

from plotutils import setfig

from astropy import units as u
from astropy.units.quantity import Quantity
from astropy import constants as const
MSUN = const.M_sun.cgs.value
AU = const.au.cgs.value
DAY = 86400
G = const.G.cgs.value

from .kepler import Efn #

def semimajor(P,M):
    """P, M can be ``Quantity`` objects; otherwise default to day, M_sun
    """
    if type(P) != Quantity:
        P = P*u.day
    if type(M) != Quantity:
        M = M*u.M_sun
    a = ((P/2/np.pi)**2*const.G*M)**(1./3)
    return a.to(u.AU)

def random_spherepos(n):
    """returns SkyCoord object with n positions randomly oriented on the unit sphere

    Parameters
    ----------
    n : int
        number of positions desired

    Returns
    -------
    c : ``SkyCoord`` object with random positions 
    """
    signs = np.sign(rand.uniform(-1,1,size=n))
    thetas = Angle(np.arccos(rand.uniform(size=n)*signs),unit=u.rad) #random b/w 0 and 180
    phis = Angle(rand.uniform(0,2*np.pi,size=n),unit=u.rad)
    c = SkyCoord(phis,thetas,1,representation='physicsspherical')
    return c

def orbitproject(x,y,inc,phi=0,psi=0):
    """Transform x,y planar coordinates into observer's coordinate frame.
    
    x,y are coordinates in z=0 plane (plane of the orbit)

    observer is at (inc, phi) on celestial sphere (angles in radians);
    psi is orientation of final x-y axes about the (inc,phi) vector.

    Returns x,y,z values in observer's coordinate frame, where
    x,y are now plane-of-sky coordinates and z is along the line of sight.

    Parameters
    ----------
    x,y : float or arrray-like
        Coordinates to transorm

    inc : float or array-like
        Polar angle(s) of observer (where inc=0 corresponds to north pole
        of original x-y plane).  This angle is the same as standard "inclination."

    phi : float or array-like, optional
        Azimuthal angle of observer around z-axis

    psi : float or array-like, optional
        Orientation of final observer coordinate frame (azimuthal around
        (inc,phi) vector.

    Returns
    -------
    x,y,z : ``ndarray``
        Coordinates in observers' frames.  x,y in "plane of sky" and z
        along line of sight.
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

    Returns positions and velocities as SkyCoord objects.  Uses
    ``orbitutils.kepler.Efn`` to calculate eccentric anomalies using interpolation.    

    Parameters
    ----------
    Ms, eccs, semimajors, mreds : float or array-like
        Mean anomalies, eccentricities, semimajor axes (AU), reduced masses (Msun).

    obspos : ``None``, (x,y,z) tuple or ``SkyCoord`` object
        Locations of observers for which to return coordinates.
        If ``None`` then populate randomly on sphere.  If (x,y,z) or
        ``SkyCoord`` object provided, then use those.

    Returns
    -------
    pos,vel : ``SkyCoord``
        Objects representing the positions and velocities, the coordinates
        of which are ``Quantity`` objects that have units.  Positions are in
        projected AU and velocities in km/s. 
    """

    Es = Efn(Ms,eccs) #eccentric anomalies by interpolation

    rs = semimajors*(1-eccs*np.cos(Es))
    nus = 2 * np.arctan2(np.sqrt(1+eccs)*np.sin(Es/2),np.sqrt(1-eccs)*np.cos(Es/2))

    xs = semimajors*(np.cos(Es) - eccs)         #AU
    ys = semimajors*np.sqrt(1-eccs**2)*np.sin(Es)  #AU
    
    Edots = np.sqrt(G*mreds*MSUN/(semimajors*AU)**3)/(1-eccs*np.cos(Es))
        
    xdots = -semimajors*AU*np.sin(Es)*Edots/1e5  #km/s
    ydots = semimajors*AU*np.sqrt(1-eccs**2)*np.cos(Es)*Edots/1e5 # km/s
        
    n = np.size(xs)

    orbpos = SkyCoord(xs,ys,0*u.AU,representation='cartesian',unit='AU')
    orbvel = SkyCoord(xdots,ydots,0*u.km/u.s,representation='cartesian',unit='km/s')
    if obspos is None:
        obspos = random_spherepos(n) #observer position
    if type(obspos) == type((1,2,3)):
        obspos = SkyCoord(obspos[0],obspos[1],obspos[2],
                          representation='cartesian').represent_as('physicsspherical')

    if not hasattr(obspos,'theta'): #if obspos not physics spherical, make it 
        obspos = obspos.represent_as('physicsspherical')
        
    #random orientation of the sky 'x-y' coordinates
    psi = rand.random(n)*2*np.pi  

    #transform positions and velocities into observer coordinates
    x,y,z = orbitproject(orbpos.x,orbpos.y,obspos.theta,obspos.phi,psi)
    vx,vy,vz = orbitproject(orbvel.x,orbvel.y,obspos.theta,obspos.phi,psi)

    return (SkyCoord(x,y,z,representation='cartesian'),
            SkyCoord(vx,vy,vz,representation='cartesian')) 

class TripleOrbitPopulation(object):
    def __init__(self,M1s,M2s,M3s,Plong,Pshort,ecclong=0,eccshort=0,n=None,
                 mean_anomalies_long=None,obsx_long=None,obsy_long=None,obsz_long=None,
                 obspos_long=None,
                 mean_anomalies_short=None,obsx_short=None,obsy_short=None,obsz_short=None,
                 obspos_short=None):                 
        """Stars 2 and 3 orbit each other close (short orbit), far from star 1 (long orbit)

        Object that defines a triple star system, with orbits calculated approximating
        Pshort << Plong.

        Parameters
        ----------
        M1s, M2s, M3s : float or array-like
            Masses of stars.  Stars 2 and 3 are in a short orbit, far away from star 1.

        Plong, Pshort : float or array-like
            Orbital Periods.  Plong is orbital period of 2+3 and 1; Pshort is orbital
            period of 2 and 3.

        ecclong, eccshort : float or array-like, optional
            Eccentricities.  Same story (long vs. short).  Default=0 (circular).

        n : int, optional
            Number of systems to simulate (if M1s, M2s, M3s aren't arrays of size > 1
            already)

        mean_anomalies_short, mean_anomalies_long : float or array_like, optional
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
        if Plong < Pshort:
            Pshort,Plong = (Plong, Pshort)
        
        self.orbpop_long = OrbitPopulation(M1s,M2s+M3s,Plong,eccs=ecclong,n=n,
                                           mean_anomalies=mean_anomalies_long,
                                           obsx=obsx_long,obsy=obsy_long,obsz=obsz_long)

        self.orbpop_short = OrbitPopulation(M2s,M3s,Pshort,eccs=eccshort,n=n,
                                           mean_anomalies=mean_anomalies_short,
                                           obsx=obsx_short,obsy=obsy_short,obsz=obsz_short)



    @property
    def RV_1(self):
        """Instantaneous RV of star 1 with respect to system center-of-mass
        """
        return self.orbpop_long.RVs * (self.orbpop_long.M2s / (self.orbpop_long.M1s + self.orbpop_long.M2s))

    @property
    def RV_2(self):
        """Instantaneous RV of star 2 with respect to system center-of-mass
        """
        return -self.orbpop_long.RVs * (self.orbpop_long.M1s /
                                        (self.orbpop_long.M1s + self.orbpop_long.M2s)) +\
                self.orbpop_short.RVs_com1
                
    @property
    def RV_3(self):
        """Instantaneous RV of star 3 with respect to system center-of-mass
        """
        return -self.orbpop_long.RVs * (self.orbpop_long.M1s / (self.orbpop_long.M1s + self.orbpop_long.M2s)) +\
            self.orbpop_short.RVs_com2

    @property
    def Rsky(self):
        """Projected separation of star 2+3 pair from star 1 
        """
        return self.orbpop_long.Rsky
    
    def dRV_1(self,dt):
        """Returns difference in RVs (separated by time dt) of star 1.
        """
        return self.orbpop_long.dRV(dt,com=True)

    def dRV_2(self,dt):
        """Returns difference in RVs (separated by time dt) of star 2.
        """
        return -self.orbpop_long.dRV(dt) * (self.orbpop_long.M1s/(self.orbpop_long.M1s + self.orbpop_long.M2s)) +\
            self.orbpop_short.dRV(dt,com=True)

    def dRV_3(self,dt):
        """Returns difference in RVs (separated by time dt) of star 3.
        """
        return -self.orbpop_long.dRV(dt) * (self.orbpop_long.M1s/(self.orbpop_long.M1s + self.orbpop_long.M2s)) -\
            self.orbpop_short.dRV(dt) * (self.orbpop_short.M1s/(self.orbpop_short.M1s + self.orbpop_short.M2s))
        

class OrbitPopulation(object):
    def __init__(self,M1s,M2s,Ps,eccs=0,n=None,
                 mean_anomalies=None,obsx=None,obsy=None,obsz=None,
                 obspos=None):
        """Population of orbits.

        Parameters
        ----------
        M1s, M2s : float or array_like
            Primary and secondary masses (in solar masses)

        Ps : float or array_like
            Orbital periods (in days)

        eccs : float or array_like, optional
            Eccentricities.

        n : int, optional
            Number of instances to simulate.  If not provided, then this number
            will be the length of M2s (or Ps) provided.

        mean_anomalies : float or array_like, optional
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
        M1s = np.atleast_1d(M1s) * u.M_sun
        M2s = np.atleast_1d(M2s) * u.M_sun
        Ps = np.atleast_1d(Ps) * u.day

        if n is None:
            if len(M2s)==1:
                n = len(Ps)
            else:
                n = len(M2s)

        if len(M1s)==1 and len(M2s)==1:
            M1s = np.ones(n)*M1s
            M2s = np.ones(n)*M2s

        self.M1s = M1s
        self.M2s = M2s

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

        #coordinates of random observers
        if obspos is None:
            if obsx is None:
                self.obspos = random_spherepos(n)
            else:
                self.obspos = SkyCoord(obsx,obsy,obsz,representation='cartesian')
        else:
            self.obspos = obspos

        #get positions, velocities relative to M1
        positions,velocities = orbit_posvel(self.Ms,self.eccs,self.semimajors.value,
                                            self.mreds.value,
                                            self.obspos)

        self.positions = positions
        self.velocities = velocities
        
    @property
    def Rsky(self):
        """Projected sky separation of stars
        """
        return np.sqrt(self.positions.x**2 + self.positions.y**2)

    @property
    def RVs(self):
        """Relative radial velocities of two stars
        """
        return self.velocities.z

    @property
    def RVs_com1(self):
        """RVs of star 1 relative to center-of-mass
        """
        return self.RVs * (self.M2s / (self.M1s + self.M2s))

    @property
    def RVs_com2(self):
        """RVs of star 2 relative to center-of-mass
        """
        return -self.RVs * (self.M1s / (self.M1s + self.M2s))
    
    def dRV(self,dt,com=False):
        """

        dt in days; if com, then returns the change in RV of component 1 in COM frame
        """
        if type(dt) != Quantity:
            dt *= u.day

        mean_motions = np.sqrt(G*(self.mreds)*MSUN/(self.semimajors*AU)**3)
        mean_motions = np.sqrt(const.G*(self.mreds)/(self.semimajors)**3)
        #print mean_motions * dt / (2*pi)

        newMs = self.Ms + mean_motions * dt
        pos,vel = orbit_posvel(newMs,self.eccs,self.semimajors.value,
                               self.mreds.value,
                               self.obspos)
        
        if com:
            return (vel.z - self.RVs) * (self.M2s / (self.M1s + self.M2s))
        else:
            return vel.z-self.RVs

    def RV_timeseries(self,ts,recalc=False):
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
            
