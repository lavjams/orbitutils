import numpy as np
import numpy.random as rand

class Spherepos(object):
    """
    A class to keep track of positions on a sphere. (or 3-d positions)

    Initialized with either [x,y,z] or [theta,phi].

    """
    def __init__(self,coords,normed=True):
        if len(coords)==2:
            #spherical coordinates passed
            self.theta = coords[0]
            self.phi = coords[1]
            self.x = np.sin(self.theta)*np.cos(self.phi)
            self.y = np.sin(self.theta)*np.sin(self.phi)
            self.z = np.cos(self.theta)
        elif len(coords)==3:
            #cartesian coordinates passed.
            self.x = coords[0]
            self.y = coords[1]
            self.z = coords[2]
            self.norm = np.sqrt(self.x**2 + self.y**2 + self.z**2)
            self.normed = normed
            if normed:
                self.x = self.x/self.norm
                self.y = self.y/self.norm
                self.z = self.z/self.norm
            self.theta = np.arccos(self.z/sqrt(self.x**2+self.y**2+self.z**2))
            self.phi = np.arctan2(self.y,self.x)

    def transform(self,th,ph):
        th = inc; ph = phi
        newaxis = Spherepos((th,ph))
        (x,y,z) = newaxis.cart()
        theta = np.arccos(z)
        phi = np.arctan2(x,y)  #should it be arctan2(y,x)?
        (x,y,z) = self.cart()
        x2 = np.cos(phi)*x + np.cos(theta)*np.sin(phi)*y + np.sin(theta)*np.sin(phi)*z
        y2 = -np.sin(phi)*x + np.cos(theta)*np.cos(phi)*y + np.sin(theta)*np.cos(phi)*z
        z2 = -np.sin(theta)*y + np.cos(theta)*z
        return Spherepos((x2,y2,z2),normed=self.normed)
    
    def cart(self):
        return (self.x,self.y,self.z)

    def sph(self):
        return (self.theta,self.phi)

    def angsep(self,pos2):
        assert type(pos2) == type(self)
        return np.arccos(np.dot(self.cart(),pos2.cart()))
    
def rand_inc(n,mininc=0,maxinc=np.pi/2):
    umax = np.cos(mininc)
    umin = np.cos(maxinc)
    u = rand.random(size=n)*(umax-umin) + umin
    return np.arccos(u)
    #return np.arcsin(rand_sini(n,xmin=np.sin(mininc),xmax=np.sin(maxinc)))

def rand_sini(n,xmin=0,xmax=1):
    rmax = 1-np.sqrt(1-xmax**2)
    rmin = 1-np.sqrt(1-xmin**2)
    r = scipy.random.random(n)*(rmax-rmin)+rmin
    return np.sqrt(1-(1-r)**2)

    
def rand_spherepos(n,mininc=0,maxinc=np.pi/2,randfn=None,fnarg=None):
    if n==0:
        return None
    if randfn==None:
        theta = (rand_inc(n,mininc=mininc,maxinc=maxinc)-pi/2)*np.sign(rand..random(n)-0.5)+np.pi/2
    else:
        if fnarg==None:
            theta = randfn(n)
        else:
            theta = randfn(n,fnarg)
    phi = rand.random(n)*2*np.pi-np.pi
    return spherepos((theta,phi))
