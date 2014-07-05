import numpy as np

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

    
def rand_spherepos(n,mininc=0,maxinc=pi/2,randfn=None,fnarg=None):
    if n==0:
        return None
    if randfn==None:
        theta = (rand_inc(n,mininc=mininc,maxinc=maxinc)-pi/2)*sign(scipy.random.random(n)-0.5)+pi/2
    else:
        if fnarg==None:
            theta = randfn(n)
        else:
            theta = randfn(n,fnarg)
    phi = scipy.random.random(n)*2*pi-pi
    return spherepos((theta,phi))
