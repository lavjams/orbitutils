import numpy as np


def dot(v1,v2):
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]

class Spherepos(object):
    """
    A class to keep track of positions on a sphere. (or 3-d positions)

    
    """
    def __init__(self,coords,normed=True):
        if len(coords)==2:
            self.theta = coords[0]
            self.phi = coords[1]
            self.x = np.sin(self.theta)*np.cos(self.phi)
            self.y = np.sin(self.theta)*np.sin(self.phi)
            self.z = np.cos(self.theta)
        elif len(coords)==3:
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

    def __getitem__(self,inds):
        return spherepos((self.theta[inds],self.phi[inds]))

    def __len__(self):
        return len(self.theta)

    def __call__(self):
        return self

    def __add__(self,pos2):
        if pos2==None and self != None:
            return self
        if self==None and pos2 != None:
            return pos2
        if self==None and pos2==None:
            return None
        self.theta = np.concatenate((self.theta,pos2.theta))
        self.phi = np.concatenate((self.phi,pos2.phi))
        self.x = np.concatenate((self.x,pos2.x))
        self.y = np.concatenate((self.y,pos2.y))
        self.z = np.concatenate((self.z,pos2.z))
        return self

    def transform(self,th,ph):
        if self==None:
            return None
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
        return np.arccos(dot(self.cart(),pos2.cart()))

    
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
