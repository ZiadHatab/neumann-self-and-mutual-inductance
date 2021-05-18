import numpy as np
import random

# my script
import bezier


class Inductance():
    
    def self_neumann(self,B,r):
        '''
        B: ndarray; B = [[x0,y0,z0],[x1,y1,z1], ..., [xn,yn,zn]]
        r: scalar; wire radius in meter
        '''
        mu0 = 1.25663706e-6
        
        # small wire segments
        dB = np.diff(B, axis=0)
        
        # get center points
        Bcen = np.array([np.convolve(b, np.ones(2)/2, mode='valid') for b in B.T]).T
        
        # find normal vector to plane spanned by the curve
        # choose two random points for cross product
        [inx1, inx2] = random.sample(range(0, len(dB)), 2)
        a = np.cross(dB[inx1], dB[inx2])
        b = np.cross(np.array([1,1,1]),B[-1]-B[0]) # in case the curve is a line
        normal = b/np.sqrt(np.sum(b**2)) if (a == 0).all() else a/np.sqrt(np.sum(a**2))
    
        # create second wire normally shifted by r
        Bcen2 = Bcen + r*normal
        
        # quality check of the segmentation
        dl = np.sqrt(np.sum(dB**2, axis=1))
        l = np.sum(dl)   # total length
        
        ratio =  dl/r
        numGood = np.sum(ratio < 0.6)  # number of short segments (good)
        numBad = np.sum(ratio >= 0.6)  # number of long segments (bad)
        
        print(f'Wire total length approx: {l}')
        print(f'Max length of a segment: {max(dl)}')
        print(f'Ratio of max segment length to wire radius: {max(dl)/r:0.4f} (less than 0.6 is good)')
        print(f'Good segments (<0.6): {numGood}; Bad segments (>=0.6): {numBad}')
        
        # 0.6 seems to be a good threshold (based on experimenting)
        if max(dl)/r > 0.6:
            print('The ratio of max segment length to wire radius is greater than 0.6. This can impact the solution accuracy.')
            
        if numGood <= numBad:
            print('number of Good segments (<0.6) are less than bad segments (>=0.6). Increase the sample rate to improve this!')
        
        # perform the integration
        [X, Y, Z] = np.array([Bcen[:,0]]), np.array([Bcen[:,1]]), np.array([Bcen[:,2]])
        [X2, Y2, Z2] = np.array([Bcen2[:,0]]), np.array([Bcen2[:,1]]), np.array([Bcen2[:,2]])
        [dX, dY, dZ] = np.array([dB[:,0]]), np.array([dB[:,1]]), np.array([dB[:,2]])
        
        intag = (dX*dX.T + dY*dY.T + dZ*dZ.T)/np.sqrt((X2-X.T)**2 + (Y2.T-Y)**2 + (Z2-Z.T)**2)
        
        L = (mu0/4/np.pi)*np.sum(intag)
        
        # forloop; the slow way to do things!
        '''
        intag = []
        for x1,y1,z1,dx1,dy1,dz1 in zip(Bcen[:,0], Bcen[:,1], Bcen[:,2], dB[:,0],dB[:,1],dB[:,2]):
            for x2,y2,z2,dx2,dy2,dz2 in zip(Bcen2[:,0], Bcen2[:,1], Bcen2[:,2], dB[:,0],dB[:,1],dB[:,2]):
                intag.append((dx1*dx2 + dy1*dy2 + dz1*dz2)/np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2))
        
        L = (mu0/4/np.pi)*np.sum(intag)
        print(L)
        '''
        return L
    
    def mutual_neumann(self,B1,B2):
        '''
        B1: ndarray; first wire B1 = [[x0,y0,z0],[x1,y1,z1], ..., [xn,yn,zn]]
        B2: ndarray; second wire B2 = [[x0,y0,z0],[x1,y1,z1], ..., [xn,yn,zn]]
        '''
        mu0 = 1.25663706e-6
        
        # small wire segments
        dB1 = np.diff(B1, axis=0)
        dB2 = np.diff(B2, axis=0)
        
        # center points
        Bcen1 = np.array([np.convolve(b, np.ones(2)/2, mode='valid') for b in B1.T]).T
        Bcen2 = np.array([np.convolve(b, np.ones(2)/2, mode='valid') for b in B2.T]).T
        
        # perform the integration
        [X1, Y1, Z1] = np.array([Bcen1[:,0]]), np.array([Bcen1[:,1]]), np.array([Bcen1[:,2]])
        [X2, Y2, Z2] = np.array([Bcen2[:,0]]), np.array([Bcen2[:,1]]), np.array([Bcen2[:,2]])
        [dX1, dY1, dZ1] = np.array([dB1[:,0]]), np.array([dB1[:,1]]), np.array([dB1[:,2]])
        [dX2, dY2, dZ2] = np.array([dB2[:,0]]), np.array([dB2[:,1]]), np.array([dB2[:,2]])
        
        intag = (dX1*dX2.T + dY1*dY2.T + dZ1*dZ2.T)/np.sqrt((X2-X1.T)**2 + (Y2.T-Y1)**2 + (Z2-Z1.T)**2)
        
        M = (mu0/4/np.pi)*np.sum(intag)
        
        return M

class Capacitance():
    
    def wire2wire(self,B1,B2,r,er=1):
        e0 = 8.8541878128*1e-12
    
        dB = np.diff(B1, axis=0)
        dl = np.sqrt(np.sum(dB**2, axis=1))
        
        # center points
        Bcen1 = np.array([np.convolve(b, np.ones(2)/2, mode='valid') for b in B1.T]).T
        Bcen2 = np.array([np.convolve(b, np.ones(2)/2, mode='valid') for b in B2.T]).T
        
        h = np.sqrt(np.sum((Bcen2-Bcen1)**2, axis=1))
        C = np.pi*e0*er*np.sum(dl/np.arccosh(h/r/2))
        
        return C
    

class Resistance():
    
    def gold(self,B,r):
        sigma = 4.10e7
        A = np.pi*r**2
        dB = np.diff(B, axis=0)
        dl = np.sqrt(np.sum(dB**2, axis=1))
        l = np.sum(dl)
        R = l/A/sigma
        return R
    
    def aluminum(self,B,r):
        sigma = 3.5e7
        A = np.pi*r**2
        dB = np.diff(B, axis=0)
        dl = np.sqrt(np.sum(dB**2, axis=1))
        l = np.sum(dl)
        R = l/A/sigma
        return R
    
    def generic(self,B,r,sigma):
        A = np.pi*r**2
        dB = np.diff(B, axis=0)
        dl = np.sqrt(np.sum(dB**2, axis=1))
        l = np.sum(dl)
        R = l/A/sigma
        return R
    
if __name__ == '__main__':
    P = np.array([[0,200,0], [250,250+200,0], [500,200,0]])*1e-6
    t = np.linspace(0,1,1000)
    B = bezier.curve(t, P)
    
    r = 25e-6/2
    
    wire = Inductance()
    print(wire.self_neumann(B,r))
    print(wire.mutual_neumann(B, B + 0.1e-3*np.array([0,0,1])))
    print(Resistance().gold(B, r))
    print(Capacitance().wire2wire(B, B + 0.1e-3*np.array([0,0,1]), r, 1))
    
# EOF