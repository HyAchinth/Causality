""" Multivariate RKHS.  
    datapoints is np array of shape (D x N), where D is the number of variables and N is the number of data
    points.
    F(x,M) evaluates Joint Probability Density of all variables except those marked zero in array M.
    Uses Multivariate Gaussian Density Kernel, or Multivariate Gaussian Cumulative Density Kernel, based
    on setting of "cumulative" to True for cumulative or False for density. 
"""
from math import e, sqrt, pi, log, gamma
import numpy as np
from scipy import stats

class RKHS:
    def __init__(self, data, sigmas = None, includeVars = None, delta = 3, s=1):
        """
        """
        assert type(data) == type({}), "rkhsMV: Error -- Data must be in the form of a dictionary varName -> [val1, val2, ... , valN]"
        # If includeVars is present, use it to build the samples and associated sigmas
        self.vMap = {} # Map from var name to var index
        self.iMap = {} # Map from index to varName
        self.s = s # Smoothness factor
        if includeVars is None:
            self.varNames = list(data.keys())
            self.D = len(self.varNames)
        else:
            self.varNames = includeVars
            self.D = len(self.varNames)
        self.N = len(data[self.varNames[0]])
        self.X = []
        tempA = []
        self.stdDevs = []
        for i in range(self.D):
            varName = self.varNames[i]
            dat = data[varName]
            stdDev = np.std(dat)
            self.stdDevs.append(stdDev)
            self.vMap[varName] = i
            self.iMap[i] = varName
            tempA.append(dat)
        for i in range(self.N):
            sample = []
            for j in range(self.D):
                val = tempA[j][i]
                sample.append(val)
            self.X.append(sample)
        self.X = np.array(self.X)
        # Compute covariance matrix for kernel
        coverage = 1
        #print(self.X)
        Xa = np.array(self.X).T
        #print('Xa = ',  Xa.shape)
        c = np.cov(Xa).reshape(self.D, self.D)
        #hsradius = np.linalg.norm(np.diag(c), 2) # Hyper-sphere radius
        #hsvol = pi**self.D/2.0 / gamma(self.D/2 + 1) * hsradius**self.D
        volume = 0
        dims2 = np.diag(c)
        vol = 1
        for dim2 in dims2:
            dim = sqrt(dim2)
            vol *= dim
        
        #print('hsvol = ', hsvol)
        #covdet = np.linalg.det(c)
        #elvol = hsvol * covdet
        #print('elvol, covdet = ', elvol, covdet)
        #targetelvol = elvol * sigmaScale * coverage / self.N
        #targetvol = vol * sigmaScale * coverage / self.N
        #scale = sqrt(targetvol/vol**(1/self.D))
        #scale = (1 / log(self.N, 4)) / 6
        #scale = 2 / self.D * self.N**(-1/6) * sigmaScale
        #scale = (.4 / (self.D)) * self.N**(-1/6) # ORIG ***
        #scale = (.4 / (self.D)) * self.N**(-1/6) # ORIG ***
        #scale = .8 / self.D * self.N**(-1/6) * sigmaScale
        scale = (.28 * self.N**(-1/5) * self.s)**(1/self.D)
        scale = s**(1/self.D) * 1
        scale = (.1 * s * self.N**(-1/6))**(1/self.D)
        #scale = (1.6 / (self.D)) * 1/log(self.N,3.8)
        #scale = (2 / (self.D)) * 1/log(self.N, 4) * sigmaScale
        #print('scalevol = ', scale**self.D * self.N)
        #print('scalevol = ', scale**self.D * self.N)
        #print('volume = ', vol)
        #print('targetvol = ', targetvol)
        #print('scale = ', scale)
        cS = np.zeros(shape=c.shape)
        #print('c = ')
        for i in range(self.D):
            for j in range(self.D):
                v = c[i,j]
                cS[i,j] = (sqrt(v) * scale)**2 if v >=0 else -(sqrt(-v) * scale)**2
        #print('c = ', c, c.shape)
        #cS = c * scale
        #print('cs = ', cS, c.shape)
        #5/0
        self.delta = delta
        # Calculate the covariance matrix, assuming all vars are independent
        self.covM = cS
        # Precompute inverse covariance matrix
        self.covMI = np.linalg.inv(self.covM)
        #print('covMI = ', self.covMI)
        # Precompute determinant of covM
        self.detcov = np.linalg.det(self.covM)
        # Precompute the denominator for the kernel function
        self.denom = sqrt((2*pi)**self.D * self.detcov) * 2
        #print('self.N = ', self.N)
        #print('self.D = ', self.D)
        # Precompute range variables for efficency
        self.nrange = range(self.N)
        self.drange = range(self.D)
        self.mv = stats.multivariate_normal(mean=[0]*self.D, cov=self.covM)

    
    def K(self, x1, x2=0):
        """ Multivariate gaussian density kernel
            x1 and x2 are vectors of length D.
            returns a density vector of length D.
        """
        diffs = (x1 - x2).reshape(self.D, 1)  # Vector of differences
        #diffs = np.array([0] * self.D).reshape(self.D, 1)
        return e**(-.5 * (diffs.transpose() @ self.covMI @ diffs)[0][0]) / self.denom
        #return density


    def F(self, x, cumulative=False):
        """ x is a multivariate sample of length D
        """
        v = 0.0
        N = self.N
        if not cumulative:
            # PDF
            for p in self.X:
                p = np.array(p)
                v += self.K(p, x)
        else:
            # CDF
            mv = self.mv
            for p in self.X:
                c = mv.cdf(p-x)
                #if x >= p:
                v+= c
                #else:
                #    v+= 1-c
        result = v / N
        return result



