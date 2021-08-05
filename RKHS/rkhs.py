import math
import numpy as np 
from mpmath import *
import matplotlib.pyplot as plt
from array import *
from getData import DataReader
import random                                          
                                          
Data = DataReader("../models/Mdist.csv")     #read the csv file() generated form synthDataGen.py 
Dict = Data.read()                          #Dict is of the form ['X': [0.232, -1.444 ..... ]] 
X = Dict['X']                               # X is a list of values X = [x1,x2,x3,......] 
#print(X)

class RKHS:
    def __init__(self, X):                  
        self.data = X
    
    def K(self,x1,x2=0,sigma = 1.0):       # Kernel Function 

        return ((1/(sigma * math.sqrt(2*math.pi))) * math.exp( -1 * (x1-x2)**2 / (2 * sigma**2) ))

    def F(self,x=0,sigma=1.0):             # Evaluation function, takes x2 and sigma of K() as parameter
        sum = 0.0
        for i in self.data:
            val = self.K(i,x,sigma)
            sum += val
        return sum/len(self.data)
    
    def ksaw(self,x1,x2=0, p=1.0):
        
        return max(0, (1 - (abs(x1 - x2) * p)) * p)

    def fsaw(self,x,p=1):        
        sum = 0.0
        for i in self.data:
            val = self.ksaw(x,i)
            sum += val
        return sum/len(self.data)

    def kpol(self,x1,x2=0,i=2,c=0):

        return (x1*x2 + c)**i

    def fpol(self,x,i=2):
        sum = 0.0
        for i in self.data:
            val = self.kpol(x,i)
            sum += val
        return sum/len(X)

def pdf(x,u1=2,u2=-2,s=1):              # Actual PDF fuction for the sum of logistic(u1,s) and logistic(u2,s) distributions. 
                        
    return ((1/(4*s))*sech((x-u1)/(2*s))**2 + (1/(4*s))*sech((x-u2)/(2*s))**2)/2



def sawtooth(x,p=2):
    
    return 2*( x/p - math.floor(0.5 + x/p))


def plot(sigma=1,min=-5,max=5,num=200,s=1):         #Test by plotting our F(X) vs pdf(X). X= [min,....,max] n(X)=num                          
    ker = []                                
    xax = []                                    #ker = [F(x1),....,F(xn)], xax = [x1,....,xn] {x-axis} , gdf =[pdf(x1),....,pdf(xn)]
    gdf = []

    Y = []
    for i in range(1,1001):
        Y.append(random.uniform(-1,1))

    r = RKHS(X)
    step = (max-min)/num
    #print(step)
    for i in np.arange(min,max+step,step):
        ker.append(r.F(i,sigma))            
        xax.append(i)
        gdf.append(pdf(i))

    plt.plot(xax,ker,label = 'F(X),sigma='+str(sigma))
    plt.plot(xax,gdf,label = 'pdf(X),scale='+str(s), color= '#000000')
    # plt.plot(xax,ker,label = 'F(X)')
    # plt.plot(xax,gdf,label = 'ideal()', color= '#000000')
    
    plt.legend()
    plt.show()

plot(0.6,-5,5,200,1)
