import math
import numpy as np 
from mpmath import *
import matplotlib.pyplot as plt
from array import *
from getData import DataReader                                          
                                          
Data = DataReader("./models/Mdist.csv")     #read the csv file() generated form synthDataGen.py 
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


def pdf(x,u1=2,u2=-2,s=1):              # Actual PDF fuction for the sum of logistic(u1,s) and logistic(u2,s) distributions. 
                        
    return ((1/(4*s))*sech((x-u1)/(2*s)) + (1/(4*s))*sech((x-u2)/(2*s)))/2


def plot(sigma=1,min=-5,max=5,num=200):         #Test by plotting our F(X) vs pdf(X). X= [min,....,max] n(X)=num                          
    ker = []                                
    xax = []                                    #ker = [F(x1),....,F(xn)], xax = [x1,....,xn] {x-axis} , gdf =[gdf(x1),....,gdf(xn)]
    gdf = []
    r = RKHS(X)
    step = (max-min)/num
    print(step)
    for i in np.arange(min,max+step,step):
        ker.append(r.F(i,sigma))            
        xax.append(i)
        gdf.append(pdf(i))
    #print(xax)
    print(len(xax))
    plt.plot(xax,ker,label = 'F(X),sigma='+str(sigma))
    plt.plot(xax,gdf,label = 'pdf(X)', color= '#000000')
    plt.legend()
    plt.show()

plot(1.2,-5,5)
