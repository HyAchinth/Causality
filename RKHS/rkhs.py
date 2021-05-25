import math
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

def plot(sigma=1):                          # Very basic testing function to plot in a [-20, 20] range
    arr = array('f',[])                     # Vary sigma and observe the effect on the graph
    r = RKHS(X)
    for i in range(-20,20,1):
        arr.append(r.F(i,sigma))            # add F(i) to arr. -20<i<20
    plt.plot(arr)
    plt.show()

plot(0.2)
