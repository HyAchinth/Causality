from math import *
import sys
sys.path.append('.')
sys.path.append('../')
from synth import getData
from Uprob import UPROB
import numpy as np
import math
from RKHSmod.rkhsMV import RKHS
from mpmath import *
import matplotlib.pyplot as plt


def pdf(x,u1=2,u2=-2,s=1):              # Actual PDF fuction for the sum of logistic(u1,s) and logistic(u2,s) distributions. 
                        
    return ((1/(4*s))*sech((x-u1)/(2*s))**2 + (1/(4*s))*sech((x-u2)/(2*s))**2)/2

def F(r,x=0,dat=None):             # Evaluation function, takes x2 and sigma of K() as parameter
        sum = 0.0
        for i in dat:
            val = r.K(i,x)
            sum += val
        return sum/len(dat)


path = '../models/Mdist.csv'
d = getData.DataReader(path)
data = d.read()
dat = data['X']
s=1.3
r = RKHS(data, ['X'],s=1.3)
max = 5
min = -5
num = 200
ker = []
xax = []
gdf = []


step = (max-min)/num
    #print(step)
for i in np.arange(min,max+step,step):
    ker.append(r.P(i))            
    xax.append(i)
    gdf.append(pdf(i))

# plt.plot(xax,ker,label = 'F(X),sigma='+str(sigma))
# plt.plot(xax,gdf,label = 'pdf(X),scale='+str(s), color= '#000000')
plt.plot(xax,ker,label = 'rkhsMV,sigma='+str(s))
plt.plot(xax,gdf,label = 'ideal()', color= '#000000')

plt.legend()
plt.show()





X = ['1','2','3','4','5','6','7']
print(X[-7:])
# vars = ['A','B','C']
# vals = [1,2,3]
# set = []
# for i in range(len(vars)):
#     x = (vars[i],vals[i])
#     set.append(x)

# print(set)
# k = 50
# filter_len = floor(len(X)*k*0.01)
# print(filter_len)
# filtered_vars = X[-filter_len:]
# filtered_vars2 = X[1:-filter_len]
# print(filtered_vars,filtered_vars2)

# path = '../models/nCondition.csv'
# d = getData.DataReader(path)
# data = d.read()

# vars = ['A5','B','C','D','E']
# target = 'A5'

# U =UPROB(data,includeVars=vars,k=60)

# #vals = [-1.6242107813616267,-4.882820970121719,2.034325259781949,-1.038089626559042,2.203530057472711]
# vals = [0.6540964483271905,3.242787775058589,1.8371005373984732,-1.7623330158108805,1.6103142808358637]
# #vals = [-1.2147678368913783,2.982144266622187,2.958393367399707,-0.6025040961079522,3.3676169505486553]
# vals2 = [3.242787775058589,1.8371005373984732,-1.7623330158108805,1.6103142808358637]

# vals3 = [0.1140964483271905,3.242787775058589,1.8371005373984732,-1.7623330158108805,1.6103142808358637]


# print("P=",U.condP(vals))
# print("P=",U.condP(vals3))
# print("E=",U.condE(target,vals2))
# print("E=",U.condE(target,vals2))
