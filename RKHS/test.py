from   sklearn.gaussian_process import GaussianProcessRegressor
from   rffgpr import RFFGaussianProcessRegressor
import matplotlib.pyplot as plt
import sys
if '.' not in sys.path:
    sys.path.append('../')

from synth import getData
import numpy as np
from Condtest2 import *
from RKHSmod.rkhs import RKHS


path = '../models/Cprobdata.csv'
d = getData.DataReader(path)
data = d.read()        
X1 = data['X']
Y1 = data['Y']

sigma = 0.2

r1 = RKHS(X1,kparms=[sigma])
r2 = RKHS(Y1,kparms=[sigma])

ps = ProbSpace(data)

Datasize = 10000
Featsize = 100

X = np.reshape(X1[:Datasize],(Datasize,1))
Y = np.reshape(Y1[:Datasize],(Datasize,1))


print(np.shape(X))

R = RFFGaussianProcessRegressor(rff_dim=Featsize)

R.fit(X,Y)

testPoints = []
testMin = 0
testMax = 10
tp = testMin
numTP = 200
interval = (testMax - testMin) / numTP
sq = []
probpy = []
rkhs = []
Pdev=[]
RFFdev=[]
RKHSdev = []

for i in range(numTP + 1):
    testPoints.append(tp)
    ideal = square(tp)
    sq.append(ideal)
    p = ps.distr('Y', [('X', tp)]).E()        
    probpy.append(p)
    err1 = abs(ideal-p)
    Pdev.append(err1)
    r = m(tp,r1,r2)
    rkhs.append(r)
    rerr = abs(ideal-r)
    RKHSdev.append(rerr)
    tp += interval



Xt = np.reshape(testPoints,(numTP+1,1))
y_mean, y_cov = R.predict(Xt)

print("sizes=",len(y_mean),len(sq))

for i in range(len(y_mean)):
    err = abs(y_mean[i]-sq[i])
    RFFdev.append(err)

print('Probpy Average Error: ',sum(Pdev)/len(Pdev))
print('Probpy Average Error: ',sum(RKHSdev)/len(RKHSdev))
print('RFF Average Error: ',str(sum(RFFdev)/len(RFFdev)))


RFFstring = 'RFF curve, num(F) = ' + str(Featsize)

plt.plot(testPoints,y_mean, label = RFFstring)
plt.plot(testPoints,rkhs, label = 'RKHS curve')
plt.plot(testPoints,probpy, label = 'Probpy curve')
plt.plot(testPoints,sq, label = 'Ideal curve')
plt.legend()
plt.show()



