from   sklearn.gaussian_process import GaussianProcessRegressor
from   rffgpr import RFFGaussianProcessRegressor
import matplotlib.pyplot as plt
import sys
if '.' not in sys.path:
    sys.path.append('../')
import time
from synth import getData
import numpy as np
from Condtest2 import *
from RKHSmod.rkhs import RKHS


path = '../models/Cprobdata.csv'
d = getData.DataReader(path)
data = d.read()        
X1 = data['X']
Y1 = data['Y']

Datasize = len(X1)
Featsize = 100
Featsize2 = int(Datasize/10)

X = np.reshape(X1[:Datasize],(Datasize,1))
Y = np.reshape(Y1[:Datasize],(Datasize,1))

sigma = 0.2

#Ideal Curve calculation

testPoints = []
testMin = 5
testMax = 10   
tp = testMin
numTP = 200
interval = (testMax - testMin) / numTP
sq = []
probpy = []
rkhsEX = []
rkhsPX = []
Pdev=[]
RFFdev=[]
RKHSEXdev = []
RKHSPXdev = []

for i in range(numTP + 1):
    testPoints.append(tp)
    ideal = square(tp)
    sq.append(ideal)
    tp += interval

#Probpy Calculation

t1 = time.time()
ps = ProbSpace(data)
tp = testMin
for i in range(numTP + 1):
    p = ps.distr('Y', [('X', tp)]).E()        
    probpy.append(p)
    tp += interval
t2 = time.time()
Probtime = t2-t1

#RKHS Calculation - E(Y|X=x)

t1 = time.time()
r1 = RKHS(X1,kparms=[sigma])
r2 = RKHS(Y1,kparms=[sigma])
tp = testMin
for i in range(numTP + 1):
    r = m(tp,r1,r2)
    rkhsEX.append(r)
    tp += interval
t2 = time.time()
RKHSEXtime = t2-t1


#RFF Calculation

t1 = time.time()
R = RFFGaussianProcessRegressor(rff_dim=Featsize,sigma=0.2)
R.fit(X,Y)
Xt = np.reshape(testPoints,(numTP+1,1))
y_mean, y_cov = R.predict(Xt)
t2 = time.time()
RFFtime = t2-t1

#RFF Calculation-2
t1 = time.time()
R2 = RFFGaussianProcessRegressor(rff_dim=Featsize2,sigma=0.2)
R2.fit(X,Y)
Xt = np.reshape(testPoints,(numTP+1,1))
y_mean2, y_cov2 = R2.predict(Xt)
t2 = time.time()
RFFtime2 = t2-t1

RFFdev2 = []

#Error Calculaitons
for i in range(len(y_mean)):
    err = abs(y_mean[i]-sq[i])
    RFFdev.append(err)
    err = abs(y_mean2[i]-sq[i])
    RFFdev2.append(err)
    err1 = abs(probpy[i]-sq[i])
    Pdev.append(err1)    
    rerr = abs(rkhsEX[i]-sq[i])
    RKHSEXdev.append(rerr)
    

#Print Results
print('Probpy       time taken :',str(Probtime),'  \tAverage Error: ',sum(Pdev)/len(Pdev))
print('RKHS-E(X)    time taken :',str(RKHSEXtime),'  \tAverage Error: ',sum(RKHSEXdev)/len(RKHSEXdev))

print('RFF          time taken :',str(RFFtime),'  \tAverage Error: ',str(sum(RFFdev)[0]/len(RFFdev)))
print('RFF(N/10)    time taken :',str(RFFtime2),'  \tAverage Error: ',str(sum(RFFdev2)[0]/len(RFFdev2)))

#Plot Graph
RFFstring = 'RFF curve, num(F) = ' + str(Featsize)
RFFstring2 = 'RFF curve, num(F) = N/10 =' + str(Featsize2)

plt.plot(testPoints,y_mean, label = RFFstring)
plt.plot(testPoints,y_mean2, label = RFFstring2)
plt.plot(testPoints,rkhsEX, label = 'RKHS curve-E(X|Y=y)')

plt.plot(testPoints,probpy, label = 'Probpy curve')
plt.plot(testPoints,sq, label = 'Ideal curve',color='#000000', linewidth=2, linestyle='solid')
plt.legend()
plt.show()



