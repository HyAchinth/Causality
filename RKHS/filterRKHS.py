import sys

import numpy
sys.path.append('.')
sys.path.append('../')
from Probability.Prob import *
from synth import getData
from numpy import *
from RKHSmod.rkhs import RKHS
from Condtest2 import *

def tanh(x):
    return math.tanh(x)


path = '../models/Cprobdata2.csv'
d = getData.DataReader(path)
data = d.read()

P = ProbSpace(data)

#conditional X value
conditional = 0

FilterData, parentProb, finalQuery = P.filter([('X',conditional)])

Y = FilterData['Y']
Z = FilterData['Z']

s = 0.2 #sigma

r1 = RKHS(Y,kparms=[s])
r2 = RKHS(Z,kparms=[s])

testPoints = []
testMin = -10
testMax = 10
tp = testMin
numTP = 200
interval = (testMax - testMin) / numTP

Rdev = []
cond = []
ideal = []

for i in range(numTP + 1):
    testPoints.append(tp)   
    #ans = evalYx(tp,r1,r2,choice[j],e)
    ans = m(tp,r1,r2)
    if ans != 0:
        cond.append(ans)
    elif i== 0:
        cond.append(0)
    else:
        cond.append(cond[i-1])
    r = tanh(tp)
    ideal.append(r)            
    err = abs(r-ans)
    Rdev.append(err)           
    #print(tp,ans,r,err)
    tp += interval

print("Average Error:",sum(Rdev)/len(Rdev),"Max error:",max(Rdev))

labelstr = 'ProbSpace Z|Y,X='+str(conditional)

plt.plot(testPoints,ideal, label = 'Ideal Curve')
plt.plot(testPoints,cond, label = labelstr)    
plt.legend()
plt.show()


