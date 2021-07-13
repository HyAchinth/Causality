import sys

import numpy
sys.path.append('.')
sys.path.append('../')
from Probability.Prob import *
from synth import getData
from numpy import *
from RKHSmod.rkhs import RKHS
from Condtest2 import *

def ref(x,choice):
    if choice == 1:
        return math.tanh(x) + math.sin(conditional)
    else:
        return math.sin(x) + math.tanh(conditional)


    


path = '../models/Cprobdata2.csv'
d = getData.DataReader(path)
data = d.read()

P = ProbSpace(data)

#conditional X value
conditional = 0

#choice X=1(tanh) , Y=0(sinh) (choose 1 to condition on X, 0 for Y)
choice = 1

if choice == 1:
    FilterData, parentProb, finalQuery = P.filter([('X',conditional)])
else:
    FilterData, parentProb, finalQuery = P.filter([('Y',conditional)])

X = FilterData['X']
Y = FilterData['Y']
Z = FilterData['Z']

filterlen = len(Z)

s = 0.2 #sigma

if choice == 0:
    r1 = RKHS(X,kparms=[s])
else:
    r1 = RKHS(Y,kparms=[s])

r2 = RKHS(Z,kparms=[s])

testPoints = []
testMin = -10
testMax = 10
tp = testMin
numTP = 200
interval = (testMax - testMin) / numTP

Rdev = []
Pdev = []
cond = []
ideal = []
Probpy = []
RSqerr = []

PSqerr = []
Sqmean = []

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
    if choice ==0:
        p = P.distr('Z', [('X', tp), ('Y', conditional)]).E()
    else:
        p = P.distr('Z', [('Y', tp), ('X', conditional)]).E()
    Probpy.append(p)        
    r = ref(tp,choice)
    ideal.append(r)            
    err = abs(r-ans)
    Rdev.append(err)
    err2 = abs(r-p)
    Pdev.append(err2)   
    tp += interval

ymean = sum(ideal)/len(ideal)

for i in range(numTP + 1):
    RSqerr.append(Rdev[i]**2)
    PSqerr.append(Pdev[i]**2)
    Sqmean.append((ideal[i]-ymean)**2)

R2rkhs = 1 - sum(RSqerr)/sum(Sqmean)
R2probpy = 1 - sum(PSqerr)/sum(Sqmean)


print("Filtered Datasize:",filterlen)
print("Filter-RKHS\tAverage Error:",sum(Rdev)/len(Rdev),"\tR2 error:",R2rkhs)
print("ProbSpace \tAverage Error:",sum(Pdev)/len(Pdev),"\tR2 error:",R2probpy)

labelstr = 'Filter-RKHS Z|X,Y='+str(conditional)

plt.plot(testPoints,ideal, label = 'Ideal Curve',color='#000000')
plt.plot(testPoints,Probpy, label = 'ProbSPace')
plt.plot(testPoints,cond, label = labelstr)    
plt.legend()
plt.show()


