from math import *
import sys
sys.path.append('.')
sys.path.append('../')
from synth import getData
from Uprob import UPROB

X = ['1','2','3','4','5','6','7']
print(X[:-1])
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

path = './models/nCondition.csv'
d = getData.DataReader(path)
data = d.read()

vars = ['A5','B','C','D','E']

U =UPROB(data,includeVars=vars,k=60)

#vals = [-1.6242107813616267,-4.882820970121719,2.034325259781949,-1.038089626559042,2.203530057472711]
vals = [0.6540964483271905,3.242787775058589,1.8371005373984732,-1.7623330158108805,1.6103142808358637]
#vals = [-1.2147678368913783,2.982144266622187,2.958393367399707,-0.6025040961079522,3.3676169505486553]
vals2 = [3.242787775058589,1.8371005373984732,-1.7623330158108805,1.6103142808358637]

print("P=",U.CondP(vals))
print("E=",U.CondE(vals2))
