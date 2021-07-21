"""
    Present a plot of the distributions for the given .py test file
    python3 Probabiity/probPlot.py <testfilepath>.py
    Data should previously have been generated using:
    python3 synth/synthDataGen.py <testfilepath>.py <numRecs>
"""
import sys
if '.' not in sys.path:
    sys.path.append('.')
import rv
from synth import getData
import independence
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Probability.Prob import ProbSpace
import numpy as np
from matplotlib import cm

args = sys.argv
if len(args) > 1:
    test = args[1]
if len(args) > 2:
    v1 = args[2].strip()
else:
    v1 = 'A'
if len(args) > 3:
    v2 = args[3].strip()
else:
    v2 = 'B'
if len(args) > 4:
    v3 = args[4].strip()
else:
    v3 = 'C'
f = open(test, 'r')
exec(f.read(), globals())

print('Testing: ', test, '--', testDescript)

# For dat file, use the input file name with the .csv extension
tokens = test.split('.')
testFileRoot = str.join('.',tokens[:-1])
datFileName = testFileRoot + '.csv'

d = getData.DataReader(datFileName)
data = d.read()


prob = ProbSpace(data)
traces = {}
traces['X'] = []
traces['Y'] = []
traces['Z'] = []
v1distr = prob.distr(v1)
v2distr = prob.distr(v2)
v3distr = prob.distr(v3)
v1mean = v1distr.E()
v2mean = v2distr.E()
v3mean = v3distr.E()
#ymin = targdistr.minVal()
#ymax = targdistr.maxVal()
#xmin = conddistr.minVal()
#xmax = conddistr.maxVal()
lim = 3
v1dat = prob.ds[v1]
v2dat = prob.ds[v2]
v3dat = prob.ds[v3]
numPts = len(v1dat)
xt = traces['X']
yt = traces['Y']
zt = traces['Z']
for i in range(numPts):
    #yval = ymin + i * y_incr        
    xt.append(v1dat[i])
    yt.append(v2dat[i])
    zt.append(v3dat[i])
fig = plt.figure()
x = np.array(xt)
y = np.array(yt)
z = np.array(zt)
my_cmap = plt.get_cmap('hot')
ax = fig.add_subplot(111, projection='3d')
#ax.plot_trisurf(x, y, z, cmap = my_cmap)
ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5);
plt.show()
