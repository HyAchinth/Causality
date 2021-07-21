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
from synth import getData, synthDataGen
import independence
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Probability.Prob import ProbSpace
import numpy as np
from matplotlib import cm
from RKHSmod.rkhsMV import RKHS
from math import log, tanh, sqrt, sin, cos

tries = 1
datSize = 200
# Arg format is  <datSize>
dims = 2
smoothness = 1
cumulative = False
if len(sys.argv) > 1:
    datSize = int(sys.argv[1])
if len(sys.argv) > 2:
    smoothness = float(sys.argv[2])
if 'cum' in sys.argv:
    cumulative=true
print('dims, datSize, tries = ', dims, datSize, tries)

test = 'models/nCondition.py'

f = open(test, 'r')
exec(f.read(), globals())

print('Testing: ', test, '--', testDescript)

# For dat file, use the input file name with the .csv extension
tokens = test.split('.')
testFileRoot = str.join('.',tokens[:-1])
datFileName = testFileRoot + '.csv'
jp_results = []
ps_results = []
jp_run = []
ps_run = []
for i in range(tries):
    sdg = synthDataGen.run(test, datSize)
    d = getData.DataReader(datFileName)
    data = d.read()


    prob = ProbSpace(data)

    lim = 3  # Std's from the mean to test conditionals
    numPts = 30 # How many eval points for each conditional
    print('Test Limit = ', lim, 'standard deviations from mean')
    print('Dimensions = ', dims, '.  Conditionals = ', dims - 1)
    print('Number of points to test for each conditional = ', numPts)
    N = prob.N
    evalpts = int(sqrt(N)) # How many target points to sample for expected value:  E(Z | X=x. Y=y)
    print('JPROB points for mean evaluation = ', evalpts)
    vars = prob.fieldList
    cond = []
    # Get the conditional variables
    cond.append('B')


    target = 'A2'


    #print('A: mean, std, range, incr = ', amean, astd, arange, aincr)
    R1 = RKHS(prob.ds, delta=None, includeVars=[target] + ['B'], s=smoothness)
    R2 = RKHS(prob.ds, delta=None, includeVars=['B'], s=smoothness)
    evaluations = 0
    start = time.time()
    results = []
    totalErr_jp = 0
    totalErr_ps = 0
    vars = [target, 'B']
    tps = []
    numTests = numPts**(dims)
    evaluations = 0
    means = [prob.E(c) for c in vars]
    stds = [prob.distr(c).stDev() for c in vars]
    minvs = [means[i] - stds[i] * lim for i in range(dims)]
    incrs = [(std * lim - std * -lim) / (numPts-1) for std in stds]
    print('means, stds = ', means, stds, minvs, incrs)
    #print('cond = ', cond)
    #print('means', means)
    #print('stds = ', stds)
    #print('amean = ', amean)
    #print('astd = ', astd)
    # Generate the test points
    
    
    for i in range(numPts):
        for j in range(numPts):
            minv1 = minvs[0]
            incr1 = incrs[0]
            minv2 = minvs[1]
            incr2 = incrs[1]
            tp = [minv1 + i * incr1, minv2 + j * incr2]
            tps.append(tp)
    print('tps = ', tps[:10])
    # Traces for plot
    # 1 = Actual Function, 2 = JPROB, 3 = ProbSpace 
    xt1 = []
    xt2 = []
    xt3 = []
    yt1 = []
    yt2 = []
    yt3 = []
    zt1 = []
    zt2 = []
    zt3 = []

    #print('Testpoints = ', tps)
    tnum = 0
    ssTot = 0 # Total sum of squares for R2 computation
    cmprs = []
    jp_est = []
    ps_est = []


    #print('Testing JPROB')
    jp_start = time.time()
    for t in tps:
        tnum += 1
        condVals = [t[1]]
        evaluations += 1
        zval2 = R2.F(condVals, cumulative=cumulative)
        #print('zval2 = ', zval2)
        sumYP = 0
        sumP = 0
        if tnum%100 == 0:
            print('tests ', tnum, '/', len(tps))
        evalpt = t[0]
        evaluations += 1
        zval1 = R1.F([evalpt]+condVals, cumulative=cumulative)
        if zval2 == 0:
            y_x = 0.0
        else:
            y_x = zval1 / zval2
        print('zval1, zval2, y_x = ', zval1, zval2, y_x)
        jp_est.append(zval2)
        xt2.append(t[0])
        yt2.append(t[1])
        zt2.append(y_x)
    jp_end = time.time()
    #print('Testing PS')
    ps_start = time.time()
    for t in tps:
        print('t = ', t)
        #if tnum%100 == 0:
        #    print('tests ', tnum, '/', len(tps))

        #psy_x = prob.E(v1, [(v2, v2val - .1 * v2std, v2val + .1 * v2std) , (v3, v3val - .1 * v3std, v3val + .1 * v3std)])
        condspec = []
        aval = t[0]
        bval = t[1]
        condspec = ('B', bval)
        try:
            if cumulative:
                psy_x = prob.distr(target, condspec).cdf(aval)
            else:
                psy_x = prob.P((target,aval), condspec)
            print('psy_x = ', psy_x)
        except:
            psy_x = 0
        ps_est.append(psy_x)
        xt3.append(t[0])
        yt3.append(t[1])
        zt3.append(psy_x)
    ps_end = time.time()
    totalErr_jp = 0.0
    totalErr_ps = 0.0
    results = []
    ysum = 0.0


    for result in results:
        pass
        #print('tp, y|X, ps, ref, err2_jp, err2_ps = ', result[0], result[1], result[2], result[3], result[4], result[5])

    #print('RMSE PS = ', rmse_ps)
    # Calc R2 for each

    #print('R2 JP =', R2_jp)
    #print('R2 PS =', R2_ps)

fig = plt.figure()

#fig = plt.figure()
x = np.array(xt2)
y = np.array(yt2)
z = np.array(zt2)
print('x, y, z 2 =', len(x), len(y), len(z))
my_cmap = plt.get_cmap('hot')
ax = fig.add_subplot(121, projection='3d')
ax.plot_trisurf(x, y, z, cmap = my_cmap)

x = np.array(xt3)
y = np.array(yt3)
z = np.array(zt3)
print('x, y, z 2 =', len(x), len(y), len(z))
my_cmap = plt.get_cmap('hot')
ax = fig.add_subplot(122, projection='3d')
ax.plot_trisurf(x, y, z, cmap = my_cmap)

#ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5);
plt.show()