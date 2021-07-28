"""
    Present a plot of the distributions for the given .py test file
    python3 Probabiity/probPlot.py <testfilepath>.py
    Data should previously have been generated using:
    python3 synth/synthDataGen.py <testfilepath>.py <numRecs>
"""
import sys
if '.' not in sys.path:
    sys.path.append('.')
    sys.path.append('../')
#import rv
from synth import getData, synthDataGen
#import independence
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Probability.Prob import ProbSpace
import numpy as np
from matplotlib import cm
from RKHSmod.rkhsMV import RKHS

from RKHS.Condtest2 import m
from math import log, tanh, sqrt, sin, cos

tries = 5
datSize = 200
# Arg format is  <datSize>
dims = 3
smoothness = 1
cumulative = False
if len(sys.argv) > 1:
    datSize = int(sys.argv[1])
if len(sys.argv) > 2:
    smoothness = float(sys.argv[2])
print('dims, datSize, tries = ', dims, datSize, tries)

test = '../models/doubleCondition.py'

f = open(test, 'r')
exec(f.read(), globals())

print('Testing: ', test, '--', testDescript)

# For dat file, use the input file name with the .csv extension
tokens = test.split('.')
testFileRoot = str.join('.',tokens[:-1])
datFileName = testFileRoot + '.csv'
jp_results = []
ps_results = []
fp_results = []
jp_run = []
ps_run = []
fp_run = []
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
    for i in range(len(vars)):
        var = vars[i]
        if var[0] != 'A':
            cond.append(var)

    target = 'A'

    amean = prob.E(target)
    astd = prob.distr(target).stDev()
    amin = amean - lim * astd
    arange = lim * astd - lim * -astd
    aincr = arange / (evalpts - 1)
    #print('A: mean, std, range, incr = ', amean, astd, arange, aincr)
    R1 = RKHS(prob.ds, delta=None, includeVars=[target] + cond[:dims-1], s=smoothness)
    R2 = RKHS(prob.ds, delta=None, includeVars=cond[:dims-1], s=smoothness)
    evaluations = 0
    start = time.time()
    results = []
    totalErr_jp = 0
    totalErr_ps = 0
    conds = len(cond)
    tps = []
    numTests = numPts**(dims-1)
    evaluations = 0
    means = [prob.E(c) for c in cond]
    stds = [prob.distr(c).stDev() for c in cond]
    minvs = [means[i] - stds[i] * lim for i in range(len(means))]
    incrs = [(std * lim - std * -lim) / (numPts-1) for std in stds]
    #print('cond = ', cond)
    #print('means', means)
    #print('stds = ', stds)
    #print('amean = ', amean)
    #print('astd = ', astd)
    # Generate the test points
    for i in range(numTests):
        tp = []
        for j in range(dims-1):
            minv = minvs[j]
            incr = incrs[j]
            mod = numPts**(dims - 1 - j - 1)
            #print('mod = ', mod, j)
            p = minv + int(i/mod)%numPts * incr
            tp.append(p)
        tps.append(tp)
    
    # Traces for plot
    # 1 = Actual Function, 2 = JPROB, 3 = ProbSpace , 4 = FPROB
    xt1 = []
    xt2 = []
    xt3 = []
    xt4 = []
    yt1 = []
    yt2 = []
    yt3 = []
    yt4 = []
    zt1 = []
    zt2 = []
    zt3 = []
    zt4 = []

    #print('Testpoints = ', tps)
    tnum = 0
    ssTot = 0 # Total sum of squares for R2 computation
    cmprs = []
    jp_est = []
    ps_est = []
    fp_est = []
    # Generate the target values for comparison
    for t in tps:
        cmpr = tanh(t[0]) + sin(t[1])
        cmprs.append(cmpr)
        xt1.append(t[0])
        yt1.append(t[1])
        zt1.append(cmpr)

    #print('Testing JPROB')
    jp_start = time.time()
    for t in tps:
        tnum += 1
        condVals = t
        evaluations += 1
        mean = R2.condE(target, condVals)
        #print('zval2 = ', zval2)
        # sumYP = 0
        # sumP = 0
        # if tnum%100 == 0:
        #     print('tests ', tnum, '/', len(tps))
        # for i in range(evalpts):
        #     evalpt = amin + aincr * i
        #     evaluations += 1
        #     zval1 = R1.F([evalpt]+condVals, cumulative=cumulative)
        #     if zval2 == 0:
        #         y_x = 0.0
        #     else:
        #         y_x = zval1 / zval2
        #     sumYP += y_x * evalpt
        #     sumP += y_x
        #     #cmpr = tanh(v2val)
        #     #cmpr = sin(v2val) + abs(v3val)**1.1
        # mean = sumYP/sumP if sumP > 0 else 0
        jp_est.append(mean)
        xt2.append(t[0])
        yt2.append(t[1])
        zt2.append(mean)
    jp_end = time.time()
    #print('Testing PS')
    ps_start = time.time()
    for t in tps:
        #if tnum%100 == 0:
        #    print('tests ', tnum, '/', len(tps))
        try:
            #psy_x = prob.E(v1, [(v2, v2val - .1 * v2std, v2val + .1 * v2std) , (v3, v3val - .1 * v3std, v3val + .1 * v3std)])
            condspec = []
            for c in range(dims-1):
                condVar = cond[c]
                val = t[c]
                spec = (condVar, val)
                condspec.append(spec)
            psy_x = prob.E(target, condspec)
        except:
            psy_x = 0
        ps_est.append(psy_x)
        xt3.append(t[0])
        yt3.append(t[1])
        zt3.append(psy_x)
    ps_end = time.time()

    print('Testing FProb')
    fp_start = time.time()
    for t in tps:
        FilterData, parentProb, finalQuery = prob.filter([('B',t[0])])
        A = FilterData['A']
        B = FilterData['B']
        C = FilterData['C']

        filterlen = len(C)
        s = 0.2 #sigma       

        r1 = RKHS(FilterData,includeVars=['C'])
        r2 = RKHS(FilterData,includeVars=['A'])
        
        pred = m(t[1],r1,r2)     
        #print("A, B, C=",float(pred),t[0],t[1])
        fp_est.append(float(pred))
        xt4.append(t[0])
        yt4.append(t[1])
        zt4.append(float(pred))
    fp_end = time.time()

    
    totalErr_jp = 0.0
    totalErr_ps = 0.0
    totalErr_fp = 0.0
    results = []
    ysum = 0.0
    for i in range(len(cmprs)):
        t = tps[i]
        cmpr = cmprs[i]
        ysum += cmpr
        jp_e = jp_est[i]
        ps_e = ps_est[i]
        fp_e = fp_est[i]
        error2_jp = (cmpr-jp_e)**2
        error2_ps = (cmpr-ps_e)**2
        error2_fp = (cmpr-fp_e)**2
        totalErr_jp += error2_jp
        totalErr_ps += error2_ps
        totalErr_fp += error2_fp
        results.append((t, jp_e, ps_e, fp_e, cmpr, error2_jp, error2_ps, error2_fp))

    for result in results:
        pass
        #print('tp, y|X, ps, ref, err2_jp, err2_ps = ', result[0], result[1], result[2], result[3], result[4], result[5])
    rmse_jp = sqrt(totalErr_jp) / len(tps)

    rmse_ps = sqrt(totalErr_ps) / len(tps)

    rmse_fp = sqrt(totalErr_fp) / len(tps)
    #print('RMSE PS = ', rmse_ps)
    # Calc R2 for each
    yavg = ysum / len(tps)
    ssTot = sum([(c - yavg)**2 for c in cmprs])
    R2_jp = 1 - totalErr_jp / ssTot
    R2_ps = 1 - totalErr_ps / ssTot
    R2_fp = 1 - totalErr_fp / ssTot
    print(totalErr_fp)
    #print('R2 JP =', R2_jp)
    #print('R2 PS =', R2_ps)
    print('JP:')
    #print('   RMSE = ', rmse_jp)
    print('   R2 =', R2_jp)
    jp_runtime = round((jp_end - jp_start) / evaluations * 1000, 5)
    ps_runtime = round(ps_end - ps_start, 5)
    fp_runtime = round(fp_end - fp_start, 5)
    #print('   Runtime = ', jp_runtime)
    #print('   Evaluations = ', evaluations)
    #print('   Avg Evaluaton = ', jp_runtime / evaluations)
    print('PS:')
    #print('   RMSE = ', rmse_ps)
    print('   R2 =', R2_ps)
    #print('   Runtime = ', round(ps_end - ps_start,3))
    print('FP:')
    print('    R2 =', R2_fp)

    jp_results.append(R2_jp)
    ps_results.append(R2_ps)
    fp_results.append(R2_fp)
    jp_run.append(jp_runtime / N)
    ps_run.append(ps_runtime / N)
    fp_run.append(fp_runtime / N)
jp_avg = np.mean(jp_results)
ps_avg = np.mean(ps_results)
fp_avg = np.mean(fp_results)
jp_min = np.min(jp_results)
ps_min = np.min(ps_results)
fp_min = np.min(fp_results)
jp_max = np.max(jp_results)
ps_max = np.max(ps_results)
fp_max = np.max(fp_results)
jp_std = np.std(jp_results)
ps_std = np.std(ps_results)
fp_std = np.std(fp_results)
jp_runt = np.mean(jp_run)
ps_runt = np.mean(ps_run)
fp_runt = np.mean(fp_run)
error = min(max(0, (ps_avg - jp_avg)/ ps_avg), 1)
print('dims, datSize, tries = ', dims, datSize, tries)
print('Average R2: JP, PS, FP = ', jp_avg, ps_avg, fp_avg)
print('Min R2: JP, PS, FP = ', jp_min, ps_min, fp_min)
print('Max R2: JP, PS, FP = ', jp_max, ps_max, fp_max)
print('Std R2: JP, PS, FP = ', jp_std, ps_std, fp_std)
print('Runtimes: JP, PS, FP = ', jp_runt, ps_runt, fp_runt)
print('NumTests = ', tries)
print('Error = ', error)
fig = plt.figure()
x = np.array(xt1)
y = np.array(yt1)
z = np.array(zt1)


my_cmap = plt.get_cmap('hot')
ax = fig.add_subplot(221, projection='3d')
ax.view_init(elev = 40, azim = -75)
ax.plot_trisurf(x, y, z, cmap = my_cmap)

#fig = plt.figure()
x = np.array(xt2)
y = np.array(yt2)
z = np.array(zt2)

my_cmap = plt.get_cmap('hot')
ax = fig.add_subplot(222, projection='3d')
ax.view_init(elev = 40, azim = -75)
ax.plot_trisurf(x, y, z, cmap = my_cmap)

x = np.array(xt3)
y = np.array(yt3)
z = np.array(zt3)

my_cmap = plt.get_cmap('hot')
ax = fig.add_subplot(223, projection='3d')
ax.view_init(elev = 40, azim = -75)
ax.plot_trisurf(x, y, z, cmap = my_cmap)

x = np.array(xt4)
y = np.array(yt4)
z = np.array(zt4)
my_cmap = plt.get_cmap('hot')
ax = fig.add_subplot(224, projection='3d')
ax.view_init(elev = 40, azim = -75)
ax.plot_trisurf(x, y, z, cmap = my_cmap)

#ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5);
plt.show()