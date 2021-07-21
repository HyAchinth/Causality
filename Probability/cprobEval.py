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

tries = 10
datSize = 1000
condPoints = 10
lim = 3
# Arg format is <dims> <datSize> <tries>
if len(sys.argv) > 1:
    dims = int(sys.argv[1])
else:
    dims = 2
if len(sys.argv) > 2:
    datSize = int(sys.argv[2])
if len(sys.argv) > 3:
    tries = int(sys.argv[3])
if len(sys.argv) > 4:
    condPoints = int(sys.argv[4])
if len(sys.argv) > 5:
    lim = int(sys.argv[5])
print('dims, datSize, tries, condPts, lim = ', dims, datSize, tries, condPoints, lim)

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
    print('Test Limit = ', lim, 'standard deviations from mean')
    print('Dimensions = ', dims, '.  Conditionals = ', dims - 1)
    print('Number of points to test for each conditional = ', condPoints)
    N = prob.N
    evalpts = int(sqrt(N))
    print('JPROB points for mean evaluation = ', evalpts)
    vars = prob.fieldList
    cond = []
    # Get the conditional variables
    for i in range(len(vars)):
        var = vars[i]
        if var[0] != 'A':
            cond.append(var)
    # There is a target: 'A<dims>' for each conditional dimension.  So for 3D (2 conditionals),
    # we would use A3.
    target = 'A' + str(dims)

    amean = prob.E(target)
    astd = prob.distr(target).stDev()
    amin = amean - lim * astd
    arange = lim * astd - lim * -astd
    aincr = arange / (evalpts - 1)
    #print('A: mean, std, range, incr = ', amean, astd, arange, aincr)
    smoothness=1.0
    R1 = RKHS(prob.ds, delta=None, includeVars=[target] + cond[:dims-1], s=smoothness)
    R2 = RKHS(prob.ds, delta=None, includeVars=cond[:dims-1], s=smoothness)
    evaluations = 0
    start = time.time()
    results = []
    totalErr_jp = 0
    totalErr_ps = 0
    conds = len(cond)
    tps = []
    numTests = condPoints**(dims-1)
    evaluations = 0
    means = [prob.E(c) for c in cond]
    stds = [prob.distr(c).stDev() for c in cond]
    minvs = [means[i] - stds[i] * lim for i in range(len(means))]
    incrs = [(std * lim - std * -lim) / (condPoints-1) for std in stds]
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
            mod = condPoints**(dims - 1 - j - 1)
            #print('mod = ', mod, j)
            p = minv + int(i/mod)%condPoints * incr
            tp.append(p)
        tps.append(tp)

    #print('Testpoints = ', tps)
    tnum = 0
    ssTot = 0 # Total sum of squares for R2 computation
    cmprs = []
    jp_est = []
    ps_est = []
    # Generate the target values for comparison
    for t in tps:
        cmpr1 = tanh(t[0])
        #cmpr1 = abs(t[0])**1.1
        cmpr2 = sin(t[1]) if dims > 2 else 0
        #cmpr2 = -abs(t[1])**.9 if dims > 2 else 0
        cmpr3 = tanh(t[2]) if dims > 3 else 0
        cmpr4 = cos(t[3]) if dims > 4 else 0
        cmpr5 = t[4]**2 if dims > 5 else 0
        cmprL = [cmpr1, cmpr2, cmpr3, cmpr4, cmpr5]
        cmpr = sum(cmprL[:dims - 1])
        cmprs.append(cmpr)
    #print('Testing JPROB')
    jp_start = time.time()
    for t in tps:
        tnum += 1
        condVals = t
        evaluations += 1
        zval2 = R2.F(condVals)
        #print('zval2 = ', zval2)
        sumYP = 0
        sumP = 0
        if tnum%100 == 0:
            print('JPROB: tests ', tnum, '/', len(tps))
        for i in range(evalpts):
            evalpt = amin + aincr * i
            evaluations += 1
            zval1 = R1.F([evalpt]+condVals)
            if zval2 == 0:
                y_x = 0.0
            else:
                y_x = zval1 / zval2
            sumYP += y_x * evalpt
            sumP += y_x
            #cmpr = tanh(v2val)
            #cmpr = sin(v2val) + abs(v3val)**1.1
        mean = sumYP/sumP if sumP > 0 else 0
        jp_est.append(mean)
    jp_end = time.time()
    #print('Testing PS')
    ps_start = time.time()
    tnum = 0
    for t in tps:
        tnum += 1
        if tnum%100 == 0:
            print('DPROB: tests ', tnum, '/', len(tps))
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
    ps_end = time.time()
    totalErr_jp = 0.0
    totalErr_ps = 0.0
    results = []
    ysum = 0.0
    for i in range(len(cmprs)):
        t = tps[i]
        cmpr = cmprs[i]
        ysum += cmpr
        jp_e = jp_est[i]
        ps_e = ps_est[i]
        error2_jp = (cmpr-jp_e)**2
        error2_ps = (cmpr-ps_e)**2
        totalErr_jp += error2_jp
        totalErr_ps += error2_ps
        results.append((t, jp_e, ps_e, cmpr, error2_jp, error2_ps))

    for result in results:
        pass
        #print('tp, y|X, ps, ref, err2_jp, err2_ps = ', result[0], result[1], result[2], result[3], result[4], result[5])
    rmse_jp = sqrt(totalErr_jp) / len(tps)

    rmse_ps = sqrt(totalErr_ps) / len(tps)
    #print('RMSE PS = ', rmse_ps)
    # Calc R2 for each
    yavg = ysum / len(tps)
    ssTot = sum([(c - yavg)**2 for c in cmprs])
    R2_jp = 1 - totalErr_jp / ssTot
    R2_ps = 1 - totalErr_ps / ssTot
    #print('R2 JP =', R2_jp)
    #print('R2 PS =', R2_ps)
    print('JP:')
    #print('   RMSE = ', rmse_jp)
    print('   R2 =', R2_jp)
    jp_runtime = round((jp_end - jp_start) / evaluations * 1000, 5)
    ps_runtime = round(ps_end - ps_start, 5)
    #print('   Runtime = ', jp_runtime)
    #print('   Evaluations = ', evaluations)
    #print('   Avg Evaluaton = ', jp_runtime / evaluations)
    print('PS:')
    #print('   RMSE = ', rmse_ps)
    print('   R2 =', R2_ps)
    #print('   Runtime = ', round(ps_end - ps_start,3))
    jp_results.append(R2_jp)
    ps_results.append(R2_ps)
    jp_run.append(jp_runtime / N)
    ps_run.append(ps_runtime / N)
jp_avg = np.mean(jp_results)
ps_avg = np.mean(ps_results)
jp_min = np.min(jp_results)
ps_min = np.min(ps_results)
jp_max = np.max(jp_results)
ps_max = np.max(ps_results)
jp_std = np.std(jp_results)
ps_std = np.std(ps_results)
jp_runt = np.mean(jp_run)
ps_runt = np.mean(ps_run)
error = min(max(0, (ps_avg - jp_avg)/ ps_avg), 1)
print('dims, datSize, tries = ', dims, datSize, tries)
print('Average R2: JP, PS = ', jp_avg, ps_avg)
print('Min R2: JP, PS = ', jp_min, ps_min)
print('Max R2: JP, PS = ', jp_max, ps_max)
print('Std R2: JP, PS = ', jp_std, ps_std)
print('Runtimes: JP, PS = ', jp_runt, ps_runt)
print('NumTests = ', tries)
print('Error = ', error)