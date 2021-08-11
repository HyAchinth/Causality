"""
    Present a plot of the distributions for the given .py test file
    python3 Probabiity/probPlot.py <testfilepath>.py
    Data should previously have been generated using:
    python3 synth/synthDataGen.py <testfilepath>.py <numRecs>
"""
import sys
sys.path.append('.')
sys.path.append('../')
sys.path.append('./')
#import rv

import synth.getData as getData
import synth.synthDataGen as synthDataGen  
#import independence
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Probability.Prob import ProbSpace
import numpy as np
from matplotlib import cm
from RKHSmod.rkhsMV import RKHS
from math import log, tanh, sqrt, sin, cos
from numpy.random import *

from Uprob import UPROB

tries = 3
datSize = 10000
lim = 3
dims = 6    
K = 100
RF = 0.8
# Arg format is <dims> <datSize> <tries> <K>
if len(sys.argv) > 1:
    dims = int(sys.argv[1])
if len(sys.argv) > 2:
    datSize = int(sys.argv[2])
if len(sys.argv) > 3:
    tries = int(sys.argv[3])
if len(sys.argv) > 4:
    K = int(sys.argv[4])
if len(sys.argv) > 5:
    lim = int(sys.argv[5])
#print('dims, datSize, tries, lim = ', dims, datSize, tries, lim)
numTests = 200
print('Test Limit = ', lim, 'standard deviations from mean')
print('Dimensions = ', dims, '.  Conditionals = ', dims - 1)
print('Number of points to evaluate = ', numTests)

test = '../models/nCondition.py'

f = open(test, 'r')
exec(f.read(), globals())

print('Testing: ', test, '--', testDescript)

# For dat file, use the input file name with the .csv extension
tokens = test.split('.')
testFileRoot = str.join('.',tokens[:-1])
datFileName = testFileRoot + '.csv'
jp_results = []
up_results = []
ps_results = []
jp_run = []
up_run = []
ps_run = []
for i in range(tries):
    print('\nRun', i+1)
    sdg = synthDataGen.run(test, datSize)
    d = getData.DataReader(datFileName)
    data = d.read()
    prob = ProbSpace(data)

    N = prob.N
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

    smoothness=1.0
    R1 = RKHS(prob.ds, delta=None, includeVars=[target] + cond[:dims-1], s=smoothness)
    R2 = RKHS(prob.ds, delta=None, includeVars=cond[:dims-1], s=smoothness)

    U =UPROB(prob.ds,includeVars=[target]+cond[:dims-1],k=50,rangeFactor=RF)

    print("target=",target,"conds=",cond[:dims-1])




    evaluations = 0
    start = time.time()
    results = []
    totalErr_jp = 0
    totalErr_ps = 0
    conds = len(cond)
    tps = []
    evaluations = 0
    means = [prob.E(c) for c in cond]
    stds = [prob.distr(c).stDev() for c in cond]
    minvs = [means[i] - stds[i] * lim for i in range(len(means))]
    maxvs = [means[i] + stds[i] * lim for i in range(len(means))]

    # Generate the test points
    for i in range(numTests):
        tp = []
        for j in range(dims-1):
            v = uniform(minvs[j], maxvs[j])
            tp.append(v)
        tps.append(tp)      

    tnum = 0
    ssTot = 0 # Total sum of squares for R2 computation
    cmprs = []
    jp_est = []
    ps_est = []
    up_est = []
    # Generate the target values for comparison
    for t in tps:
        cmpr1 = tanh(t[0])
        cmpr2 = sin(t[1]) if dims > 2 else 0
        cmpr3 = tanh(t[2]) if dims > 3 else 0
        cmpr4 = cos(t[3]) if dims > 4 else 0
        cmpr5 = t[4]**2 if dims > 5 else 0
        cmprL = [cmpr1, cmpr2, cmpr3, cmpr4, cmpr5]
        cmpr = sum(cmprL[:dims - 1])
        cmprs.append(cmpr)
    
    #JPROB Evaluation
    jp_start = time.time()
    
    for t in tps:
        tnum += 1
        condVals = t
        evaluations += 1
        mean = R2.condE(target, condVals)
        jp_est.append(mean)
    jp_end = time.time()

    #UPROB Evaluation    
    up_start = time.time()
    tnum = 0
    for t in tps:
        tnum += 1
        condVals = t
        evaluations += 1
        mean = U.condE(target, condVals,K)
        up_est.append(mean)
    up_end = time.time()
    if U.R2 != None:
        print("Uprob vars:",U.R2.varNames)
    #ProbSpace Evaluation
    ps_start = time.time()
    tnum = 0
    for t in tps:
        tnum += 1
        try:
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
    totalErr_up = 0.0
    results = []
    ysum = 0.0
    for i in range(len(cmprs)):
        t = tps[i]
        cmpr = cmprs[i]
        ysum += cmpr
        jp_e = jp_est[i]
        up_e = up_est[i]
        ps_e = ps_est[i]
        error2_jp = (cmpr-jp_e)**2
        error2_up = (cmpr-up_e)**2
        error2_ps = (cmpr-ps_e)**2
        totalErr_jp += error2_jp
        totalErr_up += error2_up
        totalErr_ps += error2_ps
        results.append((t, jp_e, up_e, ps_e, cmpr, error2_jp, error2_up, error2_ps))

    for result in results:
        pass
        #print('tp, y|X, ps, ref, err2_jp, err2_ps = ', result[0], result[1], result[2], result[3], result[4], result[5])
    rmse_jp = sqrt(totalErr_jp) / len(tps)
    rmse_up = sqrt(totalErr_up) / len(tps)

    rmse_ps = sqrt(totalErr_ps) / len(tps)
    #print('RMSE PS = ', rmse_ps)
    # Calc R2 for each
    yavg = ysum / len(tps)
    ssTot = sum([(c - yavg)**2 for c in cmprs])
    R2_jp = 1 - totalErr_jp / ssTot
    R2_up = 1 - totalErr_up / ssTot
    R2_ps = 1 - totalErr_ps / ssTot
    #print('R2 JP =', R2_jp)
    #print('R2 PS =', R2_ps)
    print('JP:')
    #print('   RMSE = ', rmse_jp)
    print('   R2 =', R2_jp)

    print('UP K='+str(U.k)+'%:')
    print('   R2 =', R2_up)
    jp_runtime = round((jp_end - jp_start) / evaluations * 1000, 5)
    up_runtime = round((up_end - up_start) / evaluations * 1000, 5)
    ps_runtime = round(ps_end - ps_start, 5)
    #print('   Runtime = ', jp_runtime)
    #print('   Evaluations = ', evaluations)
    #print('   Avg Evaluaton = ', jp_runtime / evaluations)
    print('PS:')
    #print('   RMSE = ', rmse_ps)
    print('   R2 =', R2_ps)
    #print('   Runtime = ', round(ps_end - ps_start,3))
    jp_results.append(R2_jp)
    up_results.append(R2_up)
    ps_results.append(R2_ps)
    jp_run.append(jp_runtime / N)
    up_run.append(up_runtime / N)
    ps_run.append(ps_runtime / N)
jp_avg = np.mean(jp_results)
up_avg = np.mean(up_results)
ps_avg = np.mean(ps_results)
jp_min = np.min(jp_results)
up_min = np.min(up_results)
ps_min = np.min(ps_results)
jp_max = np.max(jp_results)
up_max = np.max(up_results)
ps_max = np.max(ps_results)
jp_std = np.std(jp_results)
up_std = np.std(up_results)
ps_std = np.std(ps_results)
jp_runt = np.mean(jp_run)
up_runt = np.mean(up_run)
ps_runt = np.mean(ps_run)
error = min(max(0, (ps_avg - jp_avg)/ ps_avg), 1)

print('dims, datSize, tries, K, RF = ', dims, datSize, tries, K, RF)
print('Average R2: JP, UP, PS = ', jp_avg, up_avg, ps_avg)
print('Min R2: JP, UP, PS = ', jp_min, up_min, ps_min)
print('Max R2: JP, UP, PS = ', jp_max, up_max, ps_max)
print('Std R2: JP, UP, PS = ', jp_std, up_std, ps_std)
print('Runtimes: JP, UP, PS = ', jp_runt, up_runt, ps_runt)
print('NumTests = ', tries)
print('Error = ', error)