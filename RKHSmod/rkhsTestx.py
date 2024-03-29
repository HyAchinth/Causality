from math import e, sqrt, pi, erf, erfc, log
from scipy.special import erfinv
import sys
if '.' not in sys.path:
    sys.path.append('../')
import time
from synth import getData
import matplotlib.pyplot as plt
from RKHSmod.rkhs import RKHS

def ksaw(x1,x2=0, kparms=[]):
    p = kparms[0]    
    return max(0, (1 - (abs(x1 - x2) * p)) * p)

def fsaw(x,X, kparms=[]):
    
    sum = 0.0
    for i in X:
        val = ksaw(x,i,kparms)
        sum += val
    return sum/len(X)

def kQuant(x1, x2=0, kparms=[]):
    sigma = kparms[0]
    return abs(x1-x2) + sigma * sqrt(2) * erfinv(2*.5-1)

def kcdf(x1, x2=0, kparms=[]):
    # CDF Kernel
    sigma = kparms[0]
    return (1 + erf(abs(x2-x1)/sigma*sqrt(2))) / 2

def Fcdf(x,X, kparms):
    sigma = kparms[0]
    delta = kparms[1]
    cum = 0.0
    for p in X:
        if p < x:
            if delta is not None and (x-p) > delta * sigma:
                cum += 1
                continue
            cum += kcdf(p, x, kparms)
        else:
            if delta is not None and (p-x) > delta * sigma:
                continue
            cum += 1 - kcdf(p, x, kparms)
    return cum / len(X)

def Fquant(x,X, kparms):
    sigma = kparms[0]
    delta = kparms[1]
    cum = 0.0
    for p in X:
        if x > p:
            if delta is not None and (x-p) > delta * sigma:
                cum += 1
                continue
            cum += kQuant(p, x, kparms)
        else:
            if delta is not None and (p-x) > delta * sigma:
                continue
            cum += -kQuant(p, x, kparms)
    return cum / len(X)

def logistic(x, mu, s):
    return e**(-(x-mu)/s) / (s * ((1 + e**(-(x-mu)/s)))**2)

def logisticCDF(x, mu, s):
    return 1 / (1 + e**(-(x - mu)/s))

def testF(x):
    result = (logistic(x, -2, .5) + logistic(x, 2, .5)  + logistic(x, -.5, .3)) / 3
    return result

def testFCDF(x):
    result = (logisticCDF(x, -2, .5) + logisticCDF(x, 2, .5) + logisticCDF(x, -.5, .3)) / 3
    return result


if __name__ == '__main__':
    args = sys.argv
    if (len(args) > 1):
        test = args[1]
    else:
        test = '../models/rkhsTest.csv'
    d = getData.DataReader(test)
    data = d.read()
    X = data['X']
    expmean = sum(X) / len(X)
    traces = []
    dataSizes = [10, 50, 100, 1000, 10000, 100000]
    errs = {}
    maxDeviations = {}
    avgDeviations = {}
    testPoints = []
    testMin = -5
    testMax = 5
    tp = testMin
    numTP = 200
    interval = (testMax - testMin) / numTP
    tfs = []
    ctfs = []
    means = {}
    # Generate a uniform range of test points.
    # While at it, generate our expected pdf and cdf
    for i in range(numTP + 1):
        testPoints.append(tp)
        tfp = testF(tp)
        tfs.append(tfp)
        ctfp = testFCDF(tp)
        ctfs.append(ctfp)
        tp += interval
    delta = None
    start = time.time()
    evals = 0
    for size in dataSizes:
        # Choose a reasonable sigma based on data size.
        #sigma = 1 / log(size, 4)
        r1 = RKHS(X[:size],k = ksaw, f = fsaw, kparms=[2, delta])
        #r2 = RKHS(X[:size], k=kcdf, f=Fcdf, kparms=[sigma, delta])
        fs = []  # The results using a pdf kernel
        fsc = [] # Results using a cdf kernel
        totalErr = 0
        deviations = []
        for i in range(len(testPoints)):
            p = testPoints[i]
            
            fp = r1.F(p)
            evals += 1
            fs.append(fp)
            fc = 0
            fc = r2.F(p)
            evals += 1
            fsc.append(fc)
            tfp = tfs[i]
            ctfp = ctfs[i]
            err = abs(fp - tfp)
            totalErr += err
            #print('fpc, ctfp = ', fpc, ctfp)
            #deviation = abs(fpc - ctfp)
            #deviations.append(deviation)
        #maxDeviations[size] = max(deviations)
        #avgDeviations[size] = sum(deviations) / numTP
        errs[size] = totalErr / numTP
        traces.append(fs) # pdf trace
        #traces.append(fsc) # cdf trace
        #r3 = RKHS(X[:size], f=Fquant, k=kQuant, kparms=[sigma, None])
        #mean = r3.F(.5)
        #means[size] = mean
    # print('Average Errors = ', errs)
    # print('Average Deviation = ', avgDeviations)
    # print('Maximum Deviation = ', maxDeviations)
    # print('Means = ', means, expmean)
    end = time.time()
    duration = end - start
    print('elapsed = ', duration)
    print('evaluations = ', evals)
    print('elapsed per eval = ', duration / evals)
    for t in range(len(traces)):
        fs = traces[t]
        size = dataSizes[int(t/2)] # traces are alternately pdf and cdf
        if t%2 == 0:
            continue
            label = 'pdf(X)-size=' + str(size)
            linestyle = 'solid'
        
        plt.plot(testPoints, fs, label=label, linestyle=linestyle)
    #plt.plot(testPoints, tfs, label='testPDF(x)', color='#000000', linewidth=3, linestyle='solid')
    plt.plot(testPoints, ctfs, label='testCDF(x)', color='#000000', linewidth=3, linestyle='dotted')
    plt.legend()
    plt.show()

