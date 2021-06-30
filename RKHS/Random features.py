#Implementation of Random features and comparision with kernel methods
import math
import numpy as np
from scipy.special import erfinv
import sys
if '.' not in sys.path:
    sys.path.append('../')
import time
from synth import getData
import matplotlib.pyplot as plt
from RKHSmod.rkhs import RKHS
from Probability.Prob import ProbSpace
from Condtest2 import *

testMin = 5
testMax = 9

if __name__ == '__main__':
    path = '../models/Cprobdata.csv'
    d = getData.DataReader(path)
    data = d.read()        
    X = data['X']
    Y = data['Y']

    s = 0.2 #sigma value for RKHS

    r1 = RKHS(X,kparms=[s])
    r2 = RKHS(Y,kparms=[s])

    testPoints = []
    tp = testMin
    numTP = 200
    interval = (testMax - testMin) / numTP

    

