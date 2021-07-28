""""UPROB is an universal probability module that encompasses all existing methods, JPROB and FPROB. The main funcitonality is to have a 'k'
    parameter such that out of N variables/dimensions, K% of variables are filtered and JPROB is applied to the rest. Important comparisions:
    UPROB(0) == JPROB
    UPROB(N-1) == FPROB1 (filter all but 1 conditionals and return 2D RKHS result)
    UPROB(N) == FRPOB2 (filter all variables and return unidimensional RKHS result)

"""
import sys
sys.path.append('.')
sys.path.append('../')

from RKHSmod.rkhsMV import RKHS
from Probability.Prob import ProbSpace
from math import floor

class UPROB:
    def __init__(self,data, includeVars = None, delta = 3, s=1.0,k=50):
        assert type(data) == type({}), "Error -- Data must be in the form of a dictionary varName -> [val1, val2, ... , valN]"
        
        self.data = data                    # data in the form of a dictionary
        self.delta = delta                  # deviation of test points
        self.includeVars = includeVars      # variables to be included from data
        self.s = s                          # smoothness factor
        self.k = k                          # % of variables to be filtered
        self.R1 = None                      # cached rkhsMV class
        self.R2 = None                      # cached rkhsMV class2


    def CondP(self, Vals,K = None,minPoints = None,maxPoints = None):
        #Vals is a list of (x1,x2....xn) such that P(X1=x1|X2=x2.....), same UI as rkhsmv
        if(K != None):
            self.k = K
        filter_len = floor(len(self.includeVars)*self.k*0.01)
        filter_vars = self.includeVars[-filter_len:]
        filter_vals = Vals[-filter_len:]
        
        if(self.R1==None or self.R1.varNames != self.includeVars[:-filter_len]):
            P = ProbSpace(self.data)
            filter_data = []
            for i in range(filter_len):
                x = (filter_vars[i],filter_vals[i])
                filter_data.append(x)                    
            FilterData, parentProb, finalQuery = P.filter(filter_data,minPoints,maxPoints)
            self.R1 = RKHS(FilterData,includeVars=self.includeVars[:-filter_len],delta=self.delta,s=self.s)

        if(self.R2==None or self.R2.varNames != self.includeVars[1:-filter_len]):
            P = ProbSpace(self.data)
            filter_data = []
            for i in range(filter_len):
                x = (filter_vars[i],filter_vals[i])
                filter_data.append(x)                    
            FilterData, parentProb, finalQuery = P.filter(filter_data,minPoints,maxPoints)
            self.R2 = RKHS(FilterData,includeVars=self.includeVars[1:-filter_len],delta=self.delta,s=self.s)

        p1 = self.R1.P(Vals[:-filter_len])
        p2 = self.R2.P(Vals[1:-filter_len])
        if p2>0:
            return p2/p1
        else:
            return None

    def CondE(self, Vals,K = None,minPoints = None,maxPoints = None):
        #Vals is a list of (x1,x2....xn) such that E(Y|X1=x1,X2=x2.....), same UI as rkhsmv
        if(K != None):
            self.k = K
        filter_len = floor((len(self.includeVars)+1)*self.k*0.01)
        filter_vars = self.includeVars[-filter_len:]
        filter_vals = Vals[-filter_len:]

        if(self.R2==None or self.R2.varNames != self.includeVars[:-filter_len]):
            P = ProbSpace(self.data)
            filter_data = []
            for i in range(filter_len):
                x = (filter_vars[i],filter_vals[i])
                filter_data.append(x)                    
            FilterData, parentProb, finalQuery = P.filter(filter_data,minPoints,maxPoints)
            self.R2 = RKHS(FilterData,includeVars=self.includeVars[:-filter_len],delta=self.delta,s=self.s)
        
        return self.R2.condE(self.includeVars[0],Vals[:-filter_len])
    
        



            






            



