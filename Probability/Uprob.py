""""UPROB is an universal probability module that encompasses all existing methods, JPROB and FPROB. The main funcitonality is to have a 'k'
    parameter such that out of N variables/dimensions, K% of variables are filtered and JPROB is applied to the rest. Important comparisions:
    UPROB(0) == JPROB
    UPROB(99) == FPROB1 (filter all but 1 conditionals and return 2D RKHS result)
    UPROB(100) == FRPOB2 (filter all variables and return unidimensional RKHS result)

"""
import sys
sys.path.append('.')
sys.path.append('../')

from RKHSmod.rkhsMV import RKHS
from Probability.Prob import ProbSpace
from math import ceil, floor

class UPROB:
    def __init__(self,data, includeVars = None, delta = 3, s=1.0,k=50,rangeFactor = None):
        assert type(data) == type({}), "Error -- Data must be in the form of a dictionary varName -> [val1, val2, ... , valN]"
        
        self.data = data                    # data in the form of a dictionary
        self.delta = delta                  # deviation of test points
        self.includeVars = includeVars      # variables to be included from data
        self.s = s                          # smoothness factor
        self.k = k                          # % of variables to be filtered
        self.R1 = None                      # cached rkhsMV class
        self.R2 = None                      # cached rkhsMV class2
        self.r1filters = None               # cached filter values in R1
        self.r2filters = None               # cached filter values in R2
        self.minPoints = None               # min and max points for filteration
        self.maxPoints = None
        if includeVars is None:
            self.varNames = list(data.keys())
            self.D = len(self.varNames)
        else:
            self.varNames = includeVars
            self.D = len(self.varNames)
        self.N = len(data[self.varNames[0]])
        self.rangeFactor = rangeFactor



    def condP(self, Vals,K = None):
        #Vals is a list of (x1,x2....xn) such that P(X1=x1|X2=x2.....), same UI as rkhsmv
        if(K != None):
            self.k = K
        filter_len = floor((len(self.includeVars)-1)*self.k*0.01)
        dims = len(Vals)
        if(self.rangeFactor==0):
            self.rangeFactor = 0.05
        if(filter_len !=0):
            filter_vars = self.includeVars[-filter_len:]
            filter_vals = Vals[-filter_len:]
            include_vars = self.includeVars[:-filter_len]            
            self.minPoints = ceil(self.N**((dims-filter_len)/dims)*self.rangeFactor) + 20
            self.maxPoints = ceil(self.N**((dims-filter_len)/dims)/self.rangeFactor)
            print("minpoints,maxpoints=",self.minPoints,self.maxPoints)

        else:
            filter_vars = []
            filter_vals = []
            include_vars = self.includeVars
        
        # print("filter vars:",filter_vars)
        # print("include vars:",self.includeVars[:-filter_len])
        # print(self.includeVars)
        
        #Calculating R1 
        if(filter_len == (len(self.includeVars)-1)):
            P = ProbSpace(self.data)
            filter_data = []
            for i in range(filter_len):
                x = (filter_vars[i],filter_vals[i])
                filter_data.append(x)                    
            FilterData, parentProb, finalQuery = P.filter(filter_data,self.minPoints,self.maxPoints)
            # print("filter len",filter_len)
            print("filtered datapoints:",len(FilterData['B']))
            print("include vars:",self.includeVars[:-filter_len])
            self.R1 = RKHS(FilterData,includeVars=self.includeVars[:1],delta=self.delta,s=self.s)
            
            return self.R1.P(Vals[0])
            self.r1filters = filter_vals
        
        
        
        elif(filter_len != 0 and self.r1filters != filter_vals):
            P = ProbSpace(self.data)
            filter_data = []
            for i in range(filter_len):
                x = (filter_vars[i],filter_vals[i])
                filter_data.append(x)                    
            FilterData, parentProb, finalQuery = P.filter(filter_data,self.minPoints,self.maxPoints)
            # print("filter len",filter_len)
            # print("filtered datapoints:",len(FilterData['B']))
            # print("include vars:",self.includeVars[:-filter_len])
            self.R1 = RKHS(FilterData,includeVars=self.includeVars[:-filter_len],delta=self.delta,s=self.s)
            self.r1filters = filter_vals

        elif(self.R1==None):            
            self.R1 = RKHS(self.data,includeVars=self.includeVars,delta=self.delta,s=self.s)

        elif(self.R1.varNames != include_vars):            
            self.R1 = RKHS(self.data,includeVars=self.includeVars,delta=self.delta,s=self.s)

        if(filter_len != 0):                
            p = self.R1.condP(Vals[:-filter_len])
            if p>0:
                return p
            else:
                return None
        else:
            p = self.R1.condP(Vals)
            if p>0:
                return p
            else:
                return None

    def condE(self,target, Vals,K = None):
        #Vals is a list of (x1,x2....xn) such that E(Y|X1=x1,X2=x2.....), same UI as rkhsmv
        if(K != None):
            self.k = K
        filter_len = floor((len(self.includeVars)-1)*self.k*0.01)
        dims = len(Vals) + 1
        if(self.rangeFactor == None):
            self.rangeFactor = 0.05
        
        
        if(filter_len !=0):
            filter_vars = self.includeVars[-filter_len:]
            filter_vals = Vals[-filter_len:]
            include_vars = self.includeVars[1:-filter_len]
            self.minPoints = ceil(self.N**((dims-filter_len)/dims)*self.rangeFactor)
            self.maxPoints = ceil(self.N**((dims-filter_len)/dims)/self.rangeFactor)
            print("minpoints,maxpoints=",self.minPoints,self.maxPoints)

        else:
            filter_vars = []
            filter_vals = []       
            include_vars = self.includeVars
        
        #print("filter vars:",filter_vars)
        #print("include vars:",self.includeVars[:-filter_len])
        #print("self:",self.R2.varNames,"cond",self.includeVars[:-filter_len])
                
        
        if(filter_len == (len(self.includeVars)-1) ):
            self.minPoints += 20
            P = ProbSpace(self.data)
            filter_vars = self.includeVars[1:]
            filter_vals = Vals            
            filter_data = []
            for i in range(filter_len):
                x = (filter_vars[i],filter_vals[i])
                filter_data.append(x)                    
            print(self.minPoints,self.maxPoints)
            FilterData, parentProb, finalQuery = P.filter(filter_data,self.minPoints,self.maxPoints)
            X = FilterData[self.includeVars[0]]
            return sum(X)/len(X)
                
        
        elif(filter_len != 0 and self.r2filters != filter_vals):
            P = ProbSpace(self.data)
            filter_data = []
            for i in range(filter_len):
                x = (filter_vars[i],filter_vals[i])
                print(x)
                filter_data.append(x)                    
            FilterData, parentProb, finalQuery = P.filter(filter_data,self.minPoints,self.maxPoints)
            # print("filter len",filter_len)
            print("filtered datapoints:",len(FilterData['B']))
            # print("include vars:",self.includeVars[:-filter_len])
            self.R2 = RKHS(FilterData,includeVars=self.includeVars[1:-filter_len],delta=self.delta,s=self.s)
            self.r2filters = filter_vals          
        
        elif(self.R2==None):
            self.R2 = RKHS(self.data,includeVars=self.includeVars[1:],delta=self.delta,s=self.s)

        elif(self.R2.varNames != include_vars):
            self.R2 = RKHS(self.data,includeVars=self.includeVars[1:],delta=self.delta,s=self.s)
        
        if(filter_len !=0):
            return self.R2.condE(target,Vals[:-filter_len])
        else:
            return self.R2.condE(target, Vals)

        

