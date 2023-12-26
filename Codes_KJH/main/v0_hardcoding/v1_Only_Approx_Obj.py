# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:46:52 2023

@author: user

# Scenario
This is the hardcoding for simple version

It directly apply the uncertainty constraint

To show the results without constraint, apply the comments of the below it.


"""

import os, sys
from scipy import io
import numpy as np
import gurobipy as gp
from gurobipy import GRB

BASE_DIR = os.getcwd()
os.chdir("../")
os.chdir("../")
path = os.getcwd() 
sys.path.append(path) # 폴더 한 단계 위에서 file import 하기 위해서 sys path 설정
sys.path.append(f"{path}/src/Data_Generation")


### Parameters
DRO_param = {'eps_joint_cvar' : 0.05}


# Vector for Bonferroni approximation

rho_vectorC = np.linspace(0, 0.0025, 26)

# Number of individual runs (number of coupled datasets in the numerical study)

IR_max = 100
IR_sim = 100

# Number of out of sample data for each individual run (N') for testing
# dataset

OOS_max = 200
OOS_sim = 100

# Number of maximum sample size (N)

N_max = 1000

# Number of sample data in training dataset (N)

N = 100;

# Total number of data 

nScen = IR_max * (N_max + OOS_max)

#Generation of Data
np.random.seed(4)

Wmax = np.ones([6])*250
Wmin = np.zeros([6])
WD = np.eye(6)*250

nWT = len(Wmax)
wt_profile_dict = io.loadmat(f'{path}/src/Data_Generation/AV_AEMO')
wff = wt_profile_dict['AV_AEMO2'][:, :nWT]

# Cutting off very extreme values

cut_off_eps = 1e-2
wff[wff<cut_off_eps] = cut_off_eps;
wff[wff>(1-cut_off_eps)] = 1 - cut_off_eps;

# Logit-normal transformation (Eq. (1) in ref. [31])
yy = np.log(wff/(1-wff))

# Calculation of mean and variance, note that we increase the mean to have
# higher wind penetration in our test-case
mu = yy.mean(axis=0) + 1.5 # Increase 1.5 for higher wind penetration
cov_m = np.cov(yy, rowvar = False)
std_yy = yy.std(axis=0).reshape(1, yy.shape[1])
std_yy_T = std_yy.T
sigma_m = cov_m / (std_yy_T @ std_yy)

# Inverse of logit-normal transformation (Eq. (2) in ref. [31]) 
R = np.linalg.cholesky(sigma_m).T

wt_rand_pattern = np.random.randn(nScen, nWT)

y = np.tile(mu, (nScen,1)) + wt_rand_pattern @ R
Wind = (1 + np.exp(-y))**-1

# Checking correlation, mean and true mean of data
corr_check_coeff = np.corrcoef(Wind, rowvar = False)
mu_Wind = Wind.mean(axis=0)
true_mean_Wind = (1+ np.exp(-mu))**-1

# Reshaping the data structure

nWind = Wind.T
nWind = nWind.reshape(nWT, N_max + OOS_max, IR_max)


# peak N and N' samples
j = 0
WPf_max = nWind[:,0:N_max,j].transpose()
WPr_max = nWind[:,N_max:N_max + OOS_max, j].transpose()
WPf = WPf_max[0:N,:]
WPr = WPr_max[0:OOS_sim,:]

Wscen = WPf[0:N,:].transpose()
Wscen_mu = Wscen.mean(axis = 1)
Wscen_mu = Wscen_mu.reshape(len(Wscen_mu),1)
Wscen_xi = Wscen - np.tile(Wscen_mu,(1, Wscen.shape[1]))


D = np.array([100.7, 90.1, 166.95, 68.9, 66.25, 127.2, 116.6, 159, 161.65, 180.2, 246.45, 180.2, 294.15, 92.75, 310.05, 169.6, 119.25])

Pmax = np.array([152, 152, 300, 591, 60, 155, 155, 400, 400, 300, 310, 350])
Pmin = np.zeros(12, dtype = float)
ResCap = np.array([40, 40, 70, 60, 30, 30, 30, 50, 50, 50, 60, 40]) * 0.4 *10



C = np.array([17.5, 20, 15, 27.5, 30, 22.5, 25, 5, 7.5, 32.5, 10, 12.5])
Cr1 = np.array([3.5, 4, 3, 5.5, 6, 4.5, 5, 1, 1.5, 6.5, 2, 2.5])
Cr2 = np.array([3.5, 4, 3, 5.5, 6, 4.5, 5, 1, 1.5, 6.5, 2, 2.5])


nUnits = len(Pmax)
nWT = Wmax.shape[0]
nScen = Wscen.shape[1]

m = gp.Model(name='DRICC') 
p = m.addVars(nUnits, 
              lb = 0,
              ub =  [Pmax[i] for i in range(nUnits)],
              vtype = gp.GRB.CONTINUOUS, name='p') # Day-ahead power production from thermal power plants

ru = m.addVars(nUnits, 
              lb = 0,
              ub =  [ResCap[i] for i in range(nUnits)],
              vtype = gp.GRB.CONTINUOUS, name='ru') # Upward reserve dispatch from thermal power plants
rd = m.addVars(nUnits, 
              lb = 0,
              ub =  [ResCap[i] for i in range(nUnits)],
              vtype = gp.GRB.CONTINUOUS, name='rd') # Downward reserve dispatch from thermal power plants

 # Day Ahead Constraints
for i in range(nUnits):
    m.addConstr(p[i] - rd[i] >= Pmin[i], name=f'const_Pmin{i+1}')
    m.addConstr(p[i] + ru[i] <= Pmax[i], name=f'const_Pmax{i+1}')
    
m.addConstr(gp.quicksum(p) + gp.quicksum(Wmax[i] * Wscen_mu[i] for i in range(nWT)) - gp.quicksum(D) == 0, name = 'power_balance')


Y = m.addMVar((nUnits, nWT), vtype = gp.GRB.CONTINUOUS, lb = -1000, ub=1000, name='Y') #Linear decision rule for real-time power production
for j in range(nWT):
    m.addConstr( gp.quicksum(Y[i,j] for i in range(nUnits)) ==  -Wmax[j] , name = f'Y=WT{i+1}')  


s_obj = m.addVars(nScen, lb = -10000, ub = 10000, vtype = gp.GRB.CONTINUOUS, name='s_obj')
lambda_obj = m.addVar(vtype=gp.GRB.CONTINUOUS, lb = 0.0, ub= 10000.0, name='lambda_obj')
rho = rho_vectorC[-1] # 이 값을 상황 따라 바꿀 수 있음


for k in range(nScen):
    lhs = gp.quicksum(C[i] * Y[i,j] * Wscen_xi[j][k] for i in range(nUnits) for j in range(nWT))
    m.addConstr(lhs <= s_obj[k], name = f'Const_dual{k}')

# 예비력 제약조건, 이 값을 주석 넣고 뺴고로 시나리오 구분
for k in range(nScen):
    for i in range(nUnits):
        m.addConstr(gp.quicksum(Y[i,j]*Wscen_xi[j][k] for j in range(nWT)) <= ru[i] )        
        m.addConstr(-gp.quicksum(Y[i,j]*Wscen_xi[j][k] for j in range(nWT)) <= rd[i] )


# ADD Dual Norm Constraint
lhs_dual = m.addVar(name='dro_dual_norm')
YY = m.addVars(nWT, vtype = gp.GRB.CONTINUOUS, name='slack_Y')
for i in range(nWT):
    m.addConstr( YY[i] == gp.  quicksum(-Y[j,i]*C[j] for j in range(nUnits)))
m.addGenConstrNorm(lhs_dual, YY, GRB.INFINITY, "normconstr")
#m.addGenConstrNorm(lhs_dual, [YY[i] for i in range(nWT)], GRB.INFINITY, "normconstr")

m.addConstr(lhs_dual <= lambda_obj)


    
obj1 = gp.quicksum(Cr1[i]*ru[i] for i in range(nUnits))
obj2 = gp.quicksum(Cr2[i]*rd[i] for i in range(nUnits))
obj3 = gp.quicksum(C[i]*p[i] for i in range(nUnits))
obj4 = rho * lambda_obj
obj5 = 1/nScen * gp.quicksum(s_obj)

obj = obj1 + obj2 + obj3 + obj4 + obj5

set_obj = m.setObjective(obj, GRB.MINIMIZE)
m.optimize()


P_Sol = np.zeros([nUnits])
ru_Sol = np.zeros([nUnits])
rd_Sol = np.zeros([nUnits])
Y_Sol = np.zeros([nUnits, nWT])
Y2_Sol = np.zeros([nUnits])
for i in range(nUnits):
    P_Sol[i] = m.getVarByName(f"p[{i}]").X
    ru_Sol[i] = m.getVarByName(f"ru[{i}]").X
    rd_Sol[i] = m.getVarByName(f"rd[{i}]").X
    
    for j in range(nWT):
        Y_Sol[i,j] = m.getVarByName(f"Y[{i},{j}]").X
        Y2_Sol[i] = Y2_Sol[i] + Y_Sol[i,j]*Wscen_xi[j][k]        
   
s_obj_Sol = np.zeros([nScen])
lambda_Sol = m.getVarByName("lambda_obj").X
for k in range(nScen):
    s_obj_Sol[k] = m.getVarByName(f"s_obj[{k}]").X
    
obj1_Sol = 0
obj2_Sol = 0
obj3_Sol = 0
obj4_Sol = 0
obj5_Sol = 0


for i in range(nUnits):
    obj1_Sol = obj1_Sol + Cr1[i]*ru_Sol[i]
    obj2_Sol = obj2_Sol + Cr2[i]*rd_Sol[i]
    obj3_Sol = obj3_Sol + C[i]*P_Sol[i]

obj4_Sol = rho * lambda_Sol
for k in range(nScen):
    obj5_Sol = obj5_Sol + 1/nScen*s_obj_Sol[k]
    



    


    

'''
    




# lhs_dual = gp.quicksum( Y[j,i] * C[i])



# m.addConstr(gp.norm(Y[i, j] * si.C[i, j] for i in range(nUnits))
dual = []
for j in range(nWT):
    dual.append(gp.quicksum(Y[i,j]*C[i] for i in range(nUnits)))


lhs_dual = m.addVar(name='dro_dual_norm')

YY = m.addVars(nWT, vtype = gp.GRB.CONTINUOUS, name='slack_Y')

for i in range(nWT):
    m.addConstr( YY[i] == gp.quicksum(Y[j,i]*C[j] for j in range(nUnits)))


m.addGenConstrNorm(lhs_dual, [YY[i] for i in range(nWT)], GRB.INFINITY, "normconstr")

m.addConstr(lhs_dual <= lambda_obj)
'''

# def CC_matrices(si, DRO_param):

#     joint_cvar = DRO_param['eps_joint_cvar']
    
#     jcc = []
    
#     # Matrices for generation
    
#     A_g = []
    
    
#     jcc.append([A_g, B_g, C_g, D_g, joint_cvar])
#     jcc.append([A_l, B_l, C_l, D_l, joint_cvar])
#     jcc.append([A_m, B_m, C_m, D_m, joint_cvar])
    
    
#     return jcc
    
    
    
    
    