# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:46:52 2023

@author: user

It directly apply the uncertainty constraint

With chance constraint with epsilon parameters (eps)

Derived for the reference
Chen, Kuhn, and Wiesemann: mathemati
"Data-Driven Chance Constrained Programs over Wasserstein Balls - Big M method"
 
"""

import os, sys, time
from scipy import io
import numpy as np
import gurobipy as gp
from gurobipy import GRB


tic = time.time()

BASE_DIR = os.getcwd()
os.chdir("../")
os.chdir("../")
path = os.getcwd() 
sys.path.append(path) # 폴더 한 단계 위에서 file import 하기 위해서 sys path 설정
sys.path.append(f"{path}/src/Data_Generation")
from rts_data import generate_sys, generate_matrix, generate_wind

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
W_max = 250
Wmax = np.ones([6])*W_max
Wmin = np.zeros([6])
WD = np.eye(6)*W_max

nWT = len(Wmax)
wind_dict = {'nWT':nWT, 'nScen': nScen, 'N_max': N_max, 'OOS_max':OOS_max,
             'IR_max': IR_max, 'N': N, 'OOS_sim': OOS_sim}

Wscen, Wscen_mu, Wscen_xi = generate_wind(path, wind_dict)
wind_dict['Wscen'] = Wscen
wind_dict['Wscen_mu'] = Wscen_mu
wind_dict['Wscen_xi'] = Wscen_xi

#%%  Initialize DRICC Code

sys_dict = generate_sys()

Wscen_xi = Wscen_xi/2
ResCap = sys_dict['ResCap'] * 3
sys_dict['F'] = sys_dict['F'] * 2.5


sys_dict['Wscen_mu'] = Wscen_mu
sys_dict['Wmax'] = Wmax 
D = sys_dict['D']
Pmax = sys_dict['Pmax']
Pmin = sys_dict['Pmin']
C = sys_dict['C']
Cr1 = sys_dict['Cr1']
Cr2 = sys_dict['Cr2']


nUnits = sys_dict['nUnits']
nWT = sys_dict['nWT']
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
# generation min/max constarint with reserves
for i in range(nUnits):
    m.addConstr(p[i] - rd[i] >= Pmin[i], name=f'const_Pmin{i+1}')
    m.addConstr(p[i] + ru[i] <= Pmax[i], name=f'const_Pmax{i+1}')

# Power Balance Constraints
m.addConstr(gp.quicksum(p) + gp.quicksum(Wmax[i] * Wscen_mu[i] for i in range(nWT)) - gp.quicksum(D) == 0, name = 'power_balance')

Y = m.addMVar((nUnits, nWT), vtype = gp.GRB.CONTINUOUS, lb = -max(Wmax), ub= max(Wmax), name='Y') #Linear decision rule for real-time power production
#Y = m.addMVar((nUnits, nWT), vtype = gp.GRB.CONTINUOUS, lb = -max(Wmax), ub= 0, name='Y') #Linear decision rule for real-time power production

for j in range(nWT):
    m.addConstr( gp.quicksum(Y[i,j] for i in range(nUnits)) ==  - Wmax[j] , name = f'uncertainty_balance{i+1}')  


#%% DRCC Objective Formulation

# dual variables of Distributionally Robust Optimization
s_obj = m.addVars(nScen, lb = -10000, ub = 10000, vtype = gp.GRB.CONTINUOUS, name='s_obj')
lambda_obj = m.addVar(vtype=gp.GRB.CONTINUOUS, lb = 0.0, ub= 10000.0, name='lambda_obj')
rho = 0.05
#rho = rho_vectorC[-1] # 이 값을 상황 따라 바꿀 수 있음
# rho = 1/np.sqrt(nScen)

# Constraints that cTY \Xi <= Si regardless gamma term with (h-H\xi))
for n in range(nScen):
    lhs = gp.quicksum(C[i] * Y[i,j] * Wscen_xi[j][n] for i in range(nUnits) for j in range(nWT))
    m.addConstr(lhs <= s_obj[n], name = f'Const_si_dual{n}')


# Add Dual Norm Constraint for obj
lhs_dual = m.addVar(name='dro_dual_norm')
YY = m.addVars(nWT, vtype = gp.GRB.CONTINUOUS, name='slack_Y')
for i in range(nWT):
    m.addConstr( YY[i] == gp.quicksum(-Y[j,i]*C[j] for j in range(nUnits)))
m.addGenConstrNorm(lhs_dual, YY, GRB.INFINITY, "Const_norm_obj")
m.addConstr(lhs_dual <= lambda_obj)


#%% DRICC Constraint Formulation

jcc = generate_matrix(sys_dict)

nICC = 0
for nC in range(len(jcc)):
    nICC = nICC + len(jcc[nC][0])


nX = len(p) + len(ru) + len(rd)

# It needs to be scalable for generating the constraint matrix
A_g = np.concatenate([jcc[0][0], jcc[1][0]],axis = 0)
A_w = np.concatenate([jcc[0][1], jcc[1][1]],axis = 0)
b_c = np.concatenate([jcc[0][2], jcc[1][2]],axis = 0)
b_g = np.concatenate([jcc[0][3], jcc[1][3]],axis = 0)

x_var = m.addVars(nX,name='x')
for i in range(nUnits):
    m.addConstr(x_var[i] == p[i])
    m.addConstr(x_var[i+ len(p)] == ru[i])
    m.addConstr(x_var[i+ len(p)+len(ru)] == rd[i])

s_c = m.addMVar((nICC, nScen), lb = 0, name='s_c')
eps = DRO_param['eps_joint_cvar'] / 1

t_c = m.addVars(nICC, lb = 0, name='t_c')
q_c = m.addVars(nICC, vtype = gp.GRB.BINARY, name = "q_c")
#q_c = m.addMVar((nICC,nScen), vtype = gp.GRB.BINARY, name = "q_c")

M = 100000

# Add Dual Norm Constraint for
lhs_CC = m.addMVar((nICC,nWT), lb=-10000, ub = 10000, vtype = gp.GRB.CONTINUOUS, name='lhs_CC') 
lhs_dual_CC = m.addMVar((nICC),name='dro_dual_norm_CC')
const = 1


for k in range(nICC):
    lhs_x = gp.quicksum( b_g[k,j] * x_var[j] for j in range(nX)) + b_c[k] 
    lhs_bigM = M*q_c[k]  
    
    for n in range(nScen):
        # A_g : B(j,:) == jcc{j,2}
        # A_w : C(j,:) == jcc{j,3}
        # b_g : A(j,:) == jcc{j,1}
        # b_c : b(j) == -jcc{j,4}
        lhs_u = - gp.quicksum( A_g[k,gg] * Y[gg,ww] * Wscen_xi[ww][n] for gg in range(nUnits) for ww in range(nWT))
        lhs_u -=  gp.quicksum( A_w[k,ww] * Wscen_xi[ww][n] for ww in range(nWT))
        #lhs_bigM = M*q_c[k,n]  
        
        lhs = lhs_x + lhs_u + lhs_bigM
        rhs = t_c[k] - s_c[k,n]
        
        m.addConstr(lhs >= rhs)
        m.addConstr( M*(1-q_c[k]) >= rhs)  
     
    for ww in range(nWT):
        m.addConstr(lhs_CC[k,ww] ==  - ( gp.quicksum( A_g[k,gg] * Y[gg,ww] for gg in range(nUnits)) + A_w[k,ww])) 
    m.addGenConstrNorm( lhs_dual_CC[k], lhs_CC[k,:], GRB.INFINITY, "Const_norm_CC" )
    sum_sc = gp.quicksum(s_c[k,ss] for ss in range(nScen))
    lhs_sc = eps * N * t_c[k] - sum_sc 
    rhs_sc = rho * N * lhs_dual_CC[k]
    m.addConstr( lhs_sc >= rhs_sc ) 
                
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
   
X_Sol = np.zeros([nX])
for j in range(nX):
    X_Sol[j] = m.getVarByName(f"x[{j}]").X
    
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

obj_Sol = obj1_Sol + obj2_Sol + obj3_Sol + obj4_Sol + obj5_Sol 


dual_cc_lhs_Sol = np.zeros((nICC,nWT))

dual_norm_lhs_Sol = np.zeros(nICC)
dual_norm_lhs_A_Sol = np.zeros(nICC)
dual_norm_lhs_B_Sol = np.zeros(nICC)

max_dual_cc_Sol = np.zeros(nICC)
cc_Const_Sol = np.zeros(nICC)
t_c_Sol = np.zeros(nICC)
s_c_Sol = np.zeros((nICC, nScen)) 
sum_s_c_Sol = np.zeros(nICC)
q_c_Sol = np.zeros((nICC, nScen))
# q_c_Sol = np.zeros((nICC, nScen))

c_rhs_Sol = np.zeros((nICC,nScen))
dual_norm_cc_Sol = np.zeros(nICC)

dual_norm_rhs_Sol = np.zeros(nICC)


for k in range(nICC):
    t_c_Sol[k] = m.getVarByName(f"t_c[{k}]").X
    for ww in range(nWT): 
        for gg in range(nUnits):
            dual_cc_lhs_Sol[k,ww] -= A_g[k,gg] * Y_Sol[gg,ww]
        dual_cc_lhs_Sol[k,ww] -= A_w[k,ww]
    
    q_c_Sol[k] = m.getVarByName(f"q_c[{k}]").X
    
    for nn in range(nScen):
        s_c_Sol[k,nn] = m.getVarByName(f"s_c[{k},{nn}]").X
        #q_c_Sol[k,nn] = m.getVarByName(f"q_c[{k},{nn}]").X
        
        
    sum_s_c_Sol[k] = sum(s_c_Sol[k,:])
    dual_norm_lhs_B_Sol[k] = sum_s_c_Sol[k]
    max_dual_cc_Sol[k] = max(dual_cc_lhs_Sol[k,:])
    # cc_Const_Sol[k] = eps * lambda_c_Sol[k] * const
    
    dual_norm_lhs_A_Sol[k] = eps * N * t_c_Sol[k]
    dual_norm_lhs_Sol[k] = dual_norm_lhs_A_Sol[k] - dual_norm_lhs_B_Sol[k] 
    dual_norm_cc_Sol[k] = m.getVarByName(f"dro_dual_norm_CC[{k}]").X
    dual_norm_rhs_Sol[k] = rho * N * dual_norm_cc_Sol[k] 
    
lhs_c_u_Sol = np.zeros((nICC,nScen))
lhs_c_x_Sol = np.zeros((nICC,nX))
sum_lhs_c_u_Sol = np.zeros(nICC)
sum_lhs_c_x_Sol = np.zeros(nICC)

rhs_Sol = np.zeros((nICC,nScen))
sum_lhs_c_Sol = np.zeros((nICC,nScen))

for k in range(nICC):
    
    for j in range(nX):
        lhs_c_x_Sol[k,j] = b_g[k,j]*X_Sol[j] + b_c[k]
    
    sum_lhs_c_x_Sol[k] = sum(lhs_c_x_Sol[k,:])
    
    for n in range(nScen):
        rhs_Sol[k,n] = t_c_Sol[k] - s_c_Sol[k,n]
        for ww in range(nWT):
            lhs_c_u_Sol[k,n] += dual_cc_lhs_Sol[k,ww] * Wscen_xi[ww][n]
        sum_lhs_c_Sol[k,n] = lhs_c_u_Sol[k,n] + sum_lhs_c_x_Sol[k]
    
# for k in range(nScen):
#     for i in range(nUnits):
#         m.addConstr(gp.quicksum(Y[i,j]*Wscen_xi[j][k] for j in range(nWT)) <= ru[i] )        
#         m.addConstr(-gp.quicksum(Y[i,j]*Wscen_xi[j][k] for j in range(nWT)) <= rd[i] )

import matplotlib.pyplot as plt

plt.plot(P_Sol)


toc = time.time()

spent_time = toc - tic 
print("Spent Time : ", spent_time)

