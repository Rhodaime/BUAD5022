# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:53:36 2023

@author: jrbrad
"""

import gurobipy as gpy
import numpy as np
import time

''' Define problem parameters '''
num_order = 100
num_item = 100
num_fc = 3
cap_order = np.array([5000,5000,5000])

''' Mark start time of inputting data '''
start = time.time()

''' Read data '''
print('Creating/reading order data')
oq = np.genfromtxt(f'order_{num_item}_{num_order}.txt')
''' Read fulfillment center inventory data '''
fc_inv = np.genfromtxt(f'inv{num_order}_{num_item}_{num_fc}.txt')
''' Read order shipping data '''
fc_fill = np.genfromtxt(f'ship_cost{num_order}_{num_item}_{num_fc}.txt')
''' Read transshipment cost data '''
tshp = np.genfromtxt(f'tshp{num_order}_{num_fc}_{num_fc}.txt')
tshp = tshp.reshape(num_item,num_fc,num_fc)
data_complete = time.time()
print(f'Data read in {data_complete-start} seconds.')

''' Create model '''
m = gpy.Model('amazon')
m.ModelSense = gpy.GRB.MINIMIZE
#m.setParam('TimeLimit',7200)

''' Create dv for order delivery '''
y = m.addMVar((num_order,num_fc), name='y', vtype=gpy.GRB.BINARY)
#y = [[m.addVar(vtype=gpy.GRB.BINARY,name='of_'+str(o)+'_'+str(f)) for f in range(num_fc)] for o in range(num_order) ]

''' Create dv for transshipment '''
x = m.addMVar((num_item, num_fc, num_fc), name='x', vtype=gpy.GRB.CONTINUOUS)
#x = [[[m.addVar(vtype=gpy.GRB.CONTINUOUS,name='tshp_'+str(f1)+'_'+str(f2)+'_'+str(i),lb=0.0) for f1 in range(num_fc)] for f2 in range(num_fc)] for i in range(num_item)]

''' Update variables '''
m.update()

''' Constraints for inventory: order shipments can't exceed available stock '''
m.addConstrs((-oq.T @ y + fc_inv + x.sum(axis=1) - x.sum(axis=2) >= np.zeros((num_item, num_fc)) for i in range(1))) 
'''
for f in range(num_fc):
    for a in range(num_item):
        coeff = np.zeros((num_fc,num_fc))
        coeff[:,f] = 1
        coeff[f,:] = -1
        coeff[f,f] = 0
        #m.addLConstr(gpy.LinExpr([(coeff[i,j],dv_trans[a][i][j]) for i in range(len(coeff)) for j in range(len(coeff[i])) if coeff[i,j]!=0]), gpy.GRB.GREATER_EQUAL, gpy.quicksum(oq[o][a]*dv_ord_fc[o][f] for o in range(num_order))-fc_inv[f][a],name='c_inv_'+str(f)+'_'+str(a))
        m.addLConstr(fc_inv[a][f] +
                     gpy.quicksum([coeff[i,j]*x[a][i][j] for i in range(len(coeff)) for j in range(len(coeff[i])) if coeff[i,j]!=0]) 
                     - gpy.quicksum(oq[o][a]*y[o][f] for o in range(num_order)), 
                     gpy.GRB.GREATER_EQUAL, rhs=0,name='c_inv_'+str(f)+'_'+str(a)) '''
        

''' Constraint to deliver order '''
m.addConstrs((y.sum(axis=1) == np.ones((num_order)) for i in range(1)))
#for o in range(num_order):
#    m.addLConstr(gpy.quicksum(y[o]), gpy.GRB.EQUAL, rhs=1, name='order_'+str(o))

''' Constraint on maximum orders fulfilled at FCs '''
m.addConstr(y.sum(axis=0) <= cap_order)

''' Update constraints '''
m.update()

''' Create objective function '''
m.setObjective((tshp*x).sum() + (fc_fill*y).sum())  
#cost_tshp = [x[k][i][j]*tshp[k][i][j] for k in range(num_item) for i in range(num_fc) for j in range(num_fc)]
#cost_fill = [fc_fill[i][j]*y[i][j] for i in range(num_order) for j in range(num_fc)]
#m.setObjective(gpy.quicksum(cost_tshp) + gpy.quicksum(cost_fill))

''' Optimize '''
m.optimize()

print(f'Model solved in {time.time()-data_complete} seconds.')

''' Evaluate solution '''
print('\n\nComputing transshipment decision variable values')

# Find nonzero transshipment variables
tshp_nonz = {}
for i in range(num_fc): #x.shape[0]
    for j in range(num_fc): #x.shape[1]
        for k in range(num_item): #x.shape[2]
            if x[k][i][j].x != 0: #x[i,j,k].x != 0
                #print(f'{var.varName}: {var.x}')
                tshp_nonz[(i,j,k)] = x[k][i][j].x

tshp_nonz1 = {(i,j,k):x[k][i][j].x for k in range(num_item) for j in range(num_fc) for i in range(num_fc) if x[k][i][j].x != 0}
print('Nonzero decision variables are in dictionaries "tshp_nonz" and "tshp_nonz1"')

print('\n\nOptimal Order Fulfillment Decision variables')
print(y.x)
print('\nOrder  FC')
print('\n'.join([f'{i}, {np.argmax(y[i].x)}' for i in range(y.shape[0])]))

print('\n\nCheck Constraints')
print('Number of times each order is fulfilled')
print(y.x.sum(axis=1))
print('Number of orders fulfilled by each FC')
print(y.x.sum(axis=0))
print('Check inventory balance constraints')
inv_bal_lhs = (-oq.T @ y + fc_inv + x.sum(axis=1) - x.sum(axis=2))
print(f'Type of inv_bal_lhs: {type(inv_bal_lhs)}')
print(f'Type of inv_bal_lhs.getValue(): {type(inv_bal_lhs.getValue())}')
print(inv_bal_lhs.getValue())
#print((inv_bal_lhs.getValue())[np.any(inv_bal_lhs.getValue()<0,axis=1),:])

''' Save solution to file '''
print('Print solution file with rounded decision variable values')
m.params.SolutionNumber = 0
m.write(f'solution_{num_order}_{num_item}.sol')

sol_head = [f'# Solution for model {m.ModelName}', f'# Objective value = {m.getObjective().getValue()}']
sol_y = [f'{v.varName} {np.round(v.x)}' for row in y  for v in row]
sol_x = [f'{v.varName} {np.round(v.x)}' for array in x for row in array for v in row]
file_sol = 'solution_round_{num_order}_{num_item}.sol'
with open(file_sol, 'w') as f:
    f.write('\n'.join(sol_head+sol_y+sol_x))
    
    

''' redefine model with integer transshipment decision variables '''
print('Re-optimize model with integer transshipment decision variables.')
m = gpy.Model('amazon')
m.ModelSense = gpy.GRB.MINIMIZE
m.Params.MIPGap = 0.0001
y = m.addMVar((num_order,num_fc), name='y', vtype=gpy.GRB.BINARY)
x = m.addMVar((num_item, num_fc, num_fc), name='x', vtype=gpy.GRB.INTEGER)
m.update()
m.addConstrs((-oq.T @ y + fc_inv + x.sum(axis=1) - x.sum(axis=2) >= np.zeros((num_item, num_fc)) for i in range(1))) 
m.addConstrs((y.sum(axis=1) == np.ones((num_order)) for i in range(1)))
m.addConstr(y.sum(axis=0) <= cap_order)
m.update()
m.setObjective((tshp*x).sum() + (fc_fill*y).sum())  
m.update()

''' Read starting solution '''
m.read(file_sol)
m.update()

m.optimize()
