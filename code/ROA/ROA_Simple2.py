# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:47:42 2023

@author: jrbrad
"""

import gurobipy as gpy
import numpy as np

num_week = 52
d = np.array([100 for _ in range(15)] + [150 for _ in range(20)] + [300 for _ in range(5)] + [85 for _ in range(12)])
machine = np.array([[100000, 150000, 200000, 250000],
                    [135, 150, 200, 300]])
inv_beg = 0
h = 10 # Weekly inventory holding cost per week
r = 10 # Profit per unit sold

m = gpy.Model('ROA')

y = m.addMVar((machine.shape[1],), vtype=gpy.GRB.BINARY) # machine investment
x = m.addMVar((num_week,), vtype=gpy.GRB.CONTINUOUS) # production quantity
inv = m.addMVar((num_week,), vtype=gpy.GRB.CONTINUOUS) # inventory level end of period

m.addConstr(y.sum() >= 1)
m.addConstr(inv_beg + x[0] - d[0] - inv[1] >= 0)
m.addConstrs(inv[w-1] + x[w] - d[w] - inv[w] >= 0 for w in range(1,num_week))
m.addConstrs(x <= (machine@y)[1] for _ in range(1))

c = 0.01
p = (r*d).sum()
a = ((h*inv).sum() + (machine@y)[0])
m.setObjective(p - c * a, gpy.GRB.MAXIMIZE)
m.optimize()

while m.ObjVal > 0.00001:
    c = p/a.getValue()
    m.setObjective(p - c * a, gpy.GRB.MAXIMIZE)
    m.update()
    m.optimize()

print(f'Optimal ROA: {c*100:.2f}%')
print('Equipment investment:\nCost  Capacity')
print('\n'.join([f'{machine[0][i]}  {machine[1][i]}' for i in range(len(machine[0])) if y[i].x==1]))
print('\n\nProduction  Inventory')
print('\n'.join([f'{x[i].x}  {inv[i].x:.1f}' for i in range(num_week)]))
