# -*- coding: utf-8 -*-
"""
Test nict package

@author: Johannes.Wiesner
"""

import sys
sys.path.append('../nict')

import numpy as np
import pandas as pd
from network_control.energies import optimal_input,integrate_u
from nict.single_subject import state_to_state_transition,state_to_state_aggregation
from nict.single_subject import get_transition_df
from nict.single_subject import get_state_comparison_df
from nict.multi_subject import get_multi_subject_transition_df
from network_control.utils import matrix_normalization

from itertools import product

###############################################################################
# create artifical data #######################################################
###############################################################################

np.random.seed(28)
A = np.random.rand(10,10)
A_norm = matrix_normalization(A,c=1,version='continuous')
X = np.random.rand(4,10) # four example states
B = np.eye(10)
S = np.eye(10)
T = 2.01
rho = 1
c= 1
version='continous'

# create artifical state df for two subjects
node_names = [f"node_{idx}" for idx in range(0,10)]
node_groups = ['grp_1','grp_1','grp_1','grp_1','grp_1','grp_2','grp_2','grp_2','grp_2','grp_2']
state_names = ['state_1','state_2','state_3','state_4']
state_groups = ['state_group_1','state_group_1','state_group_2','state_group_2']
X_df = pd.DataFrame(np.random.rand(8,10))
X_df['subject_id'] = [1,1,1,1,2,2,2,2]
X_df['state_name'] = state_names + state_names
X_df['state_group'] = state_groups + state_groups
X_df.set_index(['subject_id','state_name','state_group'],inplace=True)
column_names = pd.MultiIndex.from_arrays(arrays=[node_groups,node_names],names=['node_group','node_label'])
X_df.columns = column_names

# create artificial matrices df for two subjects
matrices = [np.random.rand(10,10),np.random.rand(10,10)]
matrices = [A.flatten().reshape(1,-1) for A in matrices]
matrices = np.concatenate(matrices,axis=0)
A_df = pd.DataFrame(matrices)
A_df['subject_id'] = [1,2]
A_df.set_index('subject_id',inplace=True)

###############################################################################
## Testing ####################################################################
###############################################################################

# run network_control for one state transition and compare results against nict
x0_idx=0
xf_idx=1
x0 = X[x0_idx,:]
xf = X[xf_idx,:]
x,u,e = optimal_input(A=A_norm,T=T,B=B,x0=x0,xf=xf,rho=rho,S=S)
u_int = integrate_u(u)

# use nict package and check if computed control energy is the same
X_out,U_out,E_out = state_to_state_transition(A_norm,T,B,X,rho,S,order='permutations')
U_int = state_to_state_aggregation(U_out)
assert np.array_equal(u_int,U_int[0,:])

###############################################################################
# test get_transition_df with all combinations of possible keyword-arguments
###############################################################################

# create a dictionary with all combinations of possible keyword arguments
state_to_state_array = U_int
order_all = ['permutations','combinations','product','stability']
node_names_all = [node_names,None]
node_groups_all = [node_groups,None]
state_names_all = [state_names,None]
state_groups_all = [state_groups,None]
long_format_all = [True,False]
my_dict = {'order':order_all,
            'node_names':node_names_all,
            'node_groups':node_groups_all,
            'state_names':state_names_all,
            'state_groups':state_groups_all,
            'long_format':long_format_all}
keys,values = zip(*my_dict.items())
permutation_dicts = [dict(zip(keys, v)) for v in product(*values)]

all_combo_results = []

for d in permutation_dicts:
    try:
        transition_df = get_transition_df(A=A,
                                          T=T,
                                          B=B, 
                                          X=X,
                                          rho=rho,
                                          S=S,
                                          order=d['order'],
                                          node_names=d['node_names'],
                                          node_groups=d['node_groups'],
                                          state_names=d['state_names'],
                                          state_groups=d['state_groups'],
                                          long_format=d['long_format'])
        all_combo_results.append(transition_df)
    except:
        raise Exception(f"This combination of keywords did not work:\n{d}")

# run multi-subject test #####################################################

multi_subject_df = get_multi_subject_transition_df(A_df,
                                                    c=1,
                                                    version='continuous',
                                                    T=1,
                                                    X_df=X_df,
                                                    rho=1,
                                                    order='permutations',
                                                    n_jobs=2,
                                                    long_format=False)

# test state_to_state_comparison

diff_df = get_state_comparison_df(X=X,
                                  func='difference',
                                  order='permutations',
                                  node_names=node_names,
                                  node_groups=node_groups,
                                  state_names=state_names,
                                  state_groups=state_groups,
                                  long_format=False)


