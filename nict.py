# -*- coding: utf-8 -*-
"""
Extensions to https://github.com/BassettLab/control_package that might be 
offered as pull requests to that package. The purpose of this package is NOT
to provide core functionalities for NCT. These should be located in control_package.

@author: Johannes.Wiesner
"""

import numpy as np
import pandas as pd
from itertools import permutations,combinations
from network_control.energies import minimum_energy,optimal_energy

###############################################################################
## Possible pull requests for control_package #################################
###############################################################################

def state_trajectory(A,xi,T):
    """This function caclulates the trajectory for the network given our model
     if there are no constraints, and the target state is unknown, using the
     control equation precess x(t+1) = Ax(t). x(t) is the state vector, A is
     the adjacency matrix.
    
    Args:
     A             : NxN state matrix (numpy array), where N is the number of nodes in your
                   network (for example, a structural connectivity matrix 
                   constructed from DTI). A should be stable to prevent
                   uncontrolled trajectories.
         
     xi            : Nx1 initial state (numpy array) of your system where N is the number of
                   nodes. xi MUST have N rows. 
    
    T              : Number of time points, int
        
      Returns:
     x             : x is the NxT trajectory (numpy array) that results from simulating
                   x(t+1) = Ax(t)
    
     @author JStiso 
     June 2017
    """

    # Simulate trajectory
    N = np.size(A,0)

    # initialize x
    x = np.zeros((N,T))
    xt = xi
    for t in range(T):
        x[:,t] = np.reshape(xt, N) # annoying python 1d array thing
        xt_1 = np.matmul(A,xt)
        xt = xt_1
    return x

###############################################################################
## Extensions #################################################################
###############################################################################

## Control Energy Stuff #######################################################

def _get_layout(X,order):
    '''Define for-loop layout'''
    
    m,n = X.shape
    indices = np.arange(m)
    
    if order=='permutations':
        idxs = [(idx,src,tgt) for idx,(src,tgt) in enumerate(permutations(indices,2))]
    elif order=='combinations':
        idxs = [(idx,src,tgt) for idx,(src,tgt) in enumerate(combinations(indices,2))]

    n_combinations = len(idxs)
    
    return n,n_combinations,idxs
    
# FIXME: Why do minimum energy and optimal energy have different output lengths?
# TO-DO: For-loop could be parallelized using joblib?
def state_to_state_transition(A,T,B,X,rho=None,S=None,order='combinations',n_jobs=None):
    '''Compute state-to-state-transitions using network_control.energies.minimum_energy
    or network_control.energies.optimal_energy. If rho and S are provided, function
    will compute optimal energy, otherwise it will compute minimum energy.
    
    Parameters
    ----------
    A : np.array
        NxN state matrix (numpy array), where N is the number of nodes in your
        network (for example, a structural connectivity matrix constructed from DTI). 
        A should be stable to prevent uncontrolled trajectories.
    T : int
        Number of time points.
    B : np.array
        NxN input matrix (numpy array), where N is the number of nodes. B
        selects where you want your input energy to be applied to.
        For example, if B is the Identity matrix, then input energy
        will be applied to all nodes in the network. If B is a
        matrix of zeros, but B(1,1) = 1. then energy will only be
        applied at the first node.
    X : np.array
        MxN matrix, where M is the number of states and N is the number of nodes.
        Each row represents one state.
    S: None, np.array
        (NxN numpy array) Selects nodes whose distance you want to constrain, Define so
        that there is a 1 on the diagonal of elements you want to constrain, and a zero otherwise
    rho: None, float
        weights energy and distance constraints. Small rho leads to larger energy
    order: str
        Can be 'combinations' (order does not matter) or 'permutations'
        (order does matter)

    Returns
    -------
    x_out : np.array
        TxNxM matrix, where T is the number of time points, N is the number
        of nodes and M denotes each combination of states.
    u_out : np.array
        TxNxM matrix, where T is the number of time points, N is the number
        of nodes and M denotes each combination of states.

    '''

    n,n_combinations,idxs = _get_layout(X,order)

    if rho is not None and S is not None:
        shape_out = (1000*T+1,n,n_combinations) # 1001 is a fixed value in control_package
        x_out = np.zeros(shape_out)
        u_out = np.zeros(shape_out)
        
        for idx,src,tgt in idxs:
            
            x_out[:,:,idx],u_out[:,:,idx],_ = optimal_energy(A,T,B,x0=X[src,:].reshape(-1,1),xf=X[tgt,:].reshape(-1,1),rho=rho,S=S)
        
    elif rho is None and S is None:
        shape_out = (1001,n,n_combinations) # 1001 is a fixed value in control_package
        x_out = np.zeros(shape_out)
        u_out = np.zeros(shape_out)
        
        for idx,src,tgt in idxs:
            
            x_out[:,:,idx],u_out[:,:,idx],_ = minimum_energy(A,T,B,x0=X[src,:],xf=X[tgt,:])
    
    return x_out,u_out

# FIXME: network_control now offers such a function (however, not supporting riemann sum)
def aggregate_energy(u,method='auc',absolute_values=False):
    '''Aggregate control energy over all timepoints.
    https://github.com/BassettLab/control_package/blob/main/docs/pages/getting_started.rst    
    
    Parameters
    ----------
    u : numpy.array
        NxT array with control energies as returned from network_control.energies.minimum_energy 
        or network_control.energies.optimal_energy.
    
    method: str
        Method for aggregating energy for each node over time. User can choose
        between 'auc' (computing area under the curve), or 'log_ss'
        for computing the logarithm of sum of squares divided by the number of
        time points (Default = 'auc')
    
    absolute_values: boolean
        Transform the resulting vector to absolute values (Default: False)

    Returns
    -------
    u_agg : numpy.array
        Aggregated array

    '''
    if method == 'auc':
        u_agg = np.trapz(u**2,axis=0)
    elif method == 'log_ss':
        u_agg = np.log(np.sum(u**2,axis=0) / len(u))
    
    if absolute_values:
        u_agg = np.abs(u_agg)
    
    return u_agg

def state_to_state_aggregation(u_out,method='auc',absolute_values=False):
    '''Aggregate control energy as returned from nct_utils.state_to_state_transition

    Parameters
    ----------
    u_out : np.array
        TxNxM matrix, where T is the number of time points, N is the number
        of nodes and M denotes each combination of states.
    method : str, optional
        Method for aggregating energy for each node over time. User can choose
        between 'auc' (computing area under the curve), or 'log_ss'
        for computing the logarithm of sum of squares divided by the number of
        time points (Default = 'auc'). The default is 'auc'.
    absolute_values: boolean
        Transform the resulting vector to absolute values (Default: False)

    Returns
    -------
    u_agg_out : np.array
        MxN matrix where M is the number of state-to-state-combinations and N
        is the number of nodes. Values represent aggregated control energy over
        time.

    '''

    T,n,m = u_out.shape
    u_agg_out = np.zeros((m,n))
        
    for i in range(m):
        u_agg_out[i,:] = aggregate_energy(u_out[:,:,i],method,absolute_values)
        
    return u_agg_out

# TO-DO: It might be more elegant and less verbose to provide all 'style'-variables (area_labels,
# state_names,task_names,value_name,region_labels) as a dictionary and not as singular
# variables
# TO-DO: None of the style variables should be necessary to create this dataframe
def _get_state_to_state_df(a,order,area_labels=None,state_names=None,task_names=None,
                           value_name='value',region_labels=None):
    '''Produce a plottable data frame from a state-to-state-array'''
    
    # create data frame from state-to-state array
    df = pd.DataFrame(a,columns=area_labels)
    
    # add state and task infos
    if order=='permutations':
        if isinstance(state_names,(list,pd.Series)):
            state_name_combos = [(src,tgt) for (src,tgt) in permutations(state_names,2)]
            df[['source_state','target_state']] = pd.DataFrame(state_name_combos)
            df['src_tgt'] = [f"{src}-{tgt}" for (src,tgt) in permutations(state_names,2)]
        if isinstance(task_names,(list,pd.Series)):
            task_name_combos = [(src,tgt) for (src,tgt) in permutations(task_names,2)]
            df[['source_task','target_task']] = pd.DataFrame(task_name_combos)
            
    elif order=='combinations':
        if isinstance(state_names,(list,pd.Series)):
            state_name_combos = [(src,tgt) for (src,tgt) in combinations(state_names,2)]
            df[['source_state','target_state']] = pd.DataFrame(state_name_combos)
            df['src_tgt'] = [f"{src}-{tgt}" for (src,tgt) in combinations(state_names,2)]
        if isinstance(task_names,(list,pd.Series)):
            task_name_combos = [(src,tgt) for (src,tgt) in combinations(task_names,2)]
            df[['source_task','target_task']] = pd.DataFrame(task_name_combos)

    # bring data frame to long format (easier for plotting)
    id_vars = df.columns[~df.columns.isin(area_labels)]
    df = df.melt(id_vars=id_vars,value_name=value_name)
    
    if isinstance(region_labels,(list,pd.Series)):
        area_to_region_mapper = dict(zip(area_labels,region_labels))
        df['region_label'] = df['area_label'].map(area_to_region_mapper)
    
    return df

def get_u_agg_df(A,T,B,X,rho=None,S=None,order='combinations',method='auc',
                 absolute_values=False,area_labels=None,state_names=None,
                 task_names=None,value_name='value',region_labels=None):
    '''Produce a plottable data frame for a state-to-state energy array'''

    # compute each state-to-state transition and aggregate control energy
    x_out,u_out = state_to_state_transition(A,T,B,X,rho,S,order)
    u_agg = state_to_state_aggregation(u_out,method=method,absolute_values=absolute_values)
                                       
    # create data frame
    u_agg_df = _get_state_to_state_df(u_agg,order,area_labels,state_names,
                                      task_names,'u_agg',region_labels
                                      )
    
    return u_agg_df

## Other Stuff ###############################################################

def state_to_state_differences(X,order):
    '''For each combination of states compute difference between nodes'''

    m,n = X.shape
    indices = np.arange(m)
    
    if order=='permutations':
        indices_combos = [(src,tgt) for (src,tgt) in permutations(indices,2)]
    elif order=='combinations':
        indices_combos = [(src,tgt) for (src,tgt) in combinations(indices,2)]
        
    n_combos = len(indices_combos)
    
    state_diffs = np.zeros((n_combos,n))
        
    for idx,(src,tgt) in enumerate(indices_combos):
        state_diffs[idx,:] = X[src,:]-X[tgt,:]
    
    return state_diffs

def get_state_diff_df(X,order='combinations',area_labels=None,state_names=None,
                      task_names=None,region_labels=None):
    '''Produce a plottable data frame for a state-to-state difference array'''
    
    state_diffs = state_to_state_differences(X,order)
    state_diff_df = _get_state_to_state_df(state_diffs,order,area_labels,state_names,
                                           task_names,'state_diff',region_labels
                                           )

    return state_diff_df