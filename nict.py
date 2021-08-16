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
from network_control.energies import minimum_input,optimal_input,integrate_u

def _set_transition_design(X,order):
    '''Define all necessary variables for a state-to-state-analysis'''
    
    n_states,n_nodes = X.shape
    indices = np.arange(n_states)
    
    if order=='permutations':
        idxs = [(idx,src,tgt) for idx,(src,tgt) in enumerate(permutations(indices,2))]
    elif order=='combinations':
        idxs = [(idx,src,tgt) for idx,(src,tgt) in enumerate(combinations(indices,2))]

    n_transitions = len(idxs)
    
    return n_nodes,n_transitions,idxs
    
# FIXME: Why do minimum energy and optimal energy have different output lengths?
# TO-DO: For-loop could be parallelized using joblib?
def state_to_state_transition(A,T,B,X,rho,S,order):
    '''Compute state-to-state-transitions using network_control.energies.minimum_input
    or network_control.energies.optimal_input. If rho and S are provided, function
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

    n_nodes,n_transitions,idxs = _set_transition_design(X,order)
    
    x_out = [None] * n_transitions
    u_out = [None] * n_transitions

    if rho is not None and S is not None:
       
        for idx,src,tgt in idxs:
            x_out[idx],u_out[idx],_ = optimal_input(A,T,B,x0=X[src,:],xf=X[tgt,:],rho=rho,S=S)
        
    elif rho is None and S is None:

        for idx,src,tgt in idxs:
            x_out[idx],u_out[idx],_ = minimum_input(A,T,B,x0=X[src,:],xf=X[tgt,:])
    
    return x_out,u_out

def state_to_state_aggregation(u_out):
    '''Aggregate control energy as returned from nict.state_to_state_transition

    Parameters
    ----------
    u_out : np.array
        TxNxM matrix, where T is the number of time points, N is the number
        of nodes and M denotes each combination of states.
    Returns
    -------
    u_agg_out : np.array
        MxN matrix where M is the number of state-to-state-combinations and N
        is the number of nodes. Values represent aggregated control energy over
        time.

    '''

    n_states = len(u_out)
    n_nodes = u_out[0].shape[1]
    u_agg_out = np.zeros((n_states,n_nodes))
        
    for i in range(n_states):
        u_agg_out[i,:] = integrate_u(u_out[i])
        
    return u_agg_out

def get_state_to_state_df_variables(order,style_dict):
    '''Get data frame information from style dict'''
    
    src_tgt = None
    transition_names = None
    src_tgt_grp = None
    
    if 'state_labels' in style_dict:
        if order=='permutations':
            src_tgt = [(src,tgt) for (src,tgt) in permutations(style_dict['state_labels'],2)]
            transition_names = [f"{src}-{tgt}" for (src,tgt) in src_tgt]
        elif order=='combinations':
            src_tgt = [(src,tgt) for (src,tgt) in combinations(style_dict['state_labels'],2)]
            transition_names = [f"{src}-{tgt}" for (src,tgt) in src_tgt]
    
    if 'state_groups' in style_dict:
        if order=='permutations':
            src_tgt_grp = [(src,tgt) for (src,tgt) in permutations(style_dict['state_groups'],2)]
        elif order=='combinations':
            src_tgt_grp = [(src,tgt) for (src,tgt) in combinations(style_dict['state_groups'],2)]

    return src_tgt,transition_names,src_tgt_grp

def get_state_to_state_df(state_to_state_array,order,style_dict):
    '''Produce data frame from state-to-state-array'''

    # create data frame from state-to-state array
    state_to_state_df = pd.DataFrame(state_to_state_array,columns=style_dict['node_labels'])
    
    # add information from style dict
    src_tgt,transition_names,src_tgt_grp = get_state_to_state_df_variables(order,style_dict)
    
    if src_tgt:
        state_to_state_df[['source_state','target_state']] = src_tgt
        state_to_state_df['transition_name'] = transition_names
    if src_tgt_grp:
        state_to_state_df[['source_group','target_group']] = src_tgt_grp
    
    # bring data frame to long format (easier for plotting)
    id_vars = state_to_state_df.columns[~state_to_state_df.columns.isin(style_dict['node_labels'])]
    state_to_state_df = state_to_state_df.melt(id_vars=id_vars,var_name='node_label')
    
    if 'node_groups' in style_dict:
        map_dict = dict(zip(style_dict['node_labels'],style_dict['node_groups']))
        state_to_state_df['node_group'] = state_to_state_df['node_label'].map(map_dict)

    return state_to_state_df

def get_u_agg_df(A,T,B,X,rho,S,order,style_dict):
    '''Produce a plottable data frame for a state-to-state energy array'''

    # compute each state-to-state transition and aggregate control energy
    x_out,u_out = state_to_state_transition(A,T,B,X,rho,S,order)
    u_agg_out = state_to_state_aggregation(u_out)
                                       
    # create data frame
    u_agg_out_df = get_state_to_state_df(u_agg_out,order,style_dict)
    u_agg_out_df.rename(columns={'value':'u_agg'},inplace=True)
    
    return u_agg_out_df

## Other Stuff ###############################################################

def state_to_state_comparison(X,kind,order):
    '''Compare each combination of states node-wise'''

    n_nodes,n_transitions,idxs = _set_transition_design(X,order)
    state_comparison_array = np.zeros((n_transitions,n_nodes))
    
    if kind == 'state_difference':
        for idx,src,tgt in idxs:
            state_comparison_array[idx,:] = X[src,:] - X[tgt,:]
    elif kind == 'state_sum':
        for idx,src,tgt in idxs:
            state_comparison_array[idx,:] = X[src,:] + X[tgt,:]
    elif kind == 'state_mean':
        for idx,src,tgt in idxs:
            state_comparison_array[idx,:] = np.mean(np.array([X[src,:],X[tgt,:]]),axis=0)
    
    return state_comparison_array

def get_state_comparison_df(X,kind,order,style_dict):
    '''Produce a plottable data frame from a a state-to-state array'''
    
    state_comparison_array = state_to_state_comparison(X,kind,order)
    state_comparison_df = get_state_to_state_df(state_comparison_array,order,style_dict)
    state_comparison_df.rename(columns={'value':kind},inplace=True)
    
    return state_comparison_df