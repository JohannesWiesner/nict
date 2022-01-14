# -*- coding: utf-8 -*-
"""
Functions for doing network control theory related analyses for a single subject

@author: Johannes.Wiesner
"""

import numpy as np
import pandas as pd
from itertools import permutations,combinations,product
from network_control.utils import matrix_normalization
from network_control.energies import minimum_input,optimal_input,integrate_u

# We have to import integrate from scipy again to avoid the following issue:
# https://stackoverflow.com/questions/18049687/
from scipy import integrate

def _set_transition_order(n_states,order):
    '''Define all necessary variables for to compute state-to-state-transitions
    
    Parameters
    ----------
    n_states : int
        Number of states.
    order : str
        Defines transitions design. Can be 'combinations' (order does not matter), 
        'permutations' (order does matter),'product' (order does matter including self-transitions) 
        or 'stability'(only self-transitions)
    
    Returns
    -------
    n_transitions: int
        Number of all transitions
    transition_idxs: list of tuples of three integers
        The first integer in each tuple denotes the number of the transition. The 
        second and third integers denote the row indices of the respective source
        and target states

    '''
    
    indices = np.arange(n_states)
    
    if order == 'permutations':
        transition_idxs = [(idx,src,tgt) for idx,(src,tgt) in enumerate(permutations(indices,r=2))]
    elif order == 'combinations':
        transition_idxs = [(idx,src,tgt) for idx,(src,tgt) in enumerate(combinations(indices,r=2))]
    elif order == 'product':
        transition_idxs = [(idx,src,tgt) for idx,(src,tgt) in enumerate(product(indices,repeat=2))]
    elif order == 'stability':
        transition_idxs = [(idx,idx,idx) for idx,_ in enumerate(indices.tolist())]
    
    n_transitions = len(transition_idxs)
    
    return n_transitions,transition_idxs

# TO-DO: The for-loop inside this function could be parallelized using joblib?
def state_to_state_transition(A,T,B,X,rho,S,order):
    '''Compute all state-to-state-transitions of one set of states and one 
    adjaceny matrix using network_control.energies.minimum_input or 
    network_control.energies.optimal_input. If rho and S are provided, function 
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
    rho: None, float
        weights energy and distance constraints. Small rho leads to larger energy
    S: None, np.array
        (NxN numpy array) Selects nodes whose distance you want to constrain, Define so
        that there is a 1 on the diagonal of elements you want to constrain, and a zero otherwise
    order: str
        Defines transitions design. Can be 'combinations' (order does not matter), 
        'permutations' (order does matter),'product' (order does matter including self-transitions) 
        or 'stability'(only self-transitions)

    Returns
    -------
    X_out : np.array
        TxNxM matrix, where T is the number of time points, N is the number
        of nodes and M denotes each combination of states.
    U_out : np.array
        TxNxM matrix, where T is the number of time points, N is the number
        of nodes and M denotes each combination of states.
    E_out : np.array
        Mx1 matrix, where M denotes each combinations of states

    '''
    
    n_states,n_nodes = X.shape
    n_transitions,transition_idxs = _set_transition_order(n_states,order)
    
    shape_out = (round(T*1000)+1,n_nodes,n_transitions)
    X_out = np.zeros(shape_out)
    U_out = np.zeros(shape_out)
    E_out = np.zeros((n_transitions,1))

    if rho is None and S is None:        
        for idx,src,tgt in transition_idxs:
            X_out[:,:,idx],U_out[:,:,idx],E_out[idx] = minimum_input(A,T,B,x0=X[src,:],xf=X[tgt,:])

    elif rho is not None and S is not None:
        for idx,src,tgt in transition_idxs:
            X_out[:,:,idx],U_out[:,:,idx],E_out[idx] = optimal_input(A,T,B,x0=X[src,:],xf=X[tgt,:],rho=rho,S=S)
        
    return X_out,U_out,E_out

def state_to_state_aggregation(U_out):
    '''Aggregate control energy over time for a set of state-to-state transitions

    Parameters
    ----------
    U_out : np.array
        TxNxM matrix, where T is the number of time points, N is the number
        of nodes and M denotes each combination of states.

    Returns
    -------
    U_int : np.array
        MxN matrix where M is the number of state-to-state-combinations and N
        is the number of nodes. Values represent aggregated control energy over
        time.

    '''

    n_nodes,n_transitions = U_out.shape[1],U_out.shape[2]
    U_int = np.zeros((n_transitions,n_nodes))
        
    for i in range(n_transitions):
        U_int[i,:] = integrate_u(U_out[:,:,i])
        
    return U_int

def get_transition_info(state_attributes,order):
    '''Get transition information for a state-to-state-df
    
    Parameters
    ----------
    state_attributes : list 
        List that holds information about each state.
    order : str
        Defines transitions design. Can be 'combinations' (order does not matter), 
        'permutations' (order does matter),'product' (order does matter including self-transitions) 
        or 'stability'(only self-transitions)

    Returns
    -------
    src_tgt_attributes : list of tuples of source and target state attributes
        The first element in each tuple describes the attribute of the source
        state, the second element describes the attribute of the target state.

    '''
    
    # get transition order
    n_states = len(state_attributes)
    _,transition_idxs = _set_transition_order(n_states,order)
    
    # get source and target attributes as list of tuples
    src_tgt_attributes = [(state_attributes[src],state_attributes[tgt]) for (idx,src,tgt) in transition_idxs]
    
    return src_tgt_attributes

# TODO: It should be possible to already pass this function arrays in the order
# of the transition info. this would be useful to create multi-subject data frames.
# In this case order should be settable to None and all further keywords should already
# expect the right lists WITHOUT calling get_transition_info. state_names and state_groups
# would then have to be a list of tuples times the number of subjects. These lists 
# could be created by using np.tile(get_transition_info,n_subjects)
# TODO: One could make this function even more abstract by allowing the user to pass two dictionaries
# The first dictionary would describe (all sorts!) of information about nodes 
# (e.g. node_dict = {'node_name':['n1','n2'],'node_group':['g1','g2'],...}). If the 
# dictionary contains only one key than simply add this as column names, if it contains
# multipel keys than create a multiindex for columns
# The second dictinaory would describe (all sorts!) of information about the states
# (e.g. state_dict = {'state_name':['s1','s2'],'state_group':['s1','s2'],...})
# The keys of the dictionaries would corespond to the column names
# Same above: When order is set to None, these dictionaries should already contain lists
# of the same size as the input state_to_state_array
def get_state_to_state_df(state_to_state_array,order,node_names=None,node_groups=None,state_names=None,state_groups=None,long_format=False):
    '''Produce data frame from state-to-state-array. All combinations of the keywords arguments
    can produce a valid dataframe
    
    Parameters
    ----------
    state_to_state_array : np.array
        Numpy array where each row defines an operation that was done using a
        source and a target state.
    order : str
        Defines transitions design. Can be 'combinations' (order does not matter), 
        'permutations' (order does matter),'product' (order does matter including self-transitions) 
        or 'stability'(only self-transitions)
    node_names : list of str, optional
        Describes the names of the nodes. The default is None.
    node_groups : list of str, optional
        Describes possible groups for the nodes. The default is None.
    state_names : list of str, optional
        Describes names of the states. The default is None.
    state_groups : list of str, optional
        Describes possible state groups. The default is None.
    long_format : boolean, optional
        Defines if the output data frame should be returned in the long format. 
        The default is False.

    Returns
    -------
    df : pd.DataFrame
        A data frame that holds all information of the input array plus 
        additional information about the transitions.

    '''

    # create data frame from state-to-state array
    df = pd.DataFrame(state_to_state_array)
    
    # add custom node names if given (otherwise node names are just default integers)
    if node_names:
        df.columns = node_names
    else:
        node_names = df.columns
    
    # add transition info for state names if given
    if state_names:
        df[['source_state','target_state']] = get_transition_info(state_names,order)
        df['transition_name'] = df['source_state'] + '-' + df['target_state']
    
    # add transition info for state groups if given
    if state_groups:
        df[['source_group','target_group']] = get_transition_info(state_groups,order)
    
    # get all names of columns that hold information about transitions
    transition_columns = df.columns[~df.columns.isin(node_names)].tolist()
    
    if transition_columns:
        
        # convert to long format using all columns that do not describe nodes as id vars
        if long_format == True:
            df = df.melt(id_vars=transition_columns,var_name='node_name')
            
            # add node groups (at second last position)
            if node_groups:
                node_groups = df['node_name'].map(dict(zip(node_names,node_groups)))
                df.insert(loc=len(df.columns)-1,column='node_group',value=node_groups)
                
        # set all columns that do not describe node names as index
        elif long_format == False:
            df.set_index(transition_columns,inplace=True)
            
            # add node groups (as second level of a multiindex)
            if node_groups:
                df.columns = pd.MultiIndex.from_arrays(arrays=[node_groups,df.columns,],
                                                       names=['node_group','node_name'])
                
    return df

def get_transition_df(A,T,B,X,rho,S,order,**kwargs):
    '''Compute state-to-state-control energy, integrate energy over time 
    and return a data frame with information about the transitions 

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
    rho: None, float
        weights energy and distance constraints. Small rho leads to larger energy
    S: None, np.array
        (NxN numpy array) Selects nodes whose distance you want to constrain, Define so
        that there is a 1 on the diagonal of elements you want to constrain, and a zero otherwise
    order: str
        Defines transitions design. Can be 'combinations' (order does not matter), 
        'permutations' (order does matter),'product' (order does matter including self-transitions) 
        or 'stability'(only self-transitions)
    **kwargs : keyword arguments
        Further keyword-arguments that are passed to nict.get_state_to_state_df.

    Returns
    -------
    df : pd.DataFrame
        State-to-state-Dataframe with integrated control energy and additional information about
        transitions.

    '''
    

    # compute each state-to-state transition and aggregate control energy
    X_out,U_out,E_out = state_to_state_transition(A,T,B,X,rho,S,order)
    U_int = state_to_state_aggregation(U_out)
                                       
    # create data frame
    df = get_state_to_state_df(U_int,order,**kwargs)
    
    return df

# NOTE: This function would be better off in multi_subject module but had to 
# be placed here to avoid this error on Windows: https://stackoverflow.com/a/35454106/8792159
def get_subject_transition_df(A_df,subject_id,n_nodes,c,version,T,B,X_df_subject,rho,S,order,long_format,**kwargs):
    '''Compute state-to-state-control energy for a single subject, integrate energy over time 
    and return a data frame with information about the transitions.'''
    
    # get adjacency matrix for this subject and normalize
    A = A_df.loc[subject_id].values.reshape((n_nodes,n_nodes))
    A = matrix_normalization(A,c=c,version=version)
    
    # get states for this subject
    X = X_df_subject.values

    # compute transition df
    transition_df = get_transition_df(A=A,T=T,B=B,X=X,rho=rho,S=S,order=order,long_format=long_format,**kwargs)

    # add subject id
    if long_format==True:
        transition_df.insert(loc=0,column='subject_id',value=subject_id)
    
    elif long_format==False:
        current_idx = transition_df.index.to_frame()
        current_idx.insert(0,'subject_id',subject_id)
        transition_df.index = pd.MultiIndex.from_frame(current_idx)
    
    return transition_df

##############################################################################
## Not NCT-related ###########################################################
##############################################################################

# TODO: It should also be possible to set func to a custom function
def state_to_state_comparison(X,func,order):
    '''one operation for each source and target state
    
    Parameters
    ----------
    X : np.array
        MxN matrix, where M is the number of states and N is the number of nodes.
        Each row represents one state.
    func : str
        Function that should be applied using one source and one target state. Can 
        be 'difference' to compute difference between two states, 'sum' to compute
        the sum of two states or a custom function that takes in one source and one
        target state as input
    order: str
        Defines transitions design. Can be 'combinations' (order does not matter), 
        'permutations' (order does matter),'product' (order does matter including self-transitions) 
        or 'stability'(only self-transitions)

    Returns
    -------
    state_comparison_array : np.array
        Each row represents one operation using a source and a target state.

    '''

    n_states,n_nodes = X.shape
    n_transitions,transition_idxs = _set_transition_order(n_states,order)
    state_comparison_array = np.zeros((n_transitions,n_nodes))
    
    if func == 'difference':
        for idx,src,tgt in transition_idxs:
            state_comparison_array[idx,:] = np.subtract(X[src,:],X[tgt,:])
    elif func == 'sum':
        for idx,src,tgt in transition_idxs:
            state_comparison_array[idx,:] = np.add(X[src,:],X[tgt,:])
    elif callable(func):
        for idx,src,tgt in transition_idxs:
            state_comparison_array[idx,:] = func(X[src,:],X[tgt,:])

    return state_comparison_array

def get_state_comparison_df(X,func,order,**kwargs):
    '''Compute one operation for each source and target state and get a dataframe'''
    
    state_comparison_array = state_to_state_comparison(X,func,order)
    state_comparison_df = get_state_to_state_df(state_comparison_array,order,**kwargs)
    
    return state_comparison_df


if __name__ == 'main':
    pass