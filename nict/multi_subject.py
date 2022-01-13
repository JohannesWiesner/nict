# -*- coding: utf-8 -*-
"""
Functions for doing network control theory related analysis for multiple subjects

@author: Johannes.Wiesner
"""

import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from .single_subject import get_subject_transition_df

# TODO: X_df: check that values are a numpy array
# TODO: A_df: check that values are numpy array
# See FIXME below: It would be better to use sciki-learn API here. Still it would
# be convenient to recycle _check_dataframes and _get_multi_subject_transition_df
# to allow the user to come from pandas data frames to lists of numpy arrays
def _check_dataframes(X_df,A_df):
    '''Sanity check input state df and adjacency matrix df'''

    # infer number of states
    n_states = len(X_df.index.unique(level=1))
    
    # check that each subject has the same number of states
    states_per_subject = X_df.groupby(level=0).size()
    for n in states_per_subject:
        if n != n_states:
            raise ValueError('All subjects must have the same number of states')
    
    # TODO: check that each subject not only has the same number of states, but 
    # also has the same state names?
    # check that the order of states is the same for each subject
    # FIXME: The ordering should only be done if for loop throws error. Then 
    # the for-loop should be run again and if successful 
    # print('dataframe was reordered')
    X_df = X_df.sort_index(level=[0,1]) 
    state_names = X_df.index.unique(level=1).tolist()
    for subject_id,subject_df in X_df.groupby(level=0):
        subject_states = subject_df.index.get_level_values(1)
        for subject_state,state_name in zip(subject_states,state_names):
            if subject_state != state_name:
                raise ValueError('The order of states must be the same for each subject')
    
    # https://stackoverflow.com/questions/50222547/
    # FIXME: Use something like this?
    # def print_id(g):
    #     print(g.name)
    # X_df.groupby(level=0).apply(print_id)
    # check that for each subject id in X_df there is an equivalent id in A_df
    state_ids = X_df.index.unique(level=0)
    A_ids = A_df.index.unique(level=0)
    if len(state_ids) != len(A_ids):
        raise ValueError('Unequal number of subject IDs in X_df and A_df')
    for id_1,id_2 in zip(state_ids.sort_values(),A_ids.sort_values()):
        if id_1 != id_2:
            raise ValueError('X_df and A_df contain different subject IDs')

    return X_df,A_df

# TODO: It would be better to stick to the scikit-learn API here and to 
# define multi subject transition as a Transformer (or Estimator?) class
# __init__ method should accept a list of lists of arrays with the states and the adjacency 
# matrices for each subject as 'minimal' inputs (e.g. [[A,X],[A,X],...]). This 
# would serve as minimal input for minimum control energy
# __init__ method should accept c,version and T. Rho should be a keyword argument
# defaulting to None. 
# __init__ method should accept B and S. B and S could be numpy arrays or strings
# defaulting to 'identity'
# .fit method should implement setting B and S to identity matrix and appending
# these to each subject list (Note however that in future projects B, and S
# could also vary for each subject. In this case B and S should be already given
# in the input lists. In this case also, B and S should be set to None in init method)
# .fit method should normalize all A matrices using c and version
# .transform should compute either minimum or optimal control energy using
# T and order argument and energy_type = 'minimal' or 'optimal'. .transform should
# also integrate the energy 
# Class should be able to expose the transition errors using a method (e.g. get_errors)
# Class should be able to expose the transition arrays using a method (e.g. get_transition_arrays)
# Class should be able to expose the UNintegrated energy arrays? (Not sure about that
# because this could result in quite big arrays depending on the time horizon and the
# .transform method should possibly directly integrate the energy)
# The whole class should be agnostic to any descriptions. For this one should implement
# secondary functions that handle the output from .transform() to produce pandas 
# data frames (using node_names,node_groups,state_names,state_groups,long_format,etc.)
# Maybe inherit from nilearn? They also have a CacheMixin Class
# https://github.com/nilearn/nilearn/blob/1607b52458c28953a87bbe6f42448b7b4e30a72f/nilearn/input_data/nifti_masker.py#L116
def get_multi_subject_transition_df(A_df,c,version,T,X_df,rho,order,n_jobs,long_format=False):
    '''Compute optimal integrated control energy for multiple subjects
    
    Parameters
    ----------
    A_df : pd.DataFrame
        A dataframe that contains flattened adjacency matrices. The matrices 
        have to be flattened in in row-major (C-style) order.
    c : int
        normalization constant.
    version : str
        Time system (can be 'continuous' or 'discrete').
    T : int
        Time horizon.
    X_df : pd.DataFrame
        A dataframe that contains state arrays for each subject. The dataframe
        must be a multiindex dataframe where the first index contains the 
        subject id and the second index contains the name of the states. The
        data frame can have a third index that contains state groups. The 
        data frame can have two rows of colum names where the first row describes
        node groups and the second row describes node names
    rho: float
        weights energy and distance constraints. Small rho leads to larger energy
    order: str
        Defines transitions design. Can be 'combinations' (order does not matter), 
        'permutations' (order does matter),'product' (order does matter including self-transitions) 
        or 'stability'(only self-transitions)
    n_jobs : int
        Number of jobs for the parallel energy computations.
    long_format : boolean, optional
        Defines if the output data frame should be returned in the long format. 
        The default is False.

    Returns
    -------
    pd.DataFrame
        A multi-subject dataframe that contains integrated optimal control energy
        for every subject and information about the state transitions

    '''

    # sanity check state df and matrices df
    X_df,A_df = _check_dataframes(X_df,A_df)
    
    # infer transition information from X_df
    if X_df.columns.nlevels > 1:
        node_groups = X_df.columns.get_level_values(0).tolist()
        node_names = X_df.columns.get_level_values(1).tolist()
    else:
        node_groups = None
        node_names = X_df.columns
    
    # state names is (and has to be) the unique entries in the second multiindex level
    state_indices = X_df.droplevel(0).index.unique()
    state_names = state_indices.get_level_values(0).tolist()

    # state groups are the unique levels of the third multiindex level (if given)
    if X_df.index.nlevels  > 2:
        state_groups = state_indices.get_level_values(1).tolist()
    else:
        state_groups = None
    
    # set B and S to identity matrix
    n_nodes = X_df.shape[1]
    B = np.eye(n_nodes)
    S = np.eye(n_nodes)
    
    # group by subject id (first level of multiindex) and get transition df for every subject
    X_df_grouped = X_df.groupby(level=0)
    transition_dfs = Parallel(n_jobs=n_jobs)(delayed(get_subject_transition_df)(A_df=A_df,
                                                                                subject_id=subject_id,
                                                                                n_nodes=n_nodes,
                                                                                c=c,
                                                                                version=version,
                                                                                T=T,
                                                                                B=B,
                                                                                X_df_subject=X_df_subject,
                                                                                rho=rho,
                                                                                S=S,
                                                                                order=order,
                                                                                long_format=long_format,
                                                                                node_names=node_names,
                                                                                node_groups=node_groups,
                                                                                state_names=state_names,
                                                                                state_groups=state_groups) for subject_id,X_df_subject in X_df_grouped)
    
    return pd.concat(transition_dfs)

if __name__ == 'main':
    pass