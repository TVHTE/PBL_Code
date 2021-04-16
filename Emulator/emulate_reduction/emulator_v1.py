#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# In[39]:
def combi_df(paths_cubic, paths_cubicroot, reduction_cubic, reduction_cubicroot):
    """
    combine cubic and cubicroot paths to use as testset
    """

    path_combi = [[0,0,0,0,0,0,0,0,0,0,0]]
    red_combi = [0]
    
    for i in range(1, len(paths_cubic) - 1, 2):
        path_combi.append(paths_cubic.loc[i].values)
        path_combi.append(paths_cubicroot.loc[i+1].values)
        red_combi.append(reduction_cubic.loc[i*20].values[0])
        red_combi.append(reduction_cubicroot.loc[(i+1)*20].values[0])

    df_combi = pd.DataFrame(path_combi, columns=paths_cubic.columns.values)
    df_combi['reduction'] = red_combi

    # for some reason last ctax values were not integers but floats like 1399.999998
    df_combi = df_combi

    return df_combi

"""
This class will create a model which finds the reduction that comes with a certain 
carbon price path. 
"""

class CtaxRedEmulator:
    
    def __init__(self, df_lin, df_train):
        
        self.lin_path = np.asarray(df_lin.drop(['reduction'], axis = 1))
        self.lin_reduction = np.asarray(df_lin['reduction'])
        
        self.train_path = np.asarray(df_train.drop(['reduction'], axis = 1))
        self.train_reduction = np.asarray(df_train['reduction'])

        self.df_combined_lin = df_lin        
        self.df_combined_train = df_train
                
        self.weights = pd.DataFrame()
        
    def train_ctax_path(self, steps, stepsize_usd, number_of_weights):
        """
        Here the weights for each ctax step is calculated

        Load values: linear paths and random paths including the reductions from TIMER
        output: b values (weigths) for given ctax levels 
        """
        self.steps = steps
        self.stepsize_usd = stepsize_usd
        
        delta_cs = self.lin_path - self.train_path
        final_ctax = self.lin_path[:, -1]
        delta_c_norm = delta_cs / final_ctax[:, None]        
        count_weights = int(steps/number_of_weights)
        self.count_weights =  count_weights
                                
        for index in range(steps, len(delta_c_norm), steps):
                        
            delta_c_step = delta_c_norm[index:index+steps, :]
            delta_c_slice = []
            
            # get number of weights wanted, BESPREEK DIT
            for delta_c in delta_c_step:
                delta_c = delta_c[1:]
                delta_c_slice.append(np.mean(delta_c.reshape(-1, count_weights), axis=1))
                
            lin_reduction_step = self.lin_reduction[index:index+steps]
            train_reduction_step = self.train_reduction[index:index+steps]
            
            # set initial values to 0
            x0 = [i*0 for i in delta_c_slice[0]]
                
            res = minimize(self.objective_delta_c_avg, x0, args=(delta_c_slice, lin_reduction_step, train_reduction_step))
                        
            weights = pd.DataFrame([[x for x in res.x] + [index*20]])
            weights.columns = [*weights.columns[:-1], 'ctax']
            
            self.weights = pd.concat([self.weights, weights])            
            self.weights = self.weights.reset_index(drop=True)
            
        print('weights dataframe:', '\n', self.weights)
                
        # quick vis of paths found
        weights_columns = weights.columns.values
        weights_columns = weights_columns[:-1]
        
        for column in weights_columns:
            plt.plot(self.weights['ctax'], self.weights[column], label=column)
        plt.xlabel('final ctax')
        plt.ylabel('weight')
        plt.legend()
            
    @staticmethod
    def objective_delta_c_avg(x, delta_c_avg, cur_lin_reds, cur_train_reds):
        """
        Objective function that is used to find the weights 
        """
        calc_diff = sum(abs(cur_train_reds[i] - (cur_lin_reds[i] + x.dot(delta_c_avg[i]))) 
                        for i in range(len(cur_train_reds))) 

        return calc_diff
        
    def test_ctax_path(self, test_path):
        """
        here we use the calculated weights to determine the reduction for a
        random chosen ctax path
        
        input: ctax path [list]
        
        output: reduction test [int]  , reduction real [int]
        """
        
        test_path = test_path.drop(['reduction']).values
        
        # find corresponding ctax and weights for test path
        cur_ctax = test_path[10]        
        ctax_column = self.df_combined_lin.columns[-2]       
        cur_lin_path = self.df_combined_lin.loc[self.df_combined_lin[ctax_column] == cur_ctax]
        lin_path_no_red = cur_lin_path.drop('reduction', axis=1).values[0]               
        cur_lin_red = cur_lin_path['reduction'].values
        
        # calculate normalised delta C for test path
        delta_c = (lin_path_no_red - test_path) / lin_path_no_red[-1] 
        delta_c = delta_c[1:]  # BESPREEK DIT
          
        delta_c_slice = np.mean(delta_c.reshape(-1, self.count_weights), axis=1)
                
        # multiply delta C with the weights to find reduction      
        cur_weights = self.weights.iloc[(self.weights['ctax'] - cur_ctax).abs().argsort()[:1]]      
        b = cur_weights.drop('ctax', axis=1).values[0]   
        test_red = cur_lin_red - delta_c_slice @ b               
        real_red = self.df_combined_train.loc[self.df_combined_train[ctax_column] == cur_ctax]['reduction'].values[0]
        
        return (test_red, real_red)
    
    def train_ctax_multi_lin_reg(self, test_path):
        """
        Multivariate regression
        
        using sklearn
        """
        




