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
                
        self.df_tot = pd.DataFrame()
        
    def train_ctax_path(self):
        """
        Here the weights for each ctax step is calculated

        Load values: linear paths and random paths including the reductions from TIMER
        output is b values (weigths) for given ctax levels 
        """
        
        # find reduction levels for linear and training path at same ctax level @ all ctax levels        
        for index in range(0, len(self.lin_path) - 10, 10):
                    
            # ctax and reduction of train path @ index
            ctax_val_train = self.train_path[index+10][-1].round()
    
            # get the linear reduction corresponding to the ctax level of training input
            last_column = self.df_combined_lin.columns[-2]        
            cur_red_lin = self.df_combined_lin.loc[self.df_combined_lin[last_column] == ctax_val_train]         
            cur_red_lin = cur_red_lin['reduction'].values[0]
            
            # use sets of 200 dollar differences to calculate weights 
            cur_ctax = [i for i in range(index*20, index*20 + 220, 20)]
            
            # create dataframe with cur_ctax as labels and list for linear reduction
            cur_train_paths = pd.DataFrame()
            cur_lin_paths = pd.DataFrame()
            cur_lin_reds = []
                        
            for ctax in cur_ctax:

                cur_ctax_path = self.df_combined_train.loc[self.df_combined_train[last_column] == ctax]
                cur_train_paths = cur_train_paths.append(cur_ctax_path)    
                cur_lin_reds.append(self.df_combined_lin.loc[self.df_combined_lin[last_column] == ctax]['reduction'].values[0])              
                cur_ctax_path_lin = self.df_combined_lin.loc[self.df_combined_lin[last_column] == ctax]               
                cur_lin_paths = cur_lin_paths.append(cur_ctax_path_lin)
                
            # calculate delta_C for all training paths
            lin_paths_no_red = cur_lin_paths.drop('reduction', axis=1)            
            train_paths_no_red = cur_train_paths.drop('reduction', axis=1)
                        
            # empty df for all delta_c
            delta_c = pd.DataFrame()   
            
            # calculate normalised delta C for every train path with corresponding lin path CHECK DIT
            for i in range(len(lin_paths_no_red)):
                
#                print(train_paths_no_red.loc[i+index])
                
                delta_c = delta_c.append((lin_paths_no_red.loc[i+index] - train_paths_no_red.loc[i+index]) / 
                                        lin_paths_no_red.loc[i+index][10])               
                
            # random reduction values for the paths
            cur_train_reds = cur_train_paths['reduction'].values
            
            # take only the two averages of normalised delta_c HARD CODED
            delta_c_avg = []

            for delta_c in delta_c.values:
                delta_c1 = sum(delta_c[0:4])/5
                delta_c2 = sum(delta_c[5:])/6
                delta_c_avg.append([delta_c1, delta_c2])
            
            # define objective function
            def objective_delta_c_avg(x, delta_c_avg, cur_lin_reds, cur_train_reds):
                """
                Objective function that is used to find the weights 
                """
                calc_diff = sum(abs(cur_train_reds[i] - (cur_lin_reds[i] + x.dot(delta_c_avg[i]))) 
                                for i in range(len(cur_train_reds))) 

                print(calc_diff)

                return calc_diff
        
            # set initial values to 0
            x0 = [i*0 for i in delta_c_avg[0]]
            
            # minimize objective functions   
            res = minimize(objective_delta_c_avg, x0, args=(delta_c_avg, cur_lin_reds, cur_train_reds),
                          method='Nelder-Mead')

            # print results
#             print(res)
    
            # save weights to ctax level
            weights = pd.DataFrame([[res.x[0], res.x[1], 200 + (index*20)]], 
                                   columns=['b1','b2','ctax'])
            
            self.df_tot = pd.concat([self.df_tot, weights])            
            self.df_tot = self.df_tot.reset_index(drop=True)   
           
        # make df and add to self
        self.weights = self.df_tot
        
        print(self.df_tot)
        
        # quick vis of paths found
        plt.plot(self.df_tot['ctax'], self.df_tot['b1'], color='blue', linewidth=3)
        plt.plot(self.df_tot['ctax'], self.df_tot['b2'], color='red', linewidth=3)
        plt.xlabel('final ctax')
        plt.ylabel('weight')
        plt.legend(['b1','b2'])
        
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.df_tot['ctax'], self.df_tot['b1'], color='blue')
        axs[1].plot(self.df_tot['ctax'], self.df_tot['b2'], color='red')
        
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
        last_column = self.df_combined_lin.columns[-2]       
        cur_lin_path = self.df_combined_lin.loc[self.df_combined_lin[last_column] == cur_ctax]                      
        lin_path_no_red = cur_lin_path.drop('reduction', axis=1).values[0]               
        cur_lin_red = cur_lin_path['reduction'].values
        
        # calculate normalised delta C for test path
        delta_c = (lin_path_no_red - test_path) / lin_path_no_red[-1] 

        # take only the two averages of normalised delta_c NOG HARD CODED
        delta_c_avg = []
        delta_c1 = sum(delta_c[0:4])/5
        delta_c2 = sum(delta_c[5:])/6
        delta_c_avg.append([delta_c1, delta_c2])
        
        # multiply delta C with the weights to find reduction      
        df_sort = self.weights.iloc[(self.weights['ctax'] - cur_ctax).abs().argsort()[:1]]
        
        b1 = df_sort['b1'].values[0]
        b2 = df_sort['b2'].values[0]
        
        test_red = cur_lin_red - ((delta_c1 * b1) + (delta_c2 * b2))
        
        real_red = self.df_combined_train.loc[self.df_combined_train[last_column] == cur_ctax]['reduction'].values[0]
        
#         print('\n', 'weights: ', b1, b2,
#               '\n', 'delta Cs: ', delta_c1, delta_c2,
#               '\n', 'reductions (lin, test, real): ', cur_lin_red, test_red, real_red)
        
        return (test_red, real_red)

if __name__ == "__main__": 
    
    # check if model works
    paths_linear = df_11_SSP1.drop(['USD','reduction'], axis = 1)
    reduction_linear = df_11_SSP1['reduction']

    paths_cubic = df_11_SSP1_cubic.drop(['USD','reduction'], axis = 1)
    reduction_cubic = df_11_SSP1_cubic['reduction']

    paths_cubicroot = df_11_SSP1_cubicroot.drop(['USD','reduction'], axis = 1)
    reduction_cubicroot = df_11_SSP1_cubicroot['reduction']

    # DIT MOET ANDERS



    # paths train is 1/2 cubic and 1/2 root
    path_combi, red_combi = combi_df(paths_cubic, paths_cubicroot, reduction_cubic, reduction_cubicroot)
    
    x = CtaxRedEmulator(paths_linear, reduction_linear, path_combi, red_combi)

    x.train_ctax_path()
    
    test_path_root = paths_cubicroot.loc[35].values
    test_path_cubic = paths_cubic.loc[36].values

    x.test_ctax_path(test_path_cubic)

    # in training, even index was used for cubicroot so test with uneven (and exactly the other way around for root)
    cubic_test = []
    root_test = []
    
    for i in range(1, len(paths_cubicroot), 2):
        
        cubic_test.append(x.test_ctax_path(paths_cubicroot.loc[i].values))
        root_test.append(x.test_ctax_path(paths_cubic.loc[i+1].values))


    # In[40]:
    
    
    reds = [i for i in range(len(cubic_test))]
    
    for red in zip(*root_test):
        plt.plot(reds, red, label='root')
    plt.title('accuracy of emulator')
    plt.xlabel('number of carbon tax paths')
    plt.ylabel('reduction')
    
    for red in zip(*cubic_test):
        plt.plot(reds, red, label='cubic')
    
    plt.legend()
    
    
    # In[27]:
    
    
    cubic_path = paths_cubic.loc[184].values
    cubicroot_path = paths_cubicroot.loc[184].values
    lin_path = paths_linear.loc[184].values
    
    fig, axs = plt.subplots(1, 1)
    axs.plot(cubic_path, color='blue', label='cubic')
    axs.plot(cubicroot_path, color='blue', label='cubic root')
    axs.plot(lin_path, color='red', label='linear')
    axs.vlines(5,[0], 3500, color='black', label='weight division')
    # axs.fill_between(cubicroot_path, lin_path)
    axs.legend()
    
    # Turn off tick labels
    axs.set_yticklabels([])
    axs.set_xticklabels([])


