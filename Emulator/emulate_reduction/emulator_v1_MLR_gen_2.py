#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class CtaxRedEmulator:
    """
    This class will create a model which finds the reduction that comes with a certain 
    carbon price path. 
    """   
    def __init__(self, df_lin, df_train):
        
        self.lin_path = np.asarray(df_lin.drop(['reduction'], axis = 1))
        self.lin_reduction = np.asarray(df_lin['reduction'])
        
        self.train_path = np.asarray(df_train.drop(['reduction'], axis = 1))
        self.train_reduction = np.asarray(df_train['reduction'])

        self.df_combined_lin = df_lin        
        self.df_combined_train = df_train.sort_values([df_train.year])
                
        self.weights = pd.DataFrame()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.train_path, self.train_reduction,
                                                                                test_size = 0.1, random_state=9)
        
        self.year = df_train.year
        self.region = df_train.region
        
    def train_ctax_path(self, stepsize, number_of_weights):
        """
        Here the weights for each ctax step is calculated

        Load values: linear paths and random paths including the reductions from TIMER
        output: b values (weigths) for given ctax levels 
        """
        self.stepsize = stepsize
        self.number_of_weights = number_of_weights      
        
#        print(len(self.train_path))
        
        # get lin paths based on final ctax
        self.lin_train = [self.lin_path[self.lin_path[:, -1] == path[-1]] for path in self.train_path]
        self.lin_train = [path[0] for path in self.lin_train]
        self.lin_train = np.vstack(self.lin_train)
        self.lin_train = self.lin_train[self.lin_train[:, -1].argsort()]
        self.train_path = self.train_path[self.train_path[:, -1].argsort()]
                
        delta_cs = self.lin_train - self.train_path
        final_ctax = self.lin_train[:, -1]
        delta_c_norm = delta_cs / final_ctax[:, None]
        delta_c_dict = [{'delta_c': delta_c_norm[i], 'final ctax': final_ctax[i]} for i in range(len(delta_c_norm))]
        
#        print([ctax for ctax in delta_c_dict['delta_c']])
                                            
        for index in range(stepsize, len(delta_c_norm), stepsize):
                        
#            delta_c_step = delta_c_norm[index:index+stepsize_ctax, :]
            stepsize_ctax = index * 20
            
            print(index, stepsize_ctax)
            
            delta_c_step = [ctax['delta_c'] for ctax in delta_c_dict if ctax['final ctax'] <= stepsize_ctax and ctax['final ctax'] >= stepsize_ctax - 200 ] 
            delta_c_step = np.vstack(delta_c_step)
            print(len(delta_c_step))
                        
            # get number of weights wanted, BESPREEK DIT              
            delta_c_slice = [np.mean(delta_c[1:].reshape(-1, number_of_weights), axis=1) for delta_c in delta_c_step]
            lin_reduction_step = self.lin_reduction[index:index+stepsize]
            train_reduction_step = self.train_reduction[index:index+stepsize]
            
            print(train_reduction_step)
                                             
            # set initial values to 0
            x0 = [i*0 for i in delta_c_slice[0]]
                        
            res = minimize(self.objective_delta_c_avg, x0, args=(delta_c_slice, lin_reduction_step, train_reduction_step), method = 'Nelder-Mead')
                        
            weights = pd.DataFrame([[x for x in res.x] + [index*20]])
            weights.columns = [*weights.columns[:-1], 'ctax']
                        
            self.weights = pd.concat([self.weights, weights])            
            self.weights = self.weights.reset_index(drop=True)
            
        print('weights dataframe:', '\n', self.weights)
                
        # quick vis of paths found
        weights_columns = weights.columns.values
        weights_columns = weights_columns[:-1]
        
        fig1, ax1 = plt.subplots()
        for column in weights_columns:
            ax1.plot(self.weights['ctax'], self.weights[column], label=column)
        ax1.set_xlabel('final ctax')
        ax1.set_ylabel('weight')
        ax1.legend()
            
    @staticmethod
    def objective_delta_c_avg(x, delta_c_slice, lin_reductions, train_reductions):
        """
        Objective function that is used to find the weights 
        """
        calc_diff = sum(abs(train_reductions[i] - (lin_reductions[i] + x.dot(delta_c_slice[i]))) 
                        for i in range(len(train_reductions))) 
        
        return calc_diff
        
    def test_ctax_paths(self):
        """
        here we use the calculated weights to determine the reduction for a
        random chosen ctax path
        
        input: ctax paths (X_test)
        
        output: reduction test [int]  , reduction real [int]
        """
        # for multiple test_paths at once kan dus ook volgens Kaj
        lin_test_paths = np.asarray([self.lin_path[self.lin_path[:, -1] == path[-1]] for path in self.X_test])  # get linear paths
        delta_cs = lin_test_paths - self.X_test 
        final_ctax = lin_test_paths[:, -1]
        delta_c_norm = delta_cs / final_ctax[:, None]
        delta_c_slice = [np.mean(delta_c[1:].reshape(-1,  self.number_of_weights), axis=1) for delta_c in delta_c_norm[0]]
                
        # multiply delta C with the weights to find reduction              
        cur_weights = [self.weights.iloc[(self.weights['ctax'] - ctax).abs().argsort()[:1].values] for ctax in self.X_test[:, -1]]
        clean_weights = [weights.values[0] for weights in cur_weights]
        weights_columns = [column for column in range(self.number_of_weights)]
        weights_columns.append('ctax')
        weights_df = pd.DataFrame(clean_weights, columns=weights_columns)          
                
        b = weights_df.drop(['ctax'], axis=1).values   
#        test_red = self.y_test - (delta_c_slice @ b)
        
        # calculate the reduction
        test_red = [self.y_test[i] - (delta_c_slice[i] @ b[i]) for i in range(len(self.y_test))]                        
        test_red = pd.DataFrame(test_red, columns=['test reduction'])
        test_red['final ctax'] = self.X_test[:, -1]
        test_red = test_red.sort_values(by=['final ctax'])
        test_red = test_red.reset_index(drop=True)
        real_red = pd.DataFrame()
        
        real_red['real reduction'] = self.df_combined_train.loc[self.df_combined_train[self.year].isin(self.X_test[:, -1])]['reduction'].reset_index(drop=True)
        real_red['final ctax'] = self.df_combined_train.loc[self.df_combined_train[self.year].isin(self.X_test[:, -1])][self.year].reset_index(drop=True)
    
        final_tested_reduction = real_red.merge(test_red)
        
        print(final_tested_reduction)
        
        final_tested_reduction.plot(y=['real reduction', 'test reduction'])
        
        return final_tested_reduction
                    
    def test_ctax_1by1(self, test_path):
        # for only one test_path
        test_path = test_path.drop(['reduction']).values
        
        # find corresponding ctax and weights for test path
        cur_ctax = test_path[-1]        
        ctax_column = self.df_combined_lin.columns[-2]       
        cur_lin_path = self.df_combined_lin.loc[self.df_combined_lin[ctax_column] == cur_ctax]
        lin_path_no_red = cur_lin_path.drop('reduction', axis=1).values[0]               
        cur_lin_red = cur_lin_path['reduction'].values
        
        # calculate normalised delta C for test path
        delta_c = (lin_path_no_red - test_path) / lin_path_no_red[-1] 
        delta_c = delta_c[1:]  # BESPREEK DIT
                
        delta_c_slice = np.mean(delta_c.reshape(-1, self.number_of_weights), axis=1)
                
        # multiply delta C with the weights to find reduction      
        cur_weights = self.weights.iloc[(self.weights['ctax'] - cur_ctax).abs().argsort()[:1]]      
        b = cur_weights.drop('ctax', axis=1).values[0]   
        test_red = cur_lin_red - delta_c_slice @ b               
        real_red = self.df_combined_train.loc[self.df_combined_train[ctax_column] == cur_ctax]['reduction'].values[0]
                
        return (test_red[0], real_red)
    
    def emulate_ctax_MLR(self):
        """
        Multivariate Linear Regression
        
        using sklearn
        """    
        lin_reg_mod = LinearRegression()
        
        lin_reg_mod.fit(self.X_train, self.y_train)
        
        pred = lin_reg_mod.predict(self.X_test)
        
        test_set_rmse = (np.sqrt(mean_squared_error(self.y_test, pred)))
        
        test_set_r2 = r2_score(self.y_test, pred)
        
        print('RMSE: ', test_set_rmse)
        print('R-squared: ', test_set_r2)
        
        fig2, ax2 = plt.subplots()
        ax2.plot(pred, label='predicted')
        ax2.plot(self.y_test, label='true value')
        ax2.plot(pred-self.y_test, label='difference')
        ax2.set_ylabel('reduction')
        ax2.set_xlabel('test values')
        ax2.legend()

    def emulate_ctax_LR(self):
        """
        Multivariate Logistic Regression
        
        using sklearn
        """                    
        polynomial_features= PolynomialFeatures(degree=2)
        x_poly = polynomial_features.fit_transform(self.X_train)
        
        log_reg_mod = LinearRegression()
        
        log_reg_mod.fit(x_poly, self.y_train)
        
        pred = log_reg_mod.predict(self.X_test)
        
        test_set_rmse = (np.sqrt(mean_squared_error(self.y_test, pred)))
        
        test_set_r2 = r2_score(self.y_test, pred)
        
        print('RMSE: ', test_set_rmse)
        print('R-squared: ', test_set_r2)
        
        fig3, ax3 = plt.subplots()
        ax3.plot(pred, label='predicted')
        ax3.plot(self.y_test, label='true value')
        ax3.plot(pred-self.y_test, label='difference')
        ax3.set_ylabel('reduction')
        ax3.set_xlabel('test values')
        ax3.legend()

    def emulate_ctax_ridge(self):
        """
        Ridge regression
        """

