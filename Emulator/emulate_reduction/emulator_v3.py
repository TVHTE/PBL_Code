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
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor

#import graphviz 

class CtaxRedEmulator:
    """
    This class will create a model which finds the reduction that comes with a certain 
    carbon price path. 
    """   
    def __init__(self, df_lin, df_train, test_size):
        
        self.lin_path = np.asarray(df_lin.drop(['reduction'], axis = 1))
        self.lin_reduction = np.asarray(df_lin['reduction'])
        
        self.train_path = np.asarray(df_train.drop(['reduction'], axis = 1))
        self.train_reduction = np.asarray(df_train['reduction'])

        self.df_combined_lin = df_lin        
#        self.df_combined_train = df_train.sort_values([df_train.year])
                
        self.weights = pd.DataFrame()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.train_path, self.train_reduction,
                                                                                test_size = test_size, random_state=9)
        
        # create validation set
#        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
#                                                                                test_size = test_size, random_state=9)
        
        # kan ook zo
#        train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
        
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
        
        # for some reason, some values are not exactly rounded
        self.train_path[:, -1] = np.round(self.train_path[:, -1])
        
        # get lin paths based on final ctax
        self.lin_train = [self.lin_path[self.lin_path[:, -1] == path[-1]] for path in self.train_path]   
        self.lin_train = np.vstack(self.lin_train)
        self.lin_train = self.lin_train[self.lin_train[:, -1].argsort()]
        self.train_path = self.train_path[self.train_path[:, -1].argsort()]
                        
        delta_cs = self.lin_train - self.train_path
        final_ctax = self.lin_train[:, -1]
        delta_c_norm = delta_cs / final_ctax[:, None]
        delta_c_norm = np.nan_to_num(delta_c_norm)  # for first two rows
        delta_c_dict = [{'delta_c': delta_c_norm[i], 'final ctax': final_ctax[i]} for i in range(len(delta_c_norm))]
                                   
        # stepsize is number of paths used to calculate the weights 20 is the dollar step used in TIMER                         
        for index in range(0, 200 + stepsize, stepsize):
                        
            stepsize_ctax = index * 20
                        
            delta_c_step = [ctax['delta_c'] for ctax in delta_c_dict if ctax['final ctax'] <= stepsize_ctax and ctax['final ctax'] >= stepsize_ctax - (stepsize * 20)] 
                        
            # get number of weights wanted
            self.count_weights = int(len(delta_c_step[0]) / number_of_weights)             
            delta_c_slice = [np.mean(delta_c[1:].reshape(-1, self.count_weights), axis=1) for delta_c in delta_c_step]
                        
            lin_reduction_step = self.df_combined_lin[(self.df_combined_lin[self.year] <= stepsize_ctax) &
                                                          (self.df_combined_lin[self.year] >= stepsize_ctax - (stepsize * 20))]     
                                                   
            train_reduction_step = self.df_combined_train[(self.df_combined_train[self.year] <= stepsize_ctax) &
                                                          (self.df_combined_train[self.year] >= stepsize_ctax - (stepsize * 20))]    
            
            train_reduction_step[self.year] = train_reduction_step[self.year].round(0)
            lin_reduction_step[self.year] = lin_reduction_step[self.year].round(0)

            lin_reduction_step = pd.merge(lin_reduction_step, train_reduction_step, on=self.year)
            lin_reduction_step = lin_reduction_step[['reduction_x']].values
            train_reduction_step = train_reduction_step[['reduction']].values
            
#            print(len(delta_c_slice), len(lin_reduction_step), len(train_reduction_step))

            # set initial values to 0
            x0 = [i*0 for i in delta_c_slice[0]]
                        
            res = minimize(self.objective, x0, args=(delta_c_slice, lin_reduction_step, train_reduction_step))
                        
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
    def objective(x, delta_c_slice, lin_reductions, train_reductions):
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
        
        output: final reduction, calculated with the weights
        """
        # for multiple test_paths at once kan dus ook volgens Kaj
        self.X_test[:,-1] = np.round(self.X_test[:,-1])
        lin_test_paths = np.asarray([self.lin_path[self.lin_path[:, -1] == path[-1]] for path in self.X_test])  # get linear paths
        lin_test_paths = np.vstack(lin_test_paths)
                        
        delta_cs = lin_test_paths - self.X_test 
        final_ctax = lin_test_paths[:, -1]
        delta_c_norm = delta_cs / final_ctax[:, None]
        delta_c_slice = [np.mean(delta_c[1:].reshape(-1,  self.count_weights), axis=1) for delta_c in delta_c_norm]
                
        # multiply delta C with the weights to find reduction              
        cur_weights = [self.weights.iloc[(self.weights['ctax'] - ctax).abs().argsort()[:1].values] for ctax in self.X_test[:, -1]]
        clean_weights = [weights.values[0] for weights in cur_weights]
        weights_columns = [column for column in range(self.number_of_weights)]
        weights_columns.append('ctax')
        weights_df = pd.DataFrame(clean_weights, columns=weights_columns)          
                
        b = weights_df.drop(['ctax'], axis=1).values   
        
        # calculate the reduction
        test_red = [self.y_test[i] - (delta_c_slice[i] @ b[i]) for i in range(len(self.y_test))]                        
        test_red = pd.DataFrame(test_red, columns=['test reduction'])
        test_red['final ctax'] = self.X_test[:, -1]
        test_red = test_red.sort_values(by=['final ctax'])
        test_red = test_red.reset_index(drop=True)
        real_red = pd.DataFrame()
        
        real_red['real reduction'] = self.df_combined_train.loc[self.df_combined_train[self.year].isin(self.X_test[:, -1])]['reduction'].reset_index(drop=True)
        real_red['final ctax'] = self.df_combined_train.loc[self.df_combined_train[self.year].isin(self.X_test[:, -1])][self.year].reset_index(drop=True)
        
        final_tested_reduction = test_red.merge(real_red)
        final_tested_reduction.plot(y=['real reduction', 'test reduction'], x=1, grid=True, ylabel='reduction [%]')
        
        return final_tested_reduction
    
    def train_ctax_MLR(self):
        """
        Multivariate Linear Regression using sklearn
        """    
        self.method = 'multivariate linear regression'        

        lin_regr_mod = LinearRegression()
        
        lin_regr_mod.fit(self.X_train, self.y_train)
        
        pred = lin_regr_mod.predict(self.X_test)
        
        return pred

    def train_ctax_PR(self, degree):
        """
        Multivariate Logistic Regression
        
        using sklearn
        """
        self.method = 'polynomial regression'        
                    
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(self.X_train)
        x_poly_test = poly.fit_transform(self.X_test)
        
        log_reg_mod = LinearRegression()
        
        log_reg_mod.fit(x_poly, self.y_train)
        
        pred = log_reg_mod.predict(x_poly_test)
        
        return pred

    def train_ctax_ridge(self, alpha):
        """
        Ridge regression
        """
        self.method = 'ridge regression'        

        ridge_regr_mod = Ridge(alpha=alpha)
        
        ridge_regr_mod.fit(self.X_train, self.y_train)
        
        pred = ridge_regr_mod.predict(self.X_test)
        
        return pred
    
    def train_ctax_lasso(self, alpha):
        """
        Lasso regression
        """
        self.method = 'lasso regression'        

        ridge_regr_mod = Lasso(alpha=alpha)
        
        ridge_regr_mod.fit(self.X_train, self.y_train)
        
        pred = ridge_regr_mod.predict(self.X_test)
        
        return pred
            
    def train_ctax_tree(self, max_depth):
        """
        Regression trees
        """
        
        # bootstrap methods, dataset opsplitsen zodat je ook test met je testsets
        self.method = 'regression tree'
        
        clf = tree.DecisionTreeRegressor(random_state=0, max_depth=max_depth)
        
        clf.fit(self.X_train, self.y_train)
        
        tree.plot_tree(clf)
        
#        dot_data = tree.export_graphviz(clf, out_file=None)
#        graph = graphviz.Source(dot_data)
#        graph.render('ctax')
        
        pred = clf.predict(self.X_test)
        
        return pred
    
    def train_ctax_forest(self, max_depth):
        """
        Regression trees
        """
        
        # bootstrap methods, dataset opsplitsen zodat je ook test met je testsets
        self.method = 'regression forest'
        
        clf = RandomForestRegressor(random_state=0, max_depth=max_depth)
        
        clf.fit(self.X_train, self.y_train)
        
        pred = clf.predict(self.X_test) 
        
        clf.decision_path(self.X_train)
        
        return pred
    
    def test_regr(self, pred):
        """
        Method to test the multivariate regressions
        """
        sorted_y_test = np.sort(self.y_test)         
        sorted_pred = np.sort(pred)
                
        test_set_rmse = (np.sqrt(mean_squared_error(sorted_y_test, sorted_pred)))
        
        test_set_r2 = r2_score(sorted_y_test, sorted_pred)
        
        print('\033[1m' + 'method: ' + '\033[0m', self.method, '\n', 'RMSE: ', test_set_rmse, '\n', 'R-squared: ', test_set_r2)
        
        fig, ax = plt.subplots()
        ax.plot(sorted_pred, label='predicted')
        ax.plot(sorted_y_test, label='true value')
        ax.plot(abs(sorted_pred - sorted_y_test), label='difference')
        ax.plot([i*0 for i in range(len(sorted_pred))], '--', color='red', linewidth=0.8)
        ax.set_ylabel('reduction [%]')
        ax.set_xlabel('# test value')
        ax.grid()
        ax.legend()
        ax.set_title(self.method)
        
        return fig
    
