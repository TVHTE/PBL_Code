#!/usr/bin/env python
# coding: utf-8

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math

from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

# TensorFlow
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

#import graphviz 

class CtaxRedEmulator:
    """
    This class will create a model which finds the reduction that comes with a certain 
    carbon price path. 
    """   
    def __init__(self, year, region, df_lin, df_train, test_size):
        
        self.year = year
        self.region = region
        self.columns = df_lin.columns
        self.pred = pd.DataFrame()
        
        self.lin_path = np.asarray(df_lin.drop(['reduction'], axis = 1))
        self.lin_reduction = np.asarray(df_lin['reduction'])
         
        df_train = df_train.drop_duplicates()
        
        self.train_path = np.asarray(df_train.drop(['reduction'], axis = 1))
        self.train_reduction = np.asarray(df_train['reduction'])
        
        self.train_path
                
        self.df_combined_lin = df_lin        
        self.df_combined_train = df_train.sort_values(str(year)).reset_index(drop=True)
#        self.df_combined_train = df_train
        
        self.weights = pd.DataFrame()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.train_path, self.train_reduction,
                                                                                test_size = test_size, random_state=9)
        
        # create validation set
#        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
#                                                                                test_size = test_size, random_state=9)
        
        # kan ook zo
#        train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
        
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
        self.lin_train = np.asarray([self.lin_path[self.lin_path[:, -1] == self.find_nearest(self.lin_path[:, -1], path[-1])][0] for path in self.train_path])   
        self.lin_train = self.lin_train[self.lin_train[:, -1].argsort()]
        self.train_path = self.train_path[self.train_path[:, -1].argsort()]
        self.lin_train = self.lin_train[:len(self.train_path)]
                
        delta_cs = self.lin_train - self.train_path
        final_ctax = self.lin_train[:, -1]
        delta_c_norm = delta_cs / final_ctax[:, None]
        delta_c_norm = np.nan_to_num(delta_c_norm)  # for first two rows
        self.delta_c_df = pd.DataFrame(delta_c_norm, columns=self.columns[:-1])
        delta_c_dict = [{'delta_c': delta_c_norm[i], 'final ctax': final_ctax[i]} for i in range(len(delta_c_norm))]
        
        self.lin_train_df = pd.DataFrame(self.lin_train, columns=self.columns[:-1])
        self.train_path_df = pd.DataFrame(self.train_path, columns=self.columns[:-1])
        self.train_path_df
        
        self.all_lin = pd.merge(self.lin_train_df.round(decimals=1), self.df_combined_lin.round(decimals=1), how='inner')
        self.all_lin = self.all_lin[:len(self.lin_train_df)]
                
        # stepsize is number of paths used to calculate the weights 20 is the dollar step used in TIMER                         
        for index in range(0, 200 + stepsize, stepsize):
                        
            stepsize_ctax = index * 20
            delta_c_step = [ctax['delta_c'] for ctax in delta_c_dict if ctax['final ctax'] <= stepsize_ctax and ctax['final ctax'] >= stepsize_ctax - (stepsize * 20)]       
            year = str(self.year)
         
            indices_train = self.train_path_df[(self.train_path_df[year] <= stepsize_ctax) &
                                         (self.train_path_df[year] >= stepsize_ctax - (stepsize * 20))]

            train_reduction_step = self.df_combined_train.iloc[indices_train.index.values]
            lin_reduction_step = self.all_lin.iloc[indices_train.index.values]
            
            delta_c_step = self.delta_c_df.iloc[indices_train.index.values]
            self.count_weights = int(len(delta_c_step.iloc[0]) / number_of_weights)             
            delta_c_slice = [np.mean(delta_c[1:].reshape(-1, self.count_weights), axis=1) for delta_c in delta_c_step.values] 
           
            train_reduction_step[year] = train_reduction_step[year].round(0)
            lin_reduction_step[year] = lin_reduction_step[year].round(0)

            lin_reduction_step = lin_reduction_step['reduction'].values
            train_reduction_step = train_reduction_step['reduction'].values
            
            lin_reduction_step = lin_reduction_step.reshape((len(lin_reduction_step),1))
            train_reduction_step = train_reduction_step.reshape((len(train_reduction_step),1))
            
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
     
    @staticmethod
    def find_nearest(array, value):
        
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
                
        return array[idx]    
    
    def test_ctax_paths(self):
        """
        here we use the calculated weights to determine the reduction for a
        random chosen ctax path
        
        input: ctax paths (X_test)
        
        output: final reduction, calculated with the weights
        """
        # for multiple test_paths at once        
        self.X_test[:,-1] = np.round(self.X_test[:,-1])
               
        lin_test_paths = np.asarray([self.lin_path[self.lin_path[:, -1] == self.find_nearest(self.lin_path[:, -1], path[-1])][0] for path in self.X_test]) # get linear paths
                    
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
        
        self.v1_pred = test_red
        
        return test_red
    
    def train_ctax_MLR(self):
        """
        Multivariate Linear Regression using sklearn
        """    
        self.method = 'Multivariate linear regression'        

        self.mlr = LinearRegression()
        
        self.mlr.fit(self.X_train, self.y_train)
        
        pred = self.mlr.predict(self.X_test)
        
        self.pred['MLR'] = pred
        
        return pred

    def train_ctax_PR(self, degree):
        """
        Multivariate Logistic Regression
        
        using sklearn
        """
        self.method = 'Polynomial regression'        
                    
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(self.X_train)
        x_poly_test = poly.fit_transform(self.X_test)
        
        self.pr = LinearRegression()
        
        self.pr.fit(x_poly, self.y_train)
        
        pred = self.pr.predict(x_poly_test)
        
        self.pred['PR'] = pred
        
        return pred

    def train_ctax_ridge(self, alpha):
        """
        Ridge regression
        """
        self.method = 'Ridge regression'        

        self.ridge = Ridge(alpha=alpha)
        
        self.ridge.fit(self.X_train, self.y_train)
        
        pred = self.ridge.predict(self.X_test)
        
        self.pred['Ridge'] = pred
        
        return pred
    
    def train_ctax_lasso(self, alpha):
        """
        Lasso regression
        """
        self.method = 'Lasso regression'        

        self.lasso = Lasso(alpha=alpha)
        
        self.lasso.fit(self.X_train, self.y_train)
        
        pred = self.lasso.predict(self.X_test)
        
        self.pred['Lasso'] = pred
        
        return pred
            
    def train_ctax_tree(self, max_depth):
        """
        Regression trees
        """
        
        # bootstrap methods, dataset opsplitsen zodat je ook test met je testsets
        self.method = 'Regression tree'
        
        self.tree = tree.DecisionTreeRegressor(random_state=0, max_depth=max_depth)
        
        self.tree.fit(self.X_train, self.y_train)
        
#        tree.plot_tree(clf)
        
#        dot_data = tree.export_graphviz(clf, out_file=None)
#        graph = graphviz.Source(dot_data)
#        graph.render('ctax')
        
        pred = self.tree.predict(self.X_test)
        
        self.pred['Tree'] = pred
        
        return pred
    
    def train_ctax_forest(self, max_depth):
        """
        Random forest
        """
        
        # bootstrap methods, dataset opsplitsen zodat je ook test met je testsets
        self.method = 'Regression forest'
                
        parameters = { 
                'n_estimators': [100, 200, 300],
                'max_features': ['auto', 'sqrt', 'log2']
                }   
        
        self.forest = GridSearchCV(RandomForestRegressor(), parameters, cv=5, verbose=1, n_jobs=-1)
        
        self.forest.fit(self.X_train, self.y_train)
        print('best params: ', self.forest_grid.best_params_)

        pred = self.forest_grid.predict(self.X_test) 
        
        self.pred['Forest'] = pred
                
        return pred
    
    def train_SVM(self):
        """
        Support Vector Machine Regression
        """
        self.method = 'SVM'
        
        parameters = {
            "kernel": ["rbf"],
            "C": [10, 100, 1000],
            "gamma": [1e-6, 1e-5, 1e-4]
            }
        
        self.svm = GridSearchCV(SVR(), parameters, cv=5, verbose=1, n_jobs=-1)
        self.svm.fit(self.X_train, self.y_train)
        print('best params: ', self.svm.best_params_)
        pred = self.svm.predict(self.X_test)
        
        self.pred['SVM'] = pred
        
        return pred
    
    def train_MLPRegressor(self):
        
        self.method = 'MLPRegressor'
        
        parameters = {"hidden_layer_sizes": [(50,),(100,),(500,)],
                                             "alpha": [0.00005,0.0005],
                                             "max_iter":[10000]}
#                                             "solver":['lbfgs']}

        grid = GridSearchCV(MLPRegressor(), parameters, cv=3, verbose=1, n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        print('best params: ', grid.best_params_)
        pred = grid.predict(self.X_test)
        
        self.pred['MLP'] = pred
        
        return pred
    
    def train_TF(self):
        
        self.method = 'TensorFlow'
        
        print(self.X_train)
        
        model = keras.Sequential([
                self.X_train,
                layers.Dense(64, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(1)
        ])
                
        model.summary()
                
        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.001))
        
        history = model.fit(self.X_train)
    
    def test_regr(self, pred):
        """
        Method to test the multivariate regressions
        """
        sorted_y_test = np.sort(self.y_test)         
        sorted_pred = np.sort(pred)
                
        test_set_rmse = (np.sqrt(mean_squared_error(sorted_y_test, sorted_pred)))
        
        test_set_r2 = r2_score(sorted_y_test, sorted_pred)
        
        print('\033[1m' + 'method: ' + '\033[0m', self.method, '\n', 'RMSE: ', test_set_rmse, '\n', 'R-squared: ', test_set_r2)
    
    def scatter_and_mac(self, pred):
        """
        scatterplot with on x-axes true reduction and yaxes emulated reduction and a MAC curve 
        """       
        sorted_y_test = np.sort(self.y_test)         
        sorted_pred = np.sort(pred)
        
        final_ctax = [x_test[-1] for x_test in self.X_test]
    
        p1 = max(max(sorted_y_test), max(sorted_pred))
        p2 = min(min(sorted_y_test), min(sorted_pred))
        
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].scatter(sorted_y_test, sorted_pred, c='crimson', s=10)
        ax[0].plot([p1, p2], [p1, p2], 'b--')
        ax[0].set_ylabel('Predicted reduction [%]')
        ax[0].set_xlabel('True reduction [%]')
        ax[0].grid()
        ax[0].set_xlim([p2-5, p1+5])

        ax[1].scatter(pred, final_ctax, label='predicted', s=5)
        ax[1].scatter(self.y_test, final_ctax, label='true', s=5)
        ax[1].set_ylabel('Final ctax [USD/tCO2]')
        ax[1].set_xlabel('Reduction [%]')
        ax[1].legend()
        ax[1].grid()

    def calc_miti_costs(self, method, region, step_ctax, emissions, baseline, timer_data, colormap=None):
        """
        scale ctax path
        construct MAC
        calculate area under MAC
        """    
        ctax_path = timer_data.iloc[np.random.randint(0, high=900)].drop(['reduction'])
        ctax_path = np.asarray(ctax_path)
        max_ctax = ctax_path.max()
        norm_ctax = ctax_path / max_ctax
        ctaxes_for_scale = [i for i in range(step_ctax, int(max_ctax) + step_ctax, step_ctax)]
        
#        print(norm_ctax, final_ctax, ctaxes_for_scale)
                        
        scaled_ctaxes = []
        
        for ctax in ctaxes_for_scale:
            scaled_ctaxes.append(norm_ctax * ctax)
            
        scaled_ctaxes = np.asarray(scaled_ctaxes)
        scaled_ctaxes = np.vstack(scaled_ctaxes)
        mask = np.all(np.isnan(scaled_ctaxes), axis=1)
        scaled_ctaxes = scaled_ctaxes[~mask]
        
        for scaled_ctax in scaled_ctaxes:
            plt.plot(timer_data.columns[:-1], scaled_ctax)
        
        # now use emulator
        emu_reductions = self.grid_svm.predict(scaled_ctaxes)
        
        if self.year != 2100:
            miti_year = self.year + 1
        else:
            miti_year = self.year
        
        baseline = float(baseline.loc[baseline.region == self.region][miti_year].values)
        abs_emissions = [(emu_reduction/100) * baseline for emu_reduction in emu_reductions]
        prices = ctaxes_for_scale 
        
        costs = np.trapz(prices, x=abs_emissions) * 0.001  # kg to tonnes
        print('costs : ', '{:e}'.format(costs))
        
        cmap = plt.get_cmap(colormap)
        new_cmap = self.truncate_colormap(cmap, 0.4, 0.8)
        
        plt.figure()
        plt.plot(emu_reductions, prices)
        plt.grid(True)
        plt.scatter(emu_reductions, prices, c=emu_reductions, cmap=new_cmap, zorder=10)
        plt.xlabel('reduction [%]')
        plt.ylabel('ctax (final value) [USD/tCO2]')
        plt.title(f'MAC curve for region {timer_data.region} in {timer_data.year}')
        
        return costs
        
    def calc_miti_timer(self, ctax_paths, timer_data, baseline):
        
        df_max = ctax_paths.max(axis=1)
        norm_df = ctax_paths.divide(df_max, axis=0)
        norm_df = norm_df.round(decimals=3)
        indices_paths = norm_df.drop_duplicates(keep='first').index
        
        if self.year != 2100:
            miti_year = self.year + 1
        else:
            miti_year = self.year
        
        splitted_df = np.array_split(timer_data, indices_paths[1:])
        baseline = float(baseline.loc[baseline.region == self.region][miti_year].values)        
        self.miti_timer = []
                
        for df in splitted_df:
            abs_emissions = [(reduction/100) * baseline for reduction in df['reduction']]
            prices = df[str(self.year)]
            costs = np.trapz(prices, x=abs_emissions) * 0.001
            self.miti_timer.append(costs)
                 
        return self.miti_timer
     
    def calc_miti_emu(self, ctax_paths, timer_data, baseline):
        
        df_max = ctax_paths.max(axis=1)
        norm_df = ctax_paths.divide(df_max, axis=0)
        norm_df = norm_df.round(decimals=3)
        indices_paths = norm_df.drop_duplicates(keep='first').index
        
        if self.year != 2100:
            miti_year = self.year + 1
        else:
            miti_year = self.year
        
        splitted_df = np.array_split(timer_data, indices_paths[1:])
        self.splitted = splitted_df
        baseline = float(baseline.loc[baseline.region == self.region][miti_year].values)        
        self.miti_emu = []
        self.emu_reductions = []
                
        for df in splitted_df:
            paths = df.drop('reduction', axis=1)
#            reductions = self.lin_regr_mod.predict(df)
            reductions = self.svm.predict(paths)
            paths['emu reduction'] = reductions
            self.emu_reductions.append(paths)
            abs_emissions = [(reduction/100) * baseline for reduction in reductions]
            prices = df[str(self.year)]
            costs = np.trapz(prices, x=abs_emissions) * 0.001
            self.miti_emu.append(costs)
                 
        return self.miti_emu
    
    def plot_timer_vs_emu(self):
                        
        p1 = max(max(self.miti_timer), max(self.miti_emu))
        p2 = min(min(self.miti_timer), min(self.miti_emu))
        
        plt.figure()
        plt.plot(self.miti_timer, label='TIMER')
        plt.plot(self.miti_emu, label='emulated')
        plt.ylabel('costs')
        plt.legend()
        plt.figure()
        plt.scatter(self.miti_timer, self.miti_emu, c='crimson', s=10)
        plt.ylabel('emulated costs (predicted) [USD]')
        plt.xlabel('TIMER costs (true) [USD]')
        plt.plot([p1, p2], [p1, p2], 'b--')
        plt.grid()
    
    @staticmethod    
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap 

    def subplot_results(self, list_of_preds):
        """
        plot all tested fits in a figure with subplots
        """
        plt.figure(1, figsize=(100, 60), dpi=180)
        
        for index, pred in enumerate(list_of_preds):
    
            rows = int(math.ceil(len(list_of_preds)/2))
            plt.subplot(rows,2,index + 1)
            pred.T.plot(legend=False, ax=plt.gca(), sharex='year', sharey='ctax [USD]', title=self.method, grid=True,
                       figsize=(10,15), ylabel='ctax [USD]', xlabel='year')
                
        
        
    
        
        
        
        