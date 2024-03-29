#!/usr/bin/env python
# coding: utf-8

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import math
import pylab

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
    def __init__(self, year, region, df_lin, df_train, df_train_tot, ctax_paths, test_size):
        
        self.year = year
        self.region = region
        self.columns = df_train.columns
        self.years = list(self.columns[:-1].values)
        
        self.lin_path = np.asarray(df_lin.drop(['reduction'], axis = 1))
        self.lin_reduction = np.asarray(df_lin['reduction'])
         
        # mss nog even checken
        df_train_tot = df_train_tot.drop_duplicates(subset=self.years)
        df_train_tot = df_train_tot.reset_index(drop=True)
        df_train = df_train.drop_duplicates(subset=self.years)
        df_train = df_train.reset_index(drop=True)
        ctax_paths = ctax_paths.drop_duplicates(subset=self.years)
        ctax_paths = ctax_paths.reset_index(drop=True)
        self.df_train_tot = df_train_tot
        self.df_lin = df_lin
        self.ctax_paths = ctax_paths
        
        test_indices = []
        
        for path_type in ctax_paths['type'].unique():
            df_type = ctax_paths.loc[ctax_paths['type'] == path_type]
            count_values = len(df_type)
            test_values = int(count_values * 0.1)  #0.1 for 10% test values per type
            test_indices.append(df_type.sample(n=test_values).index.values)

        test_indices = np.concatenate(test_indices)
        col_types = ctax_paths['type'][test_indices]
        
        self.df_test = pd.DataFrame(df_train_tot.iloc[test_indices])
        self.df_test['type'] = col_types.values
        self.df_train = self.df_train_tot.drop(df_train_tot.index[test_indices])
        self.pred = pd.DataFrame(index=pd.Index(test_indices))
                            
        self.train_path = df_train.drop(['reduction'], axis = 1)
        self.train_reduction = df_train['reduction']
                                        
        self.df_combined_lin = df_lin        
        self.df_combined_train = df_train_tot.sort_values(str(year)).reset_index(drop=True)
        
        self.weights = pd.DataFrame()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.train_path, self.train_reduction,
                                                                                test_size = test_size, random_state=9)
        
        self.train_path = np.asarray(self.train_path)
        self.train_reduction = np.asarray(self.train_reduction)
        
        self.y_test_indices = self.y_test
        self.X_test_indices = self.X_test
        self.y_train_indices = self.y_train
        self.x_train_indices = self.X_train
        
        self.X_train = np.asarray(self.X_train)
        self.X_test = np.asarray(self.X_test)
        self.y_train = np.asarray(self.y_train)
        self.y_test = np.asarray(self.y_test)
        
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
           
            if len(delta_c_norm[0]) % self.count_weights == 0: 
                delta_c_slice = [np.mean(delta_c.reshape(-1,  self.count_weights), axis=1) for delta_c in delta_c_step.values]
            else:
                delta_c_slice = [np.mean(delta_c[1:].reshape(-1,  self.count_weights), axis=1) for delta_c in delta_c_step.values]
            
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
        self.X_test = self.df_test.drop(['reduction', 'type'], axis=1).values
        self.X_test[:,-1] = np.round(self.X_test[:,-1])
               
        lin_test_paths = np.asarray([self.lin_path[self.lin_path[:, -1] == self.find_nearest(self.lin_path[:, -1], path[-1])][0] for path in self.X_test]) # get linear paths
        df_lin_test = pd.DataFrame(lin_test_paths, columns=self.years)
        lin_reductions = pd.merge(df_lin_test, self.df_lin, on=self.years, how='inner')
        lin_reductions = lin_reductions['reduction']
        
        delta_cs = lin_test_paths - self.X_test 
        final_ctax = lin_test_paths[:, -1]
        delta_c_norm = delta_cs / final_ctax[:, None]
                
        if len(delta_c_norm[0]) % self.count_weights == 0: 
            delta_c_slice = [np.mean(delta_c.reshape(-1,  self.count_weights), axis=1) for delta_c in delta_c_norm]
        else:
            delta_c_slice = [np.mean(delta_c[1:].reshape(-1,  self.count_weights), axis=1) for delta_c in delta_c_norm]
                  
        # multiply delta C with the weights to find reduction              
        cur_weights = [self.weights.iloc[(self.weights['ctax'] - ctax).abs().argsort()[:1].values] for ctax in self.X_test[:, -1]]
        clean_weights = [weights.values[0] for weights in cur_weights]
        weights_columns = [column for column in range(self.number_of_weights)]
        weights_columns.append('ctax')
        weights_df = pd.DataFrame(clean_weights, columns=weights_columns)          
                
        b = weights_df.drop(['ctax'], axis=1).values   
        
        # calculate the reduction
#        test_red = [lin_reductions[i] - (delta_c_slice[i] @ b[i]) for i in range(len(self.df_test))]
        
        test_red = []
        for i in range(len(lin_reductions)):
            if self.X_test[i, -1] <= lin_test_paths[i, -1]:
                red = lin_reductions[i] - (delta_c_slice[i] @ b[i]) 
                test_red.append(red)
            else:
                red = lin_reductions[i] + (delta_c_slice[i] @ b[i]) 
                test_red.append(red)

        self.v1_pred = test_red
        
        self.pred['V1'] = self.v1_pred
        
        return test_red
    
    def train_ctax_MLR(self):
        """
        Multivariate Linear Regression using sklearn
        """    
        self.method = 'MLR'        

        self.mlr = LinearRegression()
        
        self.mlr.fit(self.X_train, self.y_train)
        
        pred = self.mlr.predict(self.X_test)
        test_pred = self.mlr.predict(self.df_test.drop(['reduction', 'type'], axis=1))
        
        self.pred[self.method] = test_pred
        
        return pred

    def train_ctax_PR(self, degree):
        """
        Multivariate Logistic Regression
        
        using sklearn
        """
        self.method = 'PR'        
                    
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(self.X_train)
        x_poly_test = poly.fit_transform(self.df_test.drop(['reduction', 'type'], axis=1))
        
        self.pr = LinearRegression()
        
        self.pr.fit(x_poly, self.y_train)
        
        pred = self.pr.predict(x_poly_test)
        test_pred = self.pr.predict(x_poly_test)
        
        self.pred[self.method] = test_pred
        
        return pred

    def train_ctax_ridge(self, alpha):
        """
        Ridge regression
        """
        self.method = 'RR'        

        parameters = {'alpha':[1, 10]}
        
        self.ridge = GridSearchCV(Ridge(), parameters, scoring='neg_mean_squared_error',cv=5)
        
        self.ridge.fit(self.X_train, self.y_train)
        
        pred = self.ridge.predict(self.X_test)
        test_pred = self.ridge.predict(self.df_test.drop(['reduction', 'type'], axis=1))
        
        self.pred[self.method] = test_pred

        print(self.method, 'best params: ', self.ridge.best_params_)
        
        return pred
    
    def train_ctax_lasso(self, alpha):
        """
        Lasso regression
        """
        self.method = 'LR'        

        self.lasso = Lasso(alpha=alpha)
        
        self.lasso.fit(self.X_train, self.y_train)
        
        pred = self.lasso.predict(self.X_test)
        test_pred = self.lasso.predict(self.df_test.drop(['reduction', 'type'], axis=1))
        
        self.pred[self.method] = test_pred
        
        return pred
            
    def train_ctax_tree(self, max_depth):
        """
        Regression trees
        """
        self.method = 'RT'

        parameters = { 
        'max_depth': np.arange(3, 10),
        }
        
        self.tree = GridSearchCV(tree.DecisionTreeRegressor(), parameters)
        
        self.tree.fit(self.X_train, self.y_train)
        print(self.method, 'best params: ', self.tree.best_params_)
        
#        tree.plot_tree(self.tree)
#        plt.show()
        
#        dot_data = tree.export_graphviz(clf, out_file=None)
#        graph = graphviz.Source(dot_data)
#        graph.render('ctax')
        
        pred = self.tree.predict(self.X_test)
        test_pred = self.tree.predict(self.df_test.drop(['reduction', 'type'], axis=1))
        
        self.pred[self.method] = test_pred
        
        return pred
    
    def train_ctax_forest(self, max_depth):
        """
        Random forest
        """
        
        # bootstrap methods, dataset opsplitsen zodat je ook test met je testsets
        self.method = 'RF'
                
        parameters = { 
                'n_estimators': [200, 300, 400, 500],
                'max_features': ['auto', 'sqrt', 'log2']
                }   
        
        self.forest = GridSearchCV(RandomForestRegressor(), parameters, cv=5, verbose=1, n_jobs=-1)
        
        self.forest.fit(self.X_train, self.y_train)
        print(self.method, 'best params: ', self.forest.best_params_)

        pred = self.forest.predict(self.X_test)
        test_pred = self.forest.predict(self.df_test.drop(['reduction', 'type'], axis=1))
        
        self.pred[self.method] = test_pred
                
        return pred
    
    def train_SVM(self):
        """
        Support Vector Machine Regression
        """
        self.method = 'SVM'
        
        parameters = {
            "kernel": ["rbf"],
            "C": [10, 100, 500],
            "gamma": [1e-6, 1e-5, 1e-4]
            }
        
        self.svm = GridSearchCV(SVR(), parameters, cv=5, verbose=1, n_jobs=-1)
        self.svm.fit(self.X_train, self.y_train)
        print(self.method, 'best params: ', self.svm.best_params_)
        pred = self.svm.predict(self.X_test)
        test_pred = self.svm.predict(self.df_test.drop(['reduction', 'type'], axis=1))
        
        self.pred[self.method] = test_pred
        
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
        
        self.pred[self.method] = pred
        
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
    
    def score(self):
        """
        Method to test the multivariate regressions
        """
        score = pd.DataFrame(columns=self.pred.columns)
        
        for col in self.pred.columns:
                        
            score.loc[0, col] = (np.sqrt(mean_squared_error(self.df_test['reduction'], self.pred[col])))
        
            score.loc[1, col] = r2_score(self.df_test['reduction'], self.pred[col])
            
        score['type'] = ['RMSE', 'R-squared']
        score = score.set_index('type', drop=True)
        
        return score
    
    def pred_vs_true(self):
        """
        plot predicted versus true outcomes
        """
        
        grid_fig_x = 1
        grid_fig_y = math.ceil(len(self.pred.columns) / grid_fig_x)
        fig  = plt.figure(figsize=(11,3.7)) 
        fig.text(0.5, 0.03, 'True reduction [%]', ha='center')
        fig.text(0.07, 0.5, 'Predicted reduction [%]', va='center', rotation='vertical')
        c= {'linear': '#FBD1A2',
            'cubic': '#7DCFB6',
            'cubicroot': '#00B2CA',
            'capped linear': '#1D4E89',
            'scaled IAMC': '#F79256',
            'random': '#571F4E',
            'capped cubic': '#5D5179',
            'capped cubicroot': '#4F759B'}
        
        marker = {'linear': 'X',
                'cubic': 'X',
                'cubicroot': 'X',
                'capped linear': 'o',
                'scaled IAMC': '^',
                'random': 'D',
                'capped cubic': 'o',
                'capped cubicroot': 'o'}
        
        for index, col in enumerate(self.pred.columns):
            
            p1 = max(max(self.y_test), max(self.pred[col]))
            p2 = min(min(self.y_test), min(self.pred[col]))
                                    
            ax = plt.subplot(grid_fig_x, grid_fig_y, index + 1)
            for path_type in self.ctax_paths['type'].unique():
                ix = self.df_test.loc[self.df_test['type'] == path_type].index
                ax.scatter(self.df_test['reduction'][ix], self.pred[col][ix], c=c[path_type], label=path_type,
                           s=10, alpha=0.9, marker=marker[path_type])
            ax.plot([p1, p2], [p1, p2], color='0.6', linestyle='dashed')
            ax.title.set_text(col)
            ax.grid()
            handles, labels = ax.get_legend_handles_labels()
        
        plt.figure(figsize=(3,2))
        plt.legend(handles, labels, frameon=False, framealpha=1, markerscale=2)
        plt.axis('off')

        def export_legend(legend, filename="legend.png"):
            fig  = legend.figure
            fig.canvas.draw()
            bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(filename, dpi="figure", bbox_inches=bbox)
        
#        export_legend(legend)
        plt.show()
            
    def check_difference(self):
        """
        find where the large differences are
        """
        
        difference_SVM_test = abs(self.pred['SVM'] - self.y_test)
        difference_SVM_test = difference_SVM_test.loc[difference_SVM_test > 40]
        X_test_large_difference = self.X_test[difference_SVM_test.index.values]
        
        print(difference_SVM_test.index.values)

        xaxes = list(map(int, self.columns[:-1]))
        plt.figure()
#        for path in X_test_large_difference:         
#            plt.plot(xaxes, path)
#            plt.xlabel('years')
#            plt.ylabel('ctax')
#            plt.grid()
            
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        for index, reduction in enumerate(difference_SVM_test):    
            ax[0].scatter(self.y_test[difference_SVM_test.index.values[index]], reduction, s=10,
                          label=index)
        ax[0].set_ylabel('Predicted reduction [%]')
        ax[0].set_xlabel('True reduction [%]')
        ax[0].grid()
        ax[0].legend()

        for index, path in enumerate(X_test_large_difference):         
            ax[1].plot(xaxes, path, label=index)
        ax[1].set_ylabel('ctax [USD/tCO2]')
        ax[1].set_xlabel('year')
        ax[1].legend()
        ax[1].grid()
            
        print(self.df_train.columns.values)    
        paths_high_2040 = self.df_train[(self.df_train['2020'] < 50) & (self.df_train['2030'] < 150) & 
                                        (self.df_train['2040'] < 250)]
        print(len(paths_high_2040))
        
    def scatter_and_mac(self, pred):
        """
        scatterplot with on x-axes true reduction and yaxes emulated reduction and a MAC curve 
        """       
       
        final_ctax = [x_test[-1] for x_test in self.X_test]
        path_type = self.ctax_paths.iloc[self.X_test_indices.index.values]['type']
#        cdict = {key: value for key, value in enumerate(self.df_test['type'].unique())}
    
        p1 = max(max(self.y_test), max(pred))
        p2 = min(min(self.y_test), min(pred))
        
        fig, ax = plt.subplots(1,2, figsize=(10,4))
    
        ax[0].scatter(self.y_test, pred, c='crimson', s=10, label=path_type)
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
            reductions = self.svm.predict(paths)
            paths['emu reduction'] = reductions
            self.emu_reductions.append(paths)
            abs_emissions = [(reduction/100) * baseline for reduction in reductions]
            prices = df[str(self.year)]
            costs = np.trapz(prices, x=abs_emissions) * 0.001
            self.miti_emu.append(costs)
                 
        return self.miti_emu
    
    def calc_miti_emu_2(self, baseline, step_ctax):
                
        paths_only = self.df_test[self.years].values
        emu_costs = []
        
        for path in paths_only:
            norm_ctax = path / path.max()
            ctaxes_for_scale = [i for i in range(step_ctax, int(path.max()) + step_ctax, step_ctax)]
            paths_for_mac = [norm_ctax * scaled_ctax for scaled_ctax in ctaxes_for_scale]
            reductions = self.mlr.predict(paths_for_mac)
            abs_emissions = [(reduction/100) * baseline for reduction in reductions]
            prices = ctaxes_for_scale
            emu_costs.append(np.trapz(prices, x=abs_emissions) * 0.001)
        
        self.emu_costs = emu_costs
#        ctax_path = np.asarray(paths_only)
#        max_ctax = ctax_path.max()
#        norm_ctax = ctax_path / max_ctax
#        print(len(norm_ctax))
#        ctaxes_for_scale = [i for i in range(step_ctax, int(max_ctax) + step_ctax, step_ctax)]
#        
#        print(max_ctax)
#                        
#        scaled_ctaxes = []
#        
#        for ctax in ctaxes_for_scale:
#            scaled_ctaxes.append(norm_ctax * ctax)
#            
#        scaled_ctaxes = np.asarray(scaled_ctaxes)
#        scaled_ctaxes = np.concatenate(scaled_ctaxes)
#        mask = np.all(np.isnan(scaled_ctaxes), axis=1)
#        scaled_ctaxes = scaled_ctaxes[~mask]
#        
#        print(scaled_ctaxes)
#                
#        self.miti_emu = []
#        self.emu_reductions = []
#                
#        for df in splitted_df:
#            paths = df.drop('reduction', axis=1)
#            print(paths)
#            reductions = self.mlr.predict(paths)
##            reductions = self.svm.predict(paths)
#            paths['emu reduction'] = reductions
#            self.emu_reductions.append(paths)
#            abs_emissions = [(reduction/100) * baseline for reduction in reductions]
#            prices = df[str(self.year)]
#            costs = np.trapz(prices, x=abs_emissions) * 0.001
#            self.miti_emu.append(costs)
#                 
#        return self.miti_emu
    
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
                
        
        
    
        
        
        
        