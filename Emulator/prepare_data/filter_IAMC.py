# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:41:38 2021

@author: toonv
"""

import pandas as pd
import numpy as np

class IAMC:
    
    def __init__(self, database):
    
        self.database = database
    
    def filter_iamc(self, year_step, variable, max_ctax, models):
        """
        get data (given the variable) from IAMC database
        
        """
        
        self.max_ctax = max_ctax
        self.database["model stripped"] = self.database["Model"].str.split(' ').str[0] # add model stripped column
        self.years = [str(2020 + i) for i in range(0, 85, year_step)]  # years to use
        self.columns = ['model stripped', 'Scenario'] + self.years  # columns to use
        self.variable = variable
        
        filtered = self.database[self.database['Variable'] == variable].reset_index(drop=True)
        filtered = filtered[self.columns]
#        filtered = filtered[filtered.loc['model stripped'].str.match([models])]
        
        selected_models = pd.DataFrame()
        
        for model in models:
            filtered = filtered.loc[filtered['model stripped'].str.match(model)]
            selected_models = pd.concat([selected_models, filtered])
                    
        if self.variable == 'Price|Carbon':
                        
            for year in self.years:
                filtered.drop(filtered.loc[filtered[year] >= max_ctax].index, inplace=True)  # remove values larger than 4000
#                filtered.drop(filtered.loc[filtered['model stripped'].str.match('C-ROADS-5.005')].index, inplace=True)  # drop specific models
            
        unique_models = self.database['model stripped'].unique()  # all models reported
        
        self.unique_models = unique_models
            
        self.values_only = filtered.drop(['model stripped', 'Scenario'], axis=1)        
        
        return filtered, unique_models
    
    def scale_ctax(self, step_ctax):
        """
        normalise the ctax paths and scale them accordingly until max ctax
        """
        
        final_ctax = self.values_only.max(axis=1).values
        norm_ctax = self.values_only.values / final_ctax[:, None]
        ctaxes_for_scale = [i for i in range(step_ctax, self.max_ctax + step_ctax, step_ctax)]
        
        scaled_ctaxes = []
        
        for ctax in ctaxes_for_scale:
            scaled_ctaxes.append(norm_ctax * ctax)

        scaled_ctaxes = np.asarray(scaled_ctaxes)
        scaled_ctaxes = np.vstack(scaled_ctaxes)
        mask = np.all(np.isnan(scaled_ctaxes), axis=1)
        scaled_ctaxes = scaled_ctaxes[~mask]
        self.scaled_ctaxes = pd.DataFrame(scaled_ctaxes, columns=self.years)
        
#        self.scaled_ctaxes.T.plot(legend=False)
        print(self.scaled_ctaxes)

    def prepare_mym(self):
        """
        Dan moet je dus nog een functie maken die van één rij uit je huidige dataframe een andere dataframe maakt, 
        nl met jaren als kolommen en 26 keer dezelfde rij onder elkaar voor de regio's (plus lege plus nog een rij dezelfde). 
        Dan kun je makkelijk loopen over de rijen (met df.iterrows) en al die dataframes los maken
        
        26 copies, 1x0, 1xzelfde
        """
        self.mym_ctaxes = self.scaled_ctaxes.loc[self.scaled_ctaxes.index.repeat(26)]
        print(self.mym_ctaxes.index.values)

    def plot_iamc(self, filtered):
        
        plot_prices = filtered[self.years]  
        plot_prices.T.plot(legend=False)
#        final_zero = filtered.loc[filtered['2100'] == 0]
#        final_zero = final_zero.drop(['model stripped', 'Scenario'], axis=1)
#        final_zero.T.plot(legend=True)
        
        self.empty_models = []  # to store empty models
                
        for model in self.unique_models:
            
             model_set = filtered[filtered['model stripped'].str.match(model)]
             ctax_only = model_set.drop(['model stripped', 'Scenario'], axis=1) 
             
             if ctax_only.empty == False: 
                 ctax_only.T.plot(legend=False, title=model)
                
             else:
                self.empty_models.append(model)
                
             