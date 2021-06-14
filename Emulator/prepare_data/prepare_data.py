# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:41:38 2021

@author: toonv
"""

import pandas as pd
import numpy as np
from pym import pym
import os

def combine_azure_ctax(year, region, ctax_paths, reductions):
    """
    combine azure reduction output with ctax paths input
    based on year and region
    """
    
    if year != 2100:
        year = year + 1
    
    reduction_indices = reductions.loc[reductions.region == region][year].index
    cur_reduction = reductions.loc[reduction_indices][year]
    baseline_reduction = reductions.loc[0][year]
    percentage_reduction = (((cur_reduction - baseline_reduction) * -1) / baseline_reduction) * 100
    cur_reduction = percentage_reduction
    ctax_index = reductions.loc[reduction_indices]['ctax_index']
    
    combined = pd.DataFrame()
    combined['ctax_index'] = ctax_index
    combined['reduction'] = cur_reduction
    
    ctax_paths.index.name = 'ctax_index'    
    cur_ctax_df = pd.merge(ctax_paths, combined, on=['ctax_index'])
    
    data_for_emulator = cur_ctax_df.drop(columns=['ctax_index', 'type'])
    columns = data_for_emulator.columns
    columns = [column for column in columns[:-1] if int(column) <= year]
    columns.append('reduction')
    
    data_for_emulator = data_for_emulator[data_for_emulator.columns.intersection(columns)]
    
    return data_for_emulator
    
class prepare_data:
    
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
        self.filtered = filtered
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
        
        self.final_ctax = self.values_only.max(axis=1).values
        norm_ctax = self.values_only.values / self.final_ctax[:, None]
        ctaxes_for_scale = [i for i in range(step_ctax, self.max_ctax + step_ctax, step_ctax)]
                
        scaled_ctaxes = []
        
        for ctax in ctaxes_for_scale:
            scaled_ctaxes.append(norm_ctax * ctax)

        scaled_ctaxes = np.asarray(scaled_ctaxes)
        scaled_ctaxes = np.vstack(scaled_ctaxes)
        mask = np.all(np.isnan(scaled_ctaxes), axis=1)
        scaled_ctaxes = scaled_ctaxes[~mask]
        self.scaled_ctax_paths = pd.DataFrame(scaled_ctaxes, columns=self.years)
        self.scaled_ctax_paths = self.scaled_ctax_paths.drop_duplicates()
                
        return self.scaled_ctax_paths
        
    def get_linear(self, max_ctax):
        """
        get the linear paths that come with the final ctaxes of the scaled ctax paths
        """
        path = []
        num = len(self.years)
        lin_ctaxes = range(0, max_ctax + 40, 40)
        
        for ctax in lin_ctaxes:
            path.append(np.linspace(0, ctax, num=num))
        
        self.lin_ctax_paths = pd.DataFrame(path, columns=self.years)
                
        return self.lin_ctax_paths
    
    def get_cubic(self, max_ctax):
        """
        generate cubic paths
        """
        paths = []
        num = len(self.years)
        ctaxes = range(0, max_ctax+40, 40)
        
        for ctax in ctaxes:
            a = ctax/((num - 1)**(3))
            
            path =[] 
            for step in range(0, num): 
                price = a * (step**3)               
                path.append(price)
                
            paths.append(path)
        self.cubic_paths = pd.DataFrame(paths, columns=self.years)
                
        return self.cubic_paths
    
    def get_cubicroot(self, max_ctax):
        """
        generate cubicroot paths
        """
        paths = []
        num = len(self.years)
        ctaxes = range(0, max_ctax+40, 40)
        
        for ctax in ctaxes:
            a = ctax/((num - 1)**(1/3))
            path =[]
            
            for step in range(0, num): 
                price = a * (step**(1/3))               
                path.append(price)
                
            paths.append(path)
                
        self.cubicroot_paths = pd.DataFrame(paths, columns=self.years)
                
        return self.cubicroot_paths
    
    def sparse_cubicroot(self, max_ctax):

        path = []        
        nums = len(self.years)

        for num in range(1, nums):
            
            a = max_ctax/((num)**(1/3))
            first_part_path =[]
                        
            for step in range(0, num): 
                price = a * (step**(1/3))               
                first_part_path.append(price)
             
            second_part_path = np.array([max_ctax] * (nums - num))
                        
            path.append(np.concatenate([first_part_path, second_part_path]))
                 
        columns = [str(2020 + i) for i in range(0, 90, 10)]
                                                   
        self.sparse_cubicroot_paths = pd.DataFrame(path, columns=columns)
                
        return self.sparse_cubicroot_paths        
        
    def sparse_cubic(self, max_ctax):

        path = []        
        nums = len(self.years)

        for num in range(1, nums):
            
            a = max_ctax/((num)**(3))
            first_part_path =[]
                        
            for step in range(0, num): 
                price = a * (step**(3))               
                first_part_path.append(price)
             
            second_part_path = np.array([max_ctax] * (nums - num))
                        
            path.append(np.concatenate([first_part_path, second_part_path]))
                 
        columns = [str(2020 + i) for i in range(0, 90, 10)]
                                                   
        self.sparse_cubic_paths = pd.DataFrame(path, columns=columns)
                
        return self.sparse_cubic_paths 

    def sparse_linear(self, max_ctax):
        """
        get linear paths that cover the left top area of the plot (very steep ctaxes)
        """
        path = []        
        nums = len(self.years)

        for num in range(nums):
            first_part_path = np.linspace(0, max_ctax, num=num)
            second_part_path = np.array([max_ctax] * (nums - num))
                        
            path.append(np.concatenate([first_part_path, second_part_path]))
                 
        columns = [str(2020 + i) for i in range(0, 90, 10)]
                                                   
        self.sparse_ctax_paths = pd.DataFrame(path, columns=columns)
        self.sparse_ctax_paths = self.sparse_ctax_paths.iloc[1:]        
                
        return self.sparse_ctax_paths
    
    def get_random(self, max_rand, max_ctax):
        """
        use random number between 0 and 2 and multiply with the IAMC ctax paths to generate random ctax paths
        """
        random_matrix = np.random.rand(len(self.scaled_ctax_paths), len(self.years))
        random_matrix = random_matrix * max_rand
        self.scaled_random_paths = self.scaled_ctax_paths.multiply(random_matrix)
        
        for year in self.years:
            self.scaled_random_paths.drop(self.scaled_random_paths.loc[self.scaled_random_paths[year] >= max_ctax].index, inplace=True)
        
        return self.scaled_random_paths
    
    def merge_all(self, path, filename):
        """
        merge all dataframes
        """                
        self.all_paths = pd.concat([self.lin_ctax_paths, self.sparse_ctax_paths, self.scaled_ctax_paths,
                                    self.scaled_random_paths, self.sparse_cubic_paths, self.sparse_cubicroot_paths])
        self.all_paths = self.all_paths.reset_index(drop=True)
        
        self.lin_ctax_paths['type'] = 'linear'
        self.sparse_ctax_paths['type'] = 'capped linear'
        self.scaled_ctax_paths['type'] = 'scaled IAMC'         
        self.scaled_random_paths['type'] = 'scaled random'
        self.sparse_cubic_paths['type'] = 'capped cubic'
        self.sparse_cubicroot_paths['type'] = 'capped cubicroot'
        
        self.all_for_excel = pd.concat([self.lin_ctax_paths, self.sparse_ctax_paths, self.scaled_ctax_paths,
                                        self.scaled_random_paths, self.sparse_cubic_paths, self.sparse_cubicroot_paths])
        self.all_for_excel = self.all_for_excel.reset_index(drop=True)
                
        self.all_for_excel.to_excel(path + filename)
    
        return self.all_paths

    def prepare_mym(self, ctax_paths, path_mym, path_csv, filename):
        """
        Dan moet je dus nog een functie maken die van één rij uit je huidige dataframe een andere dataframe maakt, 
        nl met jaren als kolommen en 26 keer dezelfde rij onder elkaar voor de regio's (plus lege plus nog een rij dezelfde). 
        Dan kun je makkelijk loopen over de rijen (met df.iterrows) en al die dataframes los maken
        
        26 copies, 1x0, 1xzelfde
        
        mym format: (YEAR)_(ctax).dat
        """
        self.mym_ctaxes = ctax_paths.loc[ctax_paths.index.repeat(27)]
        indices = self.mym_ctaxes.index.values
        unique_indices = np.unique(indices)
        zeros = [i*0 for i in range(len(self.mym_ctaxes.columns))]
        df_zeros = pd.DataFrame(columns = self.mym_ctaxes.columns)
        df_zeros.loc[27] = zeros 
        variable_name = 'main.em.EXOCarbonTax'
        
        for index in unique_indices:
            cur_df = self.mym_ctaxes[self.mym_ctaxes.index == index].reset_index(drop=True)
            cur_df = pd.concat([cur_df.iloc[:26], df_zeros, cur_df.iloc[26:]]).reset_index(drop=True)
#            cur_df = pd.concat([cur_df.iloc[:26], df_zeros]).reset_index(drop=True)
            cur_df = cur_df.apply(pd.to_numeric)
#            cur_df.to_csv(path_or_buf = path_csv + filename + str(index) + '_scaled_' + str(int(cur_df['2100'][0].round(0))))
            pym.write_mym(cur_df, filename = filename + str(index) + '.dat', path=path_mym, variable_name = variable_name)
        
#        write .sce files to same folder
        for index in ctax_paths.index:
            
            sce_filepath = os.path.join(path_mym, f"{filename}{index}.sce")
            with open(sce_filepath, "w+") as in_file:
    
                in_file.write('DIRECTORY("../scenlib/$1/ctaxinput"); \n')
                in_file.write(f'FILE("{filename}{index}.dat", "r")         = main.em.EXOCarbonTax;') # TOON CHECK DIT
        
    def plot_ctax_paths(self, ctax_paths):
        
        plot_prices = ctax_paths[self.years]  
        plot_prices.T.plot(legend=False)
        

                
             