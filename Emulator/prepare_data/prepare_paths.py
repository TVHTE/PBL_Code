# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:41:38 2021

@author: toonv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors


from pym import pym

def combine_azure_ctax(year, region, ctax_paths, emissions, baseline):
    """
    combine azure reduction output with ctax paths input
    based on year and region
    """
        
    if year != 2100:
        year = year + 1
                        
    abs_emission = np.array(emissions.loc[emissions.region == region][year].values).astype(float)
    baseline = float(baseline.loc[baseline.region == region][year].values)
            
    reduction = 100 - (abs_emission / baseline) * 100
    ctax_index = emissions.loc[emissions.region == region]['ctax_index']
    
    combined = pd.DataFrame()
    combined['ctax_index'] = ctax_index
    combined['reduction'] = reduction
    
    ctax_paths.index.name = 'ctax_index'    
    cur_ctax_df = pd.merge(ctax_paths, combined, on=['ctax_index'])
    
    emulator_data = cur_ctax_df.drop(columns=['ctax_index', 'type'], errors='ignore')
    columns = emulator_data.columns
    columns = [column for column in columns[:-1] if int(column) <= year]
    columns.append('reduction')
        
    emulator_data = emulator_data[emulator_data.columns.intersection(columns)]
            
    if year != 2100:
        emulator_data.year = year - 1
    else:
        emulator_data.year = year
    
    emulator_data.region = region
    
    return emulator_data

def world_MAC_data(year, ctax_paths, emissions, world_baseline):
    """
    combine azure reduction output with ctax paths input
    based on year and region
    """
        
    world_emissions = np.array([emissions.loc[emissions.ctax_index == i][year].sum() for i in emissions.ctax_index.unique()])
        
    world_reduction = (world_emissions / world_baseline) * 100
            
    ctax_index = [ctax for ctax in range(11)]
    combined_world = pd.DataFrame()
    combined_world['ctax_index'] = ctax_index
    combined_world['reduction'] = world_reduction
        
    costs = np.trapz(ctax_paths[str(year)].values, x=world_emissions) * -0.001 # 0.001 is for kg to tonnes
    print('costs: ', '{:e}'.format(costs))
    
    ctax_paths.index.name = 'ctax_index'    
    ctax_world = pd.merge(ctax_paths, combined_world, on=['ctax_index'])
    ctax_world = ctax_world.drop(['ctax_index'], axis=1)    

    if year != 2100:
        ctax_world.year = year - 1
    else:
        ctax_world.year = year    
    
    ctax_world.region = 27
    
    return ctax_world

def output_costs_timer(t_system_cost, t_system_cost_rel, year, region, ctax_paths, baseline):
    """
    overview of costs output variables from TIMER
    """
    
    costs_dfs = [t_system_cost, t_system_cost_rel]
    
    world_costs = t_system_cost.loc[t_system_cost.region == region][str(year)]
    
#    print(t_system_cost.loc[t_system_cost.region == region][str(year)], baseline.loc[baseline.ctax_index == 0][str(year)])
    
    compared_to_baseline = np.array(world_costs) - np.array(baseline.loc[baseline.ctax_index == 0][str(year)])
        
    variables_combined = pd.DataFrame()
         
    for index, costs in enumerate(costs_dfs):
        world_costs = costs.loc[costs.region == region][str(year)].values
        variables_combined[index] = world_costs
                
    variables_combined.columns = ['t_system_cost [USD]', 't_system_cost_rel [% of GDP]']
    end_ctax = ctax_paths[str(year)].values
    variables_combined['final_c_price'] = end_ctax
    variables_combined['mitigation costs [USD]'] = compared_to_baseline
    variables_combined = variables_combined.set_index('final_c_price')
    variables_combined = variables_combined.round(3)
    
    return variables_combined

def plot_MAC(emulator_data, label, colormap=None):
    """
    plot MAC curves
    """
    year = emulator_data.year
        
    price = emulator_data[str(year)]
        
    reduction = emulator_data.reduction
    
    cmap = plt.get_cmap(colormap)
    new_cmap = truncate_colormap(cmap, 0.4, 0.8)
    
    plt.plot(reduction, price, label=label)
    plt.grid(True)
    plt.scatter(reduction, price, c=reduction, cmap=new_cmap, zorder=10)
    plt.xlabel('reduction [%]')
    plt.ylabel('ctax (final value) [USD/tCO2]')
    plt.legend()
    plt.title(f'MAC curve for region {emulator_data.region} in {emulator_data.year}')
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap 
    
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

        self.scaled_ctax_paths.method = 'scaled IAMC'        
                
        return self.scaled_ctax_paths
        
    def get_linear(self, max_ctax, step_ctax):
        """
        get the linear paths that come with the final ctaxes of the scaled ctax paths
        """
        path = []
        num = len(self.years)
        lin_ctaxes = range(0, max_ctax + step_ctax, step_ctax)
        
        for ctax in lin_ctaxes:
            path.append(np.linspace(0, ctax, num=num))
                    
        self.lin_ctax_paths = pd.DataFrame(path, columns=self.years)
        
        self.lin_ctax_paths.method = 'linear'
        
        return self.lin_ctax_paths
    
    def get_tree_costs(self, max_ctax, step_ctax):
        
        tree_paths = []
        num = len(self.years)
        ctaxes = range(0, max_ctax + step_ctax, step_ctax)
        
        for ctax in ctaxes:
            lin_ctax = np.linspace(0, max_ctax, num=num)
            lin_ctax[-1] = ctax    
            tree_paths.append(lin_ctax)
        
        self.tree_paths = pd.DataFrame(tree_paths, columns=self.years)
                        
        return self.tree_paths
    
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
        
        self.cubic_paths.method = 'cubic'
                
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
         
        self.cubicroot_paths.method = 'cubicroot'
        
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
        
        self.sparse_cubicroot_paths.method = 'capped cubicroot'
                
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
        
        self.sparse_cubic_paths.method = 'capped cubic'
        
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
        
        self.sparse_ctax_paths.method = 'capped linear'
                
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
        
        self.scaled_random_paths.method = 'scaled random'
        
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
        
#        ctax_paths.to_csv(path_or_buf = path_csv + 'tree_pahts.csv')
        
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
        
    def plot_ctax_paths(self, ctax_paths, colormap):
        
        cmap = plt.get_cmap(colormap)
        new_cmap = truncate_colormap(cmap, 0.4, 1)        
                
        plot_prices = ctax_paths[self.years]  
        plot_prices.T.plot(legend=False, colormap=new_cmap, grid=True, xlabel='year' , ylabel='ctax (final value) [USD/tCO2]')
        

  
        
        

                
             