# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 10:41:14 2021

@author: toonv
"""

import zipfile
import os
import pandas as pd

PATH = 'C:/Users/toonv/Documents/PBL/Data/ctax_runs_azure'
outputlib = 'C:/Users/toonv/Documents/PBL/Data/ctax_runs_clean'

folders = sorted(os.listdir(PATH))
years = [1971, 1981, 1991, 2001, 2011, 2021, 2031, 2041, 2051, 2061, 2071, 2081, 2091, 2100]
red_df = pd.DataFrame()
total_df = pd.DataFrame() 
ctax_indices = []
all_dfs = pd.DataFrame() 
all_test = pd.DataFrame() 


for folder in folders:    
   folder_number = folder[10:20]
   print(folder_number)
   zipdata = zipfile.ZipFile(PATH + '/' + folder + f'/output_ctax_runs_SSP2_ctax_train_{folder_number}_{folder_number}.zip', 'r')
   zipinfos = zipdata.infolist()
   
   ctax_indices.append(folder_number)
   
   for zipinfo in zipinfos:
       if zipinfo.filename == f'TIMER_3_11/ctax_runs/SSP2_{folder_number}/co2tax':
           zipinfo.filename = f'co2tax_{folder_number}.dat'
   
   zipdata.extract(f'TIMER_3_11/ctax_runs/SSP2_{folder_number}/co2tax', path=outputlib)

   co2tax = pd.read_table(f'C:/Users/toonv/Documents/PBL/Data/ctax_runs_clean/co2tax_{folder_number}.dat',
                          skiprows = [0], sep = ',', names = ['reduction'], index_col=False)
  
   co2tax.loc[-1] = 1971
   co2tax.index = co2tax.index + 1
   co2tax = co2tax.sort_index()
   co2tax.iloc[-1] = 0  # without ;]
   
   year_indices = [co2tax.loc[co2tax['reduction'] == str(year)].index.values for year in years]
    
   year_indices = year_indices[1:]   
   year_indices.insert(0, [0])
   
   for count, year_index in enumerate(year_indices):
       red_df['ctax_index'] = folder_number
       year_index = year_index[0]
       column = years[count]
       reduction_10_years = co2tax[year_index:year_index+28]
       red_df[column] = reduction_10_years['reduction'][1:].values
   
#   print(red_df)
   all_dfs = all_dfs.append(red_df)

all_dfs = all_dfs.set_index(['ctax_index', all_dfs.index])
print(all_dfs)
all_dfs.to_csv('C:/Users/toonv/Documents/PBL/Data/co2tax_total.csv')

