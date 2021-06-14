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

folders = os.listdir(PATH)
years = [1971, 1981, 1991, 2001, 2011, 2021, 2031, 2041, 2051, 2061, 2071, 2081, 2091, 2100]
red_df = pd.DataFrame() 

for count, folder in enumerate(folders):    
   zipdata = zipfile.ZipFile(PATH + '/' + folder + f'/output_ctax_runs_SSP2_ctax_train_{count}_{count}.zip', 'r')
   zipinfos = zipdata.infolist()
   
   for zipinfo in zipinfos:
       if zipinfo.filename == f'TIMER_3_11/ctax_runs/SSP2_{count}/co2tax':
           zipinfo.filename = f'co2tax_{count}.dat'
   
   zipdata.extract(f'TIMER_3_11/ctax_runs/SSP2_{count}/co2tax', path=outputlib)

   co2tax = pd.read_table(f'C:/Users/toonv/Documents/PBL/Data/ctax_runs_clean/co2tax_{count}.dat',
                          skiprows = [0], sep = ',', names = ['reduction'], index_col=False)
  
   co2tax.loc[-1] = 1971
   co2tax.index = co2tax.index + 1
   co2tax = co2tax.sort_index()
   co2tax.iloc[-1] = 0  # without ;]
   
   year_indices = [co2tax.loc[co2tax['reduction'] == str(year)].index.values for year in years]
    
   year_indices = year_indices[1:]   
   year_indices.insert(0, [0])
   
   for count_2, year_index in enumerate(year_indices):
       year_index = year_index[0]
       column = years[count_2]
       reduction_10_years = co2tax[year_index:year_index+28]
       red_df[column] = reduction_10_years['reduction'][1:].values
       
   red_df.to_csv(f'C:/Users/toonv/Documents/PBL/Data/ctax_runs_csv/co2tax_{count}.csv')
   print(red_df)