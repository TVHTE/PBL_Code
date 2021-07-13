#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

def reduction_df(df, year, region):
    """
    outputs the reduction for a specific region and timeframe
    """
    
    if year < 2020 or year > 2100:
        raise Exception("Only years ranging from 2020 to 2100 in timesteps of 10")
        
    if region < 1 or region > 26:
        raise Exception("Only regions ranging from 1 to 26")
    
    # empty df
    index = list(range(0,4020,20))
    
    # first calculate the costs (0~4000) for the given period   
    year_index = []
    start_year = 2020
    steps_new_year = 202 # until new year in df
    
    for i in range(0, 20, 10):               
        cur_index = df.index[int(((year + i) - start_year)/10 * steps_new_year - 1)]
        year_index.append(cur_index)

    df_cur = df[year_index[0]+1:year_index[1]]
    df_cur = df_cur[region]
    df_cur = pd.DataFrame(df_cur)
    
    #  set index to USD (0:4000)     
    df_cur['USD'] = index       
    df_cur.set_index('USD', inplace=True, drop=True)       
    df_cur.columns = ['reduction']     
    
    df_cur.year = year
    df_cur.region = region
    
    return df_cur

def find_path(reductions, path, timerstep):
    """
    Calculate the ctax paths from the repsonse curve

    Input
    -------
    pandas dataframe, number of steps taken (0 to 10)
    and define the path (string) linear, cubic, cubicroot
    
    Output
    -------
    ctax paths pandas df
    
    """

    possible_paths = ['linear', 'cubic', 'cubicroot']
        
    if path not in possible_paths:      
         raise Exception("given path not one of linear, cubic or cubicroot")
    
    if path == 'linear':
        num_path = 1
    elif path == 'cubic':
        num_path = 2
    elif path == 'cubicroot':
        num_path = 3
        
    #empty list for all paths    
    path = []
        
    # amount of columns is year - 2020 / 5 (5 year timesteps) 
    count_columns = int((reductions.year - 2020) / timerstep)
             
    columns = [str(i*timerstep + 2020) for i in range(count_columns + 1)]
        
    for index in reductions.index.values:
        num = len(columns)
        
        # calculate path (1,2 and 3 and append the paths to list)
        if num_path == 1:                          
            path.append(np.linspace(0, index, num=num))
        
        if num_path == 2:
            # find a in y = ax^3              
            a = (index/(count_columns**(3)))
            price_path = []

            for step in range(0, num):
                price = a * (step**3)               
                price_path.append(price)
            
            path.append(price_path)
            
        if num_path == 3:                              
            # find a in y = ax^(1/3)                
            a = (index/(count_columns**(1/3)))
            price_path = []

            for step in range(0, num):
                price = a * (step**(1/3))
                price_path.append(price)

            price_path[-1] = price_path[-1].round()
            path.append(price_path)       
                
    # convert to pandas dataframe        
    df_paths = pd.DataFrame(path, columns=columns)
            
    # combine dataframes
    df = pd.concat([df_paths.reset_index(drop=True),reductions.reset_index(drop=True)], axis=1)   
    
    df.year = reductions.year
    df.region = reductions.region
    
    return df 






