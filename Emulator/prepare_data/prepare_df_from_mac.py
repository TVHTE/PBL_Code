#!/usr/bin/env python
# coding: utf-8

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
    
    return df_cur

def find_path(df, step_columns, path):
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
        
    # make sure columns are same length as given step_columns    
    steps = 10       
    df_columns = []
    
    for i in range(0, steps, step_columns):                
        df_columns.append(i)        
        
        if i == steps - step_columns:        
            df_columns.append(steps)
        
    for index in df.index.values:                     
        num = len(df_columns)
        
        # calculate path (1,2 and 3 and append the paths to list)
        if num_path == 1:                          
            path.append(np.linspace(0, index, num=num))
        
        if num_path == 2:
            # find a in y = ax^3              
            a = (index/(steps**(3)))
            price_path = []

            for step in range(0, num):
                price = a * (step**3)               
                price_path.append(price)
            
            path.append(price_path)
            
        if num_path == 3:                              
            # find a in y = ax^(1/3)                
            a = (index/(steps**(1/3)))
            price_path = []

            for step in range(0, num):
                price = a * (step**(1/3))
                price_path.append(price)

            price_path[-1] = price_path[-1].round()
            path.append(price_path)       
                
    # convert to pandas dataframe        
    df_paths = pd.DataFrame(path, columns=df_columns)
            
    # combine dataframes
    df = pd.concat([df_paths.reset_index(drop=True),df.reset_index(drop=True)], axis=1)   
    
    return df 


# In[13]:

if __name__ == "main":

    reduction_lin = reduction_df(df_linear, 2040, 11)
    reduction_cubic = reduction_df(df_cubic, 2040, 11)
    reduction_cubicroot = reduction_df(df_cubicroot, 2040, 11)
    
    df_lin_path = find_path(reduction_lin, 1, 'linear')
    
    df_cubic_path = find_path(reduction_lin, 1, 'cubicroot')
    
    df_cubic_path.tail()



