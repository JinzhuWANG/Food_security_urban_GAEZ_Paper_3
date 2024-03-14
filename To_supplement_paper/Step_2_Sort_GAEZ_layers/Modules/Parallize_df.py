import os
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import numpy as np
from multiprocessing import  Pool
import ee
import concurrent
from concurrent.futures import ThreadPoolExecutor

# function to split df into multiple dfs, and feed each 
# of them to a seperate core
def parallelize_dataframe(df, func, n_cores=4):

    with ThreadPoolExecutor(n_cores) as executor:
        
        # submit tasks to the executor
        futures = [executor.submit(func,df_split)
                      for df_split in np.array_split(df, n_cores)]
        
        # iterate over completed tasks and update the pbar
        out_dfs = []
        for future in concurrent.futures.as_completed(futures):
            out_dfs.append(future.result())

    return  pd.concat(out_dfs)

########
def compute_urban_production(in_df,in_shp,name='EN_Name'):
  # hack to show process bar in multi-processing module
  print(' ',end='',flush=True)

  df_list = []
  for idx,row in tqdm(in_df.iterrows(),total=len(in_df)):
    crop = row['crop']
    water = row['water']
    year = row['year']
    rcp = row['rcp']
    ssp = row['SSP']

    # get the farmland_area gap between now and then, (e.g., 2000 and 2005)
    #   here divide(2) for ensuring rainfed/irrigated famland shares the cropland loss in the same magitude
    gap_area = row['future_farmland_area'].select('encroachment_led_farmland')\
           .subtract(row['lag_area'].select('encroachment_led_farmland').divide(2)) 
    #   here divide(4) because the 1/2sd = top - bot, then the 1/2(1/2sd) to eunsure 
    #   rainfed/irrigated shared the 1/2sd as the confidence interval 
    gap_area_sd = ee.Image(row['lag_area'].select('encroachment_led_farmland_bot')\
             .subtract(row['lag_area'].select('encroachment_led_farmland_top')).divide(4)).abs()

    # compute the crop production gap caused by the framland loss
    crop_yield = row['img_future']
    gap_production  = gap_area.multiply(crop_yield)
    gap_production_sd = gap_area_sd.multiply(crop_yield)
     
    in_img = ee.Image([gap_production,gap_production_sd]).rename(['val','sd'])

    stats = in_img.reduceRegions(collection = in_shp, reducer = 'sum', scale = GAEZ_pix_scale).getInfo()
    stats_df = pd.DataFrame({i['properties'][name]:[i['properties']['val']] for i in stats['features']}).T.reset_index()
    stats_df['sd'] = [i['properties']['sd'] for i in stats['features']]
    

    # append infomation
    stats_df.columns=['Province','Gap Production','Gap Production sd']
    stats_df['crop'] = crop
    stats_df['water'] = water
    stats_df['year'] = year
    stats_df['rcp'] = rcp
    stats_df['ssp'] = ssp

    # store df                        
    df_list.append(stats_df)

  # concat dfs
  out_df = pd.concat(df_list,ignore_index=True)
  
  return out_df