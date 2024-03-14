import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import difflib
from tqdm.auto import tqdm
from pprint import pprint

# function to jude if (all filter words in the string) and (none excluesion word in the string) 
def filter_exclude(string_list,word_list:list,exclusion=[]):
  return_list = []
  for string in string_list:
    if all(True if i in string else False for i in word_list) and \
      all(True if j not in string else False for j in exclusion):
      return_list.append(string)
  return return_list


# function to filter GAEZ imgs and make a df
def get_img_df(img_path, exclusion = 'None', **kwargs):

  # get k,v
  keys = kwargs.keys()
  vals = [i if isinstance(i,list) else [i] for i in kwargs.values()]

  # force the exclusion be a list
  if not isinstance(exclusion,list):
    exclusion = [exclusion]

  # get the combinations from vals. E.g. [[A,B],[C]] ==> [[A,C],[B,C]]
  cmobination = list(itertools.product(*vals))

  # fitler the img_path using combination
  filtered_list = []
  for word in cmobination:
    filtered = filter_exclude(img_path,word)     
    if len(filtered) == 0:
      print(f'{word} have no coresponding img!')
    elif len(filtered) == 1:
      filtered_list.append(word+(filtered[0],))
    else:
      filtered = filter_exclude(img_path,word,exclusion)
      if len(filtered) == 1:
        filtered_list.append(word+(filtered[0],))
      else:
        print(f'{word} have {len(filtered)} coresponding img!')
  
  # construc the df
  if len(filtered_list) > 0:
    df = pd.DataFrame(filtered_list)
    df.columns = list(keys) + ['GEE_path']
    return df
  else:
    print('\nNone img was filterd, find below to exclud the replicates\n')
    return filtered


# function to compute the mean and confidence interval 
# from a list of img_path
def compute_mean_ci(img_list):

  # img_list to imageCollection
  img_col = ee.ImageCollection([ee.Image(i) for i in img_list])

  # compute mean
  img_mean = img_col.mean()

  # standard deviation
  sd = img_col.reduce(reducer=ee.Reducer.stdDev())

  # compute the 95% confidence interval distance
  ci = sd.divide((len(img_list)**(1/2))).multiply(1.96)

  return img_mean,sd,ci


# function to match the names between GAEZ and yearbook
def match_GAEZ_Yearbook_name(GAEZ_stats_dict,yearbook_dict):

  '''Match stats between GAEZ_stats_dict and yearbook_dict. 
  Both dicts will have a close city names [Jiao_zuo/Jiaozuo], 
  so this function can use the name as keys to pair stats.
  
  e.g., 
  GAEZ_stats_dict = {'Jiao_zuo':}'''

  names_tmp = []
  production_GAEZ_tmp = []
  production_yearbook_tmp = []

  for name in GAEZ_stats_dict.keys():
    name_re = name.replace('_','').title()
    yearbook_names = yearbook_dict.keys()
    close_match = difflib.get_close_matches(name_re, yearbook_names,1)[0]

    # judge if there will be a match
    if len(close_match) > 0:
      # add record to list
      names_tmp.append(name_re)
      production_GAEZ_tmp.append(GAEZ_stats_dict[name])
      production_yearbook_tmp.append(yearbook_dict[close_match])
    else:
      print(f'No match for {name}')

  return names_tmp,production_GAEZ_tmp,production_yearbook_tmp

# function to perform zonal statistics on each ee_tif,
# and link the result to yearbook

def stats_GAEZ_Yearbook(img_df,yearbook_df,city_shp,unit_scale=1000):

  # filter the admin boundaries using names from yearbook
  yearbook_names = yearbook_df['Zone'].tolist()
  Filtered_citys = city_shp.filter(ee.Filter.inList('NAME', ee.List(yearbook_names)))

  # sanity check --> how many recores (admin units) are paried?
  filtered_names = set([i['properties']['NAME'] for i in Filtered_citys.getInfo()['features']])
  print(f'The GEE_SHP has {len(filtered_names)} coresponding records given {len(yearbook_names)} total yearbook records.')

  # updata the yearbook using only the filtered_names
  yearbook_df_filterd = yearbook_df[yearbook_df['Zone'].isin(filtered_names)]

  # compute the stats of each img of GEE
  names_list = []
  production_GAEZ_list = []
  production_yearbook_list = []

  for idx,row in tqdm(list(img_df.iterrows())):

    # first perform zonal statistics on each ee_img from the img_df
    img = ee.Image(row['GEE_path'])
    crop = row['crop']
    stats = img.reduceRegions(collection=Filtered_citys, 
                  reducer=ee.Reducer.sum(), 
                  scale=img.projection().nominalScale()).getInfo()

    # The unit of GAEZ is 1,000t, while which is 1t in yearbook, so here need to multiply a scale
    GAEZ_dict = {i['properties']['NAME']:i['properties']['sum']*unit_scale for i in stats['features']}
    

    # fetch the yearbook record, note some same crops have differenet names between GAEZ and yearbook.
    # GAEZ:yearbook --> {Wetland_rice:Rice, Soybean:Legume, Potato:Tubers} 
    if crop == 'Wetland_rice':
      yearbook_dict = dict(zip(yearbook_df_filterd['Zone'],yearbook_df_filterd['Rice']))
    elif crop == 'Soybean':
      yearbook_dict = dict(zip(yearbook_df_filterd['Zone'],yearbook_df_filterd['Legume']))
    elif crop == 'Potato':
      yearbook_dict = dict(zip(yearbook_df_filterd['Zone'],yearbook_df_filterd['Tubers']))
    else:
      yearbook_dict = dict(zip(yearbook_df_filterd['Zone'],yearbook_df_filterd[crop]))


    # add record_list to list
    names_list.append(filtered_names)
    production_GAEZ_list.append([GAEZ_dict[i] for i in filtered_names])
    production_yearbook_list.append([yearbook_dict[i] for i in filtered_names])

  # add record to img_Df
  img_df['City'] = names_list
  img_df['GAEZ'] = production_GAEZ_list
  img_df['Yearbook'] = production_yearbook_list

  # change variable type for ploting
  img_df_exp = img_df.explode(['City','GAEZ','Yearbook'])
  img_df_exp['GAEZ'] = img_df_exp['GAEZ'].astype(float)
  img_df_exp['Yearbook'] = img_df_exp['Yearbook'].astype(float)

  return img_df_exp

def zonal_sum_GAEZ_production(img_df,
                  zone_cityes,
                  img_field = 'GEE_path',
                  zone_name='CityNameC'):

  # compute the stats of each img of GEE
  names_list = []
  production_GAEZ_list = []

  for idx,row in tqdm(list(img_df.iterrows())):

    # first perform zonal statistics on each ee_img from the img_df
    img = ee.Image(row[img_field])
    crop = row['crop']
    stats = img.reduceRegions(collection=zone_cityes, 
                  reducer=ee.Reducer.sum(), 
                  scale=img.projection().nominalScale()).getInfo()

    # The unit of GAEZ is 1,000t, while which is 1t in yearbook, so here need to multiply a scale
    GAEZ_dict = {i['properties']['CityNameC']:i['properties']['sum'] for i in stats['features']}
    
    # add record_list to list
    names_list.append(list(GAEZ_dict.keys()))
    production_GAEZ_list.append(list(GAEZ_dict.values()))


  # add record to img_Df
  img_df['City'] = names_list
  img_df['Production'] = production_GAEZ_list


  # change variable type for ploting
  img_df_exp = img_df.explode(['City','Production'])
  img_df_exp['Production'] = img_df_exp['Production'].astype(float)
  
  return img_df_exp
