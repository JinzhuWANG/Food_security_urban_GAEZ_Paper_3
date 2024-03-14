import ee
import numpy as np


#_______________________________
# define the year-range
year = [f'{i}_{i+2}' for i in range(1990,2020,3)]
year_img_val_dict = {yr:i for yr,i in zip(year,range(10,0,-1))}


# show the years used for train/pred
proj_yr_selected = [year[0],year[-1]]