import ee
import numpy as np
# import tensorflow as tf


# Specify the export folder that stores the whole img to be classified
model_folder = 'Saved_models'
sample_folder = 'TF_record_samples'


# Specify the size and shape of patches expected by the model.
KERNEL_SIZE = 512
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
KERNEL_BUFFER = 128


# define if the train/eval samples are from a same source?
# i.e., they both trained from t0-->t1
train_eval_same_source = True

# Specify training features and parsing dictionary
FEATURES_train  = ['built_up_t0', 'elevation', 'slope', 'built_up_t1']
# FEATURES_DICT_train = {k:tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in  FEATURES_train}

# Specify eval features and parsing dictionary
if train_eval_same_source:
  FEATURES_eval  = FEATURES_train
else:
  FEATURES_eval  = ['built_up_t1', 'elevation', 'slope', 'built_up_t2']
# FEATURES_DICT_eval = {k:tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in FEATURES_eval}


# Sizes of the training and evaluation datasets.
TRAIN_SIZE = 8000
EVAL_SIZE = 2000


#_______________________________
# define the year-range
year = [f'{i}_{i+2}' for i in range(1990,2020,3)]
year_img_val_dict = {yr:i for yr,i in zip(year,range(10,0,-1))}

# # get all possible years for traning the projection
# proj_yr = []
# for k,v in year_img_val_dict.items():
#   # get the t0 val/img
#   t0 = k
      
#   # get the t1;t2
#   if v == 2:
#     pass
#   else:
#     for val in range(v-1,1,-1):
#       prj_val = val - (v - val)
          
#       if  prj_val>0:
          
#         prj_yr  = [k for k,v in year_img_val_dict.items() if v==prj_val][0]
        
#         t1 = [k for k,v in year_img_val_dict.items() if v==val][0]
#         t2 = [k for k,v in year_img_val_dict.items() if v==prj_val][0]
                        
#         proj_yr.append((t0,t1,t2))


# # define the limit_list that controls the periods for prediction
# limit_list = 13

# show the years used for train/pred
proj_yr_selected = [year[0],year[-1]]