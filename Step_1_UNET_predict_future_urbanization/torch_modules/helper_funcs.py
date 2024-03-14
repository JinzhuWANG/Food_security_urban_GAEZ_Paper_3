import ee
import numpy as np
import pandas as pd

import torch
import tensorflow as tf

from Global_Variables import *
import glob
from tqdm.auto import tqdm


# construc a function to export train samples
def export_sample(img,
          region,
          year_select,
          patch_n,
          shard_per_patch,
          export_type='train'):

  # simplify the sampling region, then buffer inwardly
  # to make sure smaples are not sampled from nondata area
  Simplified_geo = region.union().geometry().simplify(50000)
  Simplified_buffer = Simplified_geo.buffer(-15000)

  # get the total shards
  total_shards = patch_n * shard_per_patch

  # 1). split the exporting into seperate tasks (num. of patch_n)
  for patch in range(patch_n):

    # 2). sampling and exporting
    geomSample = ee.FeatureCollection([])
    for i in range(shard_per_patch):

      # get the seed and sample_size for this patch
      if export_type == 'train':
        seed = np.random.randint(1,40000)
        sample_size = TRAIN_SIZE
        fe_select = FEATURES_train
      elif export_type == 'eval':
        seed = np.random.randint(40000,65535)
        sample_size = EVAL_SIZE
        fe_select = FEATURES_eval
      else:
        print('Wrong export type, chose from ["train","eval"]')
        break

      # 3). then sample int(total_size/total_shards) from each task
      sample = img.sample(
        region = Simplified_buffer, 
        scale = 30,
        numPixels = int(sample_size/total_shards),
        seed = seed,
        tileScale = 8
      )

      geomSample = geomSample.merge(sample)

    # 4). exporting
    desc = f"{export_type}_{year_select[0]}_{year_select[1]}_patch_{patch:03}"

    task = ee.batch.Export.table.toDrive(
        collection = geomSample,
        description = desc,
        folder = sample_folder,
        fileNamePrefix = desc,
        fileFormat = 'TFRecord',
        selectors = fe_select
    )
    task.start()

    print(desc)

def parse_tfrecord(example_proto,fe_dict):
  """The parsing function.
  Read a serialized example into the structure defined by FEATURES_DICT.
  Args:
    example_proto: a serialized Example.
  Returns:
    A dictionary of tensors, keyed by feature name.
  """
  return tf.io.parse_single_example(example_proto, fe_dict)


def to_tuple(inputs,fe,mode='train'):
  """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
  Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
  Args:
    inputs: A dictionary of tensors, keyed by feature name.
  Returns:
    A tuple of (inputs, outputs).
  """
  inputsList = [inputs.get(key) for key in fe]
  stacked = tf.stack(inputsList, axis=0)

  # return tuple for train, and tensor for pred
  if mode == 'train':
    return stacked[:len(fe)-1,:,:], stacked[len(fe)-1:,:,:]
  elif mode == 'pred':
    return stacked


def get_dataset(pattern,fe,fe_dict,mode='train'):
  """Function to read, parse and format to tuple a set of input tfrecord files.
  Get all the files matching the pattern, parse and convert to tuple.
  Args:
    pattern: A file pattern to match in a Cloud Storage bucket.
  Returns:
    A tf.data.Dataset
  """
  globs = sorted(glob.glob(f"{pattern}*"))
  dataset = tf.data.TFRecordDataset(globs, compression_type='GZIP')
  dataset = dataset.map(lambda x: parse_tfrecord(x,fe_dict), num_parallel_calls=2)
  dataset = dataset.map(lambda x: to_tuple(x,fe,mode), num_parallel_calls=2)
  return dataset

# parse TFRecord to npy
def TF_to_npy(train_base,eval_base,path):

  # get TFRecord
  train_files = f'./{sample_folder}/{train_base}*'
  eval_files = f'./{sample_folder}/{eval_base}*'

  train_record = get_dataset(train_files,FEATURES_train,FEATURES_DICT_train)
  eval_record = get_dataset(eval_files,FEATURES_eval,FEATURES_DICT_eval)

  # save TFRecord to npy  
  for data,data_type in zip([train_record,eval_record],['train_record','eval_record']):
    # get the sample size
    length = TRAIN_SIZE if data_type == 'train_record' else EVAL_SIZE

    idx = 0
    # save rf-record to npy
    for record in tqdm(data,total=length):

      # get the x-y data
      x = record[0].numpy()
      y = record[1].numpy()

      # combine x-y to a tuple
      x_y_tuple = np.array((x,y))
      if data_type == 'train_record': np.save(f'{path}/{train_base}_{idx:05}.npy',x_y_tuple)
      if data_type == 'eval_record': np.save(f'{path}/{eval_base}_{idx:05}.npy',x_y_tuple)

      # update the index
      idx += 1

# function to evaluate the model
def eval_model(model,loader,creterion):

  # set to eval mode
  model.eval()
  
  # compute losses
  losses = []
  with torch.no_grad():
    for data in loader:

      # get data, then send them to GPU
      x = torch.FloatTensor(data[0]).to('cuda')
      y = torch.FloatTensor(data[1]).to('cuda')

      # train the model
      score = model(x)

      # compute loss
      loss = creterion(score,y)
      losses.append(loss.cpu().numpy())
 
  # change model back to training mode    
  model.train()

  return losses

# a helper function to export tif to TF
def export_img_bufferd_tf(img,out_image_base, kernel_buffer, region):
  """Run the image export task.  Block until complete.
  """
  task = ee.batch.Export.image.toDrive(
      image = img.select(FEATURES_train[:-1]).toFloat(),
      description = out_image_base,
      folder = sample_folder,
      fileNamePrefix = out_image_base,
      region = region.geometry().bounds(),
      scale = 30,
      fileFormat = 'TFRecord',
      maxPixels = 1e10,
      formatOptions = {
        'patchDimensions': KERNEL_SHAPE,
        'kernelSize': kernel_buffer,
        'compressed': True,
        'maxFileSize': 104857600
    }
  )
  task.start()