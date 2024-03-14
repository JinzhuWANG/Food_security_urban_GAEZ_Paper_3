from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import concurrent.futures


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
