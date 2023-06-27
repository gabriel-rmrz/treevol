import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_tree_map(df, baseDir):
  plots_dir = baseDir + '/plots/'
  xyzw_df = df[['x0','x1','x2','weight']]
  rgb_df = df[['r','g','b']]
  #rgb_df = pd.DataFrame(xyzw_arr[:,4:], columns=['r', 'g', 'b'])
  # Plotting input data

  fig, ax = plt.subplots(figsize=(10,10))
  ax.scatter(xyzw_df['x0'], xyzw_df['x1'],s=0.5)
  with open('tree_info.json','r') as treeFile:
    #print(treeFile.read())
    treeInfo = json.loads(treeFile.read())
    for idx, td in treeInfo['tree_description'].items():
      x = td['x1']
      y = td['y1']
      delta_x = td['x2'] - td['x1']
      delta_y = td['y2'] - td['y1']
      cx = x + delta_x/2
      cy = y + delta_y/2

      ecolor = 'b'
      fcolor = 'b'
      if (td['isValid'] == 0):
        ecolor = 'r'
        fcolor = 'r'
      if (td['isBifurcated'] == 1):
        ecolor = 'g'
        fcolor = 'g'
      if (td['isFelled'] == 1):
        ecolor = 'black'
        fcolor = 'black'

      rect = patches.Rectangle((x,y), delta_x, delta_y, linewidth=1, edgecolor=ecolor, facecolor=fcolor,alpha=0.5)
      ax.add_artist(rect)
      ax.annotate('T '+idx, (cx, cy), color='black', weight='bold', fontsize=6, ha='center', va='center')

  ax.grid()
  plt.savefig(plots_dir+'patches.png')

def get_df(baseDir, slice_range):
  infile_name = 'slice_z_%d_%d' % (slice_range[0], slice_range[1])
  h5_infile = h5py.File(baseDir + '/data/'+infile_name+'.h5', 'r')

  
  xyzw_arr = h5_infile['slice'][:]
  df = pd.DataFrame(xyzw_arr, columns=['x0', 'x1', 'x2', 'weight','r','g','b'])
  df = df.sample(frac=0.4, random_state=1)
  return df 

def eff_per_layer(df_clusters, baseDir):
  print(df_clusters)
  with open('tree_info.json','r') as treeFile:
    #print(treeFile.read())
    treeInfo = json.loads(treeFile.read())
    total = 0 
    matches = 0
    for idx, td in treeInfo['tree_description'].items():
      if( td['isValid'] == 1 and td['isBifurcated'] == 0 and td['isFelled'] == 0 and td['isComplete']):
        print('searching for matches')
        df_matches = df_clusters.query(f"x_center_fit >  {td['x1']} and x_center_fit < {td['x2']} and y_center_fit >  {td['y1']} and y_center_fit < {td['y2']}")
        print(df_matches)
        total +=1
        if(len(df_matches.index) > 0):
          matches += 1
        else:
          print(f"Tree {idx} was not found!")
    if(total > 0):
      eff = matches/total
      print(f"Total: {total},    Marches: {matches},       Efficiency: {eff}")

        
        



def calc_eff(baseDir, config):

  slice_limits = config.slice_limits()
  slice_range = [slice_limits[0], slice_limits[1]]

  h5_inclustersfile = h5py.File(baseDir + '/results/cluster_fitted_center.h5','r')

  xyzw_arr = h5_inclustersfile['slice_ref_0'][:]
  df_clusters = pd.DataFrame(xyzw_arr, columns=['x_center_fit', 'y_center_fit', 'r_fit'])
  #tree_positions = h5_inclustersfile[
  df = get_df(baseDir, slice_range)

  draw_tree_map(df, baseDir)
  eff_per_layer(df_clusters,baseDir)

  with open('tree_info.json','r') as treeFile:
    #print(treeFile.read())
    treeInfo = json.loads(treeFile.read())
    for idx, td in treeInfo['tree_description'].items():
      pass
