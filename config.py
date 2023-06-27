import os
import json
import numpy as np

class Configuration(dict):
  '''
  This class will contain all the information relevant to the analysis of a given point cloud.
  As the density of points, type of tree, and other properties will affect the parameters used
  by the clustering, fitting,... algoriths it is important to have a default set of values and
  values personalised for every particular point cloud.
  '''
  def __init__(self,cData):
    print(cData)
    if isinstance(cData,dict):
      super().__init__(cData)
    elif isinstance(cData, str) and os.path.exists(cData):
      with open(cData) as f:
        super().__init__(json.load(f))
    else:
      raise TypeError('Wrong configuration data type', type(cData), cData)

  def __getattr__(self, attr):
    return lambda: self[attr]

  def pointCloud(self):
    '''
    This is the input pointcloud right after the reconstruction
    '''
    return self.get('pointCloud_file', 'data/Area_2_LAS_5.las')
    
  def slice_limits(self):
    '''
    This define the layers that will be used for the clustering algorithm to find the trees
    TODO: maybe a definition where the slices are not together can be added.
    '''
    return self.get('slice_limits',  [1300, 1500])


  def clue_params(self):
    '''
    These are the parameters for the CLUEStering. This values should vary depending on the
    point cloud density, diamater of the tree, separation of the trees and other factors

    '''
    default = {
      "dc" : 0.3,
      "rhoc" : 20,
      "delta" : 1.5
    }
    default.update(self.get('clue_parms',{}))
    return default


