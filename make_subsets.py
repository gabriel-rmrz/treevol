import os
import laspy
import numpy as np

def bounding_box_filter(points, colors, min_bound, max_bound):
  mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
  return points[mask], colors[mask]

def save_points_to_file(points, filename, las, colors=None):
  header = laspy.LasHeader(point_format=3, version="1.2")
  las_data = laspy.LasData(header)

  # Assign the new point data
  las_data.x = points[:, 0]
  las_data.y = points[:, 1]
  las_data.z = points[:, 2]

  if colors is not None:
    las_data.red = colors[:, 0]
    las_data.green = colors[:, 1]
    las_data.blue = colors[:, 2]

  las_data.write(filename)

def main():
  # Reading the point cloud
  input_file = 'Area_2_LAS_15.las'
  las = laspy.read(input_file)
  
  point_data = np.vstack([las.X, las.Y, las.Z]).transpose()
  if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
    point_color = np.vstack([las.red, las.green, las.blue]).transpose()
    point_color  = point_color
  
  ''' 
  Initial limits
  X
  952870
  913310
  Y
  63934
  24158
  Z
  10285
  -4026
  '''
  min_bound_i = np.array([913310, 24158, -4026])
  #max_bound = [952870, 63934, 10285]
  #min_lim = np.array([0, 0,0])
  #max_lim = np.array([10, 10, 100])

  min_lims = np.array([
    [0, 0,0], 
    [10,0,0],
    [20,0,0],
    [30,0,0],
    [0, 10,0], 
    [10,10,0],
    [20,10,0],
    [30,10,0],
    [0, 20,0], 
    [10,20,0],
    [20,20,0],
    [30,20,0],
    [0, 30,0], 
    [10,30,0],
    [20,30,0],
    [30,30,0],
    [0, 0,0], 
    [10,0,0],
    [20,0,0],
    [0, 10,0], 
    [10,10,0],
    [20,10,0],
    [0, 20,0], 
    [10,20,0],
    [20,20,0],
    [0, 0,0], 
    [10,0,0],
    [0, 10,0], 
    [10,10,0],
    [0, 0,0], 
    ])
  max_lims = np.array([
    [10, 10, 100], 
    [20, 10, 100], 
    [30, 10, 100], 
    [40, 10, 100],
    [10, 20, 100], 
    [20, 20, 100], 
    [30, 20, 100], 
    [40, 20, 100],
    [10, 30, 100], 
    [20, 30, 100], 
    [30, 30, 100], 
    [40, 30, 100],
    [10, 40, 100], 
    [20, 40, 100], 
    [30, 40, 100], 
    [40, 40, 100],
    [20, 20, 100], 
    [30, 20, 100], 
    [40, 20, 100],
    [20, 30, 100], 
    [30, 30, 100], 
    [40, 30, 100],
    [20, 40, 100], 
    [30, 40, 100], 
    [40, 40, 100],
    [30, 30, 100], 
    [40, 30, 100],
    [30, 40, 100], 
    [40, 40, 100],
    [40, 40, 100],
    ])
  
  for i, (min_lim, max_lim) in enumerate(zip(min_lims, max_lims)):
    min_bound = min_bound_i + 1000* min_lim
    max_bound = min_bound_i + 1000* max_lim
    pc_reduced, colors_reduced = bounding_box_filter(point_data, point_color , min_bound, max_bound)

    save_points_to_file(pc_reduced, f"outputs/Region_{i}.las", las, colors_reduced)

if __name__ == "__main__":
  main()
