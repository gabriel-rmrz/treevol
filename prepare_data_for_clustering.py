import os
import open3d as o3d
import h5py
import laspy
import numpy as np
import pandas as pd


def make_slice(baseDir,outliers, slice_limits):
  for i in range(len(slice_limits) -1):
    outfile_name = '%s/data/slice_z_%d_%d.h5' % (baseDir, slice_limits[i], slice_limits[i+1])
    h5_outfile = h5py.File(outfile_name, 'w')
    '''
    min
    -1743.438
    -2445.0
    -157.327
    max
    9023.125
    4425.0
    481.514
    '''
    x_min = 913310
    y_min = 24158
    x_max = 952870
    y_max = 63934
    tree_bounding_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(913310, 24158, slice_limits[i]),
        max_bound=(952870, 63934, slice_limits[i+1])
    #    min_bound=(-1743438, -2445000, slice_limits[i]),
    #    max_bound=(9023125, 4425000, slice_limits[i+1])
    #    min_bound=(-1743438, -2445000, -157327),
    #    max_bound=(9023125, 4425000, 481514)
    #    min_bound=(913310, 24158, -4025),
    #    max_bound=(952870, 63934, 10285)
    #    min_bound=(-00, 101955, slice_limits[i]),
    #    max_bound=(-27774, 3839063, slice_limits[i+1])
    )
    
    # Filtraggio della nuvola di punti sull'albero
    tree_geom = outliers.crop(tree_bounding_box)
    #tree_geom = outliers

    '''
    # Visualization of the results.
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(tree_geom)
    #viewer.add_geometry(tree_mesh)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()
    '''
    
    xyz_arr = np.asarray(tree_geom.points)/1000 # Changing unit from mm to m
    xyz_colors_arr = np.asarray(tree_geom.colors)/256 #Normalising colors 
    print(xyz_arr[:,0])

    print("min")
    print(xyz_arr[:,0].min())
    print(xyz_arr[:,1].min())
    print(xyz_arr[:,2].min())
    print("max")
    print(xyz_arr[:,0].max())
    print(xyz_arr[:,1].max())
    print(xyz_arr[:,2].max())
    xyz_df = pd.DataFrame(xyz_arr, columns=['x0', 'x1', 'x2'])
    xyz_df['x0'] = xyz_df['x0'] - x_min/1000 
    xyz_df['x1'] = xyz_df['x1'] - y_min/1000 
    xyz_df['x2'] = xyz_df['x2'] - xyz_df['x2'].min()
    xyz_df['weights']=1
    xyz_df['r'] = xyz_colors_arr[:,0] 
    xyz_df['g'] = xyz_colors_arr[:,1] 
    xyz_df['b'] = xyz_colors_arr[:,2] 
    
    #xyz_df.to_hdf('z_slice.h5','df', mode='w')
    h5_outfile.create_dataset('slice', data=xyz_df)
    h5_outfile.close()
  return 0
  


def make_dir_structure(baseDir):
  if not os.path.exists(baseDir):
    print("Creating directory '{}'...".format(baseDir))
    os.makedirs(baseDir)
    print("Creating directory '{}/data'...".format(baseDir))
    os.makedirs(baseDir+'/data')
  else:
    print("Directory '{}' already exists. Please delete it or change it the value of baseDir".format(baseDir))
    return 0
  return 1

def prepare(baseDir, config):
  if not make_dir_structure(baseDir):
    return 0
  # Lettura della nuvola di punti
  inputFile = config.pointCloud()
  #input_file = '../pointnet2/data/Area1_01_Output_laz1_4_colourised.laz'
  las = laspy.read(inputFile)
  '''
  print(list(las.point_format.dimension_names))
  exit()  
  '''
  
  point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1,0))
  point_color = np.stack([las.red, las.green, las.blue], axis=0).transpose((1,0))
  point_color  = point_color/point_color.max()
  
  geom = o3d.geometry.PointCloud()
  geom.points = o3d.utility.Vector3dVector(point_data)
  geom.colors = o3d.utility.Vector3dVector(point_color)
  

  # Definizione del filtro di rimozione del pavimento
  #plane_model, inliers = geom.segment_plane(distance_threshold=.1,  ransac_n=3, num_iterations=500)
  plane_model, inliers = geom.segment_plane(distance_threshold=500.,  ransac_n=3, num_iterations=500)
  outliers = geom.select_by_index(inliers, invert=True)
  
  #o3d.visualization.draw_geometries([outliers])
  
  # Definizione della bounding box dell'albero
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
  #slice_limits = [1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100]
  #slice_limits = [2000, 2200, 2400, 2600, 2800, 3000]
  #slice_limits = [2000, 2300, 2600, 2900, 3200, 3500]
  slice_limits = config.slice_limits()
  #slice_limits = [30000, 40000]
  make_slice(outliers, slice_limits)

