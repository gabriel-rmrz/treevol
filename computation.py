import open3d as o3d
import copy
import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_pointcloud(las, isRGB=False):
  point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1,0))
  geom = o3d.geometry.PointCloud()
  geom.points = o3d.utility.Vector3dVector(point_data)

  if isRGB:
    point_color = np.stack([las.red, las.green, las.blue], axis=0).transpose((1,0))
    point_color  = point_color/point_color.max()
    geom.colors = o3d.utility.Vector3dVector(point_color)
  return geom

def get_mesh(pointcloud, box=None):
  if(box):
    point_cloud_box = pointcloud.crop(box)
  mesh, volume = point_cloud_box.compute_convex_hull()
  mesh.compute_vertex_normals()
  mesh.remove_degenerate_triangles()
  mesh.remove_duplicated_triangles()
  mesh.remove_duplicated_vertices()
  mesh.remove_non_manifold_edges()
  return point_cloud_box, mesh, volume

def run_viewer(pointcloud, meshes=None):
  viewer = o3d.visualization.Visualizer()
  viewer.create_window()
  if(pointcloud):
    viewer.add_geometry(pointcloud)
  #viewer.add_geometry(tree_mesh_0)
  if(meshes):
    for m in meshes:
      viewer.add_geometry(m)
  opt = viewer.get_render_option()
  opt.show_coordinate_frame = True
  #opt.background_color = np.asarray([0.0, 0.0, 0.0])
  viewer.run()
  viewer.destroy_window()

def get_tree_pointcloud(pointcloud):
  tree_bounding_box = o3d.geometry.AxisAlignedBoundingBox(
      min_bound=(914310, 28158, -4026),
      max_bound=(918310, 32158, 10285)
  )
  
  # Filtraggio della nuvola di punti sull'albero
  return outliers.crop(tree_bounding_box)

def get_trunk_pointcloud(pointcloud, slice_width):

  tree_bounding_box_slice_0 = o3d.geometry.AxisAlignedBoundingBox(
      min_bound=(914310, 28158, slices['min_height']),
      max_bound=(918310, 32158, slices['min_height'] + slices['width'])
      #max_bound=(918310, 32158, max_height)
  )
  
  tree_geom_slice_0, tree_mesh_0, tree_volume_0 = get_mesh(outliers, tree_bounding_box_slice_0)
  center_of_mass = tree_mesh_0.get_center()
  slice_arr = np.asarray(tree_geom_slice_0.points)
  r = np.sqrt((slice_arr[:,0] - center_of_mass[0])**2 + (slice_arr[:,1] - center_of_mass[1])**2)
  rcut = 3*r.max()
  tree_coord_arr = np.asarray(outliers.points)
  tree_colors_arr = np.asarray(outliers.colors)
  tree_arr = np.concatenate([tree_coord_arr,tree_colors_arr], axis=1)
  trunk_arr = tree_arr[np.sqrt((tree_arr[:,0] - center_of_mass[0])**2 + (tree_arr[:,1] - center_of_mass[1])**2) < rcut]
  #trunk_colors_arr = tree_colors_arr[np.sqrt((tree_arr[:,0] - center_of_mass[0])**2 + (tree_arr[:,2] - center_of_mass[1])**2) < rcut]
  
  trunk_df = pd.DataFrame(trunk_arr,columns=['X','Y','Z','red', 'green', 'blue'])
  trunk_pointcloud = get_pointcloud(trunk_df, isRGB=True)

  return center_of_mass, trunk_pointcloud

if __name__ == '__main__':
  # Lettura della nuvola di punti
  input_file = '../pointnet2/data/Area_2_LAS_5.las'
  las = laspy.read(input_file)
  geom = get_pointcloud(las, isRGB=True)
  
  # Definizione del filtro di rimozione del pavimento
  #plane_model, inliers = geom.segment_plane(distance_threshold=.1,  ransac_n=3, num_iterations=500)
  plane_model, inliers = geom.segment_plane(distance_threshold=500.,  ransac_n=3, num_iterations=500)
  outliers = geom.select_by_index(inliers, invert=True)
  inliers = geom.select_by_index(inliers, invert=False)
  
  # o3d.visualization.draw_geometries([inliers])

  
  # Definizione della bounding box dell'albero
  tree_geom = get_tree_pointcloud(outliers)

  slices = {
      'min_height': -2300,
      'max_height': 9000,
      'width': 100  #mm
      }
  center_of_mass, trunk_geom = get_trunk_pointcloud(tree_geom, slices)
  trunk_mesh_all = []
  h_slices = range(slices['min_height'],slices['max_height'], slices['width'])
  for idh, h in enumerate(h_slices):
    trunk_bounding_box_slice = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(914310, 28158, h),
        max_bound=(918310, 32158, h + slices['width'])
    )
    trunk_geom_slice, trunk_mesh, trunk_volume = get_mesh(trunk_geom, trunk_bounding_box_slice)
    trunk_mesh_all.append(trunk_mesh)
    #trunk_volume -= np.pi * trunk_trunk_radius ** 2 * trunk_mesh.get_volume()
    #trunk_volume = trunk_mesh.get_volume()
    
    #center_of_mass = trunk_mesh.get_center()
    #print("Volume del tronco dell'albero:", trunk_volume)
    #print("Center of mass:", center_of_mass)
  
    # to polar coordinates
    slice_arr = np.asarray(trunk_geom_slice.points)
    X = slice_arr[:,0] 
    Y = slice_arr[:,1]
    Z = slice_arr[:,2]
  
    r = np.sqrt((X- center_of_mass[0])**2 + (Y - center_of_mass[1])**2)
    #r = np.sqrt((X - 914310)**2 + (Y - 28158)**2)
    r /= 1000
    #step = (r.max()- r.min()) /50
    #bins = np.arange(r.min(), r.max(), step)
    step = (r.max()) /50
    bins = np.arange(0.0, r.max(), step)
    plt.figure(1)
    plt.hist(r, bins)
    plt.savefig('plots/hist_slices/hist_slice_'+str(idh) +'.png')
    plt.clf()
    plt.plot(X - center_of_mass[0],Y - center_of_mass[1],marker='o', linestyle='',color='black', markersize=4)
    plt.axhline(0)
    plt.axvline(0)
    plt.savefig('plots/scatter_slices/scatter_slice_'+str(idh) +'.png')
    plt.clf()
    plt.figure(2)
    plt.plot(center_of_mass[0], center_of_mass[1], marker='o', linestyle='', markersize=8)
  
  plt.figure(2)
  plt.savefig('plots/cm_scatter')
  
  
  # Visualization of the results.

  run_viewer(trunk_geom, trunk_mesh_all)
