import open3d as o3d
import copy
import laspy
import numpy as np
import matplotlib.pyplot as plt

# Lettura della nuvola di punti
input_file = '../pointnet2/data/Area_2_LAS_5.las'
las = laspy.read(input_file)

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
print('X') # Red axis
print(las.X.max())
print(las.X.min())
print('Y') # Green axis
print(las.Y.max())
print(las.Y.min())
print('Z') # Blue axis 
print(las.Z.max())
print(las.Z.min())
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
tree_bounding_box = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=(914310, 28158, -4026),
    max_bound=(918310, 32158, 10285)
)

# Filtraggio della nuvola di punti sull'albero
tree_geom = outliers.crop(tree_bounding_box)



min_height = -2300
max_height = 9000
tree_trunk_slice_width = 100 #mm 
# Calcolo del volume del tronco dell'albero
tree_bounding_box_slice_0 = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=(914310, 28158, min_height),
    max_bound=(918310, 32158, min_height + tree_trunk_slice_width)
)

tree_geom_slice_0 = outliers.crop(tree_bounding_box_slice_0)
tree_mesh_0, tree_volume_0 = tree_geom_slice_0.compute_convex_hull()
tree_mesh_0.compute_vertex_normals()
tree_mesh_0.remove_degenerate_triangles()
tree_mesh_0.remove_duplicated_triangles()
tree_mesh_0.remove_duplicated_vertices()
#tree_mesh.remove_non_manifold_edges()

# Calcolo delle coordinate del punto GPS centrale del tronco dell'albero
center_of_mass = tree_mesh_0.get_center()

slice_arr = np.asarray(tree_geom_slice_0.points)
X = slice_arr[:,0]
Y = slice_arr[:,1]
Z = slice_arr[:,2]
r = np.sqrt((X - center_of_mass[0])**2 + (Y - center_of_mass[1])**2)
rcut = 2*r.max()

print(rcut)

tree_arr = np.asarray(outliers.points)
X_tree = tree_arr[:,0]
Y_tree = tree_arr[:,1]
Z_tree = tree_arr[:,1]

tree_colors_arr = np.asarray(outliers.colors)
r_tree = tree_arr[:,0]
g_tree = tree_arr[:,1]
b_tree = tree_arr[:,1]

trunk_arr = tree_arr[np.sqrt((X_tree - center_of_mass[0])**2 + (Y_tree - center_of_mass[1])**2) < rcut]
trunk_colors_arr = tree_colors_arr[np.sqrt((X_tree - center_of_mass[0])**2 + (Y_tree - center_of_mass[1])**2) < rcut]

trunk_point_data = np.stack([trunk_arr[:,0], trunk_arr[:,1], trunk_arr[:,2]], axis=0).transpose((1,0))
trunk_colors = np.stack([trunk_colors_arr[:,0], trunk_colors_arr[:,1], trunk_colors_arr[:,2]], axis=0).transpose((1,0))
trunk_colors  = trunk_colors/trunk_colors.max()

trunk_geom = o3d.geometry.PointCloud()
trunk_geom.points = o3d.utility.Vector3dVector(trunk_point_data)
trunk_geom.colors = o3d.utility.Vector3dVector(trunk_colors)


#o3d.visualization.draw_geometries([trunk_geom])

#exit()


trunk_mesh_all = []
h_slices = range(min_height, max_height, tree_trunk_slice_width)
for idh, h in enumerate(h_slices):
  trunk_bounding_box_slice = o3d.geometry.AxisAlignedBoundingBox(
      min_bound=(914310, 28158, h),
      max_bound=(918310, 32158, h + tree_trunk_slice_width)
  )
  trunk_geom_slice = trunk_geom.crop(trunk_bounding_box_slice)
  trunk_mesh, trunk_volume = trunk_geom_slice.compute_convex_hull()
  trunk_mesh.compute_vertex_normals()
  trunk_mesh.remove_degenerate_triangles()
  trunk_mesh.remove_duplicated_triangles()
  trunk_mesh.remove_duplicated_vertices()
  #trunk_mesh.remove_non_manifold_edges()
  trunk_mesh_all.append(trunk_mesh)
  #trunk_volume -= np.pi * trunk_trunk_radius ** 2 * trunk_mesh.get_volume()
  #trunk_volume = trunk_mesh.get_volume()
  
  # Calcolo delle coordinate del punto GPS centrale del tronco dell'albero
  #center_of_mass = trunk_mesh.get_center()

  # Stampa del volume del tronco e delle coordinate GPS del punto centrale
  #print("Volume del tronco dell'albero:", trunk_volume)
  print("Center of mass:", center_of_mass)



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

viewer = o3d.visualization.Visualizer()
viewer.create_window()
viewer.add_geometry(trunk_geom)
#for tm in trunk_mesh_all:
#  viewer.add_geometry(tm)
opt = viewer.get_render_option()
opt.show_coordinate_frame = True
opt.background_color = np.asarray([0.5, 0.5, 0.5])
viewer.run()
viewer.destroy_window()
