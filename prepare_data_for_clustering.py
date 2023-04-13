import open3d as o3d
import h5py
import laspy
import numpy as np

# Lettura della nuvola di punti
input_file = '../pointnet2/data/Area_2_LAS_5.las'
h5_outfile = h5py.File('z_slice.h5', 'w')
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
    min_bound=(913310, 24158, 1800),
    max_bound=(952870, 63934, 2200)
#    min_bound=(913310, 24158, -4025),
#    max_bound=(952870, 63934, 10285)
)

# Filtraggio della nuvola di punti sull'albero
tree_geom = outliers.crop(tree_bounding_box)

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
# Plotting xy projection

xyz_arr = np.asarray(tree_geom.points)/1000 # Changing unit from mm to m
h5_outfile.create_dataset('slice', data=xyz_arr)
h5_outfile.close()




