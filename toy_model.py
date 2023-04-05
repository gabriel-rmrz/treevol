import open3d as o3d
import laspy
import numpy as np

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
    max_bound=(920310, 32158, 10285)
)

# Filtraggio della nuvola di punti sull'albero
tree_geom = outliers.crop(tree_bounding_box)

# Definizione del raggio del tronco dell'albero
tree_trunk_radius = 0.1

# Calcolo del volume del tronco dell'albero
tree_mesh, tree_volume = tree_geom.compute_convex_hull()
print("tree_volume")
print(type(tree_volume))
print("tree_mesh")
print(type(tree_mesh))
tree_mesh.compute_vertex_normals()
tree_mesh.remove_degenerate_triangles()
tree_mesh.remove_duplicated_triangles()
tree_mesh.remove_duplicated_vertices()
tree_mesh.remove_non_manifold_edges()
#tree_volume -= np.pi * tree_trunk_radius ** 2 * tree_mesh.get_volume()
tree_volume = tree_mesh.get_volume()

# Calcolo delle coordinate del punto GPS centrale del tronco dell'albero
center_of_mass = tree_mesh.get_center()

#gps_coords = (latitude, longitude, altitude) # calcolo delle coordinate GPS in base alla posizione dell'albero

# Stampa del volume del tronco e delle coordinate GPS del punto centrale
print("Volume del tronco dell'albero:", tree_volume)
print("Center of mass:", center_of_mass)
#print("Coordinate GPS del punto centrale del tronco dell'albero:", gps_coords)

# Visualization of the results.

viewer = o3d.visualization.Visualizer()
viewer.create_window()
viewer.add_geometry(tree_geom)
viewer.add_geometry(tree_mesh)
opt = viewer.get_render_option()
opt.show_coordinate_frame = True
opt.background_color = np.asarray([0.5, 0.5, 0.5])
viewer.run()
viewer.destroy_window()
