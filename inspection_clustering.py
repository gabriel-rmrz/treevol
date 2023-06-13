import numpy as np
import open3d as o3d
import laspy
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
  #opt = viewer.get_render_option()
  #opt.show_coordinate_frame = True
  #camera_params = ctr.convert_to_pinhole_camera_parameters()
  ctr = viewer.get_view_control()
  parameters = o3d.io.read_pinhole_camera_parameters("test_camera.json")
  if(pointcloud):
    viewer.add_geometry(pointcloud)
  #viewer.add_geometry(tree_mesh_0)
  if(meshes):
    for m in meshes:
      viewer.add_geometry(m)
  #opt.background_color = np.asarray([0.0, 0.0, 0.0])
  ctr.convert_from_pinhole_camera_parameters(parameters)
  ctr.rotate(10.0, 0.0)
  img = viewer.capture_screen_float_buffer(True)
  plt.imshow(np.asarray(img))
  ctr.convert_from_pinhole_camera_parameters(parameters)
  #viewer.get_view_control().set_front([-0.31514718595924335, -0.81724359719245454, 0.48248850144838423])
  #viewer.get_view_control().set_lookat([933147.0, 44109.5, 3129.5])
  #viewer.get_view_control().set_up([-0.19318902082705131, 0.55299295208179833, 0.81047936258719344])
  #ctr.set_zoom(1.7)
  viewer.run()
  plt.savefig('test_o3d.png')

  viewer.destroy_window()


if __name__ == '__main__':
  # Lettura della nuvola di punti
  input_file = 'data/Area_2_LAS_5.las'
  las = laspy.read(input_file)
  geom = get_pointcloud(las, isRGB=True)

  plane_model, inliers = geom.segment_plane(distance_threshold=200.,  ransac_n=3, num_iterations=500)
  outliers = geom.select_by_index(inliers, invert=True)
  inliers = geom.select_by_index(inliers, invert=False)
  ''' 
  Initial limits
  X
  913310
  952870
  Y
  24158
  63934
  Z
  -4026
  10285
  '''

  delta_x1 = 2
  delta_y1 = 23

  delta_x2 = 5
  delta_y2 = 28

  tree_bounding_box = o3d.geometry.AxisAlignedBoundingBox(
      #min_bound=(913310, 24158, -4026),
      #max_bound=(952870, 63934, 10285)
      min_bound=(913310+1000*delta_x1, 24158+1000*delta_y1, -4026),
      max_bound=(913310+1000*delta_x2, 24158+1000*delta_y2, 10285)
      )
  tree_geom = outliers.crop(tree_bounding_box)

  run_viewer(tree_geom)
  '''
  o3d.visualization.draw_geometries([outliers],
                                  zoom=0.7,
                                  front=[-0.31514718595924335, -0.81724359719245454, 0.48248850144838423],
                                  lookat=[933147.0, 44109.5, 3129.5],
                                  up=[-0.19318902082705131, 0.55299295208179833, 0.81047936258719344])
  '''
