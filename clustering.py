import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Lettura della nuvola di punti
h5_infile = h5py.File('z_slice.h5', 'r')
xyz_arr = h5_infile['slice'][:]
h5_infile.close()

# Plotting input data
plt.scatter(xyz_arr[:,0], xyz_arr[:,1],s=0.5)
plt.savefig('plots/scatter.png')
plt.clf()

xy_prj = xyz_arr[:,0:2]
print(xy_prj.shape)

# Clusterization
min_n_clusters = 50
max_n_clusters = 110
step_n_clusters = 5

wcss = []
for n_clusters in range(min_n_clusters, max_n_clusters, step_n_clusters):
  kmeans = KMeans(n_clusters=n_clusters, n_init = 10, random_state=8)
  pred = kmeans.fit_predict(xy_prj)
  wcss.append(kmeans.inertia_)
  pred = pred.reshape(pred.size,1)
  
  xy_pred = np.concatenate((xy_prj,pred), axis=1)
  column_values = ['x','y', 'cluster']
  df = pd.DataFrame(data= xy_pred,
                    columns = column_values)
  
  # ploting with differnt color for every cluster
  clusters = df.groupby('cluster')
  
  for name, cluster in clusters:
    plt.figure(1)
    plt.plot(cluster.x, cluster.y, marker='o', linestyle='', markersize=0.5, label=name)
    plt.figure(2)
    plt.plot(cluster.x, cluster.y, marker='o', linestyle='', markersize=0.5, label=name)
    plt.savefig('plots/scatter_colors_'+str(n_clusters)+'_'+ str(name) + '.png')
    
    plt.clf()
  
  #plt.legend()
  plt.figure(1)
  plt.savefig('plots/scatter_colors_'+str(n_clusters) + '.png')

plt.clf()
plt.plot(range(min_n_clusters,max_n_clusters, step_n_clusters),wcss)
plt.savefig('plots/inertia.png')



