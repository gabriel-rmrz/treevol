import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CLUEstering as clue

from sklearn.cluster import KMeans

# Lettura della nuvola di punti
h5_infile = h5py.File('z_slice.h5', 'r')
xyzw_arr = h5_infile['slice'][:]
xyzw_df = pd.DataFrame(xyzw_arr, columns=['x0', 'x1', 'x2', 'weight'])
# Plotting input data
plt.scatter(xyzw_df['x0'], xyzw_df['x1'],s=0.5)
plt.savefig('plots/scatter.png')
plt.clf()

xyw_df = xyzw_df[['x0','x1','weight']]

# Clusterization
min_n_clusters = 50
max_n_clusters = 110
step_n_clusters = 5
''' k-means
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

'''
blob_test = clue.makeBlobs(1000,2)
print(blob_test.head())
#exit()

#clust = clue.clusterer(50,50,1000)
clust = clue.clusterer(1,5,1.5)
clust.readData(xyzw_df)
#clust.readData(blob_test)
clust.runCLUE()
clust.clusterPlotter()

