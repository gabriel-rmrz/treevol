import os
import math
import h5py
import json
import numpy as np
import pandas as pd
import awkward as awk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import CLUEstering as clue
import cv2 as cv

from sklearn.cluster import KMeans
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
def add_subplot_axis(ax, rect, axisbg='w'):
  fig = plt.gcf()
  box = ax.get_position()
  width = box.width
  height = box.height
  inax_position =  ax.transAxes.transform(rect[0:2])
  transFigure = fig.transFigure.inverted()
  infig_position = transFigure.transform(inax_position)
  x = infig_position[0]
  y = infig_position[1]
  width *= rect[2]
  height *= rect[3]
  subax = fig.add_axes([x, y, width, height], facecolor=axisbg)
  x_labelsize = subax.get_xticklabels()[0].get_size()
  y_labelsize = subax.get_yticklabels()[0].get_size()
  x_labelsize *= rect[2]**0.5
  y_labelsize *= rect[3]**0.5
  subax.xaxis.set_tick_params(labelsize=x_labelsize)
  subax.yaxis.set_tick_params(labelsize=y_labelsize)
  return subax 
def clusterization_kmeans():
  # Clusterization
  # k-means
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

def calc_eff(df):
  xyzw_df = df[['x0','x1','x2','weight']]
  rgb_df = df[['r','g','b']]
  #rgb_df = pd.DataFrame(xyzw_arr[:,4:], columns=['r', 'g', 'b'])
  # Plotting input data
  
  fig, ax = plt.subplots(figsize=(10,10))
  ax.scatter(xyzw_df['x0'], xyzw_df['x1'],s=0.5)
  with open('tree_info.json','r') as treeFile:
    #print(treeFile.read())
    treeInfo = json.loads(treeFile.read())
    for idx, td in treeInfo['tree_description'].items():
      x = td['x1']
      y = td['y1']
      delta_x = td['x2'] - td['x1']
      delta_y = td['y2'] - td['y1']
      cx = x + delta_x/2
      cy = y + delta_y/2

      ecolor = 'b'
      fcolor = 'b'
      if (td['isValid'] == 0):
        ecolor = 'r'
        fcolor = 'r'
      if (td['isBifurcated'] == 1):
        ecolor = 'g'
        fcolor = 'g'
      if (td['isFelled'] == 1):
        ecolor = 'black'
        fcolor = 'black'

      rect = patches.Rectangle((x,y), delta_x, delta_y, linewidth=1, edgecolor=ecolor, facecolor=fcolor,alpha=0.5)
      ax.add_artist(rect)
      ax.annotate('T '+idx, (cx, cy), color='black', weight='bold', fontsize=6, ha='center', va='center')

      print(td['x1'])
  ax.grid()
  plt.savefig('plots/patches.png')


def get_df(slice_range):
  infile_name = 'slice_z_%d_%d' % (slice_range[0], slice_range[1])
  h5_infile = h5py.File('data/'+infile_name+'.h5', 'r')
  xyzw_arr = h5_infile['slice'][:]
  df = pd.DataFrame(xyzw_arr, columns=['x0', 'x1', 'x2', 'weight','r','g','b'])
  df = df.sample(frac=0.4, random_state=1)
  return df

def clusterization_clue(df, slice_range):
  infile_name = 'slice_z_%d_%d' % (slice_range[0], slice_range[1])
  xyzw_df = df[['x0','x1','x2','weight']]
  rgb_df = df[['r','g','b']]
  #rgb_df = pd.DataFrame(xyzw_arr[:,4:], columns=['r', 'g', 'b'])
  # Plotting input data
  
  plt.figure(figsize=(10,10))
  plt.scatter(xyzw_df['x0'], xyzw_df['x1'],s=0.5)
  plt.grid()
  if not os.path.exists('plots/'+infile_name):
    os.makedirs('plots/' + infile_name)
  if not os.path.exists('plots/'+infile_name+'/subcluster'):
    os.makedirs('plots/' + infile_name+ '/subcluster')
  if not os.path.exists('plots/'+infile_name+'/cluster'):
    os.makedirs('plots/' + infile_name+ '/cluster')
  print('Saving plots/' + infile_name + '_scatter.png image')
  plt.savefig('plots/' + infile_name + '_scatter.png')
  plt.clf()
  
  xyw_df = xyzw_df[['x0','x1','weight']]
  
  '''
  dc = 0.3
  rhoc = 120
  delta = 1.5
  '''
  dc = 0.3
  rhoc = 20
  delta = 1.5
  clust = clue.clusterer(dc, rhoc, delta)
  clust.readData(xyzw_df)
  clust.runCLUE()
  labels = np.asarray(clust.clusterIds)
  plt.figure(2)
  plt.plot(xyw_df.x0, xyw_df.x1, marker='o', linestyle='', markersize=0.5)
  cl_meanx = []
  cl_meany = []
  fit_x_center = []
  fit_y_center = []
  fit_radius = []
  for i in range(0, labels.max(),1):
    cluster = xyzw_df[labels==i]
    cluster_rgb = rgb_df[labels==i]
    plt.figure(2)
    plt.scatter(cluster.x0.mean(), cluster.x1.mean(), s=50, facecolors='none', edgecolor='black')
    plt.plot(cluster.x0, cluster.x1, marker='o', linestyle='', color='red', markersize=0.7)
    #plt.figure(3)
    plt.figure(1)
    #plt.savefig('plots/CLUE/histos/histo_cl_'+str(i)+'.png')
    ax = plt.figure(1).add_subplot(111)
    ax.plot(cluster.x0, cluster.x1, marker='o', linestyle='', color='black', markersize=0.7)
    ax.scatter(cluster.x0.mean(), cluster.x1.mean(), s=500, facecolors='none', edgecolor='blue')
    rect = [0.65, 0.05, 0.3, 0.3]
    ax1 = add_subplot_axis(ax,rect)
    ax1.hist(cluster_rgb.r, 30, color='red', alpha=0.3)
    ax1.hist(cluster_rgb.g, 30, color='green', alpha=0.3)
    ax1.hist(cluster_rgb.b, 30, color='blue', alpha=0.3)
    plt.savefig('plots/'+infile_name + '/cluster/scatter_cl_'+str(i)+'.png')
    print("%d: %d" % (i,cluster.x0.size))
    print("%d x mean: %f" % (i,cluster.x0.mean()))
    print("%d y mean: %f" % (i,cluster.x1.mean()))
    print("%d z mean: %f" % (i,cluster.x2.mean()))
    cl_meanx.append(cluster.x0.mean())
    cl_meany.append(cluster.x1.mean())
    plt.clf()
    #plt.figure(3)
    plt.figure(3)
    plt.hist2d(cluster.x0, cluster.x1, bins=(20,20), cmap=plt.cm.jet)
    plt.savefig('plots/'+infile_name + '/cluster/histo_cl_'+str(i)+'.png')
    plt.clf()

    resolution = 0.01
    xbins = np.arange(cluster.x0.min(), cluster.x0.max(), resolution)
    ybins = np.arange(cluster.x1.min(), cluster.x1.max(), resolution)
    dist_map = [[0 for n in range(len(xbins)-1)] for j in range(len(ybins) -1)]
    


    for index, cl in cluster.iterrows():
      for n in range(len(xbins) -1):
        if(cl['x0'] > xbins[n] and cl['x0'] < xbins[n+1]):
          for j in range(len(ybins) -1):
            if(cl['x1'] > ybins[j] and cl['x1'] < ybins[j+1]):
              dist_map[j][n] += math.fabs(cl['x2'])
    

    if (len(dist_map) != 0):
      plt.imshow(dist_map, interpolation='none', cmap=plt.cm.Purples, origin='lower')
      plt.savefig('plots/'+infile_name + '/cluster/map_cl_'+str(i)+'.png')
      plt.clf()
      img =30*np.asarray(dist_map)
      img_cv = img.astype(np.uint8)
      #img_cv = cv.medianBlur(img_cv,3)
      img_cv = cv.GaussianBlur(img_cv,(3,3),0)
      plt.imshow(img_cv, interpolation='none', cmap=plt.cm.Greens, origin='lower')
      plt.savefig('plots/'+infile_name + '/cluster/map_blurred_cl_'+str(i)+'.png')

      #img_gray = cv.cvtColor(img_cv, cv.COLOR_RGB2GRAY)
      circles = cv.HoughCircles(img_cv,cv.HOUGH_GRADIENT,1.5, 20, param1=10,param2=15,minRadius=5, maxRadius=30)
      plt.clf()

      if circles is not None:
        plt.figure(4)
        circles_i = np.uint16(np.around(circles))
        output = np.asarray([[0 for n in range(len(xbins)-1)] for j in range(len(ybins) -1)])
        output = cv.medianBlur(output.astype(np.uint8),3)
        img_cv = cv.cvtColor(img_cv, cv.COLOR_GRAY2BGR)
        output = cv.cvtColor(output, cv.COLOR_GRAY2BGR)
        plt.figure(4)
        fig, ax = plt.subplots()
        ax.cla()
        #plt.subplots()
        ax.plot(cluster.x0, cluster.x1, marker='o', linestyle='', color='black', markersize=0.7)
        for (xcir,ycir,rcir) in circles_i[0,:]:
          # draw the outer circle
          cv.circle(img_cv,(xcir,ycir),rcir,(0,255,0),1)
          cv.circle(img_cv, (xcir, ycir), 1, (0, 0, 255), 2)

          x_center = cluster.x0.min() + xcir*resolution
          y_center = cluster.x1.min() + ycir*resolution
          radius = rcir*resolution
          circle_fig = plt.Circle((cluster.x0.min() + xcir*resolution, cluster.x1.min() + ycir*resolution), rcir*resolution, color='b', fill = False) #the fit is made in cm
          ax.add_patch(circle_fig)
          print ("x_center: %4.2f" % x_center)
          print ("y_center: %4.2f" % y_center)
          print ("radius: %4.2f" % radius)
        plt.savefig('plots/'+infile_name + '/cluster/scatter_fit_cl_'+str(i)+'.png')
        plt.clf()
        plt.figure(5)
        plt.clf()
        print(cluster.x0.min() + circles[0,:][0][0] *resolution)
        print(cluster.x1.min() + circles[0,:][0][1] *resolution)
        print( circles[0,:][0][2] *resolution)
        x_center_0 = cluster.x0.min() + circles[0,:][0][0]*resolution
        y_center_0 = cluster.x1.min() + circles[0,:][0][1]*resolution
        radius_0 = circles[0,:][0][2]*resolution
        fit_x_center.append(cluster.x0.min() + circles[0,:][0][0] *resolution)
        fit_y_center.append(cluster.x1.min() + circles[0,:][0][1] *resolution)
        fit_radius.append(circles[0,:][0][2] *resolution)


        cluster_filtered = cluster[np.sqrt((cluster.x0- x_center_0)**2 + (cluster.x1 -y_center_0)**2) < 1.2*radius_0]
        print("normal vector x component: ",cluster_filtered.x0.mean() - x_center_0)
        print("normal vector y component: ",cluster_filtered.x1.mean() - y_center_0)
        print("normal vector z component: ",cluster_filtered.x2.mean() - cluster_filtered.x2.max() )
        print(np.sqrt((cluster.x0- x_center_0)**2 + (cluster.x1 -y_center_0)**2) < 1.2*radius_0)
        plt.plot(cluster_filtered.x0, cluster_filtered.x1, marker='o', linestyle='', markersize=1.0, color='midnightblue')
        plt.savefig('plots/'+infile_name+'/cluster/scatter_filtered_'+str(i) + '.png')

        #cv.imshow('detected circles',np.hstack([img_cv,output]))
        cv.imshow('detected circles',output)
        #cv.imwrite('plots/'+infile_name + '/cluster/cicle_fit_cl_'+str(i)+'.png', np.hstack([img_cv,output]))
        cv.imwrite('plots/'+infile_name + '/cluster/cicle_fit_cl_'+str(i)+'.png', img_cv)
      
    '''
  
    clust = clue.clusterer(0.2, 40, 1.5)
    clust.readData(cluster)
    clust.runCLUE()
  
    labels_per_cluster = np.asarray(clust.clusterIds)
    print(labels_per_cluster)
    for j in range(0, labels_per_cluster.max()+1,1):
      subcluster = cluster[labels_per_cluster==j]
  
      plt.figure(1)
      plt.plot(subcluster.x0, subcluster.x1, marker='o', linestyle='', markersize=0.7, label=r"Number of points: %d" % (subcluster.x0.size))
      plt.legend(fontsize=14)
      plt.savefig('plots/'+infile_name+'/subcluster/scatter_subcl_'+str(i)+'_'+str(j)+'.png')
      print("%d: %d" % (i,subcluster.x0.size))
      print("%d subcluster x mean: %f" % (i,subcluster.x0.mean()))
      print("%d subcluster y mean: %f" % (i,subcluster.x1.mean()))
      print("%d subcluster z mean: %f" % (i,subcluster.x2.mean()))
    '''

  
    ## finding circles in the z slices
    
  
  plt.figure(2)
  plt.savefig('plots/' + infile_name + '_scatter_all.png')
  plt.clf()
  return cl_meanx, cl_meany, fit_x_center, fit_y_center, fit_radius
  
  #clust.clusterPlotter()

def main():
  #slice_limits = [1300, 1400, 1500]
  #slice_limits = [1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100]
  #slice_limits = [2000, 2200, 2400, 2600, 2800, 3000]
  #slice_limits = [2000, 2300, 2600, 2900, 3200, 3500]
  slice_limits = [2000, 2150]

  x_all = []
  y_all = []
  z_all = []
  xc_all = []
  yc_all = []
  r_all = []
  for i in range(len(slice_limits) -1):
    slice_range = [slice_limits[i],slice_limits[i+1]]
    df = get_df(slice_range)
    calc_eff(df)
    x, y, fit_x_center, fit_y_center, fit_radius = clusterization_clue(df,slice_range)
    z =np.ones(len(x))*slice_limits[i]
    x_all.append(x)
    y_all.append(y)
    z_all.append(z)
    z =np.ones(len(fit_x_center))*slice_limits[i]
    xc_all.append(fit_x_center)
    yc_all.append(fit_y_center)
    r_all.append(fit_radius)
    
    #ax.plot(x, y, 'r+', zdir='y', zs=1.5)
  fig, ax=plt.subplots(figsize=(10,10))
  fig = plt.gcf()
  ax = plt.gca()
  ax.cla()
  #ax= fig.add_subplot(111, projection= '3d')
  for i in range(len(slice_limits) -1):
    ax.scatter(x_all[i],y_all[i], color='goldenrod')
    for xc, (yc, r) in zip(xc_all[i],zip(yc_all[i],r_all[i])):
      print(xc)
      print(yc)
      print(r)
      circle = plt.Circle( (xc, yc), 3*r, color='b', fill=False)
      ax.add_artist(circle)
  plt.savefig('test_magico.png')
  ax2= fig.add_subplot(111)
  for i in range(len(slice_limits) -1):
    ax2.scatter(x_all[i],y_all[i])
  plt.savefig('test_magico_2.png')

  n_sections = len(slice_limits)
  h5_outfile = h5py.File("results/tree_clusters_matching.h5", 'w')
  for i in range(n_sections -1):
    tracks = [[[] for k in range(len(x_all[i]))] for l in range(n_sections-1-i)]
    tracks_dist = [[[] for k in range(len(x_all[i]))] for l in range(n_sections-1-i)]
    xcord = [[[] for k in range(len(x_all[i]))] for l in range(n_sections-1-i)]
    ycord = [[[] for k in range(len(x_all[i]))] for l in range(n_sections-1-i)]
    print("###############################################################################")
    print("################## Layer: %d                ####################################" % (i))
    print("###############################################################################")
    ax2.scatter(x_all[i],y_all[i])
    for k, (p1x, p1y) in enumerate(zip(x_all[i],y_all[i])):
      for j in range(n_sections -1):
        if j >= i :
          min_dist = 100
          threshold = 0.4
          l_min_dist=-100
          for l, (p2x, p2y) in enumerate(zip(x_all[j],y_all[j])):
            dist = math.sqrt((p2x-p1x)**2+(p2y-p1y)**2)
            if(dist < min_dist):
              min_dist = dist
              l_min_dist = l
              xcord_min_dist = p2x
              ycord_min_dist = p2y
          if(min_dist > threshold):
            min_dist = -1
            l_min_dist =-99
          tracks[j-i][k].append(l_min_dist)
          tracks_dist[j-i][k].append(min_dist)
          xcord[j-i][k].append(p2x)
          ycord[j-i][k].append(p2y)

    df = pd.DataFrame()
    for m in range(n_sections -1):
      if m >= i :
        df["s%d_cl_dist"%(m)] = np.squeeze(np.asarray(tracks_dist[m-i]))
        df["s%d_cl_xcord"%(m)] = np.squeeze(np.asarray(xcord[m-i]))
        df["s%d_cl_ycord"%(m)] = np.squeeze(np.asarray(ycord[m-i]))
        #print("i %d, m %d"%(i,m))
        #print(tracks[m-i])
    print(df)
    h5_outfile.create_dataset('slice_ref_%d'%(i), data=df)

    #print(tracks)
    #print(tracks_dist)
  h5_outfile.close()
  
if __name__ == '__main__':
  main()
