# treevol
Computation of tree volumes using machine learning.

## How to run

### Conda environment
Install miniconda.

Create the a new condda environment:

```
conda env create -f env_las.py  
```


### Scripts

#### toy_model.py
- Reads a point cloud from a .las file
- 
```
python toy_model.py
```

#### prepare_data_for_clustering.py

```
python prepare_data_for_clustering.py
```

#### clustering.py
```
python clustering.py
```

## TO DO
#### DTM
- maybe get a sample with more uneven ground. I think Open 3d is just making squere cuts to take away the ground points.

#### Clustering

- [ ] convert point cloud proyections and used CNN as performed in https://cds.cern.ch/record/2209337/files/SKalgorithm_CERN_Report.pdf
- [x] Add RGB histogram in the per tree plots
- [ ] Use bigger sample ( with more trees) 
  - [ ] Find an automise way to set the parameters for clusterizer taking into account the total sample information.
    - [ ] ASK GAETANO: Is there a way to find the density of points of the sample?
    - [ ] of the tree and the size of the slide is also important at the moment of defining the parameters in CLUE
- [ ] Try the tools for clustering given by Emanuele.
- [ ] Take slices at different height and compare them to make the selection more robust.
- [ ] Investigate how the ouliers are defined in CLUE.
  - [ ] Are all the point put into a cluster?
- [ ] use the RGB information
  - [ ] in the case of CLUE we have to find a way to define the weights.
- [ ] if still using k-means, research the different metrics.


#### Segmentation
- for this part pointnet++ seems to be still the best option.
  - Make the test with the segmetation part.


#### CD/CI
- Start the implementation.
- is it possible to add two different repositories to one pipeline?

## DONE
#### research and implement the clusterization of the point clouds.
- [x] probably the z direction can be ignored as first aproximaximation. Plot total points ignoring z and contour curves. << it certainly can be ignored, but considering the number of point(around 1000) the z coordinate can be kept. This is using CLUEstering.>>
  - in fact a section in z can be used for this task. Let's say from 2 m - 2.2 meters, this would reduce the size of the input data and also the complexity << this seems to be working >>
  - test as well what taking away the outliers do to the point of the leave, which are noise at the moment. << This depends on every method, we have to look deeper into this >>
  - we could apply this method at two different heights and compere between them to improve the accuracy << still to do >>
  - make a script to prepare data for clustering << DONE >>
- Add more features for the training like the RGB information (maybe the number id of the foto used for the reconstruction of the different points) << still to do >>
- Try other metrics. << still to do >>


## Main milestones
- identify individual trees.
  - find a way to tell trees from bushes (and probabaly other non tree objects)
- identify features in every tree (branches, trunk, soil) (poinet++?)
- more precise volume computation

