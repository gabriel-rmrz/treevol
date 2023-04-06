# treevol
Computation of tree volumes using machine learning.

## How to run

### Conda environment
Install miniconda.

Create the a new condda environment:

'''
conda env create -f env_las.py  
'''
### Scripts

#### toy_model.py
- Reads a point cloud from a .las file
- 
'''
python toy_model.py
'''

#### prepare_data_for_clustering.py

'''
python prepare_data_for_clustering.py
'''

#### clustering.py
'''
python clustering.py
'''

## TO DO
#### research and implement the clusterization of the point clouds.
- probably the z direction can be ignored as first aproximaximation. Plot total points ignoring z and contour curves.
  - in fact a section in z can be used for this task. Let's say from 2 m - 2.2 meters, this would reduce the size of the input data and also the complexity
  - test as well what taking away the outliers do to the point of the leave, which are noise at the moment.
  - we could apply this method at two different heights and compere between them to improve the accuracy
  - make a script to prepare data for clustering
#### Segmentation
- for this part pointnet++ seems to be still the best option.
  - Make the test with the segmetation part.

#### CD/CI
- Start the implementation.
- is it possible to add two different repositories to one pipeline?




## Main milestones
- identify individual trees.
  - find a way to tell trees from bushes (and probabaly other non tree objects)
- identify features in every tree (branches, trunk, soil) (poinet++?)
- more precise volume computation

