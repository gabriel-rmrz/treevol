# treevol
Computation of tree volumes using machine learning.



## TO DO
- research and implement the clusterization of the point clouds.
  - probably the z direction can be ignored as first aproximaximation. Plot total points ignoring z and contour curves.
    - in fact a section in z can be used for this task. Let's say from 2 m - 2.2 meters, this would reduce the size of the input data and also the complexity
    - test as well what taking away the outliers do to the point of the leave, which are noise at the moment.
    - we could apply this method at two different heights and compere between them to improve the accuracy
    - make a script to prepare data for clustering



## Main milestones
- identify individual trees.
  - find a way to tell trees from bushes (and probabaly other non tree objects)
- identify features in every tree (branches, trunk, soil) (poinet++?)
- more precise volume computation

