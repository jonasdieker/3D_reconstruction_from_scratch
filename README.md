# Depth Reconstruction from Scratch

The first goal of this project is to provide a modular way to the 3D reconstruction problem. Any of the intermediate steps listed below can be replaced by different algorithms of the same task. The second goal is to write everything from scratch, i.e. using numpy only.

## Feature Matching

 - corner detection (Shi-Tomasi)
 - feature descriptor (Histogram of Oriented Gradients)
 - finding feature correspondences

![key_point_matching](assets/key_point_matching.png)

## Determine Relativ Transformation of Cameras

 - computing fundamental matrix E from point correspondences using coplanarity constraint $x'^T E x'' = 0$
 - obtain a homogeneous linear system -> solve using SVD
 - From E can recover R, b but not the scale

## Depth Reconstruction

 - Triangulation


## Future Extensions

- [ ] bundle adjustments -> going from a pair of images to n many images