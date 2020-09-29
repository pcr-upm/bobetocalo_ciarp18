# Facial Landmarks Detection using a Cascade of Recombinator Networks

We provide C++ code in order to replicate the CRN experiments in our paper https://bobetocalo.github.io/pdf/paper_ciarp18.pdf

If you use this code for your own research, you must reference our CIARP paper:

```
Facial Landmarks Detection using a Cascade of Recombinator Networks
Pedro Diego López, Roberto Valle, Luis Baumela.
Conference on Progress in Pattern Recognition, Image Analysis, Computer Vision and Applications, 23nd Iberoamerican Congress, CIARP 2018, Madrid, Spain, November 19-22, 2018.
```

#### Requisites
- faces_framework https://github.com/bobetocalo/faces_framework

#### Installation
This repository must be located inside the following directory:
```
faces_framework
    └── alignment 
        └── bobetocalo_ciarp18
```
You need to have a C++ compiler (supporting C++11):
```
> mkdir release
> cd release
> cmake ..
> make -j$(nproc)
> cd ..
```
#### Usage
Use the --database option to load the proper trained model.
```
> ./release/face_alignment_bobetocalo_ciarp18_test --database 300w_public
```
