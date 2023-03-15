# DL systems || Homework 4

This is assignment is a part of [DL-systems course presented by CMU](https://dlsyscourse.org/). 

In this HW, we try to merge the backend implemented in [HW3](https://github.com/ahmedtarek1325/dlsys_hw3) with the system we have built in [HW2](https://github.com/ahmedtarek1325/dlsys_hw2). Then we introduce other important functionality as CONV2D and RNN. 
## What you will find in this repo


This HW is the final step for building our own DL library. In this HW will finally build


Navigating this repo you will find an implementation for the following: 
1.  Extending some Ops files as 
    - **Conv**
    - Flip
    - Dilate
    - UnDilate
    - Tanh 
    - Stack 
    - Split 
2. EXtending some nn layers as: 
    - Flatten
    - Conv
3.  Adding a ResNet9 model. 





## What does a tensor consists of? 



## Tests
To run tests you can use the following commands in the repo directory: 
ps if you are running in jupyter add `!`  before the following commands

- `python3 -m pytest -l -v -k "nd_backend"`
- `python3 -m pytest -l -v -k "dilate"`
- `python3 -m pytest -l -v -k "flip"`
- Conv Forward  and Bacward
    - `python3 -m pytest -l -v -k "nn_conv_backward"`
    - `python3 -m pytest -l -v -k "nn_conv_backward"`





---
## my AHA moments
1. Convolution and correlation are two fundamental concepts that have wide applications in statistics, Signals, classical computer vision and DL. But how to make them more efficient? 

    In DL, we have multiple CNN layers in just one model; in addition, in one convolution, we may have multiple channels **doing this via looping would be extremely slow** 
    To mitigate this, U can see fast Fourier or something, or u may see the im2col algorithm u can see this [video for Zico kolter](https://youtu.be/7kclgMIcMq0) to see how to implement it here, or you can see his [notebook] directly. (https://github.com/dlsyscourse/public_notebooks/blob/main/convolution_implementation.ipynb) 

2. Final Comment on **"Abstraction and building complex systems on diffeerent amount of time"**
    
    After working on this project for almost three months, I struggled a lot in the last part because I was violating some assumptions that were put. 
    
    For instance, in HW1 and HW2, I've been working on the Numpy backend. Therefore a lot of easy, nice features, such as using -ve numbers in indexing arrays from the end. 

    Then in HW3, We put an assumption that we will not handle -ve numbers in indexing arrays while working on the CPU and GPU backends. 

    However, on the last homework, when we started to copy and paste the stuff we had added in HW3 to HW1 and HW2, **I faced tons of errors**after trying just to run the code for two days without adding anything fancy yet in HW4. I found out that all the problems came from the idea I was using -ve indices in HW1 and HW2 without noticing that I no longer have this feature. 

    Consequently, the learned lessons that I'll use myself while working on big projects would try: 
    - try to always write ur assumptions in bold and always keep your eyes on them. 
    - Always think of the communication between different parts of the project, which parts are eligible to engage which parts. 
    - Try to make a schematic for the constructed part alongside having a plan for the bigger picture. 

--- 

## Want to see more of the assignments ? 
### Click here to see the rest of the assignments and my take outts
1. [HW1](https://github.com/ahmedtarek1325/dlsys_hw1)
2. [HW2](https://github.com/ahmedtarek1325/dlsys_hw2)
3. [HW3](https://github.com/ahmedtarek1325/dlsys_hw3)
4. [HW4](https://github.com/ahmedtarek1325/dlsys_hw4)

## Refrences:  
1. [Lectures [10,14,15,18,19] to understand the theory](https://dlsyscourse.org/lectures/)

