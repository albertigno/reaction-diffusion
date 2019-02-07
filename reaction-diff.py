#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:22:49 2019

@author: alberto

reaction - diffusion (Gray-Scott model)
implemented from http://www.karlsims.com/rd.html
and https://www.youtube.com/watch?v=BV9ny785UNc&list=PLRqwX-V7Uu6ZiZxtDDRCi6uhfTH4FilpH&index=17&t=0s


It models the change in space and time of the concentration of one or more
chemical substances: local chemical reactions that transform the substances and
their diffusion or spread over a surface in space.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

grid_size=200
sim_time = 10000

Da = 1.0
Db = 0.2
f = 0.055
k = 0.062 # kill rate
dt = 0.5
# initial grid
#grid = np.array([np.random.rand(grid_size,grid_size),
#                 np.zeros((grid_size,grid_size)),
#                 np.random.rand(grid_size,grid_size)]).swapaxes(0,2)

grid = np.array([np.ones((grid_size,grid_size)),
                 np.zeros((grid_size,grid_size)), 
                 np.zeros((grid_size,grid_size))]).swapaxes(0,2)

square_size=2
for i in range(square_size):
    for j in range(square_size):
        grid[i+grid_size/2-square_size/2, j+grid_size/2-square_size/2, 2] = 1
        grid[i+grid_size/2-square_size/2, j+grid_size/2-square_size/2, 0] = 0

# compute laplacian
def laplace(input_grid):
    mask = np.array([[0.05,0.2,0.05],[0.2,-1.0,0.2],[0.05,0.2,0.05]])
    laplacian = signal.convolve2d(input_grid, mask, boundary='symm', mode='same')
    return laplacian

def update_grid(grid):
    lap_A = laplace(grid[:,:,0])
    lap_B = laplace(grid[:,:,2])
    new_grid = np.zeros(grid.shape)
    for x in range(grid_size):
        for y in range(grid_size):
            A = grid[x,y,0];
            B = grid[x,y,2];
            new_grid[x,y,0] = (A + (Da * lap_A[x,y] - A*B*B + f*(1-A) ) * dt)
            new_grid[x,y,2] = (B + (Db * lap_B[x,y] + A*B*B - B*(k+f) ) * dt)
    return new_grid
    
fig=plt.figure('reaction-diffusion')
im = plt.imshow(np.zeros((grid_size,grid_size,3)))

for t in range(sim_time):
    grid = update_grid(grid)
    im.set_data(grid)
    plt.draw()
    print (t)
    plt.pause(0.0001)
    

plt.show()