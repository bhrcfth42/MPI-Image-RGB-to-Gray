# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 13:23:42 2020

@author: fatih
"""

import numpy as np
from PIL import Image
import time
from mpi4py import MPI

def to_gray(list_of_pixels):
    for px in list_of_pixels:
        gray=0
        for rgb in px:
            gray+=rgb
        gray/=3
        px[0]=gray
        px[1]=gray
        px[2]=gray
    return list_of_pixels.ravel()

comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

start_time = time.time()
if rank==0:
    im = Image.open('Fruits-Watermelon.jpg', 'r')
    list_of_pixels=np.array(list(im.getdata())).ravel()
    new_list=np.empty([int(len(list_of_pixels)/size)],dtype=int)
    resim_new=np.empty([int(len(list_of_pixels))],dtype=int)
else:
    list_of_pixels=None
    new_list=np.empty([int(3000000/size)],dtype=int)
    resim_new=np.empty([3000000],dtype=int)

comm.Scatter(list_of_pixels,new_list,root=0)

new_list=to_gray(new_list.reshape(-1,3))

print("Islemci : ",rank,"\n")

comm.Gather(new_list,resim_new)

if(rank==0):
    im_new = Image.fromarray(resim_new.reshape(-1,1000,3).astype(np.uint8))
    im_new.save("Fruits-Watermelon_gray_mpi.jpg")

print("--- %s saniye ---" % (time.time() - start_time))