
import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
import Functions

num_im = 70000
num_pixels = 56 # 28
theta = np.zeros(num_im)
theta_noisy = np.zeros(num_im)
x1 = np.zeros(num_im)
y1 = np.zeros(num_im)
x2 = np.zeros(num_im)
y2 = np.zeros(num_im)
prop_inc = 1
width, height = num_pixels, num_pixels
sigma = 0.1

#directory = 'New_Images_' + str(num_pixels)
directory = 'New_Images_56noise01'
os.makedirs(directory, exist_ok=True)

os.chdir(directory)
angles = open("angles.txt","w+")
angles_noisy = open("angles_noisy.txt","w+")

periodicity = 2*np.pi # np.pi
for i in range(num_im):
    theta[i] = periodicity*random.random()
    theta_noisy[i] = Functions.circ_shift(torch.tensor(np.random.normal(theta[i], sigma)))

    line_thickness = 1
    r1 = 3*line_thickness*num_pixels/28
    r2 = 12*line_thickness*num_pixels/28
    center = (num_pixels-1)/2
    x1[i] = center+r1*np.cos(theta[i])
    y1[i] = center+r1*np.sin(theta[i])
    x2[i] = center+r2*np.cos(theta[i])
    y2[i] = center+r2*np.sin(theta[i])
    image = np.ones((width, height), dtype = "uint8")

    x = np.linspace(x1[i],x2[i],500)
    m = (y2[i]-y1[i])/(x2[i]-x1[i])
    n = (y1[i]*x2[i]-y2[i]*x1[i])/(x2[i]-x1[i])
    y = np.zeros(len(x))

    for j in range(len(x)):
        y[j] = m*x[j]+n

    x_pixel = np.zeros(len(x))
    y_pixel = np.zeros(len(x))

    for l in range(len(x)):
        x_pixel[l] = round(x[l])
        y_pixel[l] = round(y[l])

    for v in range(len(x_pixel)):
        image[int(x_pixel[v])][int(y_pixel[v])]=0
        for w in range(line_thickness+1):
            image[int(x_pixel[v])-w][int(y_pixel[v])]=0
            image[int(x_pixel[v])+w][int(y_pixel[v])]=0
            image[int(x_pixel[v])][int(y_pixel[v]-w)]=0
            image[int(x_pixel[v])][int(y_pixel[v]+w)]=0
            #image[int(x_pixel[v])-w][int(y_pixel[v]+w)]=0
            #image[int(x_pixel[v])+w][int(y_pixel[v]-w)]=0
            #image[int(x_pixel[v])-w][int(y_pixel[v]-w)]=0
            #image[int(x_pixel[v])+w][int(y_pixel[v]+w)]=0

    #Guardamos los datos en la carpeta Modificaciones
    name = 'Imagen' + str(i) + '.png'
    plt.imsave(name,image,cmap='gray')

    angles.write(str(theta[i])+"\n")
    angles_noisy.write(str(theta_noisy[i])+"\n")

angles.close()
angles_noisy.close()
