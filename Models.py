#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:30:56 2022

@author: benjapases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Functions

class ReLUNet(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.layer1 = nn.Linear(784, 100)
        torch.nn.init.xavier_uniform_(self.layer1.weight)#esto es para inicializar los pesos; lo hacemos en cada capa

        self.layer2 = nn.Linear(100, 100)
        torch.nn.init.xavier_uniform_(self.layer2.weight)

        self.layer3 = nn.Linear(100, 2)
        torch.nn.init.xavier_uniform_(self.layer3.weight)

        self.hidden_activation = nn.ReLU()
        self.temperature = temperature


    def forward(self, x):
        x = self.hidden_activation(self.layer1(x))
        x = self.hidden_activation(self.layer2(x))
        return self.layer3(x) / self.temperature


class TanhNet(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.layer1 = nn.Linear(784, 100)
        torch.nn.init.xavier_uniform_(self.layer1.weight)

        self.layer2 = nn.Linear(100, 100)
        torch.nn.init.xavier_uniform_(self.layer2.weight)

        self.layer3 = nn.Linear(100, 2)
        torch.nn.init.xavier_uniform_(self.layer3.weight)

        self.hidden_activation = nn.Tanh()
        self.temperature = temperature

    def forward(self, x):
        x = self.hidden_activation(self.layer1(x))
        x = self.hidden_activation(self.layer2(x))
        return self.layer3(x) / self.temperature

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        num_pixels = 56 # 28 # 56
        num_hneurons = 100 # round(100*num_pixels/28)
        num_bneurons = 2 # 1 # 2 # 5

        self.encoder = nn.Sequential(nn.Linear(num_pixels*num_pixels,num_hneurons),nn.ReLU(),nn.Linear(num_hneurons,num_bneurons))
        for i in np.arange(0,len(self.encoder),2):
            torch.nn.init.xavier_uniform_(self.encoder[i].weight)

        self.decoder = nn.Sequential(nn.Linear(num_bneurons,num_hneurons),nn.ReLU(),nn.Linear(num_hneurons,num_pixels*num_pixels))
        for i in np.arange(0,len(self.decoder),2):
            torch.nn.init.xavier_uniform_(self.decoder[i].weight)

        self.out_activation_tanh = nn.Tanh()
        self.out_activation_sigmoid = nn.Sigmoid()
        self.out_activation_relu = nn.ReLU()
        self.out_activation_elu = nn.ELU()

    def forward(self, x):
        o_encoder = self.encoder(x)
        first_slice = o_encoder[:,0]
        second_slice = o_encoder[:,1]
        out_encoder1 = self.out_activation_tanh(first_slice)
        out_encoder2 = self.out_activation_elu(second_slice) + 1
        out_encoder = torch.stack([out1,out2],dim=1)
        out_decoder = self.out_activation_sigmoid(self.decoder(out_encoder))
        return out_decoder

class piecewise_linear(nn.Module):
    #def __init__(self, in_features, alpha = None):
    def __init__(self, alpha = None):
        super(piecewise_linear, self).__init__()
        #self.in_features = in_features

        # initialize alpha
        if alpha == None:
            self.alpha = torch.tensor(0.0) # create a tensor out of alpha
        else:
            self.alpha = torch.tensor(alpha) # create a tensor out of alpha

        #self.alpha.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        x[x>1] = 1 + self.alpha*x
        x[x<0] = 0 + self.alpha*x
        return x

class FFN(nn.Module):
    def __init__(self):
        super().__init__()

        num_pixels = 56 # 28 # 56
        num_hneurons = 100 # round(100*num_pixels/28)

        #self.layers = nn.Sequential(nn.Linear(num_pixels*num_pixels,num_hneurons),nn.ReLU(),nn.Linear(num_hneurons,num_hneurons),nn.ReLU(),nn.Linear(num_hneurons,2))
        self.layers = nn.Sequential(nn.Linear(num_pixels*num_pixels,num_hneurons),nn.ReLU(),nn.Linear(num_hneurons,2))
        #self.layers = nn.Sequential(nn.Linear(num_pixels*num_pixels,num_hneurons),nn.SiLU(),nn.Linear(num_hneurons,50),nn.SiLU(),nn.Linear(50,10),nn.SiLU(),nn.Linear(10,2))
        #self.layers = nn.Sequential(nn.Linear(num_pixels*num_pixels,num_hneurons),nn.SiLU(),nn.Linear(num_hneurons,2))
        for i in np.arange(0,len(self.layers),2):
            torch.nn.init.xavier_uniform_(self.layers[i].weight)
        self.out_activation_tanh = nn.Tanh()
        self.out_activation_sigmoid = nn.Sigmoid()
        self.out_activation_relu = nn.ReLU()
        self.out_activation_elu = nn.ELU()

    def forward(self,x):
        olayer = self.layers(x)
        first_slice = olayer[:,0]
        second_slice = olayer[:,1]
        #out1 = first_slice
        #out1 = Functions.circ_shift(first_slice, np.pi)
        #out1 = -np.pi/4 + 5/4*np.pi*self.out_activation_sigmoid(first_slice)
        #out1 = -np.pi/4 + self.out_activation_relu(first_slice)
        #out1 = np.pi*self.out_activation_sigmoid(first_slice)
        out1 = self.out_activation_tanh(first_slice)
        out2 = self.out_activation_elu(second_slice) + 1
        out = torch.stack([out1,out2],dim=1)

        return out



class Auto_encod_trained(nn.Module):
    def __init__(self):
        super().__init__()

        num_pixels = 56 # 28 # 56
        num_hneurons = 100 # round(100*num_pixels/28)

        self.layers = nn.Sequential(nn.Linear(num_pixels*num_pixels,num_hneurons),nn.ReLU(),nn.Linear(num_hneurons,2))

        #self.layer1 = nn.Linear(2,100)
        #torch.nn.init.xavier_uniform_(self.layer1.weight)

        #self.layer2 = nn.Linear(100,784)
        #torch.nn.init.xavier_uniform_(self.layer2.weight)

        self.decoder = nn.Sequential(nn.Linear(2,num_hneurons),nn.ReLU(),nn.Linear(num_hneurons,num_pixels*num_pixels),nn.Sigmoid())
        for i in np.arange(0,len(self.decoder),2):
            torch.nn.init.xavier_uniform_(self.decoder[i].weight)

        self.hidden_activation = nn.ReLU()
        self.angle_activation = nn.Sigmoid()
        self.error_activation = nn.ELU()
        self.out_activation_tanh = nn.Tanh()
        self.out_activation_sigmoid = nn.Sigmoid()
        self.out_activation_relu = nn.ReLU()
        self.out_activation_elu = nn.ELU()


    def forward(self,x):
        #self.encoded.load_state_dict(torch.load('/home/benjapases/Desktop/TFM_Benja/Early_stopping/model_trained_encoder.pth'))
        #self.encoded.eval()
        out1 = self.layers(x)
        first_slice = out1[:,0]
        second_slice = out1[:,1]
        #out2 = np.pi*self.angle_activation(first_slice)
        out2 = self.out_activation_tanh(first_slice)
        out3 = self.out_activation_elu(second_slice) + 1
        out4 = torch.stack([out2,out3],dim=1)
        #out5 = self.hidden_activation(self.layer1(out4))
        #out6 = self.angle_activation(self.layer2(out5))
        outfinal = self.decoder(out4)

        return outfinal


#Inicilalizacion autoencoder con los pesos finales de Auto_encod_trained

class Auto_encdec_fix(nn.Module):
    def __init__(self):
        super().__init__()
        num_pixels = 56 # 28 # 56
        num_hneurons = 100 # round(100*num_pixels/28)

        self.layers = nn.Sequential(nn.Linear(num_pixels*num_pixels,num_hneurons),nn.ReLU(),nn.Linear(num_hneurons,2))

        self.decoder = nn.Sequential(nn.Linear(2,num_hneurons),nn.ReLU(),nn.Linear(num_hneurons,num_pixels*num_pixels),nn.Sigmoid())

        self.hidden_activation = nn.ReLU()
        self.angle_activation = nn.Sigmoid()
        self.error_activation = nn.ELU()
        self.out_activation_tanh = nn.Tanh()
        self.out_activation_sigmoid = nn.Sigmoid()
        self.out_activation_relu = nn.ReLU()
        self.out_activation_elu = nn.ELU()

    def forward(self,x):
        #self.encoded.load_state_dict(torch.load('/home/benjapases/Desktop/TFM_Benja/Early_stopping/model_trained_encoder.pth'))
        #self.encoded.eval()
        out1 = self.layers(x)
        first_slice = out1[:,0]
        second_slice = out1[:,1]
        #out2 = np.pi*self.angle_activation(first_slice)
        out2 = self.out_activation_tanh(first_slice)
        out3 = self.out_activation_elu(second_slice) + 1
        out4 = torch.stack([out2,out3],dim=1)
        #out5 = self.hidden_activation(self.layer1(out4))
        #out6 = self.angle_activation(self.layer2(out5))
        outfinal = self.decoder(out4)

        return outfinal
