
import numpy as np
import torch
import torchvision

#def normalization(train_tensor,valid_tensor,test_tensor):
def normalization(tensor):

    mean_pixels = torch.zeros(len(tensor))
    norm_tensor = tensor

    for i in range(len(tensor)):
        mean_pixels = torch.mean(tensor[i])
        std_pixels = torch.std(tensor[i])
        norm_tensor[i] = (tensor[i] - mean_pixels)/std_pixels

    return norm_tensor

    #mean_pixels = torch.zeros(len(train_tensor))

    #for i in range(len(train_tensor)):
    #    mean_pixels[i] = torch.mean(train_tensor[i])
    #    if len(torch.unique(train_tensor[i])) > 2:
    #        print(torch.unique(train_tensor[i]))

    #train_mean = torch.mean(mean_pixels)
    #train_std = torch.std(mean_pixels)

    #trainnorm_tensor = (train_tensor-train_mean)/train_std
    #validnorm_tensor = (valid_tensor-train_mean)/train_std
    #testnorm_tensor = (test_tensor-train_mean)/train_std

    # return trainnorm_tensor,validnorm_tensor,testnorm_tensor

def loss_mse(y_actual,y_predicted):
    loss = torch.mean((y_actual-y_predicted)**2,dim=1)
    return loss

def circ_shift(theta, periodicity = 2*np.pi):
    factor = 2*np.pi/periodicity
    tmp = torch.atan2(torch.sin(factor*theta), torch.cos(factor*theta))
    return torch.where(tmp<0 , 2*np.pi+tmp, tmp)/factor

def circ_dist(theta0, theta1, periodicity = 2*np.pi):
    #max_dist = 0.5*periodicity
    #dist = abs(max_dist - abs(max_dist - abs(theta0 - theta1)))

    diff = theta0 - theta1
    pdiff = torch.stack((diff - periodicity, diff, diff + periodicity), dim=1)
    indices = torch.argmin(torch.abs(pdiff), dim=1, keepdim=True)
    min_diff = torch.gather(pdiff, dim=1, index=indices)
    return min_diff

def loss_circ_gll(y_actual,y_predicted):
    error = circ_dist(y_actual, y_predicted[:,0], periodicity = np.pi)
    loss = 0.5*(error/y_predicted[:,1])**2 + torch.log(y_predicted[:,1]*np.sqrt(2*np.pi))
    return loss

def loss_gll(y_actual,y_predicted):
    y_predicted[:,1] += 0.001 # to allow for accurate predictions
    loss = 0.5*((y_actual-y_predicted[:,0])/y_predicted[:,1])**2 + torch.log(y_predicted[:,1]*np.sqrt(2*np.pi))

    return loss

#def loss_vmll(y_actual, y_predicted):
    #vmll_norm = y_actual
    #step = 0.05
    #for idx in np.arange(len(y_actual)):
        #vmll_norm[idx] = step*torch.sum(torch.exp(torch.cos(x)/y_predicted[:,1]**2)) for x in torch.arange(-np.pi,np.pi,step)
    #print(vmll_norm)
    #loss = -torch.cos(2*(y_actual-y_predicted[:,0]))/y_predicted[:,1]**2 + torch.log(vmll_norm)
    #return loss
