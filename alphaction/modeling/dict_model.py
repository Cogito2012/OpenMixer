# Simple pytorch implementation of Dictionary Learning based on stochastic gradient descent
#
# June 2018
# Jeremias Sulam


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


####################################
    ## Dict. Learning ##
####################################

class DictLearn(nn.Module):
    def __init__(self, num_basis, dim_basis, SC='FISTA', sc_iters=None):
        super(DictLearn, self).__init__()

        self.W = nn.Parameter(torch.randn(dim_basis, num_basis, requires_grad=False))
        
        # normalization
        self.W.data = NormDict(self.W.data)
        self.SC = SC
        self.sc_iters = sc_iters
        
        if self.sc_iters is None:
            self.sc_iters = 20 if SC=='FISTA' else 50
        
    
        
    def forward(self, Y, K):
        
        # normalizing Dict
        self.W.requires_grad_(False)
        self.W.data = NormDict(self.W.data)
        
        # Sparse Coding
        if self.SC == 'IHT':
            Gamma, residual, errIHT = IHT(Y,self.W,K, self.sc_iters)
        elif self.SC == 'FISTA':
            Gamma, residual, errIHT = FISTA(Y,self.W,K, self.sc_iters)
        else: print("Oops!")
        
        # Reconstructing
        self.W.requires_grad_(True)
        X = torch.mm(Gamma,self.W.transpose(1,0))
        
        # sparsity
        # NNZ = np.count_nonzero(Gamma.cpu().data.numpy())/Gamma.shape[0]

        return X, Gamma, errIHT
        

        
#--------------------------------------------------------------
#         Auxiliary Functions
#--------------------------------------------------------------

def hard_threshold_k(X, k):
    Gamma = X.clone()
    m = X.data.shape[1]
    a,_ = torch.abs(Gamma).data.sort(dim=1,descending=True)
    T = torch.mm(a[:,k].unsqueeze(1),torch.Tensor(np.ones((1,m))).to(device))
    mask = Variable(torch.Tensor((np.abs(Gamma.data.cpu().numpy())>T.cpu().numpy()) + 0.)).to(device)
    Gamma = Gamma * mask
    return Gamma#, mask.data.nonzero()

#--------------------------------------------------------------


def soft_threshold(X, lamda):
    #pdb.set_trace()
    Gamma = X.clone()
    Gamma = torch.sign(Gamma) * F.relu(torch.abs(Gamma)-lamda)
    return Gamma.to(device)


#--------------------------------------------------------------


def IHT(Y,W,K, ITER=50):
    
    c = PowerMethod(W)
    eta = 1/c
    Gamma = hard_threshold_k(torch.mm(Y,eta*W),K)    
    residual = torch.mm(Gamma, W.transpose(1,0)) - Y
    
    norms = np.zeros((ITER,))

    for i in range(ITER):
        Gamma = hard_threshold_k(Gamma - eta * torch.mm(residual, W), K)
        residual = torch.mm(Gamma, W.transpose(1,0)) - Y
        norms[i] = np.linalg.norm(residual.cpu().numpy(),'fro')/ np.linalg.norm(Y.cpu().numpy(),'fro')
    
    return Gamma, residual, norms


#--------------------------------------------------------------


def FISTA(Y,W,lamda, ITER=20):
    
    c = PowerMethod(W)
    eta = 1/c
    norms = np.zeros((ITER,))
    
    Gamma = soft_threshold(torch.mm(Y,eta*W),lamda)
    Z = Gamma.clone()
    Gamma_1 = Gamma.clone()
    t = 1
    
    for i in range(ITER):
        Gamma_1 = Gamma.clone()
        residual = torch.mm(Z, W.transpose(1,0)) - Y
        Gamma = soft_threshold(Z - eta * torch.mm(residual, W), lamda/c)
        
        t_1 = t
        t = (1+np.sqrt(1 + 4*t**2))/2
        #pdb.set_trace()
        Z = Gamma + ((t_1 - 1)/t * (Gamma - Gamma_1)).to(device)
        
        norms[i] = np.linalg.norm(residual.cpu().numpy(),'fro')/ np.linalg.norm(Y.cpu().numpy(),'fro')
    
    return Gamma, residual, norms


#--------------------------------------------------------------

def NormDict(W):
    Wn = torch.norm(W, p=2, dim=0).detach()
    W = W.div(Wn.expand_as(W))
    return W

#--------------------------------------------------------------

def PowerMethod(W):
    ITER = 100
    m = W.shape[1]
    X = torch.randn(1, m).to(device)
    for i in range(ITER):
        Dgamma = torch.mm(X,W.transpose(1,0))
        X = torch.mm(Dgamma,W)
        nm = torch.norm(X,p=2)
        X = X/nm
    
    return nm

#--------------------------------------------------------------


def showFilters(W,ncol,nrows):
    p = int(np.sqrt(W.shape[0]))+2
    Nimages = W.shape[1]
    Mosaic = np.zeros((p*ncol,p*nrows))
    indx = 0
    for i in range(ncol):
        for j in range(nrows):
            im = W[:,indx].reshape(p-2,p-2)
            im = (im-np.min(im))
            im = im/np.max(im)
            Mosaic[ i*p : (i+1)*p , j*p : (j+1)*p ] = np.pad(im,(1,1),mode='constant')
            indx += 1
            
    return Mosaic
