import numpy as np
import os
import torch

def cos_similarity(x1,x2):
    t1 = x1.dot(x2.T)
    x1_linalg = np.linalg.norm(x1,axis=1)
    x2_linalg = np.linalg.norm(x2,axis=1)
    x1_linalg = x1_linalg.reshape((x1_linalg.shape[0],1))
    x2_linalg = x2_linalg.reshape((1,x2_linalg.shape[0]))
    t2 = x1_linalg.dot(x2_linalg)
    cos = t1/t2
    
    return cos

def calculate_S(z):
    S = cos_similarity(z,z)


    return S

def calculate_rou(S,p,rate=0.4):

    m = S.shape[0]
    
    t = int(rate*m*m)
    temp = np.sort(S.reshape((m*m,)))
    Sc = temp[-t]
    
    rou = np.sum(np.sign(S-Sc),axis=1)
    rank = np.argsort(rou)[:p]
    
    return rank

def get_prototypes(fea0, fea1, k, p):
    prototypes_list = []
    z_samples = []
    z_samples.append(fea0)
    z_samples.append(fea1)
    for i in range(k):
        S = calculate_S(z_samples[i])
        prototype_index = calculate_rou(S,p)
        print(prototype_index)
        prototypes = z_samples[i][prototype_index]
        prototypes_list.append(prototypes)

    return prototypes_list


def produce_pseudo_labels(prototypes_list, z):
    prototypes = prototypes_list

    new_y = np.zeros([7742, 2])
    n = z.shape[0]
    sigma = np.zeros((n, 2))
    for c in range(2):
        proto = prototypes[c]
        S = cos_similarity(z, proto)
        sigma[:, c] = np.mean(S, axis=1)

    y_pseudo = np.argmax(sigma, axis=1)
    new_y[:, 0] = y_pseudo
    new_y[:, 1] = 1 - new_y[:, 0]


    return new_y