import time
import time
import os
import random
import numpy as np                                                
import matplotlib.pyplot as plt                                   
import autograd, autograd.core, autograd.extend, autograd.tracer  
import autograd.numpy as anp      
import scipy, scipy.ndimage, scipy.sparse, scipy.sparse.linalg 

import torch 
                                                     
import gymnasium as gym
# from gym import spaces
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter

WIDTH = 12
HEIGHT = 6


class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

def get_args(normals, forces, density=1e-4):
    width = normals.shape[0] - 1
    height = normals.shape[1] - 1
    fixdofs = np.flatnonzero(normals.ravel())
    alldofs = np.arange((width+1) * (height+1) * 2)
    freedofs = np.sort(list(set(alldofs) - set(fixdofs)))
    params = {
        "young" : 1, "young_min": 1e-9, "poisson" : 0.3, "g": 0,
        "density" : density, "xmin": 0.001, "xmax": 1.0,
        "nelx": width, "nely": height, "mask": 1, "penal": 3.0, 
        "filter_width": 1,
        "freedofs": freedofs, "fixdofs": fixdofs, "forces": forces.ravel(),
        "opt_steps": 80, "print_every": 10, "plot": False
    }
    return ObjectView(params)


#TODO MAYBE HERE YOU CAN AJDUST THE NORMALS TO MAKE THEM CHANGEBLE
def mbb_beam(width = WIDTH, height = HEIGHT, density=1e-4, y=1, x=0, rd=1):
    normals = np.zeros((width+1, height+1), 2)
    normals[0, 0, x] = 1
    normals[0, 0, y] = 1
    normals[0, -1, x] = 1
    normals[0, -1, y] = 1
    forces = np.zeros((width+1, height+1, 2))
    forces[-1, rd, y] = -1
    return normals, forces, density

def young_modulus(x, e_0, e_min, p=3):
    return e_min + x ** p * (e_0 - e_min)

def physical_density(x, args, volume_constraint=False, use_filter=True):
    
    x = args.mask * x.reshape(args.nelx, args.nely) #unflatten x
    return gaussians_filter(x, args.filter_width) if use_filter else x

def mean_density(x, args, volume_constraint=False, use_filter=True):
    mean_density = anp.mean(physical_density(x, args, volume_constraint, 
                                             use_filter))
    mean = mean_density / anp.mean(args.mask)
    return mean

def objective(x, args, volume_constraint=False, use_filter=True):
    kwargs = dict(penal=args.penal, e_min=args.young_min, e_0=args.young)
    x_phys = physical_density(x, args, volume_constraint = volume_constraint,
                              use_filter = use_filter)
    ke = get_stiffness_matrix(args.young, args.poisson)
    u = displace(x_phys, ke, args.forces, args.freedofs, args.fixdofs, **kwargs)
    c = compliance(x_phys, u, ke, **kwargs)
    return c

@autograd.extend.primitive
def gaussian_filter(x, width):
    return scipy.ndimage.gaussian_filter(x, width, mode="reflect")

def _gaussian_filter_vjp(ans, x, width):
    del ans, x
    return lambda g: gaussian_filter(g, width)
autograd.extend.defvjp(gaussian_filter, _gaussian_filter_vjp)

def compliance(x_phys, u, ke, *, penal=3, e_min=1e-9, e_0=1):
    nely, nelx = x_phys.shape
    ely, elx = anp.meshgrid(range(nely), range(nelx))

    n1 = (nely+1) * (elx+0) + (ely+0)
    n2 = (nely+1) * (elx+1) + (ely+0)
    n3 = (nely+1) * (elx+1) + (ely+1)
    n4 = (nely+1) * (elx+0) + (ely+1)
    all_ixs = anp.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1])
    u_selected = u[all_ixs]

    ke_u = anp.einsum("ij", "jkl->ikl", ke, u_selected)

    ce = anp.einsum("ijk", "ijk->jk", u_selected, ke_u)
    C =young_modulus(x_phys, e_0, e_min, p=penal) * ce.T
    return anp.sum(C)

#WIE IN BESO ALGO
def get_stiffness_matrix(e,nu):
    k = anp.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    return e/(1-nu**2)*anp.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                               [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                               [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                               [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                               [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                               [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                               [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                               [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])

def get_k(stiffness, ke):

    nely, nelx = stiffness.shape
    ely, elx = anp.meshgrid(range(nely), range(nelx))
    ely, elx = ely. reshape(-1, 1), elx. reshape(-1, 1)

    n1 = (nely+1) * (elx+0) + (ely+0)
    n2 = (nely+1) * (elx+1) + (ely+0)
    n3 = (nely+1) * (elx+1) + (ely+1)
    n4 = (nely+1) * (elx+0) + (ely+1)
    edof = anp.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1])
    edof = edof.T[0]
    x_list = anp.repeat(edof, 8)
    y_list = anp.tile(edof, 8).flatten()

    kd = stiffness.T.reshape(nelx*nely, 1, 1)
    value_list = (kd* anp.tile(ke, kd.shape)).flatten()
    return value_list, y_list, x_list


def inverse_permutation(indices):
    inverse_perm = np.zeros(len(indices), dtype=anp.int64)
    inverse_perm[indices] = np.arange(len(indices),dtype=anp.int64)
    return inverse_perm

# TODO IF THERE IS AN ERROR CHECK THE K_YLIST AND K_XLIST ORDER
def _get_dof_indices(freedofs, fixdofs, k_ylist, k_xlist):
    index_map = inverse_permutation(anp.concatenate([freedofs, fixdofs]))
    keep = anp.isin(k_ylist, freedofs) & anp.isin(k_xlist, freedofs)
    i = index_map[k_ylist][keep]
    j = index_map[k_xlist][keep]

def _get_solver(a_entries, a_indices, size, sym_pos):
    a = scipy.sparse.coo_atrix(a_entries, a_indices, shape=(size,)*2).tocsc()
    if sym_pos:
        return scipy.sparse.linalg.factorized(a)
    else:
        return scipy.sparse.linalg.splu(a).solve

@autograd.primitive
def solve_coo(a_entries, a_indices, b, sym_pos=False):
    solver = _get_solver(a_entries, a_indices, b.size, sym_pos)
    return solver(b)

def grad_solve_coo_entries(ans, a_entries, a_indices, b, sym_pos=False):
    def jvp(grad_ans):
        lambda_ = solve_coo(a_entries, a_indices if sym_pos else a_indices[::-1], grad_ans, sym_pos)
        i, j = a_indices
        return -lambda_[i] * ans[j]
    return jvp

autograd.extend.defvjp(solve_coo, grad_solve_coo_entries, lambda: print("err:gradien undefined"), lambda: print("err:gradient not implmented"))


def displace(x_phys, ke, forces, freedofs, fixdofs, *, penal=3, e_min=1e-9, e_0=1):
    stiffness = young_modulus(x_phys, e_0, e_min, p=penal)
    k_entries, k_ylist, k_xlist = get_k(stiffness, ke)

    index_map, keep, indices = _get_dof_indices(freedofs, fixdofs, k_ylist, k_xlist)

    u_nonzero = solve_coo(k_entries[keep], indices, forces[freedofs], sym_pos=True)
    u_values = anp.concatenate([u_nonzero, anp.zeros(len(fixdofs))])
    return u_values[index_map]