import sys
import os
import time
import random
import math
import datetime

import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt


import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


import tensorflow as tf

HEIGHT = 5
"""The height of the design space."""
WIDTH = 10
"""The width of the design space."""
bound_nodes_list = [(0, 0), (HEIGHT-1, 0)]
"""A list containing the coordinates of the nodes that are bound in the 
design space."""
loaded_nodes_list = [(HEIGHT-1, WIDTH-1)]
"""A list containing the coordinates of the nodes that are loaded in the 
design space."""

#Aluminium 6061

YOUNG_MODULUS = 68
POISSON_RATIO = 0,32

YIELD_STRENGTH = 275 #MPa
STRAIN_LIMIT = 0.002

NUMBER_SUBPROCESSES = 1
"""The number of subprocesses to be used in the multiprocessing 
environment."""
LOG_DIR = "log/"
"""The directory of where to save the best model."""
TS_BOARD_DIR = "ts_board/"
"""The directory of where to save the tensorboard logs."""
TIMESTEPS = 5e6
"""The number of timesteps to be used during the training of the model."""

DESIGN = 0
"""The Constant Used to access the design dimension of the design space"""
BOUND = 1
"""The Constant used to access the Boundary layout of the design space"""
LOADED_X = 2
"""The Constant used to access the nodes which are loaded in the x direction"""
LOADED_Y = 3
"""The Constant used to access the nodes which are loaded in the y direction"""

mpl.rcParams['image.cmap'] = 'YlOrBr'

def shape(xi):
	"""Shape functions for a 4-node, isoparametric element
		N_i(xi,eta) where i=[1,2,3,4]
		Input: 1x2,  Output: 1x4"""
	xi,eta = tuple(xi)
	N = [(1.0-xi)*(1.0-eta), (1.0+xi)*(1.0-eta), (1.0+xi)*(1.0+eta), (1.0-xi)*(1.0+eta)]
	return 0.25 * np.array(N)
def gradshape(xi):
	"""Gradient of the shape functions for a 4-node, isoparametric element.
		dN_i(xi,eta)/dxi and dN_i(xi,eta)/deta
		Input: 1x2,  Output: 2x4"""
	xi,eta = tuple(xi)
	dN = [[-(1.0-eta),  (1.0-eta), (1.0+eta), -(1.0+eta)],
		  [-(1.0-xi), -(1.0+xi), (1.0+xi),  (1.0-xi)]]
	return 0.25 * np.array(dN)

def assemble_load(nodes, load, f):
  for load_data in load:
    node_index = load_data[0]
    dof = load_data[1]
    magnitude = load_data[2]
    global_dof = 2 * node_index + dof - 1
    f[global_dof] += magnitude

def convert_relative_to_absolute(coord, matrix_dim):
    """
    Convert relative indices in a coordinate to absolute indices.

    Parameters:
    coord (tuple): The coordinate with potentially relative indices.
    matrix_dim (tuple): The dimensions of the matrix.

    Returns:
    tuple: The coordinate with absolute indices.
    """
    return tuple(dim + i if i < 0 else i for i, dim in zip(coord, matrix_dim))

def convert_all(coords, matrix_dim):
    """
    Convert all relative indices in a list of coordinates to absolute indices.

    Parameters:
    coords (list): The list of coordinates with potentially relative indices.
    matrix_dim (tuple): The dimensions of the matrix.

    Returns:
    list: The list of coordinates with absolute indices.
    """
    return list(map(lambda coord: convert_relative_to_absolute(coord, matrix_dim), 
                    coords))

def extract_nodes_and_elements(design):
    nodes = []
    elements = []
    node_id = 0
    node_map = {}

    filled_indices = np.where(design == 1)
    
    for i, j in zip(*filled_indices):
        # Define nodes for the filled pixel
        pixel_nodes = [
            (i, j), (i, j+1),
            (i+1, j), (i+1, j+1)
        ]
        
        for node in pixel_nodes:
            if node not in node_map:
                node_map[node] = node_id
                nodes.append([float(node[0]), float(node[1])])
                node_id += 1
        
        # Define elements by connecting nodes
        n1, n2, n3, n4 = (node_map[(i, j)], node_map[(i, j+1)], 
                          node_map[(i+1, j)], node_map[(i+1, j+1)])
        elements.append([n1, n3, n4, n2])
        elements.append([n4, n2, n1, n3])


    return nodes, elements, node_map


def extract_boundary_conditions(bounded, loaded_x, loaded_y, node_map,):
    boundary_conditions = []
    rows, cols = bounded.shape



    # for i in range(rows):
    #     for j in range(cols):
    #         if loaded_x[i, j] != 0 or loaded_y[i, j] != 0:
    #             # Apply loads to all corners of the pixel
    #             pixel_nodes = [
    #                 (i, j), (i, j+1),
    #                 (i+1, j), (i+1, j+1)
    #             ]
    #             for node in pixel_nodes:
    #                 node_id = node_map.get(node)
    #                 if node_id is not None:
    #                     if loaded_x[i, j] != 0:
    #                         #boundary_conditions.append([node_id, 1, 1, loaded_x[i, j]])
    #                         boundary_conditions.append([node_id, 2, 2, 0.0])
                            
    #                     if loaded_y[i, j] != 0:
    #                         #boundary_conditions.append([node_id, 2, 2, loaded_y[i, j]])
    #                         boundary_conditions.append([node_id, 1, 1, 0.0])


    for i in range(rows):
        for j in range(cols):
            if bounded[i, j] != 0:
                # Apply boundary conditions to all corners of the pixel
                pixel_nodes = [
                    (i, j), (i, j+1),
                    (i+1, j), (i+1, j+1)
                ]
                for node in pixel_nodes:
                    node_id = node_map.get(node)
                    if node_id is not None:
                        if(bounded[i, j] == 1):
                            boundary_conditions.append([node_id, 1, 1, 0.0])  # Example: fixed in x-direction
                        if(bounded[i, j] == 2):
                            boundary_conditions.append([node_id, 2, 2, 0.0])  # Example: fixed in y-direction
                        if(bounded[i, j] == 3):
                            boundary_conditions.append([node_id, 1, 1, 0.0])  # Example: fixed in x-direction
                            boundary_conditions.append([node_id, 2, 2, 0.0])  # Example: fixed in y-direction


    return boundary_conditions

# n, 1, 2, 0.0 'fixed in both directions
# n, 1, 1 ,0.0 'fixed in dof 1 direction
# n, 1, 1, 0.1 moved in dof1 free in dof2
# n , 2, 2 0.0 fixed in dof2 direction

def extract_loads(loaded_x, loaded_y, node_map):
    """Extracts load information from the loaded matrices.

    Args:
        loaded_x: Numpy array representing the x-component of the load.
        loaded_y: Numpy array representing the y-component of the load.
        node_map: Dictionary mapping node coordinates to node IDs.

    Returns:
        List of load information, where each element is a tuple (node_id, load_x, load_y).
    """

    loads = []
    rows, cols = loaded_x.shape

    for i in range(rows):
        for j in range(cols):
            if loaded_x[i, j] != 0 or loaded_y[i, j] != 0:
                # Apply loads to all corners of the pixel
                pixel_nodes = [
                    (i, j), (i, j+1),
                    (i+1, j), (i+1, j+1)
                ]
                for node in pixel_nodes:
                    node_id = node_map.get(node)
                    if node_id is not None:
                        if loaded_x[i, j] != 0:
                            loads.append([node_id, 1, loaded_x[i, j]])
                        if loaded_y[i, j] != 0:
                            loads.append([node_id, 2, loaded_y[i, j]])

    return loads

def extract_fem_data(matrix):
    design = matrix[DESIGN, :, :]
    bounded = matrix[BOUND, :, :]
    loaded_x = matrix[LOADED_X, :, :]
    loaded_y = matrix[LOADED_Y, :, :]

    nodes, elements, node_map = extract_nodes_and_elements(design)
    boundary_conditions = extract_boundary_conditions(bounded, loaded_x, loaded_y, node_map)
    loads = extract_loads(loaded_x, loaded_y, node_map)

    return (nodes, elements, boundary_conditions, loads)

def scale_matrix(matrix, target_rows, target_cols):
    """A function that scales a given matrix to a target size. 
    The scaling is done by repeating the elements of the matrix.
    ----------
    Parameters:\n
    - matrix : numpy.ndarray
        - The matrix that should be scaled.
    - target_rows : int
        - The target number of rows of the scaled matrix.
    - target_cols : int
        - The target number of columns of the scaled matrix.
    -------
    Returns:\n
    - numpy.ndarray
        - The scaled matrix.
    """
    
    original_rows = len(matrix)
    original_cols = len(matrix[0]) if original_rows > 0 else 0
    
    # Calculate scale factors
    row_scale_factor = target_rows // original_rows
    col_scale_factor = target_cols // original_cols
    
    # Scale the matrix
    scaled_matrix = []
    for row in matrix:
        # Scale each row horizontally
        scaled_row = []
        for element in row:
            scaled_row.extend([element] * col_scale_factor)
        
        # Scale the matrix vertically
        for _ in range(row_scale_factor):
            scaled_matrix.append(list(scaled_row))
    
    # Handle any remaining rows due to non-integer scale factors
    additional_rows = target_rows % original_rows
    if additional_rows > 0:
        for i in range(additional_rows):
            scaled_matrix.append(list(scaled_matrix[i]))
    
    # Handle any remaining columns due to non-integer scale factors
    additional_cols = target_cols % original_cols
    if additional_cols > 0:
        for row in scaled_matrix:
            row.extend(row[:additional_cols])
    
    return np.array(scaled_matrix)

def FEM(nodes, conn, boundary, load):
    nodes = np.array(nodes)
    num_nodes = len(nodes)
    #print('   number of nodes:', len(nodes))
    #print('   number of elements:', len(conn))
    #print('   number of displacement boundary conditions:', len(boundary))

    ###############################
    # Plane-strain material tangent (see Bathe p. 194)
    # C is 3x3
    E = 100.0
    v = 0.3
    C = E/(1.0+v)/(1.0-2.0*v) * np.array([[1.0-v, v, 0.0], [v, 1.0-v, 0.0], [0.0, 0.0, 0.5-v]])
    ###############################
    # Make stiffness matrix
    # if N is the number of DOF, then K is NxN
    K = np.zeros((2*num_nodes, 2*num_nodes))    # square zero matrix
    # 2x2 Gauss Quadrature (4 Gauss points)
    # q4 is 4x2
    q4 = np.array([[-1,-1],[1,-1],[-1,1],[1,1]]) / math.sqrt(3.0)
    # print('\n** Assemble stiffness matrix')
    # strain in an element: [strain] = B    U
    #                        3x1     = 3x8  8x1
    #
    # strain11 = B11 U1 + B12 U2 + B13 U3 + B14 U4 + B15 U5 + B16 U6 + B17 U7 + B18 U8
    #          = B11 u1          + B13 u1          + B15 u1          + B17 u1
    #          = dN1/dx u1       + dN2/dx u1       + dN3/dx u1       + dN4/dx u1
    B = np.zeros((3,8))
    # conn[0] is node numbers of the element
    for c in conn:     # loop through each element
        # coordinates of each node in the element
        # shape = 4x2
        # for example:
        #    nodePts = [[0.0,   0.0],
        #               [0.033, 0.0],
        #               [0.033, 0.066],
        #               [0.0,   0.066]]
        nodePts = nodes[c,:]
        Ke = np.zeros((8,8))	# element stiffness matrix is 8x8
        for q in q4:			# for each Gauss point
            # q is 1x2, N(xi,eta)
            dN = gradshape(q)       # partial derivative of N wrt (xi,eta): 2x4
            J  = np.dot(dN, nodePts).T # J is 2x2
            dN = np.dot(np.linalg.inv(J), dN)    # partial derivative of N wrt (x,y): 2x4
            # assemble B matrix  [3x8]
            B[0,0::2] = dN[0,:]
            B[1,1::2] = dN[1,:]
            B[2,0::2] = dN[1,:]
            B[2,1::2] = dN[0,:]
            # element stiffness matrix
            Ke += np.dot(np.dot(B.T,C),B) * np.linalg.det(J)
        # Scatter operation
        for i,I in enumerate(c):
            for j,J in enumerate(c):
                K[2*I,2*J]     += Ke[2*i,2*j]
                K[2*I+1,2*J]   += Ke[2*i+1,2*j]
                K[2*I+1,2*J+1] += Ke[2*i+1,2*j+1]
                K[2*I,2*J+1]   += Ke[2*i,2*j+1]
    ###############################
    # Assign nodal forces and boundary conditions
    #    if N is the number of nodes, then f is 2xN
    f = np.zeros((2*num_nodes))          # initialize to 0 forces
    assemble_load(nodes, load, f)
    # How about displacement boundary conditions:
    #    [k11 k12 k13] [u1] = [f1]
    #    [k21 k22 k23] [u2]   [f2]
    #    [k31 k32 k33] [u3]   [f3]
    #
    #    if u3=x then
    #       [k11 k12 k13] [u1] = [f1]
    #       [k21 k22 k23] [u2]   [f2]
    #       [k31 k32 k33] [ x]   [f3]
    #   =>
    #       [k11 k12 k13] [u1] = [f1]
    #       [k21 k22 k23] [u2]   [f2]
    #       [  0   0   1] [u3]   [ x]
    #   the reaction force is
    #       f3 = [k31 k32 k33] * [u1 u2 u3]
    for i in range(len(boundary)):  # apply all boundary displacements
        nn  = boundary[i][0]
        dof = boundary[i][1]
        val = boundary[i][2]
        j = 2*nn
        if dof == 2: j = j + 1
        K[j,:] = 0.0
        K[j,j] = 1.0
        f[j] = val
    ###############################
    # print('\n** Solve linear system: Ku = f')	# [K] = 2N x 2N, [f] = 2N x 1, [u] = 2N x 1
    u = np.linalg.solve(K, f)
    ###############################
    # print('\n** Post process the data')
    # (pre-allocate space for nodal stress and strain)
    node_strain = []
    node_stress = []
    for ni in range(len(nodes)):
        node_strain.append([0.0, 0.0, 0.0])
        node_stress.append([0.0, 0.0, 0.0])
    node_strain = np.array(node_strain)
    node_stress = np.array(node_stress)

    #print(f'   min displacements: u1={min(u[0::2]):.4g}, u2={min(u[1::2]):.4g}')
    #print(f'   max displacements: u1={max(u[0::2]):.4g}, u2={max(u[1::2]):.4g}')

    avg_u1 = np.mean(u[0::2])
    avg_u2 = np.mean(u[1::2])

    emin = np.array([ 9.0e9,  9.0e9,  9.0e9])
    emax = np.array([-9.0e9, -9.0e9, -9.0e9])
    smin = np.array([ 9.0e9,  9.0e9,  9.0e9])
    smax = np.array([-9.0e9, -9.0e9, -9.0e9])

    # Initialize variables for average calculation
    total_strain = np.zeros(3)
    total_stress = np.zeros(3)
    num_elements = len(conn)

    for c in conn:  # for each element (conn is Nx4)
        nodePts = nodes[c,:]  # 4x2, eg: [[1.1,0.2], [1.2,0.3], [1.3,0.4], [1.4, 0.5]]
        for q in q4:  # for each integration pt, eg: [-0.7,-0.7]
            dN = gradshape(q)  # 2x4
            J  = np.dot(dN, nodePts).T  # 2x2
            dN = np.dot(np.linalg.inv(J), dN)  # 2x4
            B[0,0::2] = dN[0,:]  # 3x8
            B[1,1::2] = dN[1,:]
            B[2,0::2] = dN[1,:]
            B[2,1::2] = dN[0,:]

            UU = np.zeros((8,1))  # 8x1
            UU[0] = u[2*c[0]]
            UU[1] = u[2*c[0] + 1]
            UU[2] = u[2*c[1]]
            UU[3] = u[2*c[1] + 1]
            UU[4] = u[2*c[2]]
            UU[5] = u[2*c[2] + 1]
            UU[6] = u[2*c[3]]
            UU[7] = u[2*c[3] + 1]
            # get the strain and stress at the integration point
            strain = B @ UU  # (B is 3x8) (UU is 8x1) => (strain is 3x1)
            stress = C @ strain  # (C is 3x3) (strain is 3x1) => (stress is 3x1)
            emin[0] = min(emin[0], strain[0][0])
            emin[1] = min(emin[1], strain[1][0])
            emin[2] = min(emin[2], strain[2][0])
            emax[0] = max(emax[0], strain[0][0])
            emax[1] = max(emax[1], strain[1][0])
            emax[2] = max(emax[2], strain[2][0])

            node_strain[c[0]][:] = strain.T[0]
            node_strain[c[1]][:] = strain.T[0]
            node_strain[c[2]][:] = strain.T[0]
            node_strain[c[3]][:] = strain.T[0]
            node_stress[c[0]][:] = stress.T[0]
            node_stress[c[1]][:] = stress.T[0]
            node_stress[c[2]][:] = stress.T[0]
            node_stress[c[3]][:] = stress.T[0]
            smax[0] = max(smax[0], stress[0][0])
            smax[1] = max(smax[1], stress[1][0])
            smax[2] = max(smax[2], stress[2][0])
            smin[0] = min(smin[0], stress[0][0])
            smin[1] = min(smin[1], stress[1][0])
            smin[2] = min(smin[2], stress[2][0])

            # Accumulate total strain and stress for average calculation
            total_strain += strain.T[0]
            total_stress += stress.T[0]

    # Calculate average strain and stress
    average_strain = total_strain / num_elements
    average_stress = total_stress / num_elements

    print(f'   min strains: e11={emin[0]:.4g}, e22={emin[1]:.4g}, e12={emin[2]:.4g}')
    print(f'   max strains: e11={emax[0]:.4g}, e22={emax[1]:.4g}, e12={emax[2]:.4g}')
    print(f'   min stress:  s11={smin[0]:.4g}, s22={smin[1]:.4g}, s12={smin[2]:.4g}')
    print(f'   max stress:  s11={smax[0]:.4g}, s22={smax[1]:.4g}, s12={smax[2]:.4g}')
    print(f'   average strains: e11={average_strain[0]:.4g}, e22={average_strain[1]:.4g}, e12={average_strain[2]:.4g}')
    print(f'   average stress:  s11={average_stress[0]:.4g}, s22={average_stress[1]:.4g}, s12={average_stress[2]:.4g}')
    print(f'   average strain/nodes:  s11={abs((average_stress[0])/len(conn))*10000:.4g}, s22={(abs(average_stress[1])/len(conn))*10000:.4g}, s12={(abs(average_stress[2])/len(conn))*10000:.4g}')
    print(f'   average strain/nodes:  e11={abs((average_strain[0])/len(conn))*10000:.4g}, e22={(abs(average_strain[1])/len(conn))*10000:.4g}, e12={(abs(average_strain[2])/len(conn))*10000:.4g}')    
        ###############################
    # print('\n** Plot displacement')
    xvec = []
    yvec = []
    res  = []
    plot_type = 'e11'
    for ni, pt in enumerate(nodes):
        xvec.append(pt[1] + u[2*ni+1])  # Swap x and y
        yvec.append(-pt[0] - u[2*ni])   # Swap x and y, negate y to rotate 90 degrees
        if plot_type == 'u1':  res.append(u[2*ni])                # x-disp
        if plot_type == 'u2':  res.append(u[2*ni+1])              # y-disp
        if plot_type == 's11': res.append(node_stress[ni][0])     # s11
        if plot_type == 's22': res.append(node_stress[ni][1])     # s22
        if plot_type == 's12': res.append(node_stress[ni][2])     # s12
        if plot_type == 'e11': res.append(node_strain[ni][0])     # e11
        if plot_type == 'e22': res.append(node_strain[ni][1])     # e22
        if plot_type == 'e12': res.append(node_strain[ni][2])     # e12

    tri = []
    for c in conn:
        tri.append([c[0], c[1], c[2]])  # First triangle
        tri.append([c[0], c[2], c[3]])  # Second triangle

    t = plt.tricontourf(xvec, yvec, res, triangles=tri, levels=14, cmap=plt.cm.jet)
    plt.scatter(xvec, yvec, marker='o', c='b', s=0.5) # (plot the nodes)
    plt.grid()
    plt.colorbar(t)
    plt.title(plot_type)
    plt.axis('equal')
    plt.show()
    print('Done.')

    return max(smax), max(emax), avg_u1, avg_u2, len(conn), max(average_stress), max(average_strain), max(u[0::2]), max(u[1::2])

def reward_function(smax, emax, avg_disp_x, avg_disp_y, element_count, avg_stress, max_displacement_1, max_displacement_2):

  max_allowed_elements = HEIGHT * WIDTH * 2
  # print("max_allowed_elements", max_allowed_elements)
  #print("element_count", element_count)
  normalized_element_count =  (max_allowed_elements - element_count) / max_allowed_elements 
  strain_ratio =  YIELD_STRENGTH /abs(smax*1000)
  displacement_factor = 1 / (max_displacement_1+max_displacement_2)
  # Define weights for different components
  w_stress = 10
  w_strain = 0.3
  w_disp_x = 0.1
  w_disp_y = 0.1
  w_material = 0.1
  # print("normalized_element_count", normalized_element_count, "\nstrain_ratio", strain_ratio, "\nsmax", abs(smax*1000) )
  # Reward function (example)
  reward = (normalized_element_count ** 2 + strain_ratio ** 2 + displacement_factor) * 10
  #+ avg_stress**2



  return reward

def plot_mesh(nodes, conn, width=10, height=8):
    nodes = np.array(nodes)  # Convert nodes to a NumPy array
    plt.figure(figsize=(width, height))  # Set the figure size
    for element in conn:
        x = [nodes[i][1] for i in element]  # Swap x and y
        y = [-nodes[i][0] for i in element]  # Swap x and y and negate y
        plt.plot(x, y, 'b-')
    plt.scatter(nodes[:, 1], -nodes[:, 0], color='red', s=10)  # Swap x and y and negate y
    
    # Add text annotations for each node
    for i, (x, y) in enumerate(nodes):
        plt.text(y, -x, str(i+1), fontsize=20, ha='right')
    
    plt.axis('equal')
    plt.show()

def get_filled_cells(matrix):
    return sum(cell == 1 for row in matrix for cell in row)

def is_continuous(matrix, targets):
    if not targets:
        return False
    filled_cells = get_filled_cells(matrix)
    visited = dfs(matrix, targets)
    return False if len(visited) == 0 else len(visited) == filled_cells
 

def dfs(matrix, targets):
    """This function is the depth first search algorithm that is used to check 
    if all the filled cells in the design space are connected to a specified 
    start point.

    Parameters
    ----------
    matrix : numpy.ndarray
        The design space.
    start : tuple
        The coordinates of the start point.
    targets : tuple
        The coordinates of the end point.

    Returns
    -------
    set
        The set containing the visited nodes."""
    targets_copy = convert_all(targets.copy(), (height, width))
    start = targets_copy.pop(0)
    rows, cols = len(matrix), len(matrix[0]) 
    visited = set() # Set to keep track of visited nodes    
    stack = [start] # Start the stack with the start node
    if not stack:  # Check if the stack is empty
        raise RuntimeError("The stack is empty")
    while stack:
        (row, col) = stack.pop()    # pop the last coordinates from the stack
        if (row < 0 or 
            row >= rows or 
            col < 0 or 
            col >= cols or 
            (row, col) in visited or 
            matrix[row][col] == 0): 
            #checking for discarding conditions
            continue
        if (row, col) in targets_copy:

            targets_copy.remove((row, col))
        visited.add((row, col))     # Add the current node to the visited set
        stack.extend([(row-1, col), (row+1, col), 
                      (row, col-1), (row, col+1),
                      ])
        # Add the neighbours of the current node to the stack
 
    return visited if not targets_copy else []

def get_scatter_coordinates(bound_list, loaded_list):
    """this is a simple function to provide an easy way to get the coordinates
    used for the scatter plot
    -------
    Parameters:\n
    - matrix : numpy.ndarray
        - The design space.
    -------
    Returns:\n
    - x_bound_positions_for_scatter : list
        - the list of needed x coordinates for the scatter plot of the bound nodes
    - y_bound_positions_for_scatter : list
        - the list of needed y coordinates for the scatter plot of the bound nodes
    - x_loaded_positions_for_scatter : list
        - the list of needed x coordinates for the scatter plot of the loaded nodes
    - y_loaded_positions_for_scatter : list
        - the list of needed y coordinates for the scatter plot of the loaded nodes 
    """
    x_bound_for_scatter = [coord[1] for coord in bound_list]
    y_bound_for_scatter = [coord[0] for coord in bound_list]

    x_loaded_for_scatter = [coord[1] for coord in loaded_list]
    y_loaded_for_scatter = [coord[0] for coord in loaded_list]
    return (x_bound_for_scatter, y_bound_for_scatter,
            x_loaded_for_scatter, y_loaded_for_scatter)

class TopOptEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, mode="train", threshold=0.5, 
                 bound_nodes_list=bound_nodes_list, 
                 force_nodes_list=loaded_nodes_list, 
                 height=HEIGHT, width=WIDTH):
        # The constructor of the environment
        super().__init__()

        self.mode = mode
        self.threshold = threshold

        self.height = height
        self.width = width
        
        
        self.bound_nodes_list = bound_nodes_list
        self.force_nodes_list = force_nodes_list
        self.initial_max_strain = 0
        self.design_space = np.zeros((4, self.height, self.width))


        self.action_space = gym.spaces.Discrete(self.height * self.width)
        self.observation_space = gym.spaces.Box(low=-1, 
                                                high=3, 
                                                shape=(self.design_space.flatten().shape), 
                                                dtype=np.float64)
        
        # A Dictionary is used to map each coordinate tuple of the designspace
        # to a singular distinct integer for use in the optimization
        # 0 = (0,0), 1 = (0,1), 2 = (0,2), ... , number_of_nodes=(height,width)
        self._actions_to_coordinates = {}  
        k=0
        for i in range(self.height):
            for j in range(self.width):
                self._actions_to_coordinates[k] = (i,j)
                k += 1
        
        self.reward = 0
        self.step_count = 0

        self.reset()
        
    def _is_illegal_action(self, action):
        bound_matrix = self.design_space[BOUND, :, :]
        force_matrix = np.logical_or(self.design_space[LOADED_X, :, :], 
                                     self.design_space[LOADED_Y,:,:]).astype(int)
        design_matrix = self.design_space[DESIGN, :, :]

       
        # Check if the selected Action has already been performed on this state
        if action in self.performed_actions:
            return True
        #check if the selected Node is either bound or force node
        if bound_matrix[self._actions_to_coordinates[action]] != 0:
            return True
        if force_matrix[self._actions_to_coordinates[action]] != 0:
            return True
        #Check if the selected Node is already removed
        if design_matrix[self._actions_to_coordinates[action]] < 1:
            return True
        if not is_continuous(design_matrix, 
                             self.bound_nodes_list + self.force_nodes_list):
            return True

    def step(self, action):
        self.print_design(self)

        self.step_count += 1
        terminated = False

        if self._is_illegal_action(action):
            self.reward = -1
            terminated = True
            return (self.design_space, self.reward, 
                    terminated, False, self.get_info())
        
        design_matrix = self.design_space[DESIGN, :,:]
        
        
        #self.last_compliance = self.compliance

        self._remove_node(action)

        


        self.constraint = (self.design_space[DESIGN, :, :].sum() / 
                          (self.height * self.width))

        # print("\n\nstep Debug ====================================")
        # print("Step: ", self.step_count)
        # print("Compliance: ", self.compliance)
        # print("initial_compliance: ", self.initial_compliance)
        # print("last_compliance: ", self.last_compliance)
        # print("Constraint: ", self.constraint)
        # print("Reward: ", self.reward)
        # print("freedofs: ", self.freedofs)
        # print("fixdofs: ", self.fixdofs)
        # print("forces: ", self.forces)
        # print("design_space[:,:,_DESIGN]:\n", self.design_space[:,:, _DESIGN])
        # print("design_space[:,:,_BOUND]:\n", self.design_space[:,:, _BOUND])
        # print("design_space[:,:,_FORCE]:\n", self.design_space[:,:, _FORCE])
        self.nodes, self.elements , self.boundary_conditions, self.node_map = extract_fem_data(self.design_space)

        self.smax, self.emax, self.avg_u1, self.avg_u2, self.element_count, self.average_stress, self.average_strain, self.max_displacement_1, self.max_displacement_2 = FEM(self.nodes, self.elements , self.boundary_conditions, self.node_map)
        
        self.reward += reward_function(self.smax, self.emax, self.avg_u1, self.avg_u2, self.element_count, self.average_stress, self.max_displacement_1, self.max_displacement_2)
        
        self.performed_actions.append(action)

        if self.constraint < self.threshold:
            terminated = True

        self.design_space[DESIGN, :, :] = design_matrix

        

        return (self.design_space.flatten(), self.reward, 
                terminated, False, self.get_info())



    def _remove_node(self, action):
        design_matrix = self.design_space[DESIGN, :, :]
        design_matrix[self._actions_to_coordinates[action]] = 0    
        self.design_space[DESIGN, :, :] = design_matrix
    



    def reset(self, seed=None):
        # The reset function of the environment

        super().reset(seed=seed)
       
        self.step_count = 0
        self.reward = 0
        self.performed_actions = []
        self.design_space = np.zeros((4, self.height, self.width))
        self.design_space[DESIGN, :, :] += 1

        design_matrix = self.design_space[DESIGN, :, :]
        bound_matrix = self.design_space[BOUND, :,:]
        force_x_matrix = self.design_space[LOADED_X, :,:]
        force_y_matrix = self.design_space[LOADED_Y, :,:]

        if self.mode == "train":
            self.bound_nodes_list = []
            self.bound_nodes_list.append(self.generate_random_coordinate()) 
            self.bound_nodes_list.append(self.generate_random_coordinate()) 

            self.force_nodes_list = []
            self.force_nodes_list.append(self.generate_random_coordinate()),
        
        if self.mode == "eval":
            self.bound_nodes_list = bound_nodes_list
            self.force_nodes_list = loaded_nodes_list
        
        for coord in convert_all(self.bound_nodes_list, (self.height, self.width)):
            bound_matrix[coord] = 3
        for coord in convert_all(self.force_nodes_list, (self.height, self.width)):
            force_y_matrix[coord] = 10  

        

        # print("\n\ncompliance Debug ====================================")
        # print("design_matrix:\n", design_matrix)
        # print("freedofs: ", self.freedofs)
        # print("fixdofs: ", self.fixdofs)
        # print("forces: ", self.forces)



        
        # self.compliance = self.initial_compliance
        # self.last_compliance = self.initial_compliance

        self.design_space[DESIGN, :, :] = design_matrix
        self.design_space[BOUND, :, :] = bound_matrix
        self.design_space[LOADED_Y, :,:] = force_y_matrix
        self.design_space[LOADED_X, :,:] = force_x_matrix

        # print("\n\nReset Debug ====================================")
        # print("items in self.bound_nodes_list", len(self.bound_nodes_list))
        # print("bound_nodes_list", self.bound_nodes_list)
        # print("Initial Compliance:\n", self.initial_compliance)
        # print("design_matrix:\n", design_matrix)
        # print("design_space[:,:,_DESIGN]:\n", self.design_space[:,:, _DESIGN])
        # print("bound_matrix:\n", bound_matrix)
        # print("design_space[:,:,_BOUND]:\n", self.design_space[:,:, _BOUND])
        # print("force_matrix:\n", force_matrix)
        # print("design_space[:,:,_FORCE]:\n", self.design_space[:,:, _FORCE])
        # print("compliance_matrix:\n", compliance_matrix)
        # print("Initial Reward:\n", self.reward)
    
        return self.design_space.flatten(), self.get_info()



    def is_valid_coordinate(self, coord, node_lists):
    # Check adjacent positions
        checking_positions = [
            (coord[0], coord[1]),  # Current
            (coord[0] - 1, coord[1]),  # Left
            (coord[0] + 1, coord[1]),  # Right
            (coord[0], coord[1] - 1),  # Up
            (coord[0], coord[1] + 1),  # Down
        ]
        # Check if any adjacent position is in the node lists
        for pos in checking_positions:
            if any(pos in node_list for node_list in node_lists):
                return True
        return False

    def generate_random_coordinate(self):
        axis = random.randint(0, 3)
        if axis == 0:
            coord = random.randint(0, height-1), 0  # Left edge
        elif axis == 1:
            coord = 0, random.randint(0, width-1)  # Top edge
        elif axis == 2:
            coord = random.randint(0, height-1), width-1  # Right edge
        else:
            coord = height-1, random.randint(0, width-1)  # Bottom edge

        # Check if the coordinate is in the lists or next to any node in the lists
        comp_list = [self.bound_nodes_list, self.force_nodes_list]
    
        if (coord in self.bound_nodes_list or 
            coord in self.force_nodes_list or 
            self.is_valid_coordinate(coord, comp_list)):
        
            return self.generate_random_coordinate()
        else:
            return coord
    

    
    def print_design(self, mode="human"):
        # This function is used to render the environment
        # This function is not necessary for the optimization
        print("current Design")
        print(self.force_nodes_list)
        fig, ax = plt.subplots()
        ax.imshow(self.design_space[DESIGN, :, :])
        xb, yb, xl, yl = get_scatter_coordinates(self.bound_nodes_list, 
                                                 self.force_nodes_list)
        ax.scatter(xb, yb, s=20, color='k', marker='x')
        ax.scatter(xl, yl, s=20, color='k', marker='$↓$')
        plt.show()


    def get_info(self):
        # This function returns the information about the environment
        # This function is used to monitor the environment
        # The information should be a dictionary
        # The dictionary should contain the following keys:
        # - step_count: the number of steps that have been executed
        # - current Reward: the reward of the current state
        # - design_space: the current state of the environment
        return {"step_count": self.step_count, 
                "current_reward": self.reward,
                "design_space": self.design_space,
                }

def vecenv_render(env):
    plt.close()
    design_spaces = env.env_method("get_info")
    fig, axs = plt.subplots(1, NUMBER_SUBPROCESSES)
    fig.set_size_inches(20, 3)

    fig.subplots_adjust(hspace=0.05)
    
    
    for i in range(NUMBER_SUBPROCESSES * 1):
        row = i // NUMBER_SUBPROCESSES
        col = i % NUMBER_SUBPROCESSES
        
        if row == 0:
            space_index = i
        else:
            space_index = i - NUMBER_SUBPROCESSES

        # if len(design_spaces[space_index]["design_space"].shape) != 2:
        #     raise ValueError(f"Expected a 2D array, but got shape {design_spaces[space_index]['design_space'].shape}")


        current_design_space = design_spaces[space_index]["design_space"]
        current_bound_matrix = current_design_space[BOUND, :, :, ]
        current_force_matrix = np.logical_or(current_design_space[LOADED_X, :, :], 
                                     current_design_space[LOADED_Y,:,:]).astype(int)
        
        print(current_bound_matrix)
        current_bound_list = matrix_to_node_list(current_bound_matrix)
        current_force_list = matrix_to_node_list(current_force_matrix)
        
        print(current_bound_list, "\n", current_force_list)
        if row == 0:
            axs[col].imshow(current_design_space[DESIGN, :, :], 
                                 vmin=0, vmax=1)
            current_reward = design_spaces[space_index]["current_reward"]
            title = f"{i + 1}\nReward= {round(current_reward, 1)}"
        # else:
        #     axs[row, col].imshow(current_design_space[:, :, _COMPLIANCE], 
        #                          vmin=0, vmax=1)
        #     current_compliance = design_spaces[space_index]["compliance"]
        #     title = f"Compliance= {round(current_compliance, 1)}"
        
        xb, yb, xl, yl = get_scatter_coordinates(current_bound_list, 
                                                 current_force_list)
        axs[col].scatter(xb, yb, s=20, color='k', marker='x')
        axs[col].scatter(xl, yl, s=20, color='k', marker='$↓$')
        axs[col].set_title(title, fontsize=11, pad=10) 
    plt.show()
    


def matrix_to_node_list(matrix):
    """
    Finds the coordinates of all the 1s in a given matrix.

    :param matrix: A 2D list or numpy array containing 0s and 1s.
    :return: A list of tuples, where each tuple represents the 
    coordinates (row, column) of a 1 in the matrix.
    """
    coordinates = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value != 0:
                coordinates.append((i, j))
    return coordinates

def node_list_to_matrix(coordinates, rows, cols):
    """
    Creates a matrix from a list of coordinates, setting the positions 
    of the coordinates to 1.

    :param coordinates: A list of tuples, where each tuple represents 
    the coordinates (row, column) of a 1.
    :param rows: The number of rows in the matrix.
    :param cols: The number of columns in the matrix.
    :return: A 2D list (matrix) with 1s at the specified coordinates and 
    0s elsewhere.
    """
    matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    for (i, j) in coordinates:
        matrix[i][j] = 1
    return matrix

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` 
    steps)based on the training reward (in practice, we recommend using 
    ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be 
    saved. It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)s
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        print("callback baby")
        if self.save_path is not None:
            # os.makedirs(self.save_path, exist_ok=True
            return

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print("hello hello")
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("--------------------------------------------------")
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                if NUMBER_SUBPROCESSES > 1:
                    vecenv_render(env)
                else:
                    env.print_design()
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
                    

        return True
    
def main():
    if NUMBER_SUBPROCESSES == 1:
        env = TopOptEnv()
        #check_env(env, warn=True)
        env = Monitor(env, LOG_DIR)
    if NUMBER_SUBPROCESSES > 1:
        env = SubprocVecEnv([lambda: TopOptEnv() for _ in range(NUMBER_SUBPROCESSES)])
        env = VecMonitor(env, LOG_DIR)

    callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=LOG_DIR)

    start=time.time()
    model = PPO("MlpPolicy", env, tensorboard_log=TS_BOARD_DIR).learn(total_timesteps=TIMESTEPS, callback=callback)
    end=time.time()   
    print("Elapsed Time = " + str(end-start))

    if __name__=="__main__":
        main()

