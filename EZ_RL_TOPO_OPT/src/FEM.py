import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

def shape(xi):
    xi, eta = tuple(xi)
    N = [(1.0 - xi) * (1.0 - eta), (1.0 + xi) * (1.0 - eta), (1.0 + xi) * (1.0 + eta), (1.0 - xi) * (1.0 + eta)]
    return 0.25 * np.array(N)

def gradshape(xi):
	"""Gradient of the shape functions for a 4-node, isoparametric element.
		dN_i(xi,eta)/dxi and dN_i(xi,eta)/deta
		Input: 1x2,  Output: 2x4"""
	xi,eta = tuple(xi)
	dN = [[-(1.0-eta),  (1.0-eta), (1.0+eta), -(1.0+eta)],
		  [-(1.0-xi), -(1.0+xi), (1.0+xi),  (1.0-xi)]]
	return 0.25 * np.array(dN)

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

def get_max_stress_and_strain(smax, smin, emax, emin):
    max_stress = np.max(np.abs(np.concatenate((smax, smin))))
    max_strain = np.max(np.abs(np.concatenate((emax, emin))))
    return max_stress, max_strain

def assemble_load(nodes, load, f):
  for load_data in load:
    node_index = load_data[0]
    dof = load_data[1]
    magnitude = load_data[2]
    global_dof = 2 * node_index + dof - 1
    f[global_dof] += magnitude

def FEM(nodes, conn, boundary, load, plot_flag, device='cpu'):
    """
    Perform Finite Element Method (FEM) analysis on a given mesh.

    Parameters:
    nodes (list of tuples): List of node coordinates. Each tuple represents the (x, y) coordinates of a node.
    conn (list of tuples): List of element connectivity. Each tuple contains the indices of the nodes that form an element.
    boundary (list of lists): List of boundary conditions. Each sublist specifies the node index, degree of freedom (1 for x, 2 for y), and the fixed value.
    load (list of lists): List of loads applied to the nodes. Each sublist specifies the node index, degree of freedom (1 for x, 2 for y), and the load value.
    plot_flag (bool): If True, plot the mesh and results.

    Returns:
    tuple: A tuple containing:
        - smax (float): Maximum stress value.
        - emax (float): Maximum strain value.
        - avg_u1 (float): Average displacement in the x-direction.
        - avg_u2 (float): Average displacement in the y-direction.
        - element_count (int): Number of elements.
        - average_stress (float): Average stress value.
        - average_strain (float): Average strain value.
        - max_displacement_1 (float): Maximum displacement in the x-direction.
        - max_displacement_2 (float): Maximum displacement in the y-direction.
        - avg_strain_over_nodes (float): Average strain over nodes.

    Example:
    >>> nodes = [(0, 0), (1, 0), (1, 1), (0, 1)]
    >>> conn = [(0, 1, 2, 3)]
    >>> boundary = [[0, 1, 1, 0.0], [0, 2, 2, 0.0]]
    >>> load = [[2, 1, 100.0], [2, 2, 50.0]]
    >>> plot_flag = True
    >>> FEM(nodes, conn, boundary, load, plot_flag)
    (smax, emax, avg_u1, avg_u2, element_count, average_stress, average_strain, max_displacement_1, max_displacement_2, avg_strain_over_nodes)
    """
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
    K = lil_matrix((2 * num_nodes, 2 * num_nodes))    # square zero matrix
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

    K = csr_matrix(K)
    ###############################
    # print('\n** Solve linear system: Ku = f')	# [K] = 2N x 2N, [f] = 2N x 1, [u] = 2N x 1
    u = spsolve(K, f)
    ###############################
    # print('\n** Post process the data')
    # (pre-allocate space for nodal stress and strain)
    node_strain = np.zeros((num_nodes, 3))
    node_stress = np.zeros((num_nodes, 3))

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
            emin = np.minimum(emin, strain.T[0])
            emax = np.maximum(emax, strain.T[0])
            smin = np.minimum(smin, stress.T[0])
            smax = np.maximum(smax, stress.T[0])

            node_strain[c, :] = strain.T
            node_stress[c, :] = stress.T

            # Accumulate total strain and stress for average calculation
            total_strain += strain.T[0]
            total_stress += stress.T[0]




    # Calculate average strain and stress
    average_strain = total_strain / num_elements
    average_stress = total_stress / num_elements

    if plot_flag:
        # print(f'   min strains: e11={emin[0]:.4g}, e22={emin[1]:.4g}, e12={emin[2]:.4g}')
        # print(f'   max strains: e11={emax[0]:.4g}, e22={emax[1]:.4g}, e12={emax[2]:.4g}')
        # print(f'   min stress:  s11={smin[0]:.4g}, s22={smin[1]:.4g}, s12={smin[2]:.4g}')
        # print(f'   max stress:  s11={smax[0]:.4g}, s22={smax[1]:.4g}, s12={smax[2]:.4g}')
        # print(f'   average strains: e11={average_strain[0]:.4g}, e22={average_strain[1]:.4g}, e12={average_strain[2]:.4g}')
        # print(f'   average stress:  s11={average_stress[0]:.4g}, s22={average_stress[1]:.4g}, s12={average_stress[2]:.4g}')
        # print(f'   average strain/nodes:  s11={abs((average_stress[0])/len(conn))*10000:.4g}, s22={(abs(average_stress[1])/len(conn))*10000:.4g}, s12={(abs(average_stress[2])/len(conn))*10000:.4g}')
        # print(f'   average strain/nodes:  e11={abs((average_strain[0])/len(conn))*10000:.4g}, e22={(abs(average_strain[1])/len(conn))*10000:.4g}, e12={(abs(average_strain[2])/len(conn))*10000:.4g}')

        ##############################
        # print('\n** Plot displacement')
        xvec = []
        yvec = []
        res  = []
        plot_type = 'e11'
        for ni, pt in enumerate(nodes):
            xvec.append(pt[1] + u[2*ni+1])  # Swap x and y
            yvec.append(-(pt[0] + u[2*ni]))   # Swap x and y, negate y
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
        t = plt.tricontourf(xvec, yvec, res, triangles=tri, levels=14, cmap=plt.get_cmap('jet'))

        # Plot bounded nodes
        for b in boundary:
            node_index = b[0]
            plt.plot(xvec[node_index], yvec[node_index], 'ro', markersize=5, label='Bounded Node' if b == boundary[0] else "")

        # Plot loaded nodes with arrows
        arrow_scale = 10  # Adjust this scale factor as needed
        arrow_width = 0.010  # Adjust this width as needed

        # Find the maximum magnitude
        max_magnitude = max(abs(l[2]) for l in load)

        for l in load:
            node_index = l[0]
            direction = l[1]
            magnitude = l[2] / max_magnitude  # Normalize the magnitude
            if direction == 2:  # Load in x-direction
                plt.quiver(xvec[node_index], yvec[node_index], magnitude, 0, angles='xy', scale_units='xy', scale=1, color='k', width=arrow_width, label='Loaded Node' if l == load[0] else "")
            elif direction == 1:  # Load in y-direction
                plt.quiver(xvec[node_index], yvec[node_index], 0, -magnitude, angles='xy', scale_units='xy', scale=1, color='k', width=arrow_width, label='Loaded Node' if l == load[0] else "")

            # Add annotation for the magnitude of the force
            plt.annotate(f'{l[2]:.2f}', (xvec[node_index], yvec[node_index]), textcoords="offset points", xytext=(5,5), ha='center', color='k')

        plt.scatter(xvec, yvec, marker='o', c='b', s=0.5) # (plot the nodes)
        plt.grid(False)  # Turn off the grid
        plt.colorbar(t)
        plt.title(plot_type)
        plt.axis('equal')
        plt.show()
        print('Done.')

    if np.isnan(smax).any() or np.isnan(emax).any() or np.isnan(avg_u1).any() or np.isnan(avg_u2).any() or np.isnan(average_stress).any() or np.isnan(average_strain).any():
        print("NaN detected in FEM results")

    avg_strain_over_nodes = max(abs((average_strain[0]) / len(conn)) * 10000,abs((average_strain[1]) / len(conn)) * 10000, abs((average_strain[2]) / len(conn)) * 10000)
    max_stress, max_strain = get_max_stress_and_strain(smax, smin, emax, emin)
    return  max_stress, max_strain, avg_u1, avg_u2, len(conn), np.max(average_stress), np.max(average_strain), np.max(u[0::2]), np.max(u[1::2]), avg_strain_over_nodes


# def FEM(nodes, conn, boundary, load, plot_flag, device='cpu'):
#     """
#     Perform Finite Element Method (FEM) analysis on a given mesh.

#     Parameters:
#     nodes (list of tuples): List of node coordinates. Each tuple represents the (x, y) coordinates of a node.
#     conn (list of tuples): List of element connectivity. Each tuple contains the indices of the nodes that form an element.
#     boundary (list of lists): List of boundary conditions. Each sublist specifies the node index, degree of freedom (1 for x, 2 for y), and the fixed value.
#     load (list of lists): List of loads applied to the nodes. Each sublist specifies the node index, degree of freedom (1 for x, 2 for y), and the load value.
#     plot_flag (bool): If True, plot the mesh and results.

#     Returns:
#     tuple: A tuple containing:
#         - smax (float): Maximum stress value.
#         - emax (float): Maximum strain value.
#         - avg_u1 (float): Average displacement in the x-direction.
#         - avg_u2 (float): Average displacement in the y-direction.
#         - element_count (int): Number of elements.
#         - average_stress (float): Average stress value.
#         - average_strain (float): Average strain value.
#         - max_displacement_1 (float): Maximum displacement in the x-direction.
#         - max_displacement_2 (float): Maximum displacement in the y-direction.
#         - avg_strain_over_nodes (float): Average strain over nodes.

#     Example:
#     >>> nodes = [(0, 0), (1, 0), (1, 1), (0, 1)]
#     >>> conn = [(0, 1, 2, 3)]
#     >>> boundary = [[0, 1, 1, 0.0], [0, 2, 2, 0.0]]
#     >>> load = [[2, 1, 100.0], [2, 2, 50.0]]
#     >>> plot_flag = True
#     >>> FEM(nodes, conn, boundary, load, plot_flag)
#     (smax, emax, avg_u1, avg_u2, element_count, average_stress, average_strain, max_displacement_1, max_displacement_2, avg_strain_over_nodes)
#     """
#     nodes = np.array(nodes)
#     num_nodes = len(nodes)
#     #print('   number of nodes:', len(nodes))
#     #print('   number of elements:', len(conn))
#     #print('   number of displacement boundary conditions:', len(boundary))
# ###############################
#     # Plane-strain material tangent (see Bathe p. 194)
#     # C is 3x3
#     E = 100.0
#     v = 0.3
#     C = E/(1.0+v)/(1.0-2.0*v) * np.array([[1.0-v, v, 0.0], [v, 1.0-v, 0.0], [0.0, 0.0, 0.5-v]])
#     ###############################
#     # Make stiffness matrix
#     # if N is the number of DOF, then K is NxN
#     K = np.zeros((2*num_nodes, 2*num_nodes))    # square zero matrix
#     # 2x2 Gauss Quadrature (4 Gauss points)
#     # q4 is 4x2
#     q4 = np.array([[-1,-1],[1,-1],[-1,1],[1,1]]) / math.sqrt(3.0)
#     # print('\n** Assemble stiffness matrix')
#     # strain in an element: [strain] = B    U
#     #                        3x1     = 3x8  8x1
#     #
#     # strain11 = B11 U1 + B12 U2 + B13 U3 + B14 U4 + B15 U5 + B16 U6 + B17 U7 + B18 U8
#     #          = B11 u1          + B13 u1          + B15 u1          + B17 u1
#     #          = dN1/dx u1       + dN2/dx u1       + dN3/dx u1       + dN4/dx u1
#     B = np.zeros((3,8))
#     # conn[0] is node numbers of the element
#     for c in conn:     # loop through each element
#         # coordinates of each node in the element
#         # shape = 4x2
#         # for example:
#         #    nodePts = [[0.0,   0.0],
#         #               [0.033, 0.0],
#         #               [0.033, 0.066],
#         #               [0.0,   0.066]]
#         nodePts = nodes[c,:]
#         Ke = np.zeros((8,8))	# element stiffness matrix is 8x8
#         for q in q4:			# for each Gauss point
#             # q is 1x2, N(xi,eta)
#             dN = gradshape(q)       # partial derivative of N wrt (xi,eta): 2x4
#             J  = np.dot(dN, nodePts).T # J is 2x2
#             dN = np.dot(np.linalg.inv(J), dN)    # partial derivative of N wrt (x,y): 2x4
#             # assemble B matrix  [3x8]
#             B[0,0::2] = dN[0,:]
#             B[1,1::2] = dN[1,:]
#             B[2,0::2] = dN[1,:]
#             B[2,1::2] = dN[0,:]
#             # element stiffness matrix
#             Ke += np.dot(np.dot(B.T,C),B) * np.linalg.det(J)
#         # Scatter operation
#         for i,I in enumerate(c):
#             for j,J in enumerate(c):
#                 K[2*I,2*J]     += Ke[2*i,2*j]
#                 K[2*I+1,2*J]   += Ke[2*i+1,2*j]
#                 K[2*I+1,2*J+1] += Ke[2*i+1,2*j+1]
#                 K[2*I,2*J+1]   += Ke[2*i,2*j+1]
#     ###############################
#     # Assign nodal forces and boundary conditions
#     #    if N is the number of nodes, then f is 2xN
#     f = np.zeros((2*num_nodes))          # initialize to 0 forces
#     assemble_load(nodes, load, f)
#     # How about displacement boundary conditions:
#     #    [k11 k12 k13] [u1] = [f1]
#     #    [k21 k22 k23] [u2]   [f2]
#     #    [k31 k32 k33] [u3]   [f3]
#     #
#     #    if u3=x then
#     #       [k11 k12 k13] [u1] = [f1]
#     #       [k21 k22 k23] [u2]   [f2]
#     #       [k31 k32 k33] [ x]   [f3]
#     #   =>
#     #       [k11 k12 k13] [u1] = [f1]
#     #       [k21 k22 k23] [u2]   [f2]
#     #       [  0   0   1] [u3]   [ x]
#     #   the reaction force is
#     #       f3 = [k31 k32 k33] * [u1 u2 u3]
#     for i in range(len(boundary)):  # apply all boundary displacements
#         nn  = boundary[i][0]
#         dof = boundary[i][1]
#         val = boundary[i][2]
#         j = 2*nn
#         if dof == 2: j = j + 1
#         K[j,:] = 0.0
#         K[j,j] = 1.0
#         f[j] = val
#     ###############################
#     # print('\n** Solve linear system: Ku = f')	# [K] = 2N x 2N, [f] = 2N x 1, [u] = 2N x 1
#     u = np.linalg.solve(K, f)
#     ###############################
#     # print('\n** Post process the data')
#     # (pre-allocate space for nodal stress and strain)
#     node_strain = []
#     node_stress = []
#     for ni in range(len(nodes)):
#         node_strain.append([0.0, 0.0, 0.0])
#         node_stress.append([0.0, 0.0, 0.0])
#     node_strain = np.array(node_strain)
#     node_stress = np.array(node_stress)

#     #print(f'   min displacements: u1={min(u[0::2]):.4g}, u2={min(u[1::2]):.4g}')
#     #print(f'   max displacements: u1={max(u[0::2]):.4g}, u2={max(u[1::2]):.4g}')
#     avg_u1 = np.mean(u[0::2])
#     avg_u2 = np.mean(u[1::2])

#     emin = np.array([ 9.0e9,  9.0e9,  9.0e9])
#     emax = np.array([-9.0e9, -9.0e9, -9.0e9])
#     smin = np.array([ 9.0e9,  9.0e9,  9.0e9])
#     smax = np.array([-9.0e9, -9.0e9, -9.0e9])

#     # Initialize variables for average calculation
#     total_strain = np.zeros(3)
#     total_stress = np.zeros(3)
#     num_elements = len(conn)

#     for c in conn:  # for each element (conn is Nx4)
#         nodePts = nodes[c,:]  # 4x2, eg: [[1.1,0.2], [1.2,0.3], [1.3,0.4], [1.4, 0.5]]
#         for q in q4:  # for each integration pt, eg: [-0.7,-0.7]
#             dN = gradshape(q)  # 2x4
#             J  = np.dot(dN, nodePts).T  # 2x2
#             dN = np.dot(np.linalg.inv(J), dN)  # 2x4
#             B[0,0::2] = dN[0,:]  # 3x8
#             B[1,1::2] = dN[1,:]
#             B[2,0::2] = dN[1,:]
#             B[2,1::2] = dN[0,:]

#             UU = np.zeros((8,1))  # 8x1
#             UU[0] = u[2*c[0]]
#             UU[1] = u[2*c[0] + 1]
#             UU[2] = u[2*c[1]]
#             UU[3] = u[2*c[1] + 1]
#             UU[4] = u[2*c[2]]
#             UU[5] = u[2*c[2] + 1]
#             UU[6] = u[2*c[3]]
#             UU[7] = u[2*c[3] + 1]
#             # get the strain and stress at the integration point
#             strain = B @ UU  # (B is 3x8) (UU is 8x1) => (strain is 3x1)
#             stress = C @ strain  # (C is 3x3) (strain is 3x1) => (stress is 3x1)
#             emin[0] = min(emin[0], strain[0][0])
#             emin[1] = min(emin[1], strain[1][0])
#             emin[2] = min(emin[2], strain[2][0])
#             emax[0] = max(emax[0], strain[0][0])
#             emax[1] = max(emax[1], strain[1][0])
#             emax[2] = max(emax[2], strain[2][0])

#             node_strain[c[0]][:] = strain.T[0]
#             node_strain[c[1]][:] = strain.T[0]
#             node_strain[c[2]][:] = strain.T[0]
#             node_strain[c[3]][:] = strain.T[0]
#             node_stress[c[0]][:] = stress.T[0]
#             node_stress[c[1]][:] = stress.T[0]
#             node_stress[c[2]][:] = stress.T[0]
#             node_stress[c[3]][:] = stress.T[0]
#             smax[0] = max(smax[0], stress[0][0])
#             smax[1] = max(smax[1], stress[1][0])
#             smax[2] = max(smax[2], stress[2][0])
#             smin[0] = min(smin[0], stress[0][0])
#             smin[1] = min(smin[1], stress[1][0])
#             smin[2] = min(smin[2], stress[2][0])

#             # Accumulate total strain and stress for average calculation
#             total_strain += strain.T[0]
#             total_stress += stress.T[0]




#     # Calculate average strain and stress
#     average_strain = total_strain / num_elements
#     average_stress = total_stress / num_elements

#     if plot_flag:
#         # print(f'   min strains: e11={emin[0]:.4g}, e22={emin[1]:.4g}, e12={emin[2]:.4g}')
#         # print(f'   max strains: e11={emax[0]:.4g}, e22={emax[1]:.4g}, e12={emax[2]:.4g}')
#         # print(f'   min stress:  s11={smin[0]:.4g}, s22={smin[1]:.4g}, s12={smin[2]:.4g}')
#         # print(f'   max stress:  s11={smax[0]:.4g}, s22={smax[1]:.4g}, s12={smax[2]:.4g}')
#         # print(f'   average strains: e11={average_strain[0]:.4g}, e22={average_strain[1]:.4g}, e12={average_strain[2]:.4g}')
#         # print(f'   average stress:  s11={average_stress[0]:.4g}, s22={average_stress[1]:.4g}, s12={average_stress[2]:.4g}')
#         # print(f'   average strain/nodes:  s11={abs((average_stress[0])/len(conn))*10000:.4g}, s22={(abs(average_stress[1])/len(conn))*10000:.4g}, s12={(abs(average_stress[2])/len(conn))*10000:.4g}')
#         # print(f'   average strain/nodes:  e11={abs((average_strain[0])/len(conn))*10000:.4g}, e22={(abs(average_strain[1])/len(conn))*10000:.4g}, e12={(abs(average_strain[2])/len(conn))*10000:.4g}')

#         ##############################
#         # print('\n** Plot displacement')
#         xvec = []
#         yvec = []
#         res  = []
#         plot_type = 'e11'
#         for ni, pt in enumerate(nodes):
#             xvec.append(pt[1] + u[2*ni+1])  # Swap x and y
#             yvec.append(-(pt[0] + u[2*ni]))   # Swap x and y, negate y
#             if plot_type == 'u1':  res.append(u[2*ni])                # x-disp
#             if plot_type == 'u2':  res.append(u[2*ni+1])              # y-disp
#             if plot_type == 's11': res.append(node_stress[ni][0])     # s11
#             if plot_type == 's22': res.append(node_stress[ni][1])     # s22
#             if plot_type == 's12': res.append(node_stress[ni][2])     # s12
#             if plot_type == 'e11': res.append(node_strain[ni][0])     # e11
#             if plot_type == 'e22': res.append(node_strain[ni][1])     # e22
#             if plot_type == 'e12': res.append(node_strain[ni][2])     # e12

#         tri = []
#         for c in conn:
#             tri.append([c[0], c[1], c[2]])  # First triangle
#             tri.append([c[0], c[2], c[3]])  # Second triangle
#         t = plt.tricontourf(xvec, yvec, res, triangles=tri, levels=14, cmap=plt.get_cmap('jet'))

#         # Plot bounded nodes
#         for b in boundary:
#             node_index = b[0]
#             plt.plot(xvec[node_index], yvec[node_index], 'ro', markersize=5, label='Bounded Node' if b == boundary[0] else "")

#         # Plot loaded nodes with arrows
#         arrow_scale = 10  # Adjust this scale factor as needed
#         arrow_width = 0.010  # Adjust this width as needed

#         # Find the maximum magnitude
#         max_magnitude = max(abs(l[2]) for l in load)

#         for l in load:
#             node_index = l[0]
#             direction = l[1]
#             magnitude = l[2] / max_magnitude  # Normalize the magnitude
#             if direction == 2:  # Load in x-direction
#                 plt.quiver(xvec[node_index], yvec[node_index], magnitude, 0, angles='xy', scale_units='xy', scale=1, color='k', width=arrow_width, label='Loaded Node' if l == load[0] else "")
#             elif direction == 1:  # Load in y-direction
#                 plt.quiver(xvec[node_index], yvec[node_index], 0, -magnitude, angles='xy', scale_units='xy', scale=1, color='k', width=arrow_width, label='Loaded Node' if l == load[0] else "")

#             # Add annotation for the magnitude of the force
#             plt.annotate(f'{l[2]:.2f}', (xvec[node_index], yvec[node_index]), textcoords="offset points", xytext=(5,5), ha='center', color='k')

#         plt.scatter(xvec, yvec, marker='o', c='b', s=0.5) # (plot the nodes)
#         plt.grid(False)  # Turn off the grid
#         plt.colorbar(t)
#         plt.title(plot_type)
#         plt.axis('equal')
#         plt.show()
#         print('Done.')

#     if np.isnan(smax).any() or np.isnan(emax).any() or np.isnan(avg_u1).any() or np.isnan(avg_u2).any() or np.isnan(average_stress).any() or np.isnan(average_strain).any():
#         print("NaN detected in FEM results")

#     avg_strain_over_nodes = max(abs((average_strain[0]) / len(conn)) * 10000,abs((average_strain[1]) / len(conn)) * 10000, abs((average_strain[2]) / len(conn)) * 10000)
#     max_stress, max_strain = get_max_stress_and_strain(smax, smin, emax, emin)
#     return  max_stress, max_strain, avg_u1, avg_u2, len(conn), np.max(average_stress), np.max(average_strain), np.max(u[0::2]), np.max(u[1::2]), avg_strain_over_nodes
