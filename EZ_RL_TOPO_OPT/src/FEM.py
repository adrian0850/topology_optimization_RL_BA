import math
import constants as const

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
    for element_id, element in enumerate(conn):
        # Ignore the last element (is_voided flag) when accessing node indices
        x = [nodes[i][1] for i in element[:4]]  # Swap x and y
        y = [-nodes[i][0] for i in element[:4]]  # Swap x and y and negate y
        if element[4]:  # Check if the element is voided
            plt.plot(x, y, 'r--')  # Plot voided elements with red dashed lines
        else:
            plt.plot(x, y, 'b-')  # Plot non-voided elements with blue solid lines

        # Calculate the centroid of the element for annotation
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        plt.text(centroid_x, centroid_y, str(element_id + 1), fontsize=12, ha='center', color='blue')

    plt.scatter(nodes[:, 1], -nodes[:, 0], color='red', s=10)  # Swap x and y and negate y

    # Add text annotations for each node
    for i, (x, y) in enumerate(nodes):
        plt.text(y, -x, str(i + 1), fontsize=12, ha='right', color='red')

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


def FEM(nodes, conn, boundary, load, plot_flag, grid, device='cpu'):
    nodes = np.array(nodes)
    num_nodes = len(nodes)
    #print('   number of nodes:', len(nodes))
    #print('   number of elements:', len(conn))
    #print('   number of displacement boundary conditions:', len(boundary))
###############################
    # Plane-strain material tangent (see Bathe p. 194)
    # C is 3x3
    E = const.E
    v = const.v
    C_real = const.C_real
    C_dummy = const.C_dummy
    ###############################
    # Make stiffness matrix
    # if N is the number of DOF, then K is NxN
    K = lil_matrix((2 * num_nodes, 2 * num_nodes))    # square zero matrix
    # 2x2 Gauss Quadrature (4 Gauss points)
    # q4 is 4x2
    q4 = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]) / math.sqrt(3.0)
    # print('\n** Assemble stiffness matrix')
    # strain in an element: [strain] = B    U
    #                        3x1     = 3x8  8x1
    #
    # strain11 = B11 U1 + B12 U2 + B13 U3 + B14 U4 + B15 U5 + B16 U6 + B17 U7 + B18 U8
    #          = B11 u1          + B13 u1          + B15 u1          + B17 u1
    #          = dN1/dx u1       + dN2/dx u1       + dN3/dx u1       + dN4/dx u1
    B = np.zeros((3, 8))
    # conn[0] is node numbers of the element
    for c in conn:
        nodePts = nodes[c[:4], :]  # Only take the first three nodes for the element
        is_voided = c[4]  # The fourth element is the voided flag
        Ke = np.zeros((8, 8))
        for q in q4:
            dN = gradshape(q)

            J = np.dot(dN, nodePts).T
            if np.linalg.det(J) == 0:
                print("Singular Jacobian detected.")
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            dN = np.dot(np.linalg.inv(J), dN)
            B[0, 0::2] = dN[0, :]
            B[1, 1::2] = dN[1, :]
            B[2, 0::2] = dN[1, :]
            B[2, 1::2] = dN[0, :]

            if is_voided:
                C = C_dummy
            else:
                C = C_real

            Ke += np.dot(np.dot(B.T, C), B) * np.linalg.det(J)
        # Scatter operation
        for i, I in enumerate(c[:4]):  # Only take the first three nodes for the element
            for j, J in enumerate(c[:4]):  # Only take the first three nodes for the element
                K[2 * I, 2 * J] += Ke[2 * i, 2 * j]
                K[2 * I + 1, 2 * J] += Ke[2 * i + 1, 2 * j]
                K[2 * I + 1, 2 * J + 1] += Ke[2 * i + 1, 2 * j + 1]
                K[2 * I, 2 * J + 1] += Ke[2 * i, 2 * j + 1]
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
    try:
        u = spsolve(K, f)
    except Exception as e:
        print(f"Error solving linear system: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if np.isnan(u).any():
        print("NaN detected in displacement vector.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

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

    von_mises_stresses = []

    for c in conn:  # for each element (conn is Nx4)
        nodePts = nodes[c[:4], :]  # Only take the first three nodes for the element
        is_voided = c[4]  # The fourth element is the voided flag
        element_von_mises_stresses = []  # List to store von Mises stresses for the current element
        for q in q4:  # for each integration pt, eg: [-0.7,-0.7]
            dN = gradshape(q)  # 2x4
            J  = np.dot(dN, nodePts).T  # 2x2
            if np.linalg.det(J) == 0:
                print("Singular Jacobian detected during post-processing.")
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
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
            node_strain[c[:3], :] = strain.T
            node_stress[c[:3], :] = stress.T

            von_mises_stress = np.sqrt(stress[0]**2 - stress[0]*stress[1] + stress[1]**2 + 3*stress[2]**2).item()
            element_von_mises_stresses.append(von_mises_stress)

            # Accumulate total strain and stress for average calculation
            total_strain += strain.T[0]
            total_stress += stress.T[0]

        # Calculate the average von Mises stress for the current element
        avg_von_mises_stress = np.mean(element_von_mises_stresses)
        von_mises_stresses.append(avg_von_mises_stress)


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
        voided_nodes = set()
        for c in conn:
            if c[4]:  # Check if the element is voided
                voided_nodes.update(c[:4])
        
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
            if c[4]:  # Check if the element is voided
                plt.fill([xvec[c[0]], xvec[c[1]], xvec[c[2]], xvec[c[3]]],
                     [yvec[c[0]], yvec[c[1]], yvec[c[2]], yvec[c[3]]],
                     'gray', alpha=0.0)  # Plot voided elements with transparency
            else:
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
    return von_mises_stresses
    #return  von_mises_stresses, max_stress, max_strain, avg_u1, avg_u2, len(conn), np.max(average_stress), np.max(average_strain), np.max(u[0::2]), np.max(u[1::2]), avg_strain_over_nodes


