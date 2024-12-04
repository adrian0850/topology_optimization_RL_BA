import numpy as np
from skimage.measure import label

DESIGN = 0
STRESSES = 1

E_MODUL = 70000 #N/mm²
POISSON = 0.35 #Poisson's ratio

def create_grid(width, height, bounded, loaded):
    grid = np.full((2, width, height), "M", dtype=object)
    print(grid) 
    
    grid = encode_bounded_elements(grid, bounded)
    grid = encode_loaded_nodes(grid, loaded)

    grid[STRESSES][:][:] = np.zeros((width, height))
    return grid

def encode_loaded_nodes(grid, coordinate_list):
    for (row, col, val) in coordinate_list:
        print(f"Encoding loaded node at ({row}, {col}) with value '{val}'")
        grid[DESIGN][row][col] = val
    return grid

def encode_bounded_elements(grid, coordinate_list, char="B"):
    for (row, col) in coordinate_list:
        grid[DESIGN][row][col] = char
    return grid

def label_(x):
    return label(x, connectivity=1)

def get_zero_label(d, labels):
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if d[i, j] == 0:
                return labels[i, j]

def isolate_largest_group_original(x):
    x_original = np.copy(x)
    labels = label_(x)
    zerolabel = get_zero_label(x, labels)
    group_names = np.unique(labels)
    group_names = [g for g in group_names if g != zerolabel]
    vols = np.array([(labels == name).sum() for name in group_names])
    largestgroup_name = group_names[np.argmax(vols)]
    x = labels == largestgroup_name
    x = x.astype("int")
    design_ok = np.all(x == x_original)
    return x, design_ok

def rectangularmesh(Lx, Ly, ElementsX, ElementsY):
    Nx = ElementsX
    Ny = ElementsY
    xx = np.linspace(0, Lx, ElementsX + 1)
    yy = np.linspace(0, Ly, ElementsY + 1)
    nodeCoor = np.zeros(((Nx + 1) * (Ny + 1), 2))
    elementNodes = np.zeros((Nx * Ny, 4))
    for j in range(0, len(yy)):
        for i in range(0, len(xx)):
            call = (Nx + 1) * j + i
            nodeCoor[call, 0] = xx[i]
            nodeCoor[call, 1] = yy[j]
    for j in range(0, Ny):
        for i in range(0, Nx):
            d = Nx * j + i
            elementNodes[d, 0] = (Nx + 1) * j + i
            elementNodes[d, 1] = (Nx + 1) * j + i + 1
            elementNodes[d, 2] = (Nx + 1) * (j + 1) + i + 1
            elementNodes[d, 3] = (Nx + 1) * (j + 1) + i
    return nodeCoor, elementNodes

def shapeFunctionQ4(xi, eta):
    shape = (1 / 4) * np.matrix([[(1 - xi) * (1 - eta)],
                                 [(1 + xi) * (1 - eta)],
                                 [(1 + xi) * (1 + eta)],
                                 [(1 - xi) * (1 + eta)]])
    naturalDeriv = (1 / 4) * np.matrix([[-(1 - eta), -(1 - xi)],
                                        [(1 - eta), -(1 + xi)],
                                        [(1 + eta), (1 + xi)],
                                        [-(1 + eta), (1 - xi)]])
    return shape, naturalDeriv

def Jacobian(nodeCoor, naturalDeriv):
    JacobianMatrix = np.transpose(nodeCoor) * naturalDeriv
    invJacobian = np.linalg.inv(JacobianMatrix)
    XYDerivative = naturalDeriv * invJacobian
    return JacobianMatrix, invJacobian, XYDerivative

def formStiffness2D(GDof, TotElements, numberNodes, elementNodes, nodeCoor, materials, rho, thickness, VoidCheck):
    stiffness = np.zeros((GDof, GDof))
    mass = np.zeros((GDof, GDof))
    gaussLocations = np.array([[-.577350269189626, -.577350269189626],
                               [.577350269189626, -.577350269189626],
                               [.577350269189626, .577350269189626],
                               [-.577350269189626, .577350269189626]])
    gaussWeights = np.array([[1], [1], [1], [1]])
    for e in range(TotElements):
        indice = elementNodes[e, :]
        indice = indice.astype(int)
        elementDof = np.append(indice, indice + numberNodes)
        ndof = len(indice)
        for q in range(len(gaussWeights)):
            GaussPoint = gaussLocations[q, :]
            xi = GaussPoint[0]
            eta = GaussPoint[1]
            sFQ4 = shapeFunctionQ4(xi, eta)
            naturalDeriv = sFQ4[1]
            JR = Jacobian(nodeCoor[indice, :], naturalDeriv)
            JacobianMatrix = JR[0]
            XYderivative = JR[2]
            B = np.zeros((3, 2 * ndof))
            B[0, 0:ndof] = np.transpose(XYderivative[:, 0])
            B[1, ndof:(2 * ndof)] = np.transpose(XYderivative[:, 1])
            B[2, 0:ndof] = np.transpose(XYderivative[:, 1])
            B[2, ndof:(2 * ndof)] = np.transpose(XYderivative[:, 0])
            if VoidCheck[e] == 1:
                E = materials[0, 0]
                poisson = materials[0, 1]
            else:
                E = materials[1, 0]
                poisson = materials[1, 1]
            C = np.matrix([[1, poisson, 0], [poisson, 1, 0], [0, 0, (1 - poisson) / 2]]) * (E / (1 - (poisson ** 2)))
            Mat1 = np.asarray(np.transpose(B) * C * thickness * B * 1 * np.linalg.det(JacobianMatrix))
            stiffness[np.ix_(elementDof, elementDof)] = stiffness[np.ix_(elementDof, elementDof)] + Mat1
    return stiffness, mass

def solution(GDof, prescribedDof, stiffness, force):
    activeDof = []
    GDof_list = list(range(GDof))
    for i in GDof_list:
        if i not in prescribedDof[0]:
            activeDof.append(i)
    ActiveStiffness = stiffness[np.ix_(activeDof, activeDof)]
    ActiveForce = np.zeros((len(activeDof), 1))
    for i in range(len(activeDof)):
        ActiveForce[i] = force[activeDof[i]]
    U = np.matmul(np.linalg.inv(ActiveStiffness), ActiveForce)
    displacements = np.zeros((GDof, 1))
    displacements[activeDof] = U
    return displacements

def stresses2D(GDof, TotElements, nodeCoor, numberNodes, displacements, UX, UY, materials, ScaleFactor, VoidCheck, elementNodes):
    gaussLocations = np.array([[-.577350269189626, -.577350269189626],
                               [.577350269189626, -.577350269189626],
                               [.577350269189626, .577350269189626],
                               [-.577350269189626, .577350269189626]])
    gaussWeights = np.array([[1], [1], [1], [1]])
    stress = np.zeros((TotElements, len(elementNodes[0]), 3))
    for e in range(TotElements):
        indice = elementNodes[e, :]
        indice = indice.astype(int)
        elementDof = np.append(indice, indice + numberNodes)
        nn = len(indice)
        for q in range(len(gaussWeights)):
            pt = gaussLocations[q, :]
            xi = pt[0]
            eta = pt[1]
            sFQ4 = shapeFunctionQ4(xi, eta)
            naturalDeriv = sFQ4[1]
            JR = Jacobian(nodeCoor[indice, :], naturalDeriv)
            XYderivative = JR[2]
            B = np.zeros((3, 2 * nn))
            B[0, 0:nn] = np.transpose(XYderivative[:, 0])
            B[1, nn:(2 * nn)] = np.transpose(XYderivative[:, 1])
            B[2, 0:nn] = np.transpose(XYderivative[:, 1])
            B[2, nn:(2 * nn)] = np.transpose(XYderivative[:, 0])
            if VoidCheck[e] == 1:
                E = materials[0, 0]
                poisson = materials[0, 1]
            else:
                E = materials[1, 0]
                poisson = materials[1, 1]
            C = np.matrix([[1, poisson, 0], [poisson, 1, 0], [0, 0, (1 - poisson) / 2]]) * (E / (1 - (poisson ** 2)))
            strain = np.matmul(B, displacements[elementDof])
            stress[e, q, :] = np.transpose(np.dot(C, strain))
    return stress

def FEASolve(VoidCheck, Lx, Ly, ElementsX, ElementsY, LC_Nodes, Loaded_Directions, BC_Nodes, Stress):
    materials = np.zeros((2, 2))
    materials[0, 0] = E_MODUL  # Modulus of Elasticity
    materials[0, 1] = POISSON  # Poisson's Ratio
    materials[1, 0] = 1  # Modulus of Elasticity (1e-6 MPa)
    materials[1, 1] = 0.000001  # Poisson's Ratio
    P = 5e5  # Magnitude
    TotElements = ElementsX * ElementsY
    RectOutput = rectangularmesh(Lx, Ly, ElementsX, ElementsY)
    nodeCoor = RectOutput[0]
    elementNodes = RectOutput[1]
    xx = nodeCoor[:, 0]
    yy = nodeCoor[:, 1]
    numberNodes = len(xx)
    GDof = 2 * numberNodes
    fixedNodeX = list(BC_Nodes)
    fixedNodeY = list(BC_Nodes + numberNodes)
    prescribedDof = [fixedNodeX + fixedNodeY]
    force = np.zeros((GDof, 1))
    LD_Count = 0
    for F in range(len(LC_Nodes)):
        if F / 2 == int(F / 2) and F / 2 != 0:
            LD_Count += 1
        force[int(LC_Nodes[F])] = (P / 2) * Loaded_Directions[LD_Count]
    FS2D = formStiffness2D(GDof, TotElements, numberNodes, elementNodes, nodeCoor, materials, 1, 1, VoidCheck)
    stiffness = FS2D[0]
    displacements = solution(GDof, prescribedDof, stiffness, force)
    UX = np.asarray(np.transpose(displacements[0:numberNodes]))
    UY = np.asarray(np.transpose(displacements[numberNodes:GDof]))
    MaxDisplacements = displacements.min()
    StrainEnergy = np.matmul(np.transpose(displacements), stiffness)
    StrainEnergy = np.matmul(StrainEnergy, displacements)
    RectOutput = rectangularmesh(Lx, Ly, ElementsX, ElementsY)
    if Stress:
        ScaleFactor = 0.1
        StressVal = stresses2D(GDof, TotElements, nodeCoor, numberNodes, displacements, UX, UY, materials, ScaleFactor, VoidCheck, elementNodes)
        stress = StressVal
        sigmax = stress[:, :, 0]
        sigmay = stress[:, :, 1]
        tauxy = stress[:, :, 2]
        stressx_node = np.zeros((ElementsX + 1, ElementsY + 1))
        stressy_node = np.zeros((ElementsX + 1, ElementsY + 1))
        stressxy_node = np.zeros((ElementsX + 1, ElementsY + 1))
        stressx_element = np.zeros((ElementsX * ElementsY, 1))
        stressy_element = np.zeros((ElementsX * ElementsY, 1))
        stressxy_element = np.zeros((ElementsX * ElementsY, 1))
        Count = 0
        for j in range(ElementsY):
            for i in range(ElementsX):
                stressx_node[np.ix_(range(i, i + 2), range(j, j + 2))] = np.reshape(sigmax[Count, :], (2, 2))
                stressy_node[np.ix_(range(i, i + 2), range(j, j + 2))] = np.reshape(sigmay[Count, :], (2, 2))
                stressxy_node[np.ix_(range(i, i + 2), range(j, j + 2))] = np.reshape(tauxy[Count, :], (2, 2))
                Count += 1
        for ii in range(len(stressx_element)):
            stressx_element[ii] = np.mean(sigmax[ii, :])
            stressy_element[ii] = np.mean(sigmay[ii, :])
            stressxy_element[ii] = np.mean(tauxy[ii, :])
        stressx_element = np.reshape(stressx_element, (ElementsX, ElementsY))
        stressy_element = np.reshape(stressy_element, (ElementsX, ElementsY))
        stressxy_element = np.reshape(stressxy_element, (ElementsX, ElementsY))
        VonMises_element = np.sqrt((stressx_element * stressx_element) - (stressx_element * stressy_element) + (stressy_element * stressy_element) + (3 * stressxy_element * stressxy_element))
        VonMises_node = np.sqrt((stressx_node * stressx_node) - (stressx_node * stressy_node) + (stressy_node * stressy_node) + (3 * stressxy_node * stressxy_node))
        VonMises_node_nn = np.reshape(np.flip(VonMises_node, 0), (((ElementsX + 1) * (ElementsY + 1), 1)))
        VonMises_nn = np.reshape(VonMises_element, ((ElementsX * ElementsY), 1))
        VonMises_node_nn = max(VonMises_node_nn) / VonMises_node_nn
        VonMises_node_nn[VonMises_node_nn > 1e4] = 0
        VonMises_node_nn = VonMises_node_nn / max(VonMises_node_nn)
        VonMises_nn = max(VonMises_nn) / VonMises_nn
        VonMises_nn[VonMises_nn > 1e3] = 0
        VonMises_nn = VonMises_nn / max(VonMises_nn)
    else:
        VonMises_nn = []
    return MaxDisplacements, StrainEnergy, VonMises_element, VonMises_nn, VonMises_node_nn

def FEA_analysis(grid):
    # Example parameters for FEASolve
    VoidCheck = get_void_check(grid)
    Lx = grid.shape[1]
    Ly = grid.shape[2]
    ElementsX = grid.shape[1]
    ElementsY = grid.shape[2]


    LC_Nodes, Loaded_Directions = get_loaded_nodes(grid) 
    BC_Nodes = get_bounded_nodes(grid)
    Stress = True  # Example stress calculation flag

    return FEASolve(VoidCheck, Lx, Ly, ElementsX, ElementsY, LC_Nodes, Loaded_Directions, BC_Nodes, Stress)

def get_bounded_nodes(grid):
    bounded_nodes = []
    for i in range(grid.shape[1]):
        for j in range(grid.shape[2]):
            if grid[DESIGN][i][j] == "B":
                bounded_nodes.append(get_flattend_index(i, j, grid.shape[1]))
    return bounded_nodes

def get_loaded_nodes(grid):
    loaded_nodes = []
    directions = []
    for i in range(grid.shape[1]):
        for j in range(grid.shape[2]):
            if check_if_is_loaded(grid[DESIGN][i][j]): 
                if grid[DESIGN][i][j][0] == "0":
                    directions.append(0)
                else:
                    directions.append(1)
                loaded_nodes.append(get_flattend_index(i, j, grid.shape[1]))
    return loaded_nodes, directions

def get_flattend_index(i, j, width):
    return i * width + j

def check_if_is_loaded(element):
    if not isinstance(element, str):
        return False
    return element.isdigit() and (element[0] == "0" or element[0] == "1")

def get_void_check(grid):
    VoidCheck = np.ones(grid.shape[1] * grid.shape[2])
    VoidCheck[grid[DESIGN].flatten() != 0] = 0
    return VoidCheck
    

def main():
    width = int(input("Enter grid width: "))
    height = int(input("Enter grid height: "))

    bounded = [(0, 0), (5, 0)]
    loaded = [(5, 11, "011")]
    # bounded_input = input("Enter bounded coordinates as (row,col) separated by spaces: ")
    # bounded = [tuple(map(int, coord.split(','))) for coord in bounded_input.split()]
    
    # loaded_input = input("Enter loaded coordinates as (row,col,val) separated by spaces: ")
    # loaded = [tuple(map(int, coord.split(','))) for coord in loaded_input.split()]

    grid = create_grid(width, height, bounded, loaded)
    print("Grid created:")
    print(grid)
    grid[DESIGN][0][4] = 0
    VoidCheck = get_void_check(grid)
    print(VoidCheck)

    analysis_result = FEA_analysis(grid)
    print("FEA Analysis Result:")
    print(analysis_result)

if __name__ == "__main__":
    main()