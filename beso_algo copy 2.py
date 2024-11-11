import numpy as np
import matplotlib.pyplot as plt

WIDTH = 12
HEIGHT = 6

# Load conditions are the following:
# B_LEFT; B_MIDDLE; B_RIGHT; M_LEFT; M_MIDDLE; M RIGHT; T_LEFT; T_MIDDLE; T_RIGHT

LOAD_VAR = "B_RIGHT"

# Bounded conditions are the following:
# ALL_LEFT
# ALL_RIGHT
# ALL_TOP
# ALL_BOTTOM
# ALL_LEFT_RIGHT
# ALL_TOP_BOTTOM
# LT_LB (top and bottom left side)
# RT_RB (top and bottom right side)
# LT_RT (left and right side top)
# LB_RB (left and right side bottom)
# LT_RB (left top and right bottom)
# LB_RT (left bottom and right top)

BOUND_VAR = "ALL_LEFT"

def sbeso(nelx, nely, volfrac, er, rmin):
    x = np.ones((nely, nelx))
    vol = 1.0
    i = 0
    change = 1.0
    penal = 3.0
    dc = np.zeros((nely, nelx))
    c_list= []

    while change > 0.001:
        i += 1
        vol = max(vol * (1 - er), volfrac)

        if i > 1:
            olddc = dc

        # FE-ANALYSIS
        U = FE(nelx, nely, x, penal)

        # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
        KE = lk()
        c = 0.0

        for ely in range(1, nely + 1):
            for elx in range(1, nelx + 1):
                n1 = (nely + 1) * (elx - 1) + ely
                n2 = (nely + 1) * elx + ely
                Ue = U[[2 * n1 - 2, 2 * n1-1, 2 * n2 - 2, 2 * n2-1, 2 * n2 , 2 * n2 + 1, 2 * n1 , 2 * n1 + 1], 0]
                c += 0.5 * x[ely - 1, elx - 1]**penal * Ue.T @ KE @ Ue
                dc[ely - 1, elx - 1] = 0.5 * x[ely - 1, elx - 1]**(penal - 1) * Ue.T @ KE @ Ue

        c_list.append(c)

        # FILTERING OF SENSITIVITIES
        dc = check(nelx, nely, rmin, x, dc)

        # STABLIZATION OF EVOLUTIONARY PROCESS
        if i > 1:
            dc = (dc + olddc) / 2.0

        # BESO DESIGN UPDATE
        x = ADDDEL(nelx, nely, vol, dc, x)

        # PRINT RESULTS
        if i > 10:
            change = abs(sum(c_list[i - 9:i - 5]) - sum(c_list[i - 4:i])) / sum(c_list[i - 4:i])

        print(f"It.: {i:4d} Obj.: {c_list[-1]:10.4f} Vol.: {sum(sum(x)) / (nelx * nely):6.3f} ch.: {change:6.3f}")

        # PLOT DENSITIES
        plt.imshow(-x, cmap='YlOrBr_r', interpolation='nearest')
        plt.axis('equal')
        plt.axis('off')
        #plt.show(block=False)
        #plt.pause(1e-6)
        plt.savefig("beso.png")
        plt.close()

# Replace the placeholders with the actual implementations of FE, lk, check, ADDDEL, and disp
# Ensure that the data types and function signatures match the original Simp code

# Additional functions
def ADDDEL(nelx, nely, volfra, dc, x):
    l1 = np.min(dc)
    l2 = np.max(dc)

    while (l2 - l1) / l2 > 1.0e-5:
        th = (l1 + l2) / 2.0
        x = np.maximum(0.001, np.sign(dc - th))

        if sum(sum(x)) - volfra * (nelx * nely) > 0:
            l1 = th
        else:
            l2 = th

    return x

def check(nelx, nely, rmin, x, dc):
    dcf = np.zeros((nely, nelx))
    for i in range(1, nelx + 1):
        for j in range(1, nely + 1):
            sum = 0.0
            for k in range(max(i - int(rmin), 1), min(i + int(rmin), nelx) + 1):
                for l in range(max(j - int(rmin), 1), min(j + int(rmin), nely) + 1):
                    fac = rmin - np.sqrt((i - k)**2 + (j - l)**2)
                    sum += max(0, fac) * dc[l - 1, k - 1]

            dcf[j - 1, i - 1] = dcf[j - 1, i - 1] / sum

    return dcf

def FE(nelx, nely, x, penal):
    KE = lk()
    K = np.zeros((2 * (nelx + 1) * (nely + 1), 2 * (nelx + 1) * (nely + 1)))
    U = np.zeros((2 * (nely + 1) * (nelx + 1), 1))


    for elx in range(1, nelx + 1):
        for ely in range(1, nely + 1):
            n1 = (nely + 1) * (elx - 1) + ely
            n2 = (nely + 1) * elx + ely
            edof = np.array([2*n1-2, 2*n1-1, 2*n2-2, 2*n2-1, 2*n2, 2*n2 + 1, 2*n1, 2*n1+1])
            K[np.ix_(edof, edof)] += x[ely - 1, elx - 1]**penal * KE

    # DEFINE LOADS AND SUPPORT (Cantilever)
    F = get_loaded_matrix(nelx, nely)
    fixeddofs, alldofs, freedofs = get_dofs(nelx, nely)

    # SOLVING
    K_ff = K[np.ix_(freedofs, freedofs)]
    f_f = F[freedofs, 0]
    U[freedofs, 0] = np.linalg.solve(K_ff, f_f)
    U[fixeddofs, 0] = 0

    return U

def get_dofs(nelx, nely):
    alldofs = np.arange(0, 2 * (nely + 1) * (nelx + 1))
    last_node = 2 * (nely + 1) * (nelx + 1)
    brush_y = int((nely/10)*2)
    brush_x = int(nelx/10)
    match BOUND_VAR:
        case "ALL_LEFT":
            fixeddofs = np.arange(0, 2 * (nely + 1))
        case "ALL_RIGHT":
            fixeddofs = np.arange(last_node - 2 * (nely + 1), last_node)
        case "ALL_TOP":
            fixeddofs = np.arange(0, last_node- (2 * (nely + 1)), step=nely + 1)
            for i in range(fixeddofs.size):
                fixeddofs = np.append(fixeddofs, fixeddofs[i] + 1)
            print(alldofs)
            print(fixeddofs)
        case "ALL_BOTTOM":
            fixeddofs = np.arange(nely, last_node, step=nely)
        case "LB_RB":
            LB = np.arange(2*(nely), (2*nely+1) - brush_y, step=-1)
            BL = np.arange(2*(nely), brush_x * (nely+1)* (nelx+1))
            RB = np.arange(last_node, last_node - brush_y, step=-1)
            BR = np.arange(last_node, last_node - brush_x * (nely+1), step=nely+1)
            fixeddofs = LB
            print(fixeddofs)
        case "LEFT_RIGHT":
            fixeddofs = np.concatenate((np.arange(0, 2 * (nely + 1)), np.arange(last_node - 2 * (nely + 1), last_node)))
        case _:
            raise ValueError(f"Unknown BOUND_VAR: {BOUND_VAR}")

    alldofs = np.arange(0, 2 * (nely + 1) * (nelx + 1))
    freedofs = np.setdiff1d(alldofs, fixeddofs)
    return fixeddofs, alldofs, freedofs
    

def get_loaded_matrix(nelx, nely):
    F = np.zeros((2 * (nely+1) * (nelx+1), 1))
    last_node = 2 * (nely+1) * (nelx+1)

    match LOAD_VAR:
        case "B_RIGHT":
            dof_pos = last_node - 1
        case "B_MIDDLE":
            dof_pos = int(0.5 * (last_node-1) + (nely + 1)) 
        case "B_LEFT":
            dof_pos = 2 * (nely + 1) - 1
        case "M_RIGHT":
            dof_pos = int(2 * (nelx + 1) * (nely + 1) -   (nely + 1))
        case "M_MIDDLE":
            dof_pos = 0.5 * last_node
        case "M_LEFT":
            dof_pos = 0.5 * (nely + 1) - 1
        case "T_RIGHT":
            dof_pos = int(last_node - 2 * (nely + 1))
        case "T_MIDDLE":
            dof_pos = int(0.5 * last_node - (nely + 1))
            
        case "T_LEFT":
            dof_pos = 0
        case _:
            raise ValueError(f"Unknown LOAD_VAR: {LOAD_VAR}")
    F[dof_pos, 0] = -1.0
    return F

def lk():
    E = 1.0
    nu = 0.3
    k = [1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
         -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8]
    KE = E / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
        ])
    return KE


def main():
        sbeso(WIDTH, HEIGHT, 0.6, 0.02, 1.5)
if __name__ == '__main__':
    main()


