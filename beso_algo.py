import numpy as np
import matplotlib.pyplot as plt

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
        plt.imshow(-x, cmap='gray', interpolation='nearest')
        plt.axis('equal')
        plt.axis('off')
        plt.show(block=False)
        plt.pause(1e-6)

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
    F = np.zeros((2 * (nely + 1) * (nelx + 1), 1))
    U = np.zeros((2 * (nely + 1) * (nelx + 1), 1))


    for elx in range(1, nelx + 1):
        for ely in range(1, nely + 1):
            n1 = (nely + 1) * (elx - 1) + ely
            n2 = (nely + 1) * elx + ely
            edof = np.array([2*n1-2, 2*n1-1, 2*n2-2, 2*n2-1, 2*n2, 2*n2 + 1, 2*n1, 2*n1+1])
            K[np.ix_(edof, edof)] += x[ely - 1, elx - 1]**penal * KE

    # DEFINE LOADS AND SUPPORT (Cantilever)
    F[2 * (nelx + 1) * (nely + 1) - nely- 1, 0] = -1.0
    fixeddofs = np.arange(0, 2 * (nely + 1))
    alldofs = np.arange(0, 2 * (nely + 1) * (nelx + 1))
    freedofs = np.setdiff1d(alldofs, fixeddofs)

    # SOLVING
    K_ff = K[np.ix_(freedofs, freedofs)]
    f_f = F[freedofs, 0]
    U[freedofs, 0] = np.linalg.solve(K_ff, f_f)
    U[fixeddofs, 0] = 0

    return U

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
        sbeso(120, 40, 0.45, 0.02, 1.5)
if __name__ == '__main__':
    main()


