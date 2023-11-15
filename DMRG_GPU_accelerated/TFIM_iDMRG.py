import torch
import matplotlib.pyplot as plt
import numpy as np


""" Torch Setting """
device = torch.device("cpu")


sigma_x = torch.tensor([[0, 1], [1, 0]]).to(device)
sigma_z = torch.tensor([[1, 0], [0, -1]]).to(device)

LatticeLength = 16  # Final Lattice Length
MaximalStates = 16  # Maximum States for truncation

J = 1  # Interaction Strength for x direction
h = 0.8  # Interaction Strength  for z direction

""" Initialize the sysBlock and envBlock"""
sysBlock_Ham = -h * sigma_z  # Initially we have on-site potential term
sysBlock_Sigma_x = sigma_x
sysBlock_Sigma_z = sigma_z
sysBlock_Length = 1

# Environment elements:
envBlock_Ham = -h * sigma_z  # Initially we have on-site potential term
envBlock_Sigma_x = sigma_x
envBlock_Sigma_z = sigma_z
envBlock_Length = 1


Dim = 2  # Local site dimension

E_GS = []  # To store the ground state energy of eahc DMRG step
SE = []  # von Neumann Entropy for sysBlock


"""
Convert it to PyTorch Version 
"""

""" Torch Version Partial Trace"""


def partial_trace(psi, n1, n2):
    # define the density matrix for ground state psi
    # rho = psi @ psi.conj().T
    rho = torch.ger(psi, psi.T).to(device)
    rho_tensor = rho.reshape(int(n1), int(n2), int(n1), int(n2))
    RDM_sys = torch.einsum("ijkj->ik", rho_tensor)
    RDM_env = torch.einsum("ijil->jl", rho_tensor)

    return RDM_sys.to(device), RDM_env.to(device)


while (sysBlock_Length + envBlock_Length) < LatticeLength:
    # Step 1: Enlarge the Blocks by adding a new site

    """To check whether the dimensions of operators are correct"""
    sysBlock_Ham = (
        torch.kron(sysBlock_Ham, torch.eye(Dim, device=device)).to(device)
        - h
        * torch.kron(torch.eye(len(sysBlock_Ham), device=device), sigma_z).to(device)
        - J * torch.kron(sysBlock_Sigma_x, sigma_x).to(device)
    )

    envBlock_Ham = (
        torch.kron(torch.eye(Dim).to(device), envBlock_Ham).to(device)
        - h * torch.kron(sigma_z, torch.eye(len(envBlock_Ham)).to(device)).to(device)
        - J * torch.kron(sigma_x, envBlock_Sigma_x).to(device)
    )

    # Make sure both sysBlock and envBlock are Hermitian
    sysBlock_Ham[:] = 0.5 * (sysBlock_Ham + sysBlock_Ham.conj().T)
    envBlock_Ham[:] = 0.5 * (envBlock_Ham + envBlock_Ham.conj().T)

    # Step 2: Perpare the operator for Superblock Hamiltonain
    # The operators for middle two points
    sysBlock_Sigma_x = torch.kron(
        torch.eye(int(sysBlock_Ham.shape[0] // Dim)).to(device), sigma_x
    )
    sysBlock_Sigma_z = torch.kron(
        torch.eye(int(sysBlock_Ham.shape[0] // Dim)).to(device), sigma_z
    )

    envBlock_Sigma_x = torch.kron(
        sigma_x, torch.eye(int(envBlock_Ham.shape[0] // Dim)).to(device)
    )
    envBlock_Sigma_z = torch.kron(
        sigma_z, torch.eye(int(envBlock_Ham.shape[0] // Dim)).to(device)
    )

    # Update the size of both blocks
    sysBlock_Length = sysBlock_Length + 1
    envBlock_Length = envBlock_Length + 1

    # Step 3: Construct the Superblock Hamiltonain
    # print(len(sysBlock_Ham.toarray()))
    H_super = (
        torch.kron(sysBlock_Ham, torch.eye(int(envBlock_Ham.shape[0])).to(device))
        + torch.kron(torch.eye(int(sysBlock_Ham.shape[0])).to(device), envBlock_Ham)
        - J * torch.kron(sysBlock_Sigma_x, envBlock_Sigma_x)
    )

    # Return the ground state of superblock
    # val_States, vec_States = torch.linalg.svd(H_super)
    # val_GS, vec_GS = val_States[0], vec_States[:, 0]

    U, S, V_dagger = torch.linalg.svd(H_super)

    # Choose right Vec_GS from SVD, since SVD is semi-positive definite
    # Then if there are enegies level -1 , 1 --> 1 in SVD

    if torch.dot(U[:, 0], H_super @ U[:, 0]) <= 0:
        vec_GS = U[:, 0]
    else:
        vec_GS = U[:, 1]

    E_GS_local = torch.dot(vec_GS, (H_super @ vec_GS)) / (
        sysBlock_Length + envBlock_Length
    )

    print(f"Energies={E_GS_local}")
    E_GS.append(E_GS_local)

    # Step 4: Construct the RDM for sysBlock and envBlock
    sysBlock_DM, envBlock_DM = partial_trace(
        vec_GS,
        int(sysBlock_Ham.shape[0]),
        int(envBlock_Ham.shape[0]),
    )

    # Check trace of density matrix always = 1
    print(
        f" sysblock_DM Hermitian check {torch.dist(sysBlock_DM, sysBlock_DM.conj().T )}"
    )

    """ Make Sure the matrix is Hermitian"""
    sysBlock_DM = (sysBlock_DM + sysBlock_DM.conj().T) / 2
    envBlock_DM = (envBlock_DM + envBlock_DM.conj().T) / 2

    # Diagonalize the reduced density matrix
    (
        sysBlock_rotationMatrix,
        sysBlock_weight,
        sysBlock_rotationMatrix_dagger,
    ) = torch.linalg.svd(sysBlock_DM)
    (
        envBlock_rotationMatrix,
        envBlock_weight,
        envBlock_rotationMatrix_dagger,
    ) = torch.linalg.svd(envBlock_DM)

    print(f" check trace of RDM = {torch.sum(sysBlock_weight)}")

    """Make the sysBlock weigth array in descending order"""
    # Sorted Method using Numpy
    sysBlock_idx = torch.argsort(sysBlock_weight, descending=True)
    sysBlock_weight_sort = sysBlock_weight[sysBlock_idx]

    # Prevent some negative zeros
    sysBlock_weight_sort = sysBlock_weight_sort[sysBlock_weight_sort > 0]
    Isys = sysBlock_idx

    print(f"sorted {sysBlock_weight_sort}")
    # Check Entanglement
    # If the resulted array is [1,0,0,.....,0], implying our target state is unentangled
    # If the resulted array is [0.7, 0.2, 0.02, ... 1e-8] , implying our targert state is an enatangled state

    # print(np.real(sysBlock_weight_sort[:min(len(sysBlock_weight_sort), MaximalStates) ]))

    # von Neumann entropy of sysBlock
    # locally update the dummy variable SE_local
    SE_local = -torch.sum(sysBlock_weight_sort * torch.log2(sysBlock_weight_sort))
    SE.append(np.real(SE_local))
    print("sysEntropy=", np.real(SE_local))

    envBlock_idx = torch.argsort(envBlock_weight, descending=True)
    envBlock_weight_sort = envBlock_weight[envBlock_idx]
    Ienv = envBlock_idx

    # Obtain the truncated basis( There is some bugs in the truncation)
    # sysBlock is a matrix contains eigenvector, but not an array
    sysBlock_rotationMatrix = sysBlock_rotationMatrix[
        :, Isys[: min(MaximalStates, len(sysBlock_rotationMatrix))]
    ]
    envBlock_rotationMatrix = envBlock_rotationMatrix[
        :, Ienv[: min(MaximalStates, len(envBlock_rotationMatrix))]
    ]

    # Step 5: Truncation:
    # sysBlock
    sysBlock_Ham = (
        sysBlock_rotationMatrix.conj().T @ sysBlock_Ham @ sysBlock_rotationMatrix
    )
    sysBlock_Sigma_x = (
        sysBlock_rotationMatrix.conj().T @ sysBlock_Sigma_x @ sysBlock_rotationMatrix
    )
    sysBlock_Sigma_z = (
        sysBlock_rotationMatrix.conj().T @ sysBlock_Sigma_z @ sysBlock_rotationMatrix
    )

    # envBlock
    envBlock_Ham = (
        envBlock_rotationMatrix.conj().T @ envBlock_Ham @ envBlock_rotationMatrix
    )
    envBlock_Sigma_x = (
        envBlock_rotationMatrix.conj().T @ envBlock_Sigma_x @ envBlock_rotationMatrix
    )
    envBlock_Sigma_z = (
        envBlock_rotationMatrix.conj().T @ envBlock_Sigma_z @ envBlock_rotationMatrix
    )

    print("Total length =", sysBlock_Length + envBlock_Length)


loop_array = np.arange(0, len(E_GS), 1)
plt.title(r"Convergence of Ground State Energy ")
plt.scatter(
    loop_array,
    np.array([E.cpu() for E in E_GS]),
    facecolors="none",
    edgecolors="b",
    label=r"DMRG GS",
)
plt.ylabel(r" Energy per Site")
plt.xlabel(r"Iterations")
plt.legend()

plt.show()
