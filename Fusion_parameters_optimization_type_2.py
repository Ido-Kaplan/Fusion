import numpy as np
import torch
import os
from scipy.stats import unitary_group
from matplotlib import pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_m1(parameters):
    return torch.abs(parameters[0,0])**2 + torch.abs(parameters[1,0])**2

def get_m2(parameters):
    return torch.abs(parameters[0,1])**2 + torch.abs(parameters[1,1])**2

def get_m3(parameters):
    return torch.abs(parameters[0,2])**2 + torch.abs(parameters[1,2])**2

def get_m4(parameters):
    return torch.abs(parameters[0,3])**2 + torch.abs(parameters[1,3])**2


def return_entanglement_success_chance_HH(parameters):
    m1=get_m1(parameters)
    m3=get_m3(parameters)
    return 0.25*(m1*(1-m3) + m3*(1-m1))

def return_entanglement_success_chance_HV(parameters):
    m1=get_m1(parameters)
    m4=get_m4(parameters)
    return 0.25*(m1*(1-m4) + m4*(1-m1))

def return_entanglement_success_chance_VH(parameters):
    m2=get_m2(parameters)
    m3=get_m3(parameters)
    return 0.25*(m2*(1-m3) + m3*(1-m2))

def return_entanglement_success_chance_VV(parameters):
    m2=get_m2(parameters)
    m4=get_m4(parameters)
    return 0.25*(m2*(1-m4) + m4*(1-m2))

def return_entanglement_success_chance_HV_0(parameters):
    m1=get_m1(parameters)
    m2=get_m2(parameters)
    return 0.25*(m1*(1-m2) + m2*(1-m1))

def return_entanglement_success_chance_0_HV(parameters):
    m3=get_m1(parameters)
    m4=get_m2(parameters)
    return 0.25*(m3*(1-m4) + m4*(1-m3))


def loss_m1m2_are_1_m3m4_are_0(parameters):
    m1=get_m1(parameters)
    m2=get_m2(parameters)
    m3=get_m3(parameters)
    m4=get_m4(parameters)
    return m3+m4+(1-m1)**2+(1-m2)**2


def entanglement_success_chance_loss(parameters):
    return 1 - return_entanglement_success_chance_HH(parameters) - return_entanglement_success_chance_HV(parameters) \
             - return_entanglement_success_chance_VH(parameters) - return_entanglement_success_chance_VV(parameters) \
             - return_entanglement_success_chance_HV_0(parameters) - return_entanglement_success_chance_0_HV(parameters)

def get_mat_log(mat):
    mat_log = torch.tensor([[0,0],[0,0]])
    mat_log = mat_log.type(torch.complex64)
    for k in range(1,15):
        mat_log+=((-1)**(k+1))*torch.matrix_power(mat-torch.eye(2),k)/k

    return mat_log

def get_entanglement_entropy(parameters,Cv,Ch,Dv,Dh):
    part_1a = lambda Ch1, Cv1, Dh1, Dv1: (parameters[0,0] * Ch1 + parameters[0, 1] * Cv1 + parameters[0, 2] * Dh1 + parameters[0, 3] * Dv1)
    part_1b = lambda Ch1, Cv1, Dh1, Dv1: (parameters[1,0] * Ch1 + parameters[1, 1] * Cv1 + parameters[1, 2] * Dh1 + parameters[1, 3] * Dv1)
    part_2a = lambda Ch2, Cv2, Dh2, Dv2: (parameters[2,0] * Ch2 + parameters[2, 1] * Cv2 + parameters[2, 2] * Dh2 + parameters[2, 3] * Dv2)
    part_2b = lambda Ch2, Cv2, Dh2, Dv2: (parameters[3,0] * Ch2 + parameters[3, 1] * Cv2 + parameters[3, 2] * Dh2 + parameters[3, 3] * Dv2)
    if Cv==1:
        a = part_1a(0,1,0,0)*part_2a(Ch,0,Dh,Dv)+part_1a(Ch,0,Dh,Dv)*part_2a(0,1,0,0)
        b = part_1a(0,1,0,0)*part_2b(Ch,0,Dh,Dv)+part_1a(Ch,0,Dh,Dv)*part_2b(0,1,0,0)
        c = part_1b(0,1,0,0)*part_2a(Ch,0,Dh,Dv)+part_1b(Ch,0,Dh,Dv)*part_2a(0,1,0,0)
        d = part_1b(0,1,0,0)*part_2b(Ch,0,Dh,Dv)+part_1b(Ch,0,Dh,Dv)*part_2b(0,1,0,0)
    elif Ch==1:
        a = part_1a(1,0,0,0)*part_2a(0,Cv,Dh,Dv)+part_1a(0,Cv,Dh,Dv)*part_2a(1,0,0,0)
        b = part_1a(1,0,0,0)*part_2b(0,Cv,Dh,Dv)+part_1a(0,Cv,Dh,Dv)*part_2b(1,0,0,0)
        c = part_1b(1,0,0,0)*part_2a(0,Cv,Dh,Dv)+part_1b(0,Cv,Dh,Dv)*part_2a(1,0,0,0)
        d = part_1b(1,0,0,0)*part_2b(0,Cv,Dh,Dv)+part_1b(0,Cv,Dh,Dv)*part_2b(1,0,0,0)
    elif Dv==1:
        a = part_1a(0, 0, 0, 1) * part_2a(Ch, Cv, Dh, 0) + part_1a(Ch, Cv, Dh, 0) * part_2a(0, 0, 0, 1)
        b = part_1a(0, 0, 0, 1) * part_2b(Ch, Cv, Dh, 0) + part_1a(Ch, Cv, Dh, 0) * part_2b(0, 0, 0, 1)
        c = part_1b(0, 0, 0, 1) * part_2a(Ch, Cv, Dh, 0) + part_1b(Ch, Cv, Dh, 0) * part_2a(0, 0, 0, 1)
        d = part_1b(0, 0, 0, 1) * part_2b(Ch, Cv, Dh, 0) + part_1b(Ch, Cv, Dh, 0) * part_2b(0, 0, 0, 1)
    else:
        a = part_1a(0, 0, 1, 0) * part_2a(Ch, Cv, 0, Dv) + part_1a(Ch, Cv, 0, Dv) * part_2a(0, 0, 1, 0)
        b = part_1a(0, 0, 1, 0) * part_2b(Ch, Cv, 0, Dv) + part_1a(Ch, Cv, 0, Dv) * part_2b(0, 0, 1, 0)
        c = part_1b(0, 0, 1, 0) * part_2a(Ch, Cv, 0, Dv) + part_1b(Ch, Cv, 0, Dv) * part_2a(0, 0, 1, 0)
        d = part_1b(0, 0, 1, 0) * part_2b(Ch, Cv, 0, Dv) + part_1b(Ch, Cv, 0, Dv) * part_2b(0, 0, 1, 0)



    return torch.abs((a*d-b*c)/(torch.abs(a)**2+torch.abs(b)**2+torch.abs(c)**2+torch.abs(d)**2))**2

    # N_power_2 = torch.abs(a)**2 + torch.abs(b)**2 + torch.abs(c)**2 + torch.abs(d)**2
    # N_power_2 = float(N_power_2.data)

    # rho = torch.tensor([[1, 0], [0, 0]]) * (torch.abs(a) ** 2 + torch.abs(b) ** 2) + \
    #       torch.tensor([[0, 1], [0, 0]]) * (torch.conj(c) * a + torch.conj(d) * b) + \
    #       torch.tensor([[0, 0], [1, 0]]) * (torch.conj(a) * c + torch.conj(b) * d) + \
    #       torch.tensor([[0, 0], [0, 1]]) * (torch.abs(c) ** 2 + torch.abs(d) ** 2)
    #
    # rho = rho/N_power_2
    # rho = rho.type(torch.complex64)
    # entropy = -torch.trace(rho @ get_mat_log(rho))
    # if torch.abs(torch.det(rho-torch.eye(2)))<1:
    #     entropy = -torch.trace(rho@get_mat_log(rho))
    #
    # else:
    #     print("The diff between matrices isn't smaller than 1!")
    #     # if the matrix is based on a pure state:
    #     eigenvalues_of_rho = torch.abs(torch.linalg.eigvals(rho))
    #     entropy = torch.tensor(0.0)
    #     for eigenvalue in eigenvalues_of_rho:
    #         entropy -= eigenvalue * torch.log(eigenvalue)
    return entropy

def entanglement_entropy_loss(parameters,max_entropy = 0.6931471805599453,Ch=0,Cv=0,Dh=0,Dv=0):
    return max_entropy - get_entanglement_entropy(parameters,Ch=Ch,Cv=Cv,Dh=Dh,Dv=Dv)

def get_all_entanglement_entropy_loss(params):
    parameters = turn_vals_into_mat(params)
    return entanglement_entropy_loss(parameters,Ch=1,Cv=1,Dh=0,Dv=0)+entanglement_entropy_loss(parameters,Ch=0,Cv=0,Dh=1,Dv=1)+ \
          +entanglement_entropy_loss(parameters,Ch=1,Cv=0,Dh=1,Dv=0)+entanglement_entropy_loss(parameters,Ch=1,Cv=0,Dh=0,Dv=1)+ \
          +entanglement_entropy_loss(parameters,Ch=0,Cv=1,Dh=1,Dv=0)+entanglement_entropy_loss(parameters,Ch=0,Cv=1,Dh=0,Dv=1)


def unitary_loss(params):
    parameters = turn_vals_into_mat(params)
    return torch.sum(torch.abs(parameters@(torch.conj(parameters).T) - torch.eye(4)))

def modifiedGramSchmidt(A):
    """
    Gives a orthonormal matrix, using modified Gram Schmidt Procedure
    :param A: a matrix of column vectors
    :return: a matrix of orthonormal column vectors
    """
    # assuming A is a square matrix
    dim = A.shape[0]
    Q = np.zeros(A.shape, dtype=A.dtype)
    for j in range(0, dim):
        q = A[:,j]
        for i in range(0, j):
            rij = np.vdot(Q[:,i], q)
            q = q - rij*Q[:,i]
        rjj = np.linalg.norm(q, ord=2)
        if np.isclose(rjj,0.0):
            raise ValueError("invalid input matrix")
        else:
            Q[:,j] = q/rjj
    return Q



def return_vals(parameters):
    HH_chance = float(return_entanglement_success_chance_HH(parameters).data)
    HV_chance = float(return_entanglement_success_chance_HV(parameters).data)
    VH_chance = float(return_entanglement_success_chance_VH(parameters).data)
    VV_chance = float(return_entanglement_success_chance_VV(parameters).data)
    HV_0_chance = float(return_entanglement_success_chance_HV_0(parameters).data)
    _0_HV_chance = float(return_entanglement_success_chance_0_HV(parameters).data)

    tHH_entanglement = get_entanglement_entropy(parameters,Ch=1,Cv=0,Dh=1,Dv=0).data
    tHV_entanglement = get_entanglement_entropy(parameters,Ch=1,Cv=0,Dh=0,Dv=1).data
    tVH_entanglement = get_entanglement_entropy(parameters,Ch=0,Cv=1,Dh=1,Dv=0).data
    tVV_entanglement = get_entanglement_entropy(parameters,Ch=0,Cv=1,Dh=0,Dv=1).data
    tHV_0_entanglement = get_entanglement_entropy(parameters,Ch=1,Cv=1,Dh=0,Dv=0).data
    t_0_HV_entanglement = get_entanglement_entropy(parameters,Ch=0,Cv=0,Dh=1,Dv=1).data

    total_chance = HH_chance+HV_chance+VH_chance + VV_chance + HV_0_chance + _0_HV_chance
    return parameters, total_chance, (tHH_entanglement*HH_chance + tHV_entanglement*HV_chance + tVH_entanglement*VH_chance +
    tVV_entanglement*VV_chance + tHV_0_entanglement*HV_0_chance + t_0_HV_entanglement*_0_HV_chance)/(0.6931471805599453*total_chance)


def print_results(params):
    parameters = turn_vals_into_mat(params)
    HH_chance = float(return_entanglement_success_chance_HH(parameters).data)
    HV_chance = float(return_entanglement_success_chance_HV(parameters).data)
    VH_chance = float(return_entanglement_success_chance_VH(parameters).data)
    VV_chance = float(return_entanglement_success_chance_VV(parameters).data)
    HV_0_chance = float(return_entanglement_success_chance_HV_0(parameters).data)
    _0_HV_chance = float(return_entanglement_success_chance_0_HV(parameters).data)

    tHH_entanglement = get_entanglement_entropy(parameters,Ch=1,Cv=0,Dh=1,Dv=0).data
    tHV_entanglement = get_entanglement_entropy(parameters,Ch=1,Cv=0,Dh=0,Dv=1).data
    tVH_entanglement = get_entanglement_entropy(parameters,Ch=0,Cv=1,Dh=1,Dv=0).data
    tVV_entanglement = get_entanglement_entropy(parameters,Ch=0,Cv=1,Dh=0,Dv=1).data
    tHV_0_entanglement = get_entanglement_entropy(parameters,Ch=1,Cv=1,Dh=0,Dv=0).data
    t_0_HV_entanglement = get_entanglement_entropy(parameters,Ch=0,Cv=0,Dh=1,Dv=1).data

    total_chance = HH_chance+HV_chance+VH_chance + VV_chance + HV_0_chance + _0_HV_chance


    print("parameters=",parameters)
    print("Chance for entanglement:",total_chance)
    print("Average entanglement:", (tHH_entanglement*HH_chance + tHV_entanglement*HV_chance + tVH_entanglement*VH_chance +
    tVV_entanglement*VV_chance + tHV_0_entanglement*HV_0_chance + t_0_HV_entanglement*_0_HV_chance)/(0.6931471805599453*total_chance),"\n##########################\n")

    print("End result entanglement t H,H:",tHH_entanglement)
    print("End result entanglement t H,V:",tHV_entanglement)
    print("End result entanglement t V,H:",tVH_entanglement)
    print("End result entanglement t V,V:",tVV_entanglement)
    print("End result entanglement t HV,0:",tHV_0_entanglement)
    print("End result entanglement t 0,HV:",t_0_HV_entanglement)

    print("End result chance t H,H:",HH_chance)
    print("End result chance t H,V:",HV_chance)
    print("End result chance t V,H:",VH_chance)
    print("End result chance t V,V:",VV_chance)
    print("End result chance t HV,0:",HV_0_chance)
    print("End result chance t 0,HV:",_0_HV_chance)

    print("m1:",float(get_m1(parameters).data),"m2:",float(get_m2(parameters).data),"m3:",float(get_m3(parameters).data),"m4:",float(get_m4(parameters).data))
    return


def return_unitary_mat_2_x_2(theta,phi1,phi2):
    u = torch.eye(2)*torch.cos(theta)+torch.tensor([[0,1],[-1,0]])*torch.sin(theta)+0j
    u[0,0]*=torch.exp(1j*phi1)
    u[1,1]*=torch.exp(-1j*phi1)
    u[0,1]*=torch.exp(1j*phi2)
    u[1,0]*=torch.exp(-1j*phi2)
    return u

def turn_vals_into_mat(vals):
    return vals
    # u1 = return_unitary_mat_2_x_2(vals[0],vals[1],vals[2])
    # u2 = return_unitary_mat_2_x_2(vals[3],vals[4],vals[5])
    #
    #
    #
    # return torch.kron(u1,u2)


def run_opt(lamda=1,plot_loss_graph=False,print_vals=True):
    lr = 10**(-3)
    num_of_epochs = 200000
    max_legal_loss_val = 10**(-4)
    step_size = num_of_epochs//4
    gamma = 0.1
    do_optimization_steps = False


    # everything will be real:
    init_values = modifiedGramSchmidt(np.abs(unitary_group.rvs(4)))
    init_values = torch.tensor(init_values)
    #
    parameters = torch.nn.Parameter(init_values)
    # parameters = torch.nn.Parameter(torch.rand(6))


    loss_func = lambda x:  unitary_loss(parameters)+get_all_entanglement_entropy_loss(parameters)+lamda*loss_m1m2_are_1_m3m4_are_0(parameters)
#+lamda*entanglement_success_chance_loss(parameters)+get_all_entanglement_entropy_loss(parameters)

    optimizer = torch.optim.Adam([parameters], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if print_vals:
        print("Initial state:")
        print_results(parameters)
    loss_vec = []

    best_loss = loss_func(parameters)
    best_loss = abs(best_loss.item())

    for step in range(num_of_epochs):
        loss = torch.tensor(0*1j)


        loss+=loss_func(parameters)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_vec += [ abs(loss.item()) ]


        if do_optimization_steps:
            scheduler.step()

        if step%1000==0 and print_vals:
            cur_params = np.array(parameters.data)
            print("step number:",step,"current loss:",loss.item())
            if step%10000==0:
                print("current results")
                print_results(parameters)

        parameters.data = torch.nan_to_num(parameters.data)
        parameters.grad = torch.nan_to_num(parameters.grad)



    parameters = torch.tensor(modifiedGramSchmidt(np.array(parameters.data)))
    if print_vals:
        print("Results in the end of the run:")
        print_results(parameters)

    list_of_parameters = list(parameters.detach().numpy())
    if plot_loss_graph:
        plt.title("loss function")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(loss_vec)
        plt.show()


    return return_vals(parameters)

if __name__ == "__main__":
    vals = []
    lamda = 1
    vals.append(run_opt(lamda=lamda,print_vals=True))

    print(vals)
