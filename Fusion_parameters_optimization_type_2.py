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


def entanglement_success_chance_loss(parameters):
    return 1 - return_entanglement_success_chance_HH(parameters) - return_entanglement_success_chance_HV(parameters) \
             - return_entanglement_success_chance_VH(parameters) - return_entanglement_success_chance_VV(parameters) \
             - return_entanglement_success_chance_HV_0(parameters) - return_entanglement_success_chance_0_HV(parameters)


def get_t_HH(parameters):
    t_HH = torch.tensor(1.0)
    t_HH *= torch.abs(parameters[0,0]*parameters[2,2]+parameters[0,2]*parameters[2,0])**2 + \
         torch.abs(parameters[0, 0] * parameters[3, 2] + parameters[0, 2] * parameters[3, 0]) ** 2

    t_HH /=(t_HH+ torch.abs(parameters[1,0]*parameters[2,2]+parameters[1,2]*parameters[2,0])**2 +
         torch.abs(parameters[1, 0] * parameters[3, 2] + parameters[1, 2] * parameters[3, 1]) ** 2)

    return t_HH


def get_t_HV(parameters):
    t_HH = torch.tensor(1.0)
    t_HH *= torch.abs(parameters[0, 0] * parameters[2, 3] + parameters[0, 3] * parameters[2, 0]) ** 2 + \
            torch.abs(parameters[0, 0] * parameters[3, 3] + parameters[0, 3] * parameters[3, 0]) ** 2

    t_HH /= (t_HH + torch.abs(parameters[1, 0] * parameters[2, 3] + parameters[1, 3] * parameters[2, 0]) ** 2 +
             torch.abs(parameters[1, 0] * parameters[3, 3] + parameters[1, 3] * parameters[3, 0]) ** 2)

    return t_HH


def get_t_VH(parameters):
    t_HH = torch.tensor(1.0)
    t_HH *= torch.abs(parameters[0,1] * parameters[2, 2] + parameters[0, 2] * parameters[2, 1]) ** 2 + \
            torch.abs(parameters[0, 1] * parameters[3, 2] + parameters[0, 2] * parameters[3, 1]) ** 2

    t_HH /= (t_HH + torch.abs(parameters[1, 1] * parameters[2, 2] + parameters[1, 2] * parameters[2, 1]) ** 2 +
             torch.abs(parameters[1, 1] * parameters[3, 2] + parameters[1, 2] * parameters[3, 1]) ** 2)

    return t_HH


def get_t_VV(parameters):
    t_HH = torch.tensor(1.0)
    t_HH *= torch.abs(parameters[0,1] * parameters[2, 3] + parameters[0, 3] * parameters[2, 1]) ** 2 + \
            torch.abs(parameters[0, 1] * parameters[3, 3] + parameters[0, 3] * parameters[3, 1]) ** 2

    t_HH /= (t_HH + torch.abs(parameters[1, 1] * parameters[2, 3] + parameters[1, 3] * parameters[2, 1]) ** 2 +
             torch.abs(parameters[1, 1] * parameters[3, 3] + parameters[1, 3] * parameters[3, 1]) ** 2)

    return t_HH

def get_t_HV_0(parameters):
    t_HH = torch.tensor(1.0)
    t_HH *= torch.abs(parameters[0,0] * parameters[2, 1] + parameters[0, 1] * parameters[2, 0]) ** 2 + \
            torch.abs(parameters[0, 0] * parameters[3, 1] + parameters[0, 1] * parameters[3, 0]) ** 2

    t_HH /= (t_HH + torch.abs(parameters[1, 0] * parameters[2, 1] + parameters[1, 1] * parameters[2, 0]) ** 2 +
             torch.abs(parameters[1, 0] * parameters[3, 1] + parameters[1, 1] * parameters[3, 0]) ** 2)

    return t_HH

def get_t_0_HV(parameters):
    t_HH = torch.tensor(1.0)
    t_HH *= torch.abs(parameters[0,2] * parameters[2, 3] + parameters[0, 3] * parameters[2, 2]) ** 2 + \
            torch.abs(parameters[0, 2] * parameters[3, 3] + parameters[0, 3] * parameters[3, 2]) ** 2

    t_HH /= (t_HH + torch.abs(parameters[1, 2] * parameters[2, 3] + parameters[1, 3] * parameters[2, 2]) ** 2 +
             torch.abs(parameters[1, 2] * parameters[3, 3] + parameters[1, 3] * parameters[3, 2]) ** 2)

    return t_HH


def get_entanglement_entropy(parameters,get_t = get_t_HH):
    t = get_t(parameters)
    return -t*torch.log(t) - (1.0-t)*torch.log(1.0-t)

def entanglement_entropy_loss(parameters,max_entropy = 0.6931471805599453,get_t = get_t_HH):
    return max_entropy - get_entanglement_entropy(parameters,get_t)

def get_all_entanglement_entropy_loss(parameters):
    return entanglement_entropy_loss(parameters,get_t=get_t_HH)+entanglement_entropy_loss(parameters,get_t=get_t_HV)+ \
          +entanglement_entropy_loss(parameters,get_t=get_t_VH)+entanglement_entropy_loss(parameters,get_t=get_t_VV)+ \
          +entanglement_entropy_loss(parameters,get_t=get_t_HV_0)+entanglement_entropy_loss(parameters,get_t=get_t_0_HV)


def unitary_loss(parameters):
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

def print_results(parameters):
    HH_chance = float(return_entanglement_success_chance_HH(parameters).data)
    HV_chance = float(return_entanglement_success_chance_HV(parameters).data)
    VH_chance = float(return_entanglement_success_chance_VH(parameters).data)
    VV_chance = float(return_entanglement_success_chance_VV(parameters).data)
    HV_0_chance = float(return_entanglement_success_chance_HV_0(parameters).data)
    _0_HV_chance = float(return_entanglement_success_chance_0_HV(parameters).data)

    tHH_entanglement = float(get_entanglement_entropy(parameters,get_t=get_t_HH).data)
    tHV_entanglement = float(get_entanglement_entropy(parameters,get_t=get_t_HV).data)
    tVH_entanglement = float(get_entanglement_entropy(parameters,get_t=get_t_VH).data)
    tVV_entanglement = float(get_entanglement_entropy(parameters,get_t=get_t_VV).data)
    tHV_0_entanglement = float(get_entanglement_entropy(parameters,get_t=get_t_HV_0).data)
    t_0_HV_entanglement = float(get_entanglement_entropy(parameters,get_t=get_t_0_HV).data)

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

if __name__ == "__main__":
    lr = 10**(-3)
    num_of_epochs = 360000
    max_legal_loss_val = 10**(-4)
    step_size = num_of_epochs//4
    gamma = 0.1
    plot_loss_graph = True
    do_optimization_steps = False

    init_values=torch.tensor(unitary_group.rvs(4))
    parameters = torch.nn.Parameter(init_values)

    loss_func = lambda x:  20*unitary_loss(parameters)+entanglement_success_chance_loss(parameters)+get_all_entanglement_entropy_loss(parameters)

    optimizer = torch.optim.Adam([parameters], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

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

        if step%1000==0:
            cur_params = np.array(parameters.data)
            print("step number:",step,"current loss:",loss.item())
            if step%5000==0:
                print("current results")
                print_results(parameters)



    parameters = torch.tensor(modifiedGramSchmidt(np.array(parameters.data)))
    print("Results in the end of the run:")
    print_results(parameters)

    list_of_parameters = list(parameters.detach().numpy())
    if plot_loss_graph:
        plt.title("loss function")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(loss_vec)
        plt.show()
