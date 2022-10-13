import numpy as np
import torch
import os
from scipy.stats import unitary_group
from matplotlib import pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_x(parameters):
    return torch.abs(parameters[0,2])**2 + torch.abs(parameters[1,2])**2

def get_y(parameters):
    return torch.abs(parameters[0,3])**2 + torch.abs(parameters[1,3])**2

def return_entanglement_success_chance_t11(parameters):
    x=get_x(parameters)
    y=get_y(parameters)
    return 0.25*(x*(1-y) + y*(1-x))

def return_entanglement_success_chance_t10(parameters):
    x=get_x(parameters)
    y=get_y(parameters)
    return 0.25*(x*(x+y) + (1-x)*(2-x-y))

def return_entanglement_success_chance_t01(parameters):
    x=get_x(parameters)
    y=get_y(parameters)
    return 0.25*(y*(x+y) + (1-y)*(2-x-y))



def entanglement_success_chance_loss(parameters,max_chance_t11 = 0.25,max_chance_t10 = 0.375,max_chance_t01 = 0.375):
    # return 0.5 - return_entanglement_success_chance_t11(parameters) - \
    #              return_entanglement_success_chance_t10(parameters) - \
    #              return_entanglement_success_chance_t01(parameters)

    return max_chance_t11 - return_entanglement_success_chance_t11(parameters) + \
           max_chance_t10 - return_entanglement_success_chance_t10(parameters) + \
           max_chance_t01 - return_entanglement_success_chance_t01(parameters)


def get_t_01(parameters):
    t_01 = torch.tensor(1.0)
    t_01 *= torch.abs(parameters[0,0]*parameters[2,3]+parameters[0,3]*parameters[2,0])**2 + \
         torch.abs(parameters[0, 1] * parameters[2, 3] + parameters[0, 3] * parameters[2, 1]) ** 2 + \
            torch.abs(parameters[0, 0] * parameters[3, 3] + parameters[0, 3] * parameters[3, 0]) ** 2 + \
            torch.abs(parameters[0, 1] * parameters[3, 3] + parameters[0, 3] * parameters[3, 1]) ** 2

    t_01 /=(t_01+ torch.abs(parameters[1,0]*parameters[2,3]+parameters[1,3]*parameters[2,0])**2 +
         torch.abs(parameters[1, 1] * parameters[2, 3] + parameters[1, 3] * parameters[2, 1]) ** 2 +
            torch.abs(parameters[1, 0] * parameters[3, 3] + parameters[1, 3] * parameters[3, 0]) ** 2 +
            torch.abs(parameters[1, 1] * parameters[3, 3] + parameters[1, 3] * parameters[3, 1]) ** 2)
    return t_01



def get_t_10(parameters):
    t_10 = torch.tensor(1.0)
    t_10 *= torch.abs(parameters[0,2]*parameters[2,0]+parameters[0,0]*parameters[2,2])**2 + \
         torch.abs(parameters[0, 2] * parameters[2, 1] + parameters[0, 1] * parameters[2, 2]) ** 2 + \
            torch.abs(parameters[0, 2] * parameters[3, 0] + parameters[0, 0] * parameters[3, 2]) ** 2 + \
            torch.abs(parameters[0, 2] * parameters[3, 1] + parameters[0, 1] * parameters[3, 2]) ** 2

    t_10 /=(t_10+ torch.abs(parameters[1,2]*parameters[2,0]+parameters[1,0]*parameters[2,2])**2 +
         torch.abs(parameters[1, 2] * parameters[2, 1] + parameters[2, 2] * parameters[1, 1]) ** 2 +
            torch.abs(parameters[1, 2] * parameters[3, 0] + parameters[1, 0] * parameters[3, 2]) ** 2 +
            torch.abs(parameters[1, 2] * parameters[3, 1] + parameters[1, 1] * parameters[3, 2]) ** 2)
    return t_10


def get_t_11(parameters):
    t_11 = torch.tensor(1.0)
    t_11 *= torch.abs(parameters[0,2]*parameters[2,3]+parameters[0,3]*parameters[2,2])**2 + \
         torch.abs(parameters[0, 2] * parameters[3, 3] + parameters[0, 3] * parameters[3, 2]) ** 2
    t_11 /=(t_11+ torch.abs(parameters[1,2]*parameters[2,3]+parameters[1,3]*parameters[2,2])**2 +
         torch.abs(parameters[1, 2] * parameters[3, 3] + parameters[1, 3] * parameters[3, 2]) ** 2)
    return t_11

def get_entanglement_entropy(parameters,get_t = get_t_11):
    t = get_t(parameters)
    return -t*torch.log(t) - (1.0-t)*torch.log(1.0-t)

def entanglement_entropy_loss(parameters,max_entropy = 0.6931471805599453,get_t = get_t_11):
    return max_entropy - get_entanglement_entropy(parameters,get_t)

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
    t01_chance = float(return_entanglement_success_chance_t01(parameters).data)
    t10_chance = float(return_entanglement_success_chance_t10(parameters).data)
    t11_chance = float(return_entanglement_success_chance_t11(parameters).data)

    t01_entanglement = float(get_entanglement_entropy(parameters,get_t=get_t_01).data)
    t10_entanglement = float(get_entanglement_entropy(parameters,get_t=get_t_10).data)
    t11_entanglement = float(get_entanglement_entropy(parameters,get_t=get_t_11).data)


    print("\n\nAFTER:\nparameters=",parameters)
    print("Chance for entanglement:",t01_chance+t10_chance+t11_chance)
    print("Average entanglement:", (t01_entanglement*t01_chance + t10_entanglement*t10_chance + t11_entanglement*t11_chance)/(0.6931471805599453*(t01_chance+t10_chance+t11_chance)),"\n##########################\n")

    print("End result entanglement t10:",t10_entanglement)
    print("End result entanglement t11:",t11_entanglement)
    print("End result entanglement t01:",t01_entanglement)
    print("End result chance t10:",t10_chance)
    print("End result chance t11:",t11_chance)
    print("End result chance t01:",t01_chance)
    print("End result t_10:",float(get_t_10(parameters).data))
    print("End result t_11:",float(get_t_11(parameters).data))
    print("x:",float(get_x(parameters).data),"y:",float(get_y(parameters).data))
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

    loss_func = lambda x:  20*unitary_loss(parameters)+get_x(parameters)+get_y(parameters)#entanglement_success_chance_loss(parameters)+entanglement_entropy_loss(parameters,get_t=get_t_11)+entanglement_entropy_loss(parameters,get_t=get_t_10)+entanglement_entropy_loss(parameters,get_t=get_t_01)

    optimizer = torch.optim.Adam([parameters], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    print("parameters=",parameters)
    print("Initial entanglement t10:",float(get_entanglement_entropy(parameters,get_t=get_t_10).data))
    print("Initial entanglement t11:",float(get_entanglement_entropy(parameters,get_t=get_t_11).data))
    print("Initial chance t10:", float(return_entanglement_success_chance_t10(parameters).data),"Initial chance t11:",float(return_entanglement_success_chance_t11(parameters).data))
    loss_vec = []

    best_params = torch.clone(parameters.data)
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
                print("current parameters:",parameters)



    parameters = torch.tensor(modifiedGramSchmidt(np.array(parameters.data)))
    print_results(parameters)
    # print_results(best_params)

    list_of_parameters = list(parameters.detach().numpy())
    if plot_loss_graph:
        plt.title("loss function")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(loss_vec)
        plt.show()
