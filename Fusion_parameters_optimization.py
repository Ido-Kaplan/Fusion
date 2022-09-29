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
    return 0.25*(x*(1-x) + y*(1-y)+(x+y)*(2-x-y))


def entanglement_success_chance_loss(parameters,max_chance_t11 = 0.25,max_chance_t10 = 0.375):
    return max_chance_t11 - return_entanglement_success_chance_t11(parameters) + max_chance_t10 - return_entanglement_success_chance_t10(parameters)


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



if __name__ == "__main__":
    lr = 10**(-4)
    num_of_epochs = 120000
    max_legal_loss_val = 10**(-4)
    step_size = num_of_epochs//4
    gamma = 0.1
    plot_loss_graph = True
    do_optimization_steps = False

    init_values=torch.tensor(unitary_group.rvs(4))
    parameters = torch.nn.Parameter(init_values)


    loss_func = lambda x:  10*unitary_loss(parameters)+entanglement_success_chance_loss(parameters)+entanglement_entropy_loss(parameters,get_t=get_t_11)+entanglement_entropy_loss(parameters,get_t=get_t_10)

    optimizer = torch.optim.Adam([parameters], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    print("parameters=",parameters)
    print("Initial entanglement t10:",float(get_entanglement_entropy(parameters,get_t=get_t_10).data))
    print("Initial entanglement t11:",float(get_entanglement_entropy(parameters,get_t=get_t_11).data))
    print("Initial chance t10:", float(return_entanglement_success_chance_t10(parameters).data),"Initial chance t11:",float(return_entanglement_success_chance_t11(parameters).data))
    loss_vec = []
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



    print("\n\nAFTER:\nparameters=",parameters)
    print("End result entanglement t10:",float(get_entanglement_entropy(parameters,get_t=get_t_10).data))
    print("End result entanglement t11:",float(get_entanglement_entropy(parameters,get_t=get_t_11).data))
    print("End result chance t10:", float(return_entanglement_success_chance_t10(parameters).data),"End result chance t11:",float(return_entanglement_success_chance_t11(parameters).data))
    print("End result t_10:",float(get_t_10(parameters).data))
    print("End result t_11:",float(get_t_11(parameters).data))
    print("x:",float(get_x(parameters).data),"y:",float(get_y(parameters).data))
    list_of_parameters = list(parameters.detach().numpy())
    if plot_loss_graph:
        plt.title("loss function")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(loss_vec)
        plt.show()
