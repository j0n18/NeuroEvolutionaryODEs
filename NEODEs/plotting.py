import matplotlib.pyplot as plt
import torch

def plot_3D_trajectories(valid_data, node_data, model_name = "Vanilla Neural ODE"):

    #import pdb; pdb.set_trace();

    n_trials, n_timesteps, dim = valid_data.size()

    if torch.is_tensor(valid_data) or torch.is_tensor(node_data):
        valid_data = valid_data.detach().numpy()
        node_data = node_data.detach().numpy()
    

    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')

    for i in range(n_trials): #n_trials):
        x_valid = valid_data[i]
        x_node = node_data[i]

        ax.plot3D(x_valid[:,0],x_valid[:,1],x_valid[:,2],'b--', alpha = 0.3)
        ax.plot3D(x_valid[0:1,0],x_valid[0:1,1],x_valid[0:1,2],'bo')

        ax.plot3D(x_node[:,0],x_node[:,1],x_node[:,2],'r--')
        ax.plot3D(x_node[0:1,0],x_node[0:1,1],x_node[0:1,2],'ro')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.set_title(f'Trajectories for Lorenz system and {model_name} for multiple initial conditions')

    plt.show()