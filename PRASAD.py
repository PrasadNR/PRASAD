import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST

def data_loader():

    train_MNIST = MNIST('./data/MNIST', train=True, download=True, transform=transforms.ToTensor())
    test_MNIST = MNIST('./data/MNIST', train=False, download=True, transform=transforms.ToTensor())

    train_data_, train_targets = train_MNIST.data, train_MNIST.targets
    test_data_, test_targets = test_MNIST.data, test_MNIST.targets

    train_data = train_data_.reshape(-1, 784)
    test_data = test_data_.reshape(-1, 784)

    return train_data, train_targets, test_data, test_targets

def init_weights(dict_net):
    dict_net["Bias"] = dict()
    dict_net["weights"] = dict()

    for key in dict_net["layers"].keys():
        dict_net["weights"][key] = torch.randn(dict_net["layers"][key], requires_grad=True)
        dict_net["Bias"][key] = torch.randn(dict_net["layers"][key][0], requires_grad=True)

    return dict_net

def forward(x_row, dict_net):

    layer_output = x_row

    for layer_str in sorted(dict_net["layers"].keys()):
        layer_shape = dict_net["layers"][layer_str]

        x_plus_w = layer_output.repeat(layer_shape[0], 1) + dict_net["weights"][layer_str]
        joint_probability = torch.prod(x_plus_w, dim=1)
        layer_output = joint_probability + dict_net["Bias"][layer_str]

        if dict_net["ReLU"][layer_str]: layer_output = F.relu(layer_output)

    return layer_output

def update_weights(dict_net, LR):

    for layer_str in sorted(dict_net["weights"].keys()):
        w = dict_net["weights"][layer_str]
        dict_net["weights"][layer_str] = nn.Parameter(torch.Tensor(w.detach() - LR * w.grad.detach()))

    return dict_net
