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

    train_data = train_data_[:, ::4, ::4].reshape(-1, 49) / 256
    test_data = test_data_[:, ::4, ::4].reshape(-1, 49) / 256

    return train_data, train_targets, test_data, test_targets

def init_weights(dict_net):
    dict_net["Bias"] = dict()
    dict_net["weights"] = dict()

    for key in dict_net["layers"].keys():
        dict_net["weights"][key] = torch.randn(dict_net["layers"][key], requires_grad=True)
        dict_net["Bias"][key] = torch.randn(dict_net["layers"][key][0], requires_grad=True)

    return dict_net

def forward(x_row, dict_net):

    layer1_x_plus_w = x_row.repeat(dict_net["layers"]["layer1"][0], 1) + dict_net["weights"]["layer1"]
    joint_probability_layer1 = torch.prod(layer1_x_plus_w, dim=1)
    layer1_output = joint_probability_layer1 + dict_net["Bias"]["layer1"]
    layer1_relu_output = F.relu(layer1_output)

    layer2_x_plus_w = layer1_relu_output.repeat(dict_net["layers"]["layer2"][0], 1) + dict_net["weights"]["layer2"]
    joint_probability_layer2 = torch.prod(layer2_x_plus_w, dim=1)
    layer2_output = joint_probability_layer2 + dict_net["Bias"]["layer2"]

    return layer2_output

def update_weights(dict_net, LR):

    for layer_str in sorted(dict_net["weights"].keys()):
        w = dict_net["weights"][layer_str]
        dict_net["weights"][layer_str] = nn.Parameter(torch.Tensor(w.detach() - LR * w.grad.detach()))

        Bias = dict_net["Bias"][layer_str]
        dict_net["Bias"][layer_str] = nn.Parameter(torch.Tensor(Bias.detach() - LR * Bias.grad.detach()))

    return dict_net
