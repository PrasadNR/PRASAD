import torch
import torch.nn as nn

x1 = torch.ones(1)
x2 = torch.ones(1)
x3 = torch.ones(1)
x4 = torch.ones(1)

w1 = torch.randn(1, requires_grad=True)
w2 = torch.randn(1, requires_grad=True)
w3 = torch.randn(1, requires_grad=True)
w4 = torch.randn(1, requires_grad=True)

LR = 0.001; loss = 0

for i in range(100):
    print(loss)

    output = (x1 + w1) * (x2 + w2) * (x3 + w3) * (x4 + w4)
    loss = output ** 2
    loss.backward()

    w1 = nn.Parameter(torch.Tensor(w1.detach() - LR * w1.grad.detach()))
    w2 = nn.Parameter(torch.Tensor(w2.detach() - LR * w2.grad.detach()))
    w3 = nn.Parameter(torch.Tensor(w3.detach() - LR * w3.grad.detach()))
    w4 = nn.Parameter(torch.Tensor(w4.detach() - LR * w4.grad.detach()))

