import torch
import torch.nn as nn

x = torch.ones(4)
w = torch.randn(4, requires_grad=True)

LR = 0.1; loss = 0
print(x, w)

for i in range(100):
    print(loss)

    output = torch.prod(x + w)
    loss = output ** 2
    loss.backward()

    w = nn.Parameter(torch.Tensor(w.detach() - LR * w.grad.detach()))

print(w)
