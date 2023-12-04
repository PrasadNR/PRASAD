from time import time
import torch.nn as nn
from PRASAD import data_loader, forward, init_weights, update_weights

train_data, train_targets, test_data, test_targets = data_loader()

dict_net = dict()

dict_net["layers"] = dict()
dict_net["layers"]["layer2"] = (10, 16)  # (256, 10)
dict_net["layers"]["layer1"] = (16, 49)  # (784, 256)

dict_net = init_weights(dict_net)

n_epochs = 16; t0 = time(); loss = 0
for j in range(n_epochs):
    for i in range(train_targets.shape[0]):
        x = train_data[i, :]
        y = train_targets[i]

        ASAD_output = forward(x, dict_net)
        loss_criterion = nn.CrossEntropyLoss()
        loss = loss_criterion(ASAD_output, y)
        loss.backward()

        dict_net = update_weights(dict_net, LR=2 ** -16)

    print(j, float(loss))

print(time() - t0, "seconds")
