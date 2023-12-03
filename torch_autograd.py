from time import time
import torch.nn as nn
from PRASAD import data_loader, forward, init_weights, update_weights

train_data, train_targets, test_data, test_targets = data_loader()

dict_net = dict()

dict_net["layers"] = dict()
dict_net["layers"]["layer2"] = (10, 256)  # (256, 10)
dict_net["layers"]["layer1"] = (256, 784)  # (784, 256)

dict_net["ReLU"] = dict()
dict_net["ReLU"]["layer2"] = False
dict_net["ReLU"]["layer1"] = True

dict_net = init_weights(dict_net)

n_epochs = 8; t0 = time()
for j in range(n_epochs):
    for i in range(train_targets.shape[0]):
        x = train_data[i, :]
        y = train_targets[i]

        ASAD_output = forward(x, dict_net)
        loss_criterion = nn.CrossEntropyLoss()
        loss = loss_criterion(ASAD_output, y)
        loss.backward()

        dict_net = update_weights(dict_net, LR=0.001)

        print(loss)

print(time() - t0, "seconds")
