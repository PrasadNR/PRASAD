from time import time
import torch.nn as nn
from PRASAD import data_loader, forward, init_weights, update_weights, torch_eval

train_data, train_targets, test_data, test_targets = data_loader()

dict_net = dict()

dict_net["layers"] = dict()
dict_net["layers"]["layer2"] = (10, 256)  # (256, 10)
dict_net["layers"]["layer1"] = (256, 784)  # (784, 256)

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

    test_predict, test_accuracy = torch_eval(dict_net, test_data, test_targets)
    print(j, float(loss), test_accuracy)

print(time() - t0, "seconds")
