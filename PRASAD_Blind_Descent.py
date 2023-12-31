import numpy as np
from time import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

t0 = time()
X_train, X_test, y_train, y_test = train_test_split(*load_digits(return_X_y=True))
X_train = (X_train - 128) / 128; X_test = (X_test - 128) / 128
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

N_TRAIN_ROWS = y_train.shape[0]; LR = 0.001; n_epochs = 100
for i in range(n_epochs):
    dict_net = dict()
    dict_net["layer2"] = np.random.uniform(-1, 1, size=(32, 10))
    dict_net["layer1"] = np.random.uniform(-1, 1, size=(64, 32))

    X_train_3d = np.repeat(X_train[:, :, None], dict_net["layer1"].shape[1], axis=2)
    layer1_weights_3d = np.repeat(dict_net["layer1"][None, :, :], N_TRAIN_ROWS, axis=0)
    layer1_output = X_train_3d + layer1_weights_3d
    layer1_prod_output = np.prod(layer1_output, axis=1)
    layer1_relu_output = np.copy(layer1_prod_output)
    layer1_relu_output[layer1_relu_output < 0] = 0

    layer2_input_3d = np.repeat(layer1_relu_output[:, :, None], dict_net["layer2"].shape[1], axis=2)
    layer2_weights_3d = np.repeat(dict_net["layer2"][None, :, :], N_TRAIN_ROWS, axis=0)
    layer2_output = layer2_input_3d + layer2_weights_3d
    layer2_prod_output = np.prod(layer2_output, axis=1)

    np_argmax = np.argmax(layer2_prod_output, axis=1)
    print(np.sum(np_argmax == y_train) / y_train.shape[0])

print(time() - t0, "seconds")
