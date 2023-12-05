import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(*load_digits(return_X_y=True))
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

N_TRAIN_ROWS = y_train.shape[0]; LR = 0.001; dict_net = dict()
dict_net["layer2"] = np.random.uniform(-1, 1, size=(32, 10))
dict_net["layer1"] = np.random.uniform(-1, 1, size=(64, 32))

X_train_3d = np.repeat(X_train[:, :, None], dict_net["layer1"].shape[1], axis=2)
layer1_weights_3d = np.repeat(dict_net["layer1"][None, :, :], N_TRAIN_ROWS, axis=0)


print(X_train_3d.shape, layer1_weights_3d.shape)
layer1_output = X_train_3d + layer1_weights_3d
print(layer1_output)
