import numpy as np
import pandas as pd

train = pd.read_csv('../data/mnist_train.csv', sep=',')
test = pd.read_csv('../data/mnist_test.csv', sep=',')

Y_train = train['label']
Y_test = test['label']

train.drop('label', axis=1, inplace=True)
test.drop('label', axis=1, inplace=True)


Y_train = Y_train.apply(lambda x: np.eye(10)[x].astype(int))
Y_test  = Y_test.apply(lambda x: np.eye(10)[x].astype(int))

with open("../data/y_train.csv", "w+") as f:
    for a in Y_train:
        for i in range(len(a)):
            f.write('{}{}'.format(a[i], ' ' if i != len(a) - 1 else ''))
        f.write('\n')

with open("../data/y_test.csv", "w+") as f:
    for a in Y_test:
        for i in range(len(a)):
            f.write('{}{}'.format(a[i], ' ' if i != len(a) - 1 else ''))
        f.write('\n')

train.to_csv('../data/x_train.csv', header=False, index=False, sep=' ')
test.to_csv('../data/x_test.csv', header=False, index=False, sep=' ')
