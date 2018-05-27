# This is an example of training a 5-layers network with 50 units in each hidden layer.
# You need to adjust the following things:
#   - the number of layers
#   - the number of hidden units
#   - learning rate
#   - lambda_reg
#   - weight scale
#   - batch_size
#   - num_epoch
#   - ......
#  Revise this example for training your model.
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from data_utils import get_CIFAR10_data
from fcnet import FullyConnectedNet
from solver import Solver

# get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load data
data = get_CIFAR10_data()

# set up parameters for a model
h1, h2, h3, h4 = 100, 100, 100, 100
hidden_dims = [h1, h2, h3, h4]
# hidden_dims = [h1]
input_dims = 3*32*32
# h1 = 50
# hidden_dims = [h1]
num_classes = 10
lambda_reg = 0.0
# weight_scale = 1e-5
# weight_scale = 1e-2
weight_scale = 2.461858e-02

# model
model = FullyConnectedNet(hidden_dims=hidden_dims,
                          input_dims=input_dims,
                          num_classes=num_classes,
                          lambda_reg=lambda_reg,
                          weight_scale=weight_scale,
                          dtype=np.float64)

# set up parameters for training
update_rule='sgd'
learning_rate = 3.113669e-04
# learning_rate = 1e-3
batch_size=25
num_epochs=20
print_every=10

# solver
solver = Solver(model,
                data,
                update_rule='sgd',
                batch_size=25,
                num_epochs=20,
                print_every=1000,
                optim_config={
                  'learning_rate': learning_rate,
                })

# train
solver.train()

# plot
plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(solver.loss_history, 'o')
plt.xlabel('Iteration')
plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()
