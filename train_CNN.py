import json
import os
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import argparse
from struct import unpack

import mynn as nn

# set seed
np.random.seed(123)

# load train and valid set
train_images_path = r'.\dataset\MNIST\train-images.idx3-ubyte'
train_labels_path = r'.\dataset\MNIST\train-labels.idx1-ubyte'

with open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize
train_data = train_imgs / train_imgs.max()
valid_data = valid_imgs / valid_imgs.max()

# reshape
train_data = train_data.reshape(-1, 1, rows, cols)   # '1' stands for the channel
valid_data = valid_data.reshape(-1, 1, rows, cols)

valid_set = [valid_data, valid_labs]
train_set = [train_data, train_labs]

# load test set
test_images_path = r'.\dataset\MNIST\t10k-images.idx3-ubyte'
test_labels_path = r'.\dataset\MNIST\t10k-labels.idx1-ubyte'

with open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

# normalize
test_data = test_imgs / test_imgs.max()

# reshape
test_data = test_data.reshape(-1, 1, rows, cols)

test_set = [test_data, test_labs]

# get parameter from terminal
parser = argparse.ArgumentParser()
parser.add_argument('--channel_list', '-ch', type=int, nargs='+', default=[6, 16], help='Channel list of CNN, e.g. "6 16"')
parser.add_argument('--kernel_list', '-ks', type=int, nargs='+', default=[5, 5], help='Kernel size of CNN. Only support square kernel. e.g. "5 5"')
parser.add_argument('--hidden_size_list', '-hs', type=int, nargs='+', default=[128, 64], help='Kernel size of FC layers, e.g. 128 64')
parser.add_argument('--act_func', '-a', type=str, default='LeakyReLU', help='Activation function')
parser.add_argument('--weight_decay_param', '-wd', type=float, default=1e-3, help='parameter of weight decay')
parser.add_argument('--linear_weight_decay_param', '-lwd', type=float, default=1e-3, help='weight decay parameter of linear layer')
parser.add_argument('--init_lr', '-lr', type=float, default=1e-2, help='initialized learning rate')
parser.add_argument('--step_size', '-s', type=int, default=5, help='learning rate decay period')
parser.add_argument('--gamma', '-g', type=float, default=0.1, help='parameter of learning rate decay')
parser.add_argument('--batch_size','-bs', type=int, default=32)
parser.add_argument('--epoch', '-e', type=int, default=5)
parser.add_argument('--log_iter', '-l', type=int, default=100, help='period of print loss and accuracy')
parser.add_argument('--early_stop', '-es', type=bool, default=False, help='early stopping')
parser.add_argument('--patience', '-p', type=int, default=2, help='patience for early stopping')

args = parser.parse_args()

# init model
channel_list = [1] + args.channel_list
kernel_list = [(i, i) for i in args.kernel_list]
hidden_size_list = args.hidden_size_list
act_func = args.act_func
weight_decay_param = args.weight_decay_param
linear_weight_decay_param = args.linear_weight_decay_param
init_lr = args.init_lr
step_size = args.step_size
gamma = args.gamma
batch_size = args.batch_size
early_stop = args.early_stop
patience = args.patience

# calculate the first linear size
initial_size = 28
for i in range(len(kernel_list)):
    initial_size = initial_size - kernel_list[i][0] + 1
    initial_size /= 2
initial_size = int(initial_size ** 2 * channel_list[-1])

print(f'initial_size:{initial_size}')

linear_size_list = [initial_size] + hidden_size_list + [10]
weight_decay_list = [weight_decay_param for _ in range(len(kernel_list))]
linear_weight_decay_list = [linear_weight_decay_param for _ in range(len(linear_size_list) - 1)]

model = nn.models.CNN(channel_list=channel_list, kernel_list=kernel_list, linear_size_list=linear_size_list, act_func=act_func, weight_decay_list=weight_decay_list, linear_weight_decay_list=linear_weight_decay_list)
model.print_struct()
# optimizer = nn.optimizers.SGD(model, init_lr=init_lr)
optimizer = nn.optimizers.SGDMomentum(model, init_lr=init_lr)
lr_scheduler = nn.lr_schedulers.StepLR(optimizer, step_size=step_size, gamma=gamma)
metric = nn.metric.accuracy
loss_fn = nn.loss_fn.CrossEntropyLoss(model)
runner = nn.runner.Runner(model, loss_fn, metric, batch_size=batch_size, optimizer=optimizer, lr_scheduler=lr_scheduler, early_stop=early_stop, patience=patience)

# train
epoch = args.epoch
save_dir = f'./saved_models/CNN/{time.strftime('%Y-%m-%d-%H-%M', time.localtime())}'
log_iter = args.log_iter
print('[Train]Begin training...')
start_time = time.time()
runner.train(train_set, valid_set, epoch, save_dir, log_iter)
training_time = time.time() - start_time
print(f'[Train]Training completed! Total time:{training_time}')

# test
test_data, test_labels = test_set
print('[Test]Begin testing...')
loss, score = runner.eval(test_data, test_labels)
print('[Test]Testing completed!')
print(f'[Test]loss:{loss}, score:{score}')

# save result
result_path = os.path.join(save_dir, 'testing_result.json')
result = {'loss': loss, 'score': score, 'training_time': training_time}
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f)

# plot
fg, axis = plt.subplots(2, 2)
axis[0][0].plot(runner.train_loss, label='train_loss', color='r')
axis[0][0].set_xlabel('iteration')
axis[0][0].set_ylabel('loss')
axis[0][0].set_title('Train Loss')

axis[0][1].plot(runner.train_score, label='train_score', color='r')
axis[0][1].set_xlabel('iteration')
axis[0][1].set_ylabel('score')
axis[0][1].set_title('Train Score')

axis[1][0].plot(runner.valid_loss, label='valid_loss', color='b')
axis[1][0].set_xlabel('iteration')
axis[1][0].set_ylabel('loss')
axis[1][0].set_title('Valid Loss')

axis[1][1].plot(runner.valid_loss, label='valid_score', color='b')
axis[1][1].set_xlabel('iteration')
axis[1][1].set_ylabel('score')
axis[1][1].set_title('Valid Score')

fg.tight_layout()
plt.savefig(os.path.join(save_dir, 'loss_accuracy_plot.png'))

# save_params
params = {'model':{
              'type': 'CNN',
              'channel_list': channel_list,
              'kernel_list': kernel_list,
              'linear_size_list': linear_size_list,
              'act_func': act_func,
              'weight_decay_list': weight_decay_list,
              'linear_weight_decay_list': linear_weight_decay_list
          },
          'optimizer':{
              'type': 'Momentum',
              'init_lr': init_lr
          },
          'lr_scheduler':{
              'type': 'stepLR',
              'step_size': step_size,
              'gamma': gamma
          },
          'metric':{
              'type': 'accuracy'
          },
          'loss function':{
              'type': 'cross entropy'
          },
          'train':{
              'batch_size': batch_size,
              'epoch': epoch,
              'log_iter': log_iter
          }
}

json_path = os.path.join(save_dir, 'params.json')

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(params, f)