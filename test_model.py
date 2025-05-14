import mynn as nn
import re
from struct import unpack
import numpy as np
import argparse

# set seed
np.random.seed(123)

# load data
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

test_set = [test_data, test_labs]

# load model
parser = argparse.ArgumentParser()

parser.add_argument('--model_path', '-p', type=str, default='./saved_models/MLP/best_model/models/best_model.pickle')
args = parser.parse_args()

model_path = args.model_path

if re.match('^./saved_models/MLP', model_path) is not None:
    model = nn.models.MLPModel(save_dir=model_path)
elif re.match('^./saved_models/CNN', model_path) is not None:
    model = nn.models.CNN(save_dir=model_path)
    img_len_size = int(np.sqrt(test_data.shape[-1]))
    test_data = test_data.reshape(-1, 1, img_len_size, img_len_size)
    test_set = [test_data, test_labs]
else:
    raise ValueError('model_path leads to neither MLP or CNN')

# test
loss_fn = nn.loss_fn.CrossEntropyLoss(model)
metric = nn.metric.accuracy
runner = nn.runner.Runner(model, loss_fn=loss_fn, metric=metric)

test_data, test_labels = test_set
print('[Test]Begin testing...')
loss, score = runner.eval(test_data, test_labels)
print('[Test]Testing completed!')
print(f'[Test]loss:{loss}, score:{score}')