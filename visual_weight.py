import argparse
import pickle
import matplotlib.pyplot as plt

def visual(W, name):
    plt.imshow(W*10, cmap='seismic', vmin=-0.5, vmax=0.5)
    plt.title(name)
    plt.axis('off')
    plt.show()

def visual_kernel(W, name):
    in_channels, out_channels, kH, kW = W.shape
    fg, axis = plt.subplots(in_channels, out_channels, figsize=(12, 8))
    axis = axis.reshape(in_channels, out_channels)
    for in_channel in range(in_channels):
        for out_channel in range(out_channels):
            ax = axis[in_channel, out_channel]
            ax.imshow(W[in_channel, out_channel, :, :])
            ax.axis('off')
    plt.suptitle(name)
    plt.show()


# load model
parser = argparse.ArgumentParser()

parser.add_argument('--model_path', '-p', type=str, default='./saved_models/MLP/best_model/models/best_model.pickle')
args = parser.parse_args()

model_path = args.model_path

with open(model_path, 'rb') as f:
    param_list = pickle.load(f, encoding='bytes')

idx = 0
for layer in param_list:
    if isinstance(layer, dict):
        idx += 1
        name = f'Weight of the {idx} Layer'
        if layer['W'].ndim == 2:
            visual(layer['W'], name)
        elif layer['W'].ndim == 4:
            visual_kernel(layer['W'], name)