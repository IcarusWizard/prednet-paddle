import h5py
import random
import paddle
import numpy as np
from PIL import Image, ImageSequence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from prednet import PredNet

def setup_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def save_gif(imgs, path, duration=100):
    imgs = [Image.fromarray((imgs[i] * 255).astype(np.uint8)) for i in range(len(imgs))]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)

def load_gif(path):
    gif = Image.open(path)
    imgs = np.stack([np.array(f.convert('RGB')) for f in ImageSequence.Iterator(gif)])
    imgs = (imgs / 255.0).astype(np.float32)
    return imgs

def save_plot(X, X_hat, path):
    nt = X_hat.shape[0]
    aspect_ratio = float(X_hat.shape[1]) / X_hat.shape[2]
    plt.figure(figsize = (nt, 2 * aspect_ratio))
    gs = gridspec.GridSpec(2, nt)
    gs.update(wspace=0., hspace=0.)
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(X[t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Actual', fontsize=10)

        plt.subplot(gs[t + nt])
        plt.imshow(X_hat[t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted', fontsize=10)
    plt.savefig(path)
    plt.close()

def load_model_from_tensorflow_weight(weight_file):
    model = PredNet()
    with h5py.File(weight_file, 'r') as h5file:
        print(h5file.keys())
        prednet_weights = h5file['prednet_1']['prednet_1']
        print(prednet_weights.keys())
        
        for l in range(4):
            model.lstms[l].i.weight.set_value(paddle.to_tensor(prednet_weights[f'layer_i_{l}']['kernel:0'][:]).transpose((3, 2, 0, 1)))
            model.lstms[l].i.bias.set_value(paddle.to_tensor(prednet_weights[f'layer_i_{l}']['bias:0'][:]))
            model.lstms[l].f.weight.set_value(paddle.to_tensor(prednet_weights[f'layer_f_{l}']['kernel:0'][:]).transpose((3, 2, 0, 1)))
            model.lstms[l].f.bias.set_value(paddle.to_tensor(prednet_weights[f'layer_f_{l}']['bias:0'][:]))
            model.lstms[l].c.weight.set_value(paddle.to_tensor(prednet_weights[f'layer_c_{l}']['kernel:0'][:]).transpose((3, 2, 0, 1)))
            model.lstms[l].c.bias.set_value(paddle.to_tensor(prednet_weights[f'layer_c_{l}']['bias:0'][:]))
            model.lstms[l].o.weight.set_value(paddle.to_tensor(prednet_weights[f'layer_o_{l}']['kernel:0'][:]).transpose((3, 2, 0, 1)))
            model.lstms[l].o.bias.set_value(paddle.to_tensor(prednet_weights[f'layer_o_{l}']['bias:0'][:]))

            model.a_hat_layers[l][0].weight.set_value(paddle.to_tensor(prednet_weights[f'layer_ahat_{l}']['kernel:0'][:]).transpose((3, 2, 0, 1)))
            model.a_hat_layers[l][0].bias.set_value(paddle.to_tensor(prednet_weights[f'layer_ahat_{l}']['bias:0'][:]))

            if l > 0:
                model.a_layers[l][0].weight.set_value(paddle.to_tensor(prednet_weights[f'layer_a_{l-1}']['kernel:0'][:]).transpose((3, 2, 0, 1)))
                model.a_layers[l][0].bias.set_value(paddle.to_tensor(prednet_weights[f'layer_a_{l-1}']['bias:0'][:]))

    return model

if __name__ == '__main__':
    model = load_model_from_tensorflow_weight('model_data_keras2/tensorflow_weights/prednet_kitti_weights.hdf5')