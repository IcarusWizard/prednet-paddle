'''
Demo for make prediction from a gif file.
'''

import os
import argparse
import numpy as np

import paddle
from prednet import PredNet
from kitti_settings import *
from utils import load_model_from_tensorflow_weight, save_gif, load_gif

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file', type=str, default=os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.pdparams'))
    parser.add_argument('--input_gif', type=str, default='gif/demo.gif')
    parser.add_argument('--output_gif', type=str, default='gif/demo_output.gif')
    parser.add_argument('--nt', type=int, default=None, help='length of the output gif, default is the same as input.')
    args = parser.parse_args()

    weights_file = args.weight_file

    # Load trained model
    if weights_file.endswith('.hdf5'): # the pretrained weight published by the auther
        test_model = load_model_from_tensorflow_weight(weights_file)
    else:
        test_model = PredNet()
        test_model.load_dict(paddle.load(weights_file))
    test_model.output_mode = 'prediction'

    X = load_gif(args.input_gif)
    with paddle.no_grad():
        batch_x = paddle.to_tensor(X[:, np.newaxis]).transpose((0, 1, 4, 2, 3))
        batch_x_hat = test_model(batch_x, args.nt)
        X_hat = batch_x_hat[:, 0].numpy().transpose((0, 2, 3, 1))
    
    save_gif(X_hat, args.output_gif)