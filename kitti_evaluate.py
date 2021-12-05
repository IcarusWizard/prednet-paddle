'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import paddle
from paddle.io import DataLoader
from prednet import PredNet
from data import SequenceDataset
from kitti_settings import *
from tqdm import tqdm
from utils import load_model_from_tensorflow_weight

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file', type=str, default=os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.pdparams'))
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--nt', type=int, default=10)
    parser.add_argument('--n_plot', type=int, default=40)
    args = parser.parse_args()

    n_plot = args.n_plot
    batch_size = args.batch_size
    nt = args.nt

    weights_file = args.weight_file
    test_file = os.path.join(args.data_dir, 'test.h5')

    # Load trained model
    if weights_file.endswith('.hdf5'): # the pretrained weight published by the auther
        test_model = load_model_from_tensorflow_weight(weights_file)
    else:
        test_model = PredNet()
        test_model.load_dict(paddle.load(weights_file))
    test_model.output_mode = 'prediction'

    test_dataset = SequenceDataset(test_file, nt, sequence_start_mode='unique')
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    X_test = []
    X_hat = []
    with paddle.no_grad():
        for batch_x in tqdm(iter(test_loader)):
            batch_x = batch_x.transpose((1, 0, 2, 3, 4))
            X_test.append(batch_x)
            X_hat.append(test_model(batch_x))
    X_test = paddle.concat(X_test, axis=1)
    X_hat = paddle.concat(X_hat, axis=1)
    assert X_test.shape[1] == len(test_dataset), f'{X_test.shape} {len(test_dataset)}'
    X_test = np.transpose(X_test.numpy(), (1, 0, 3, 4, 2))
    X_hat = np.transpose(X_hat.numpy(), (1, 0, 3, 4, 2))

    # Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
    mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
    mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
    if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
    f = open(RESULTS_SAVE_DIR + 'prediction_scores.txt', 'w')
    f.write("Model MSE: %f\n" % mse_model)
    f.write("Previous Frame MSE: %f" % mse_prev)
    f.close()

    # Plot some predictions
    aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
    plt.figure(figsize = (nt, 2*aspect_ratio))
    gs = gridspec.GridSpec(2, nt)
    gs.update(wspace=0., hspace=0.)
    plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
    if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
    plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
    for i in plot_idx:
        for t in range(nt):
            plt.subplot(gs[t])
            plt.imshow(X_test[i,t], interpolation='none')
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
            if t==0: plt.ylabel('Actual', fontsize=10)

            plt.subplot(gs[t + nt])
            plt.imshow(X_hat[i,t], interpolation='none')
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
            if t==0: plt.ylabel('Predicted', fontsize=10)

        plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
        plt.clf()
