'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import time
import argparse
import numpy as np
import paddle
from paddle.io import DataLoader
import paddle.optimizer

from prednet import PredNet
from data import SequenceDataset, infinite_loading
from utils import setup_seed
from kitti_settings import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file', type=str, default=os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.pdparams'))
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--seed', type=str, default=1)
    parser.add_argument('--nt', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--samples_per_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--monitor_test', type=int, default=1)
    args = parser.parse_args()

    setup_seed(args.seed)

    weights_file = args.weight_file  # where weights will be saved
    if not os.path.exists(os.path.dirname(weights_file)): os.makedirs(os.path.dirname(weights_file))

    # Data files
    train_file = os.path.join(args.data_dir, 'train.h5')
    val_file = os.path.join(args.data_dir, 'val.h5')
    test_file = os.path.join(args.data_dir, 'test.h5')

    # Training parameters
    nb_epoch = args.epoch
    batch_size = args.batch_size
    samples_per_epoch = args.samples_per_epoch
    N_seq_val = 100  # number of sequences to use for validation
    steps_per_epoch = samples_per_epoch // batch_size

    # Model parameters
    n_channels, im_height, im_width = (3, 128, 160)
    input_shape = (n_channels, im_height, im_width)
    stack_sizes = (n_channels, 48, 96, 192)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3, 3, 3)
    Ahat_filt_sizes = (3, 3, 3, 3)
    R_filt_sizes = (3, 3, 3, 3)
    layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
    nt = args.nt  # number of timesteps used for sequences in training
    time_loss_weights = 1./ (nt - 1) * np.ones((nt))  # equally weight all timesteps except the first
    time_loss_weights[0] = 0

    layer_loss_weights = paddle.to_tensor(layer_loss_weights, dtype=paddle.float32)
    time_loss_weights = paddle.to_tensor(time_loss_weights, dtype=paddle.float32).reshape((nt, 1))

    model = PredNet(stack_sizes, R_stack_sizes,
                    A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                    output_mode='error')

    train_dataset = SequenceDataset(train_file, nt)
    val_dataset = SequenceDataset(val_file, nt, N_seq=N_seq_val)
    test_dataset = SequenceDataset(test_file, nt, sequence_start_mode='unique')
    train_loader = infinite_loading(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True))
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    lr_scheduler = paddle.optimizer.lr.StepDecay(args.lr, step_size=75, gamma=0.1)
    optim = paddle.optimizer.Adam(learning_rate=lr_scheduler, parameters=model.parameters())

    print('Start training!')
    start_time = time.time()
    best_val_loss = float('inf')
    for e in range(nb_epoch):
        print(f'Starting epoch {e} current lr is {lr_scheduler.get_lr()}')
        model.train()
        model.output_mode = 'error'

        step = 0
        losses = []
        while step < steps_per_epoch:
            step += 1
            batch_video = next(train_loader).transpose((1, 0, 2, 3, 4))
            errors = model(batch_video)
            loss = paddle.mean(paddle.sum(paddle.sum(errors * layer_loss_weights, axis=-1) * time_loss_weights, axis=0))
      
            optim.clear_grad()
            loss.backward()
            optim.step()
            losses.append(loss.numpy()[0])

            if step % 25 == 0:
                print(f'Epoch {e}, Step {step}, total used time {time.time() - start_time} (s), avg loss so far is {np.mean(losses)}')

        lr_scheduler.step()

        model.eval()
        model.output_mode = 'all'
        with paddle.no_grad():
            losses = []
            mses = []
            for batch_video in iter(val_loader):
                batch_video = batch_video.transpose((1, 0, 2, 3, 4))
                errors, predicted_video = model(batch_video)
                mse = paddle.mean((batch_video[1:] - predicted_video[1:]) ** 2)
                loss = paddle.mean(paddle.sum(paddle.sum(errors * layer_loss_weights, axis=-1) * time_loss_weights, axis=0))
                losses.append(loss.numpy()[0])
                mses.append(mse.numpy()[0])
            val_loss = np.mean(losses)
            val_mse = np.mean(mses)

        print(f'Epoch {e}, val loss is {val_loss}, val mse is {val_mse}.')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            paddle.save(model.state_dict(), weights_file)

        '''Note: the test data here is only use to monitor prograss, not used for selecting best model'''
        if args.monitor_test:
            model.output_mode = 'prediction'
            X_test = []
            X_hat = []
            with paddle.no_grad():
                for batch_x in iter(test_loader):
                    batch_x = batch_x.transpose((1, 0, 2, 3, 4))
                    X_test.append(batch_x)
                    X_hat.append(model(batch_x))
            X_test = paddle.concat(X_test, axis=1)
            X_hat = paddle.concat(X_hat, axis=1)
            assert X_test.shape[1] == len(test_dataset), f'{X_test.shape} {len(test_dataset)}'
            X_test = np.transpose(X_test.numpy(), (1, 0, 3, 4, 2))
            X_hat = np.transpose(X_hat.numpy(), (1, 0, 3, 4, 2))
            test_mse= np.mean((X_test[:, 1:] - X_hat[:, 1:]) ** 2)
            print(f'Epoch {e}, test mse is {test_mse}.')