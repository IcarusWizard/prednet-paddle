
import h5py
import numpy as np
from paddle.io import Dataset

# Data generator that creates sequences for input into PredNet.
class SequenceDataset(Dataset):
    def __init__(self, data_file, nt, sequence_start_mode='all', N_seq=None):
        
        super().__init__()
        with h5py.File(data_file, 'r') as h5file:
            self.X = h5file['images'][:]  # X will be like (n_images, nb_cols, nb_rows, nb_channels)
            self.sources = h5file['sources'][:] # source for each image so when creating sequences can assure that consecutive frames are from same video
        self.nt = nt # length of each sequence clip

        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode

        self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.im_shape = self.X[0].shape

        if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
        elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
            curr_location = 0
            possible_starts = []
            while curr_location < self.X.shape[0] - self.nt + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                    possible_starts.append(curr_location)
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts

        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]

        self.N_sequences = len(self.possible_starts)

    def __len__(self):
        return self.N_sequences

    def __getitem__(self, index):
        index = self.possible_starts[index]
        return self.preprocess(self.X[index:index+self.nt])

    def preprocess(self, X):
        return X.astype(np.float32) / 255

def infinite_loading(dataloader):
    while True:
        for data in iter(dataloader):
            yield data