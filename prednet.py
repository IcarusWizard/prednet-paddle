import paddle
import paddle.nn as nn

class HardSigmoid(nn.Layer):
    def forward(self, x):
        x = (0.2 * x) + 0.5
        x = paddle.clip(x, min=0, max=1)
        return x

activations = {
    'hard_sigmoid': HardSigmoid,
    'tanh' : nn.Tanh,
    'relu' : nn.ReLU
}

class ConvLSTM(nn.Layer):
    def __init__(self, input_dim=3, output_dim=32, kernel_size=3, inner_activation='hard_sigmoid', output_activation='tanh'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.padding_size = kernel_size // 2

        self.i = nn.Conv2D(self.input_dim + self.output_dim, self.output_dim, self.kernel_size, padding=self.padding_size, weight_attr=nn.initializer.XavierUniform())
        self.f = nn.Conv2D(self.input_dim + self.output_dim, self.output_dim, self.kernel_size, padding=self.padding_size, weight_attr=nn.initializer.XavierUniform())
        self.c = nn.Conv2D(self.input_dim + self.output_dim, self.output_dim, self.kernel_size, padding=self.padding_size, weight_attr=nn.initializer.XavierUniform())
        self.o = nn.Conv2D(self.input_dim + self.output_dim, self.output_dim, self.kernel_size, padding=self.padding_size, weight_attr=nn.initializer.XavierUniform())

        self.sigmoid = activations[inner_activation]()
        self.tanh = activations[output_activation]()

    def reset(self, size):
        B, H, W = size
        h = paddle.zeros((B, self.output_dim, H, W))
        c = paddle.zeros((B, self.output_dim, H, W))

        return h, c

    def forward(self, x, states):
        h, c = states
        x = paddle.concat([h, x], axis=1)
        c = c * self.sigmoid(self.f(x)) + self.sigmoid(self.i(x)) * self.tanh(self.c(x))
        h = self.sigmoid(self.o(x)) * self.tanh(c)
        return h, c

class SATLU(nn.Layer):
    def __init__(self, max_value):
        super().__init__()
        self.max_value = max_value
    
    def forward(self, x):
        return paddle.clip(x, max=self.max_value)

class PredNet(nn.Layer):
    '''PredNet architecture - Lotter 2016.
        Stacked convolutional LSTM inspired by predictive coding principles.

    # Arguments
        stack_sizes: number of channels in targets (A) and predictions (Ahat) in each layer of the architecture.
            Length is the number of layers in the architecture.
            First element is the number of channels in the input.
            Ex. (3, 16, 32) would correspond to a 3 layer architecture that takes in RGB images and has 16 and 32
                channels in the second and third layers, respectively.
        R_stack_sizes: number of channels in the representation (R) modules.
            Length must equal length of stack_sizes, but the number of channels per layer can be different.
        A_filt_sizes: filter sizes for the target (A) modules.
            Has length of 1 - len(stack_sizes).
            Ex. (3, 3) would mean that targets for layers 2 and 3 are computed by a 3x3 convolution of the errors (E)
                from the layer below (followed by max-pooling)
        Ahat_filt_sizes: filter sizes for the prediction (Ahat) modules.
            Has length equal to length of stack_sizes.
            Ex. (3, 3, 3) would mean that the predictions for each layer are computed by a 3x3 convolution of the
                representation (R) modules at each layer.
        R_filt_sizes: filter sizes for the representation (R) modules.
            Has length equal to length of stack_sizes.
            Corresponds to the filter sizes for all convolutions in the LSTM.
        pixel_max: the maximum pixel value.
            Used to clip the pixel-layer prediction.
        error_activation: activation function for the error (E) units.
        A_activation: activation function for the target (A) and prediction (A_hat) units.
        LSTM_activation: activation function for the cell and hidden states of the LSTM.
        LSTM_inner_activation: activation function for the gates in the LSTM.
        output_mode: either 'error', 'prediction', 'all' or layer specification (ex. R2, see below).
            Controls what is outputted by the PredNet.
            If 'error', the mean response of the error (E) units of each layer will be outputted.
                That is, the output shape will be (batch_size, nb_layers).
            If 'prediction', the frame prediction will be outputted.
            If 'all', the output will be the frame prediction concatenated with the mean layer errors.
                The frame prediction is flattened before concatenation.
                Nomenclature of 'all' is kept for backwards compatibility, but should not be confused with returning all of the layers of the model

    # References
        - [Deep predictive coding networks for video prediction and unsupervised learning](https://arxiv.org/abs/1605.08104)
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
        - [Convolutional LSTM network: a machine learning approach for precipitation nowcasting](http://arxiv.org/abs/1506.04214)
        - [Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects](http://www.nature.com/neuro/journal/v2/n1/pdf/nn0199_79.pdf)
    '''
    def __init__(self, 
                 stack_sizes=(3, 48, 96, 192), 
                 R_stack_sizes=(3, 48, 96, 192),
                 A_filt_sizes=(3, 3, 3), 
                 Ahat_filt_sizes=(3, 3, 3, 3), 
                 R_filt_sizes=(3, 3, 3, 3),
                 pixel_max=1., 
                 error_activation='relu', 
                 A_activation='relu',
                 LSTM_activation='tanh', 
                 LSTM_inner_activation='hard_sigmoid',
                 output_mode='error'):

        super().__init__()
        self.stack_sizes = stack_sizes
        self.nb_layers = len(stack_sizes)
        assert len(R_stack_sizes) == self.nb_layers, 'len(R_stack_sizes) must equal len(stack_sizes)'
        self.R_stack_sizes = R_stack_sizes
        assert len(A_filt_sizes) == (self.nb_layers - 1), 'len(A_filt_sizes) must equal len(stack_sizes) - 1'
        self.A_filt_sizes = A_filt_sizes
        assert len(Ahat_filt_sizes) == self.nb_layers, 'len(Ahat_filt_sizes) must equal len(stack_sizes)'
        self.Ahat_filt_sizes = Ahat_filt_sizes
        assert len(R_filt_sizes) == (self.nb_layers), 'len(R_filt_sizes) must equal len(stack_sizes)'
        self.R_filt_sizes = R_filt_sizes

        self.pixel_max = pixel_max
        self.error_activation = activations[error_activation]()
        self.up_sampling = nn.Upsample(scale_factor=2)

        default_output_modes = ['prediction', 'error', 'all']
        assert output_mode in default_output_modes, 'Invalid output_mode: ' + str(output_mode)
        self.output_mode = output_mode

        '''define the networks'''
        
        lstms = []
        a_layers = []
        a_hat_layers = []
        for l in range(self.nb_layers):
            input_dim = 2 * self.stack_sizes[l] + self.R_stack_sizes[l+1] if l < self.nb_layers - 1 else 2 * self.stack_sizes[l]
            lstms.append(
                ConvLSTM(
                    input_dim, 
                    self.R_stack_sizes[l], 
                    self.R_filt_sizes[l], 
                    LSTM_inner_activation, 
                    LSTM_activation
                )
            )

            if l == 0:
                a_hat_layers.append(nn.Sequential(
                    nn.Conv2D(self.R_stack_sizes[l], self.stack_sizes[l], kernel_size=self.Ahat_filt_sizes[l], padding=self.Ahat_filt_sizes[l]//2, weight_attr=nn.initializer.XavierUniform()),
                    activations[A_activation](),
                    SATLU(self.pixel_max)
                ))        

                a_layers.append(nn.Identity())
            else:
                a_hat_layers.append(nn.Sequential(
                    nn.Conv2D(self.R_stack_sizes[l], self.stack_sizes[l], kernel_size=self.Ahat_filt_sizes[l], padding=self.Ahat_filt_sizes[l]//2, weight_attr=nn.initializer.XavierUniform()),
                    activations[A_activation]()
                ))
            
                a_layers.append(nn.Sequential(
                    nn.Conv2D(self.stack_sizes[l-1] * 2, self.stack_sizes[l], kernel_size=self.A_filt_sizes[l-1], padding=self.A_filt_sizes[l-1]//2, weight_attr=nn.initializer.XavierUniform()),
                    activations[A_activation](),
                    nn.MaxPool2D(kernel_size=2)
                ))

        self.lstms = nn.LayerList(lstms)
        self.a_hat_layers = nn.LayerList(a_hat_layers)
        self.a_layers = nn.LayerList(a_layers)

    def reset(self, size):
        B, H, W = size
        lstm_states = [(self.lstms[l].reset((B, H//2**l, W//2**l))) for l in range(self.nb_layers)]
        errors = [paddle.zeros((B, 2 * self.stack_sizes[l], H//2**l, W//2**l)) for l in range(self.nb_layers)]
        frame = paddle.zeros((B, 3, H, W))
        return lstm_states, errors, frame

    def forward(self, sequence, nt=None):
        T, B, C, H, W = sequence.shape
        assert self.stack_sizes[0] == C
        if nt is None: nt = T

        states = self.reset((B, H, W))
        outputs = []
        for t in range(nt):
            if t < T:
                output, states = self.step(sequence[t], states)
            else:
                output, states = self.step(states[-1], states)
            outputs.append(output)

        if self.output_mode == 'all':
            outputs = (paddle.stack([o[0] for o in outputs]), paddle.stack([o[1] for o in outputs]))
        else:
            outputs = paddle.stack(outputs)

        return outputs

    def step(self, x, states):
        '''follow the pseudocode in Sec.2'''
        lstm_states, errors, _ = states

        for l in reversed(range(self.nb_layers)):
            if l < self.nb_layers - 1:
                lstm_input = paddle.concat([errors[l], self.up_sampling(lstm_states[l+1][0])], axis=1)
            else:
                lstm_input = errors[l]
            lstm_states[l] = self.lstms[l](lstm_input, lstm_states[l])

        a = x
        for l in range(self.nb_layers):
            a = self.a_layers[l](a)
            a_hat = self.a_hat_layers[l](lstm_states[l][0])
            if l == 0: frame_prediction = a_hat
            diff = a_hat - a
            errors[l] = paddle.concat([self.error_activation(diff), self.error_activation(-diff)], axis=1)
            a = errors[l]

        if self.output_mode == 'error':
            output = paddle.stack([paddle.mean(e.reshape((e.shape[0], -1)), axis=1) for e in errors], axis=-1)
        elif self.output_mode == 'prediction':
            output = frame_prediction
        elif self.output_mode == 'all':
            output = (paddle.stack([paddle.mean(e.reshape((e.shape[0], -1)), axis=1) for e in errors], axis=-1), frame_prediction)

        return output, (lstm_states, errors, frame_prediction)

if __name__ == '__main__':
    model = PredNet()
    demo_videos = paddle.rand((10, 2, 3, 128, 160))
    output = model(demo_videos)
    print(output[0][0])
