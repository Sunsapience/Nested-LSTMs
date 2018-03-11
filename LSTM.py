import tensorflow as tf
from tensorflow.contrib import rnn

class LSTMCell(rnn.RNNCell):
    """ 
    This is a basic LSTMCell.
    """

    def __init__(self,
                 n_hidden,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=None,
                 use_bias = True,
                 reuse=None,
                 name=None):
        """Initialize the basic NLSTM cell.

        Args:
            n_hidden:
                int. The number of hidden units of each cell state 
                and hidden state. 
            forget_bias:
                float. The bias added to forget gates.
            state_is_tuple=True:
                If `True`, accepted and returned states are tuples of
                the `h_state` and `c_state`s.  If `False`, they are concatenated
                along the column axis.  The latter behavior will soon be deprecated.
            Activation: function of the update values,
                including new inputs and new cell states.  Default: `tanh`
            use_bias:
                bool. Default: `True`.
            reuse:
                bool(optional). Python boolean describing whether to reuse variables
                in an existing scope.  If not `True`, and the existing scope already has
                the given variables, an error is raised.
            name:
                str`, the name of the layer. Layers with the same name will  share weights,
                but to avoid mistakes we require reuse=True in such cases.
        """

        super(LSTMCell, self).__init__(_reuse=reuse, name=name)
        if not state_is_tuple:
            tf.logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self) 
        
        self._n_hidden = n_hidden
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or tf.tanh
        self._use_bias = use_bias
        self._kernels = None  # weight
        self.built = False
    
    @property
    def state_size(self):
        if self._state_is_tuple:
            return tuple([self._n_hidden] * 2)   #(n_hidden,n_hidden)
            # return tuple([self._n_hidden,self._n_hidden])
        else:
            return self._n_hidden *2
    
    @property
    def output_size(self):
        return self._n_hidden
    
    def build(self, inputs_shape):
        if inputs_shape[1].value is None:   #inputs_shape[0]=batch_size, input_shape[1]=input_size
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                              % inputs_shape)
        
        input_size = inputs_shape[1].value
        n_hidden = self._n_hidden
        self._kernels = []  #weight

        if self._use_bias:
            self._biases = []

        input_kernel = self.add_variable(
            "input_gate_kernel",
            shape=[input_size, 4 * self._n_hidden],   # 4 gates input weight matrices(x--->w)
            initializer=tf.glorot_normal_initializer())
        hidden_kernel = self.add_variable(
            "hidden_gate_kernel",
            shape=[n_hidden, 4 * self._n_hidden],     # 4 gates hidden weight matrices(h_t--->w)
            initializer=tf.orthogonal_initializer())
        
        kernel = tf.concat([input_kernel, hidden_kernel],  
                           axis=0, name="kernel_0")
        self._kernels.append(kernel)  # the shape kernel is [input_size+n_hidden,4 * self._n_hidden]

        if self._use_bias:
            self._biases.append(
                self.add_variable(
                    "bias",
                    shape=[4 * self._n_hidden],
                    initializer=tf.zeros_initializer(dtype=self.dtype)))
    
        self.built = True

    def call(self,inputs,states):
        """forward propagation of the cell
        Args:
            inputs:
                A 2D `Tensor`. shape is [batch_size, input_size].
            states:
                a `tuple` of 2D `Tensor`,include three tensor,every shape is [batch_size, n_hidden].
        Returns:
            outputs:
                A 2D `Tensor`. shape is [batch_size, n_hidden].
            next_state:
                A `list` of 2D `Tensor`. including [state_h,state_c],shape is [batch_size, n_hidden].
        """
        sigmoid = tf.sigmoid

        h, c = states
        
        # the shape gate_inputs is  [batch_size,4 * self._n_hidden]
        gate_inputs = tf.matmul(
            tf.concat([inputs, h], 1), self._kernels[0])
        
        if self._use_bias:
            gate_inputs = tf.nn.bias_add(gate_inputs, self._biases[0])

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        # the shape of 4 gate is [bath_size,n_hidden]
        i, j, f, o = tf.split(
            value=gate_inputs, num_or_size_splits=4, axis=1)
        
        new_c = c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j)
        new_h = sigmoid(o) * self._activation(new_c)

        new_state = tuple([new_h,new_c])
        return new_h,new_state
        
        
                   