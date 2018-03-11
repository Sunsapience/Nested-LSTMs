import tensorflow as tf

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class NLSTMCell(tf.contrib.rnn.RNNCell):
    """
    The implementation is based on:
        https://arxiv.org/abs/1801.10308
        JRA. Moniz, D. Krueger.
        "Nested LSTMs"
        ACML, PMLR 77:530-544, 2017
    """

    def __init__(self,
                 n_hidden,
                 depth,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=None,
                 gate_activation=None,
                 cell_activation=None,
                 initializer=None,
                 input_gate_initializer=None,
                 use_bias=True,
                 reuse=None,
                 name=None):
        """Initialize the basic NLSTM cell.

        Args:
            n_hidden:
                int. The number of hidden units of each cell state 
                and hidden state.
            depth: 
                int, The number of layers in the nest.
            forget_bias:
                float. The bias added to forget gates.
            state_is_tuple:
                If `True`, accepted and returned states are tuples of
                the `h_state` and `c_state`s.  If `False`, they are concatenated
                along the column axis.  The latter behavior will soon be deprecated.
            activation:
                Activation function of the update values,
                including new inputs and new cell states.  Default: `tanh`.
            gata_activation:
                Activation function of the gates,
                including the input, ouput, and forget gate. Default: `sigmoid`.
            initializer: 
                Initializer of kernel(weights). Default: `orthogonal_initializer`.
            input_gate_initializer:
                Initializer of input gates.Default: `glorot_normal_initializer`.
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
        super(NLSTMCell, self).__init__(_reuse=reuse, name=name)
        if not state_is_tuple:
            tf.logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)

        self._n_hidden = n_hidden
        self._depth = depth
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or tf.tanh
        self._gate_activation = gate_activation or tf.sigmoid
        self._cell_activation = cell_activation or tf.identity
        self._initializer = initializer or tf.orthogonal_initializer()
        self._input_gate_initializer = (input_gate_initializer 
                                        or tf.glorot_normal_initializer())
        self._use_bias = use_bias
        self._kernels = None  # weight
        self._biases = None
        self.built = False

    @property
    def state_size(self):
        if self._state_is_tuple:
            return tuple([self._n_hidden] * (self._depth + 1))
        else:
            return self._n_hidden * (self._depth + 1)

    @property
    def output_size(self):
        return self._n_hidden

    @property
    def depth(self):
        return self._depth
    
    def build(self, inputs_shape):
        if inputs_shape[1].value is None:   #inputs_shape[0]=batch_size, input_shape[1]=input_size
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                              % inputs_shape)
        
        input_size = inputs_shape[1].value
        n_hidden = self._n_hidden
        self._kernels = []  #weight

        if self._use_bias:
            self._biases = []
        
        for i in range(self._depth):
            if i == 0:
                input_kernel = self.add_variable(
                    "input_gate_kernel",
                    shape=[input_size, 4 * self._n_hidden],   
                    initializer=self._input_gate_initializer)
                
                hidden_kernel = self.add_variable(
                    "hidden_gate_kernel",
                     shape=[n_hidden, 4 * self._n_hidden],     
                     initializer=self._initializer)

                kernel = tf.concat([input_kernel, hidden_kernel],
                           axis=0, name="kernel_0")
                self._kernels.append(kernel)
            else:
                self._kernels.append(
                    self.add_variable(
                    "kernel_{}".format(i),
                    shape=[2 * n_hidden, 4 * self._n_hidden],
                    initializer=self._initializer))
            
            if self._use_bias:
                self._biases.append(
                    self.add_variable(
                        "bias_{}".format(i),
                        shape=[4 * self._n_hidden],
                        initializer=tf.zeros_initializer(dtype=self.dtype)))
        self.built = True

    def recurrence(self,inputs,hidden_state,cell_states,depth):
        """ use recurrent to traverse the nested structure
        Args:
            inputs:
                A 2D `Tensor`. shape is [batch_size, input_size].
            hidden_state:
                A 2D `Tensor`. shape is [batch_size, n_hidden].
            cell_state:
                A 2D `Tensor`. shape is [batch_size, n_hidden].
            depth: 
                int. the current depth in the nested structure, begins at 0.
        Returns:
            new_h:
                A 2D `Tensor`. shape is [batch_size, n_hidden].
                the latest hidden state for current step.
            new_satte:
                A `list` of 2D `Tensor`. including [state_h,state_c],shape is [batch_size, n_hidden].
                The accumulated cell states for current step.
        """
        c = cell_states[depth]
        h = hidden_state
        add = tf.add
        multiply = tf.multiply

        gate_inputs = tf.matmul(
            tf.concat([inputs, h], 1), self._kernels[depth])

        if self._use_bias:
            gate_inputs = tf.nn.bias_add(gate_inputs, self._biases[depth])
        
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(
            value=gate_inputs, num_or_size_splits=4, axis=1)
        
        if self._use_bias:
            forget_bias_tensor = tf.constant(self._forget_bias, dtype=f.dtype)
            f = add(f, forget_bias_tensor)
        
        inner_h = multiply(c, self._gate_activation(f))
        if depth == 0:
            inner_input = multiply(self._gate_activation(i), self._cell_activation(j))
        else:
            inner_input = multiply(self._gate_activation(i), self._activation(j))
        
        if depth == (self._depth - 1):
            new_c = add(inner_h, inner_input)
            new_cs = [new_c]
        else:
            new_c, new_cs = self.recurrence(
                inputs=inner_input,
                hidden_state=inner_h,
                cell_states=cell_states,
                depth=depth + 1)
        
        new_h = multiply(self._activation(new_c), self._gate_activation(o))
        new_cs = [new_h] + new_cs

        return new_h,new_cs

    def call(self, inputs, states):
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

        h_state = states[0]
        c_state = states[1:]

        outputs, next_state = self.recurrence(inputs, h_state, c_state,0)

        if self._state_is_tuple:
            next_state = tuple(next_state)
        
        return outputs,next_state

            
