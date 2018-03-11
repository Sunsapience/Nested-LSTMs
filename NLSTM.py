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
                 forget_bias=1.0,
                 state_is_tuple=True,
                 use_peepholes=False,
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
            forget_bias:
                float. The bias added to forget gates.
            state_is_tuple:
                If `True`, accepted and returned states are tuples of
                the `h_state` and `c_state`s.  If `False`, they are concatenated
                along the column axis.  The latter behavior will soon be deprecated.
            use_peepholes:
                bool.Set True to enable diagonal/peephole connections.
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
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._use_peepholes = use_peepholes
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
            return tuple([self._n_hidden] * 3)   #(n_hidden,n_hidden)
        else:
            return self._n_hidden *3
    
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
        
        if self._use_peepholes:
            self._peep_kernels = []
        
        input_kernel = self.add_variable(
            "input_gate_kernel",
            shape=[input_size, 4 * self._n_hidden],   # 4 gates input weight matrices(x--->w)
            initializer=self._input_gate_initializer)
        hidden_kernel = self.add_variable(
            "hidden_gate_kernel",
            shape=[n_hidden, 4 * self._n_hidden],     # 4 gates hidden weight matrices(h_t--->w)
            initializer=self._initializer)
        kernel = tf.concat([input_kernel, hidden_kernel],  
                           axis=0, name="kernel_0")
        # the shape of kernel is [input_size+n_hidden,4 * self._n_hidden]
        self._kernels.append(kernel)

        # the weight matrix of inner NLSTM(Nessted)
        # the shape of inner_input(inner_x) and inner_state(inner_h ) is [batch_size,n_hidden]
        # so the shape of inner_weight_matrix is [2*n_hidden,n_hidden] 
        self._kernels.append(
            self.add_variable(
                "kernel_{}".format(1),
                shape=[ 2*n_hidden, 4 * self._n_hidden],
                initializer=self._initializer))
        
        if self._use_bias:
            self._biases.append(
                self.add_variable(
                    "bias_{}".format(0),
                    shape=[4 * self._n_hidden],
                    initializer=tf.zeros_initializer(dtype=self.dtype)))
            self._biases.append(
                self.add_variable(
                    "bias_{}".format(1),
                    shape=[4 * self._n_hidden],
                    initializer=tf.zeros_initializer(dtype=self.dtype)))
        
        if self._use_peepholes:
            self._peep_kernels.append(
                self.add_variable(
                    "peep_kernel_{}".format(0),
                     shape=[n_hidden, 3 * self._n_hidden],
                     initializer=self._initializer))
            self._peep_kernels.append(
                self.add_variable(
                    "peep_kernel_{}".format(1),
                     shape=[n_hidden, 3 * self._n_hidden],
                     initializer=self._initializer))
        
        self.built = True
    
    def recurrence(self,inputs,hidden_state,cell_state):
        """ use recurrent to traverse the nested structure
        Args:
            inputs:
                A 2D `Tensor`. shape is [batch_size, input_size].
            hidden_state:
                A 2D `Tensor`. shape is [batch_size, n_hidden].
            cell_state:
                A 2D `Tensor`. shape is [batch_size, n_hidden].
        Returns:
            new_h:
                A 2D `Tensor`. shape is [batch_size, n_hidden].
                the latest hidden state for current step.
            new_satte:
                A `list` of 2D `Tensor`. including [state_h,state_c],shape is [batch_size, n_hidden].
                The accumulated cell states for current step.
        """
        one = tf.constant(1, tf.int32)
        c = cell_state[0]
        h = hidden_state
        
        # the shape of inputs is [batch_size,input_size](input_size is not n_steps)
        # the shape of h is [batch_size,n_hidden]
        # combine [input,h](x,h_t),shape is [batch_size,input_size+n_hidden]
        # the shape kernel is [input_size+n_hidden,4 * self._n_hidden]
        # the shape gate_inputs is  [batch_size,4 * self._n_hidden]
        gate_inputs = tf.matmul(
            tf.concat([inputs, h], 1), self._kernels[0])
            
        if self._use_bias:
            gate_inputs = tf.nn.bias_add(gate_inputs, self._biases[0])
        if self._use_peepholes:
            peep_gate_inputs = tf.matmul(c, self._peep_kernels[0])  
            i_peep, f_peep, o_peep = tf.split(
                value=peep_gate_inputs, num_or_size_splits=3, axis=one)
        
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        # the shape of 4 gate is [bath_size,n_hidden]
        i, j, f, o = tf.split(
            value=gate_inputs, num_or_size_splits=4, axis=one)

        if self._use_peepholes:
            i += i_peep
            f += f_peep
            o += o_peep 

        if self._use_peepholes:
            peep_gate_inputs = tf.matmul(c, self._peep_kernels[0])
            i_peep, f_peep, o_peep = tf.split(
                value=peep_gate_inputs, num_or_size_splits=3, axis=one)
            i += i_peep
            f += f_peep
            o += o_peep
        
        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = tf.add
        multiply = tf.multiply

        if self._use_bias:
            forget_bias_tensor = tf.constant(self._forget_bias, dtype=f.dtype)
            f = add(f, forget_bias_tensor)
        
        # In[]
        # the shape of inner_h,inner_c,inner_input are [batch_size,n_hidden]
        inner_h = multiply(c, self._gate_activation(f))
        inner_c = cell_state[1]
        inner_inputs = multiply(self._gate_activation(i), self._cell_activation(j))
        
        # inner
        # the shape of inner_gate_input is [batch_size,4*n_hidden]
        # the reason is matmul([batch_size,n_hidden+n_hidden],[2*n_hidden,4*n_hidden]
        inner_gate_inputs = tf.matmul(
            tf.concat([inner_inputs, inner_h], 1), self._kernels[1])
        
        if self._use_bias:
            inner_gate_inputs = tf.nn.bias_add(inner_gate_inputs, self._biases[1])
        if self._use_peepholes:
            inner_peep_gate_inputs = tf.matmul(inner_c, self._peep_kernels[1])  
            inner_i_peep, inner_f_peep, inner_o_peep = tf.split(
                value=inner_peep_gate_inputs, num_or_size_splits=3, axis=one)
        
        new_i, new_j, new_f, new_o = tf.split(
            value=inner_gate_inputs, num_or_size_splits=4, axis=one)
        
        if self._use_peepholes:
            new_i += inner_i_peep
            new_f += inner_f_peep
            new_o += inner_o_peep
        if self._use_peepholes:
            inner_peep_gate_inputs = tf.matmul(inner_c, self._peep_kernels[1])
            inner_i_peep, inner_f_peep, inner_o_peep = tf.split(
                value=inner_peep_gate_inputs, num_or_size_splits=3, axis=one)
            new_i += inner_i_peep
            new_f += inner_f_peep
            new_o += inner_o_peep        
        if self._use_bias:
            inner_forget_bias_tensor = tf.constant(self._forget_bias, dtype=f.dtype)
            new_f = add(new_f, inner_forget_bias_tensor)  


        
        new_inner_c = add(multiply(inner_c, self._gate_activation(new_f)),
                          multiply(self._gate_activation(new_i), self._activation(new_j)))
        new_inner_h = multiply(new_o, self._activation(new_inner_c))

        new_c = new_inner_h
        new_h = multiply(o, self._activation(new_c))
        new_state = [new_h]+[new_c]+[new_inner_c]
        
        return new_h,new_state

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

        outputs, next_state = self.recurrence(inputs, h_state, c_state)

        if self._state_is_tuple:
            next_state = tuple(next_state)
        
        return outputs,next_state

        




        

        



