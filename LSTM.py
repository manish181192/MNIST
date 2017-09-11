import tensorflow as tf

def get_lstm_cell(num_units):
    cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units,
                                   input_size=None,
                                   use_peepholes=False,
                                   cell_clip=None,
                                   initializer=None,
                                   num_proj=None,
                                   proj_clip=None,
                                   num_unit_shards=1,
                                   num_proj_shards=1,
                                   forget_bias=1.0,
                                   state_is_tuple=True)
    return cell

def get_gru_cell(num_units):
    cell = tf.nn.rnn_cell.GRUCell(num_units)

    return cell

def get_cell(type,  num_units, dropout, num_lstm_layers, scope):
    cell = None
    with tf.variable_scope(scope):
        if type == 'LSTM':
            cell = get_lstm_cell(num_units)
        elif type == 'gru':
            cell = get_gru_cell(num_units)
        if num_lstm_layers>1:
            cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell]*num_lstm_layers)
        drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob= dropout)
        return drop_cell


class rnn_network():
    def __init__(self, type, num_units, num_lstm_layers, inputs, dropout, num_classes, isBidirectional):
        cell = get_cell(type, num_units,dropout, num_lstm_layers, scope="FW")

        self.outputs, self.states = tf.nn.dynamic_rnn(cell=cell,
                                            inputs=inputs,
                                            sequence_length=None,
                                            initial_state=None,
                                            dtype=tf.float32,
                                            parallel_iterations=None,
                                            swap_memory=False,
                                            time_major=False,
                                            scope=None)

        final_out = self.outputs[:, -1]

        if isBidirectional:
            scope = "BW"
            cell_bw = get_cell(type, num_units, dropout, num_lstm_layers, scope=scope)
            self.outputs_bw, self.states_bw = tf.nn.dynamic_rnn(cell=cell_bw,
                                                          inputs=inputs,
                                                          sequence_length=None,
                                                          initial_state=None,
                                                          dtype=tf.float32,
                                                          parallel_iterations=None,
                                                          swap_memory=False,
                                                          time_major=False,
                                                          scope=scope)

            out_bw = self.outputs_bw[:, -1]
            final_out = tf.concat(1, [final_out, out_bw])

        output_units = None
        if isBidirectional:
            output_units = 2*num_units
        else:
            output_units = num_units

        W = tf.Variable(initial_value= tf.truncated_normal(dtype=tf.float32, shape=[output_units,num_classes]))
        B = tf.Variable(initial_value=tf.constant(shape=[num_classes],value=0.1, dtype=tf.float32,))
        self.logits = tf.matmul(final_out, W) + B
