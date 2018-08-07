import tensorflow as tf
from models.model_ops import linear_transform,attention_han


def textcnn(X_embedded,filter_size_list=(2,3,4,5),filter_num=128,
            scope="textcnn", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        h_total = []
        for filter_size in filter_size_list:
            h = tf.layers.conv1d(inputs=X_embedded, filters=filter_num, kernel_size=filter_size,
                                 strides=1, padding='same', data_format='channels_last',
                                 activation=tf.nn.selu, use_bias=True,
                                 kernel_initializer=tf.glorot_uniform_initializer())
            h = tf.reduce_max(h, axis=-2)
            h_total.append(h)
        out_dim = filter_num * len(h_total)
        if len(h_total)>1:
            out = tf.concat(h_total, axis=-1)
        else:
            out = h_total[0]
        out = tf.reshape(out, shape=[tf.shape(X_embedded)[0], out_dim])
    return out, out_dim


def textrnn(X_embedded,X_len,state_size_list=(128,),keep_prob=1.0,
            scope="textrnn", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        batch_size= tf.shape(X_embedded)[0]
        cells_fw, cells_bw = [], []
        for i in range(len(state_size_list)):
            state_size = state_size_list[i]
            cell_fw = tf.nn.rnn_cell.GRUCell(state_size,
                                             kernel_initializer=tf.glorot_uniform_initializer())
            cell_bw = tf.nn.rnn_cell.GRUCell(state_size,
                                             kernel_initializer=tf.glorot_uniform_initializer())
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)
            cells_fw.append(cell_fw)
            cells_bw.append(cell_bw)
        if len(cells_fw) > 1:
            cells_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
            cells_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)
        else:
            cells_fw, cells_bw = cells_fw[0], cells_bw[0]
        (rnn_outputs_fw, rnn_outputs_bw), final_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cells_fw, cell_bw=cells_bw,
                                            inputs=X_embedded, sequence_length=X_len,
                                            dtype=tf.float32)
        rnn_outputs_dim = 2 * state_size_list[-1]
        rnn_outputs = tf.concat([rnn_outputs_fw, rnn_outputs_bw], axis=-1)
        rnn_outputs = tf.gather_nd(params=rnn_outputs,
                                   indices=tf.stack([tf.range(batch_size), X_len - 1], axis=1))
        return rnn_outputs, rnn_outputs_dim


def crnn(X_embedded, X_len,
         filter_size_list=(3,),filter_num=128,
         state_size_list=(128,),keep_prob=1.0,
         scope="crnn", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        batch_size,seq_len=tf.shape(X_embedded)[0],tf.shape(X_embedded)[1]
        h_total = []
        for filter_size in filter_size_list:
            h = tf.layers.conv1d(inputs=X_embedded, filters=filter_num, kernel_size=filter_size,
                                 strides=1, padding='same', data_format='channels_last',
                                 activation=tf.nn.selu, use_bias=True,
                                 kernel_initializer=tf.glorot_uniform_initializer())
            h_total.append(h)
        h_dim=filter_num*len(h_total)
        if len(h_total) > 1:
            h= tf.concat(h_total, axis=-1)
        else:
            h= h_total[0]
        h = tf.reshape(h, shape=[batch_size, seq_len, h_dim])
        cells_fw, cells_bw= [], []
        for i in range(len(state_size_list)):
            state_size = state_size_list[i]
            cell_fw = tf.nn.rnn_cell.GRUCell(state_size,
                                             kernel_initializer=tf.glorot_uniform_initializer())
            cell_bw = tf.nn.rnn_cell.GRUCell(state_size,
                                             kernel_initializer=tf.glorot_uniform_initializer())
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)
            cells_fw.append(cell_fw)
            cells_bw.append(cell_bw)
        if len(cells_fw) > 1:
            cells_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
            cells_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)
        else:
            cells_fw, cells_bw= cells_fw[0], cells_bw[0]
        (rnn_outputs_fw, rnn_outputs_bw), final_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cells_fw, cell_bw=cells_bw,
                                            inputs=h, sequence_length=X_len,
                                            dtype=tf.float32)
        out = tf.concat([rnn_outputs_fw, rnn_outputs_bw], axis=-1)
        out_dim = 2 * state_size_list[-1]
        out = tf.gather_nd(params=out, indices=tf.stack([tf.range(batch_size), X_len - 1], axis=1))
        return out,out_dim


def rcnn(X_embedded,  X_len,
         state_size_list=(128,),hidden_size=256,keep_prob=1.0,
         scope="rcnn", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        batch_size = tf.shape(X_embedded)[0]
        cells_fw, cells_bw, cells_fw_init, cells_bw_init = [], [], [], []
        for i in range(len(state_size_list)):
            state_size = state_size_list[i]
            cell_fw = tf.nn.rnn_cell.BasicRNNCell(num_units=state_size)
            cell_bw = tf.nn.rnn_cell.BasicRNNCell(num_units=state_size)
            init_fw_ = tf.get_variable(name="cell_fw_init_state_" + str(i),
                                       dtype=tf.float32, shape=[1, state_size],
                                       trainable=True, initializer=tf.glorot_uniform_initializer())
            init_fw = tf.tile(init_fw_, multiples=[batch_size, 1])
            init_bw_ = tf.get_variable(name="cell_bw_init_state_" + str(i),
                                       dtype=tf.float32, shape=[1, state_size],
                                       trainable=True, initializer=tf.glorot_uniform_initializer())
            init_bw = tf.tile(init_bw_, multiples=[batch_size, 1])
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)
            cells_fw.append(cell_fw)
            cells_bw.append(cell_bw)
            cells_fw_init.append(init_fw)
            cells_bw_init.append(init_bw)
        if len(cells_fw) > 1:
            cells_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
            cells_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)
            cells_fw_init = tf.nn.rnn_cell.MultiRNNCell(cells_fw_init)
            cells_bw_init = tf.nn.rnn_cell.MultiRNNCell(cells_bw_init)
        else:
            cells_fw, cells_bw, cells_fw_init, cells_bw_init = cells_fw[0], cells_bw[0], cells_fw_init[0], \
                                                               cells_bw_init[0]
        (rnn_outputs_fw, rnn_outputs_bw), final_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cells_fw, cell_bw=cells_bw,
                                            inputs=X_embedded, sequence_length=X_len,
                                            initial_state_fw=cells_fw_init, initial_state_bw=cells_bw_init)
        rnn_outputs_fw = tf.concat([tf.expand_dims(cells_fw_init, axis=1), rnn_outputs_fw[:, :-1, :]],
                                   axis=1)
        rnn_outputs_bw = tf.concat([rnn_outputs_bw[:, 1:, :], tf.expand_dims(cells_bw_init, axis=1)],
                                   axis=1)
        h = tf.concat([rnn_outputs_fw, X_embedded, rnn_outputs_bw], axis=-1)
        h = linear_transform(h,hidden_size,tf.nn.tanh)
        out = tf.reduce_max(h, axis=-2)
        return out,hidden_size


def han(X_embedded,  X_len,
        state_size_list=(64,),attention_dim=128,keep_prob=1.0,
        scope="han", reuse=False):
    """
    Only 1-level attention is used here.
    """
    with tf.variable_scope(scope, reuse=reuse):
        cells_fw, cells_bw= [], []
        for i in range(len(state_size_list)):
            state_size = state_size_list[i]
            cell_fw = tf.nn.rnn_cell.GRUCell(state_size,
                                             kernel_initializer=tf.glorot_uniform_initializer())
            cell_bw = tf.nn.rnn_cell.GRUCell(state_size,
                                             kernel_initializer=tf.glorot_uniform_initializer())
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)
            cells_fw.append(cell_fw)
            cells_bw.append(cell_bw)
        if len(cells_fw) > 1:
            cells_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
            cells_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)
        else:
            cells_fw, cells_bw= cells_fw[0], cells_bw[0]
        (rnn_outputs_fw, rnn_outputs_bw), final_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cells_fw, cell_bw=cells_bw,
                                            inputs=X_embedded, sequence_length=X_len,
                                            dtype=tf.float32)
        h = tf.concat([rnn_outputs_fw, rnn_outputs_bw], axis=-1)
        return attention_han(h, attention_dim, scope="attention"),attention_dim
