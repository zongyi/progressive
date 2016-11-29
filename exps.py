from train import *
from config import Config

floatX = theano.config.floatX


def numpy_floatX(data):
    return numpy.float32(data)


def test_LSTM_GRU():
    rng = numpy.random.RandomState(39287)
    # timesteps * n_samples * feature_dim
    x = numpy.array([[[1, 2], [3, 4], [1, 2], [3, 4]], [[1, 2], [3, 4], [1, 2], [3, 4]]], dtype='float32')
    y = numpy.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype='int32')[:, :, None]
    mask = numpy.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype='int32')
    input_layer = T.tensor3('x', dtype=floatX)
    input_mask = T.matrix('mask', dtype='int32')

    layer1pre = LSTMLayer(rng, 2, 2, input_layer, input_mask, False, 'lstm', 'none')
    layer1 = LinearLayer(rng, 2, 2, 'l1', layer1pre.output_lstm, 'softmax')
    layer2pre = GRULayer(rng, 2, 2, input_layer, input_mask, False, 'gru', 'none')
    layer2 = LinearLayer(rng, 2, 2, 'l2', layer2pre.output_gru, 'softmax')

    def nll(pred_prop, y):
        def logadd(prob, mask, y):
            return T.switch(T.gt(mask, 0.0), T.log(prob[y] + 1e-8), numpy_floatX(0.0))

        shape_prob = (input_layer.shape[0] * input_layer.shape[1], 2)
        shape_y = (input_layer.shape[0] * input_layer.shape[1],)
        nll_all, updates = theano.scan(fn=logadd, sequences=[T.reshape(pred_prop, shape_prob),
                                                             T.reshape(input_mask, shape_y),
                                                             T.reshape(y, shape_y)])
        return - T.sum(nll_all) / (T.sum(input_mask) + 1e-8)

    cost1 = nll(layer1.output_l, y) + 0.001 * layer1pre.l2 + 0.001 * layer1.l2
    cost2 = nll(layer2.output_l, y) + 0.001 * layer2pre.l2 + 0.001 * layer2.l2

    def param_update(cost, layer):
        gparams = [T.grad(cost, param) for param in layer.params]
        updates = [(param, param - 0.01 * gparam)
                   for param, gparam in zip(layer.params, gparams)]
        return updates

    func = theano.function(inputs=[input_layer, input_mask], outputs=[cost1, cost2],
                           updates=param_update(cost1, layer1) + param_update(cost2, layer2))
    for i in range(10):
        print(func(x, mask))


def get_cpb_basic():
    print('cpb_basic')
    cfg = Config('-dn model/cpbbasic -n cpb -t 2'.split())
    cfg.test_from_epoch = 0
    cfg.learning_rate = 0.002
    cfg.test_freq = 1000
    cfg.dumpinit()
    train_model(cfg)


def get_cpb_pkupos_basic():
    print('cpb_pkupos')
    cfg = Config('-dn model/cpb_pkupos -dn1 model/pkubasic -n cpb_pkupos -t 2'.split())
    cfg.test_from_epoch = 0
    cfg.learning_rate = 0.001
    cfg.test_freq = 1000
    cfg.end_epoch = 30
    cfg.dumpinit()
    train_model(cfg)


def get_pku_basic():
    print('pku_basic')
    cfg = Config('-dn model/pkubasic -n pku -t 2'.split())
    cfg.test_from_epoch = 0
    cfg.learning_rate = 0.001
    cfg.test_freq = 100
    cfg.dumpinit()
    train_model(cfg)


def get_progressive():
    cfg = Config('-dn model/progressive -dn1 model/pkubasic -m progressive -n cpb_pkupos -t 2'.split())
    cfg.test_from_epoch = 20
    cfg.learning_rate = 0.001
    cfg.test_freq = 1000
    cfg.end_epoch = 50
    import socket
    print(socket.gethostname())
    if socket.gethostname() == 'acl221':  # fake cpbbasic
        cfg.test_from_epoch = 10
        cfg.is_concat = True
        cfg.use_vecs = True
        cfg.dump_name = 'prog_fake_cpbbasic'
        cfg.learning_rate = 0.002
        cfg.p_n_out_new = cfg.p_n_out
        cfg.dist_n_out_new = cfg.dist_n_out
        cfg.lin1_n_out_new = cfg.lin1_n_out
        cfg.rnn_n_out_new = cfg.rnn_n_out
        cfg.lin2_n_out_new = cfg.lin2_n_out
    else:  # local sum
        cfg.use_vecs = True
        cfg.rnn_n_out_ad = cfg.rnn_n_out
        cfg.p_n_out_new = cfg.p_n_out
        cfg.dist_n_out_new = cfg.dist_n_out
        cfg.lin1_n_out_new = cfg.lin1_n_out
        cfg.rnn_n_out_new = cfg.rnn_n_out
        cfg.lin2_n_out_new = cfg.lin2_n_out
        cfg.end_epoch = 70
    cfg.dumpinit()
    train_model(cfg)


def get_progressive_concat():
    cfg = Config('-dn model/prog_concat -dn1 model/pkubasic -m progressive -n cpb_pkupos -t 2'.split())
    cfg.test_from_epoch = 6
    cfg.learning_rate = 0.002
    cfg.test_freq = 1000
    cfg.end_epoch = 50
    import socket
    print(socket.gethostname())
    if socket.gethostname() == 'acl221':  # concat
        cfg.is_concat = True
        cfg.use_vecs = True
        cfg.rnn_n_out_ad = 10
        cfg.p_n_out_new = cfg.p_n_out
        cfg.dist_n_out_new = cfg.dist_n_out
        cfg.lin1_n_out_new = cfg.lin1_n_out
        cfg.rnn_n_out_new = cfg.rnn_n_out - 10
        cfg.lin2_n_out_new = cfg.lin2_n_out
    else:  # local concat: prog_concat
        cfg.is_concat = True
        cfg.use_vecs = True
        cfg.lin1_n_out_ad = 20
        cfg.learning_rate = 0.002
        cfg.test_from_epoch = 8
        cfg.p_n_out_new = cfg.p_n_out
        cfg.dist_n_out_new = cfg.dist_n_out
        cfg.lin1_n_out_new = cfg.lin1_n_out - 20
        cfg.rnn_n_out_new = cfg.rnn_n_out
        cfg.lin2_n_out_new = cfg.lin2_n_out
    cfg.dumpinit()
    train_model(cfg)


def get_prog_cat_lin1():
    cfg = Config('-dn model/prog_concat_cat+lin1 -dn1 model/pkubasic '
                 '-m progressive -n cpb_pkupos -t 2'.split())
    cfg.test_freq = 1000
    cfg.end_epoch = 50
    # local concat: prog_concat_cat+lin1
    cfg.is_concat = True
    cfg.use_vecs = True
    cfg.concat_n_out_ad = 40
    cfg.lin1_n_out_ad = 20
    cfg.learning_rate = 0.002
    cfg.test_from_epoch = 8
    cfg.p_n_out_new = cfg.p_n_out - 10
    cfg.dist_n_out_new = cfg.dist_n_out - 10
    cfg.lin1_n_out_new = cfg.lin1_n_out - 20
    cfg.rnn_n_out_new = cfg.rnn_n_out
    cfg.lin2_n_out_new = cfg.lin2_n_out
    cfg.dumpinit()
    train_model(cfg)


def get_pku_cpbpos_basic():
    print('pku_cpbpos')
    cfg = Config('-dn model/pku_cpbpos_basic -n pku_cpbpos -t 2'.split())
    cfg.test_from_epoch = 10
    cfg.learning_rate = 0.002
    cfg.test_freq = 1000
    cfg.end_epoch = 30
    cfg.dumpinit()
    train_model(cfg)


def get_progressive_goldpos():
    cfg = Config('-dn model/prog_concat_gold -dn1 model/pku_cpbpos_basic -m progressive -n pku_cpbpos -t 1'.split())
    cfg.test_from_epoch = 10
    cfg.learning_rate = 0.002
    cfg.test_freq = 1000
    cfg.end_epoch = 50
    import socket
    print(socket.gethostname())
    if socket.gethostname() == 'acl221':  # concat
        cfg.is_concat = True
        cfg.use_vecs = True
        cfg.lin1_n_out_ad = 20
        cfg.p_n_out_new = cfg.p_n_out
        cfg.dist_n_out_new = cfg.dist_n_out
        cfg.lin1_n_out_new = cfg.lin1_n_out - 20
        cfg.rnn_n_out_new = cfg.rnn_n_out
        cfg.lin2_n_out_new = cfg.lin2_n_out
    cfg.dumpinit()
    train_model(cfg)

if __name__ == '__main__':
    # get_pku_basic()
    # get_cpb_basic()
    # get_cpb_pkupos_basic()
    # get_progressive()
    # get_progressive_concat()
    # get_prog_cat_lin1() # local
    # get_pku_cpbpos_basic() # 221
    get_progressive_goldpos()  # 221
    # get_cpb_basic() # 221
    # TODO: test (no-pos-pku-basic)+(prog-cpb-gold-pos-concat)
