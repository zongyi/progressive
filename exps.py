from dataset_api import *
from train import *
import pickle
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


def get_cpb_std():
    cfg = Config('-dn model/cpbbasic'.split())
    cfg.dumpinit()
    train_BasicNet(cfg)
    cfg = Config('-dn model/cpbbasic -t 0'.split())
    test_BasicNet(cfg)


def get_cpb_pkupos_basic():
    cfg = Config('-dn model/cpb_pkupos -n cpb_pkupos'.split())
    cfg.dumpinit()
    train_BasicNet(cfg)


if __name__ == '__main__':
    get_cpb_pkupos_basic()
    # test_LSTM_GRU()
    # print('pku'); train_BasicNet(CpbDatasets, 'pku', PKU_POSS, PKU_TAGGING, pku_train_f, pku_dev_f, pku_test_f, PKU_TRANS2, PKU_TRANS0, PKU_NOT_ENTRY_IDXS, PKU_NOT_EXIT_IDXS)
