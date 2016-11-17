from dataset_api import *
from model import *

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


def gen_masks(maxlen, n_samples, n_tagging, vis, lens):
    entry_exit_mask = numpy.zeros((maxlen, n_samples, n_tagging), dtype='float32')
    for ii, (vi, sent_len) in enumerate(zip(vis, lens)):
        entry_exit_mask[-sent_len, ii, NOT_ENTRY_IDXS] = numpy_floatX('-inf')
        if -sent_len + vi != -maxlen:
            entry_exit_mask[-sent_len + vi - 1, ii, NOT_EXIT_IDXS] = numpy_floatX('-inf')
        entry_exit_mask[-sent_len + vi, ii, NOT_ENTRY_IDXS] = numpy_floatX('-inf')
        entry_exit_mask[-1, ii, NOT_EXIT_IDXS] = numpy_floatX('-inf')
    return entry_exit_mask


def test_BasicNet():
    worddim = 50
    window = 3
    max_dist = 500
    # datasets = [[numpy.array([[[0,1],[1,0]],[[2,3],[3,2]]], dtype='int32'),  # ws
    #                            numpy.array([[[0,1],[1,0]],[[2,3],[3,2]]], dtype='int32'),  # ps
    #                            numpy.array([[0,1],[2,3]], dtype='int32'),  # dists
    #                            [],
    #                            numpy.array([[0,3],[0,3]], dtype='int32'),  # vs
    #                            [],[]],
    #                           [], []]
    # vecs = numpy.array([[0.1,0.1,0.1],[0.2,0.2,0.2],[0.3,0.3,0.3],[0.4,0.4,0.4]], dtype='float32')
    # p_vecs = numpy.array([[1,1],[2,2],[3,3],[4,4]], dtype='float32')
    # dist_vecs = numpy.array([[10,10],[20,20],[30,30],[40,40]], dtype='float32')
    datasets, vecs = CpbDatasets(worddim, window, max_dist).get()
    train_set, dev_set, test_set = datasets
    ws, ps, dists, vis, vs, rs, lens = train_set
    test_ws, test_ps, test_dists, test_vis, test_vs, test_rs, test_lens = test_set
    entry_exit_mask = gen_masks(ws.shape[0], ws.shape[1], len(TAGGING), vis, lens)
    test_entry_exit_mask = gen_masks(test_ws.shape[0], test_ws.shape[1], len(TAGGING), test_vis, test_lens)

    input_w = T.matrix('in_w', dtype='int32')
    input_p = T.matrix('in_p', dtype='int32')
    input_dist = T.vector('in_dist', dtype='int32')
    input_v = T.vector('in_v', dtype='int32')
    input_vi = T.scalar('in_vi', dtype='int32')
    input_entry_exit_mask = T.matrix('in_entry', dtype='float32')
    input_y = T.vector('in_y', dtype='int32')
    inputs = [input_w, input_p, input_dist, input_v, input_entry_exit_mask, input_vi, input_y]
    p_n_out = 20
    dist_n_out = 20
    lin1_n_out = 200
    rnn_n_out = 200
    lin2_n_out = 100
    l1 = numpy_floatX(0.0001)
    l2 = numpy_floatX(0.0001)
    lr = numpy_floatX(0.002)
    # TODO: relu 代替 tanh
    net = BasicNet(len(vecs), worddim, len(POSS), p_n_out, max_dist + 2, dist_n_out,
                   lin1_n_out, rnn_n_out, lin2_n_out, len(TAGGING),
                   window, inputs,
                   W=[vecs, None, None, None, None, None, None, None, None, None, None, None, None])

    sample_i = 0
    nll_cost = net.cal_cost(input_y, l1, l2)
    gparams = [T.grad(nll_cost, param) for param in net.params]
    updates = [(param, param - lr * gparam) for param, gparam in zip(net.params, gparams)]
    train_fun = theano.function(inputs=inputs, outputs=nll_cost, updates=updates)
    test_fun = theano.function(inputs=inputs, outputs=net.y_pred, on_unused_input='ignore')
    for e_i in range(5):
        for sample_i in range(30000):
            cost = train_fun(ws[-lens[sample_i]:, sample_i, :],
                             ps[-lens[sample_i]:, sample_i, :],
                             dists[-lens[sample_i]:, sample_i],
                             vs[-lens[sample_i]:, sample_i],
                             entry_exit_mask[-lens[sample_i]:, sample_i],
                             vis[sample_i],
                             rs[-lens[sample_i]:, sample_i])
            if sample_i % 100 == 0:
                accu = [0, 0, 0]
                for sample_j in range(1000):
                    output = test_fun(test_ws[-lens[sample_j]:, sample_j, :],
                                      test_ps[-lens[sample_j]:, sample_j, :],
                                      test_dists[-lens[sample_j]:, sample_j],
                                      test_vs[-lens[sample_j]:, sample_j],
                                      test_entry_exit_mask[-lens[sample_j]:, sample_j],
                                      test_vis[sample_j],
                                      test_rs[-lens[sample_j]:, sample_j])
                    prf = cal_f1(output, rs[-lens[sample_j]:, sample_j])
                    accu[0] += prf[0]
                    accu[1] += prf[1]
                    accu[2] += prf[2]
                precision = 0
                recall = 0
                f1 = 0
                if accu[1] != 0:
                    precision = accu[0] / accu[1]
                if accu[2] != 0:
                    recall = accu[0] / accu[2]
                if precision + recall != 0:
                    f1 = 2 * precision * recall / (precision + recall)
                print(precision, recall, f1, '  ', cost)


def cal_f1(result, answer):
    a_seq = []
    b_seq = []
    for i in range(len(result)):
        a_seq.append(TAGGING[result[i]])
        b_seq.append(TAGGING[answer[i]])
    s1 = 0
    s2 = 0
    for item in a_seq:
        if item.startswith('S-') or item.startswith('B-'):
            s1 += 1

    for item in b_seq:
        if item.startswith('S-') or item.startswith('B-'):
            s2 += 1

    s3 = 0
    i = 0
    while i < len(a_seq):
        if a_seq[i].startswith('S-'):
            if b_seq[i] == a_seq[i]:
                s3 += 1
        elif a_seq[i].startswith('B-'):
            isMatch = True
            while not a_seq[i].startswith('E-'):
                if a_seq[i] != b_seq[i]:
                    isMatch = False
                i += 1
            if a_seq[i] != b_seq[i]:
                isMatch = False
            if isMatch:
                s3 += 1
        i += 1

    return [s3, s1, s2]


if __name__ == '__main__':
    # test_LSTM_GRU()
    test_BasicNet()
