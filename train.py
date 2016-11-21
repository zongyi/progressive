from model import *
from dataset_api import *
import h5py
import timeit

floatX = theano.config.floatX


def numpy_floatX(data):
    return numpy.float32(data)


def gen_masks(maxlen, n_samples, n_tagging, vis, lens, NOT_ENTRY_IDXS, NOT_EXIT_IDXS):
    entry_exit_mask = numpy.zeros((maxlen, n_samples, n_tagging), dtype='float32')
    for ii, (vi, sent_len) in enumerate(zip(vis, lens)):
        entry_exit_mask[-sent_len, ii, NOT_ENTRY_IDXS] = numpy_floatX('-inf')
        if -sent_len + vi != -maxlen:
            entry_exit_mask[-sent_len + vi - 1, ii, NOT_EXIT_IDXS] = numpy_floatX('-inf')
        entry_exit_mask[-sent_len + vi, ii, NOT_ENTRY_IDXS] = numpy_floatX('-inf')
        entry_exit_mask[-1, ii, NOT_EXIT_IDXS] = numpy_floatX('-inf')
    return entry_exit_mask


def cal_f1(result, answer, accu, TAGGING):
    a_seq = []
    b_seq = []
    for i in range(len(result)):
        a_seq.append(TAGGING[result[i]])
        b_seq.append(TAGGING[answer[i]])
    s1 = 0
    s2 = 0
    for item in a_seq:
        if item.startswith(b'S-') or item.startswith(b'B-'):
            s1 += 1

    for item in b_seq:
        if item.startswith(b'S-') or item.startswith(b'B-'):
            s2 += 1

    s3 = 0
    i = 0
    while i < len(a_seq):
        if a_seq[i].startswith(b'S-'):
            if b_seq[i] == a_seq[i]:
                s3 += 1
        elif a_seq[i].startswith(b'B-'):
            isMatch = True
            while not a_seq[i].startswith(b'E-'):
                if a_seq[i] != b_seq[i]:
                    isMatch = False
                i += 1
            if a_seq[i] != b_seq[i]:
                isMatch = False
            if isMatch:
                s3 += 1
        i += 1
    accu[0] += s3
    accu[1] += s1
    accu[2] += s2
    return accu


def train_BasicNet(cfg):
    print('building ...')
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
    datasets, vecs = CpbDatasets(cfg.data_name, worddim, window, max_dist,
                                 cfg.POSS, cfg.TAGGING, cfg.train_f, cfg.dev_f, cfg.test_f).get()
    train_set, dev_set, test_set = datasets
    ws, ps, dists, vis, vs, rs, lens = train_set
    test_ws, test_ps, test_dists, test_vis, test_vs, test_rs, test_lens = test_set
    entry_exit_mask = gen_masks(ws.shape[0], ws.shape[1], len(cfg.TAGGING), vis, lens, cfg.NOT_ENTRY_IDXS,
                                cfg.NOT_EXIT_IDXS)
    test_entry_exit_mask = gen_masks(test_ws.shape[0], test_ws.shape[1], len(cfg.TAGGING), test_vis, test_lens,
                                     cfg.NOT_ENTRY_IDXS, cfg.NOT_EXIT_IDXS)

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
    l1_lr = numpy_floatX(0.0)
    l2_lr = numpy_floatX(0.0001)
    lr = numpy_floatX(cfg.learning_rate)
    # TODO: relu 代替 tanh
    net = BasicNet('basic', len(vecs), worddim, len(cfg.POSS), p_n_out, max_dist + 2, dist_n_out,
                   lin1_n_out, rnn_n_out, lin2_n_out, len(cfg.TAGGING),
                   window, cfg.TRANS2, cfg.TRANS0, inputs,
                   W=[vecs, None, None, None, None, None, None,
                      None, None, None, None, None, None])
    nll_cost = net.cal_cost(input_y, l1_lr, l2_lr)
    gparams = [T.grad(nll_cost, param) for param in net.params]
    updates = [(param, param - lr * gparam) for param, gparam in zip(net.params, gparams)]
    train_fun = theano.function(inputs=inputs, outputs=nll_cost, updates=updates)
    test_fun = theano.function(inputs=inputs, outputs=net.y_pred, on_unused_input='ignore')
    max_f1 = 0.0
    iter = 0
    done_loop = False
    print('training ...')
    start_time = timeit.default_timer()
    for e_i in range(20):
        if done_loop: break
        for sample_i in range(ws.shape[1]):
            if iter >= cfg.end_iter:
                done_loop = True
                break
            cost = train_fun(ws[-lens[sample_i]:, sample_i, :],
                             ps[-lens[sample_i]:, sample_i, :],
                             dists[-lens[sample_i]:, sample_i],
                             vs[-lens[sample_i]:, sample_i],
                             entry_exit_mask[-lens[sample_i]:, sample_i],
                             vis[sample_i],
                             rs[-lens[sample_i]:, sample_i])
            if e_i >= cfg.test_from_epoch and sample_i % cfg.test_freq == 0:
                precision, recall, f1 = test_prf(test_fun, test_ws, test_ps, test_dists,
                                                 test_vis, test_vs, test_rs, test_lens,
                                                 test_entry_exit_mask, cfg.TAGGING)
                print(e_i, sample_i, precision, recall, f1, '  ', cost, 'max hit!' if f1 > max_f1 else '')
                if f1 > max_f1:
                    sys.stdout.write('dumping to %s.pkl ...' % cfg.dump_name)
                    with open(cfg.dump_name + '.pkl', 'wb') as f:
                        pickle.dump(net.params, f)
                    print(' done.')
                max_f1 = max(f1, max_f1)
            iter += 1
        end_time = timeit.default_timer()
        print('%.1f s' % ((end_time - start_time) / 60.))
        start_time = end_time


def load_model(cfg):
    with open(cfg.dump_name + '.pkl', 'rb') as f:
        params = pickle.load(f)
    return params


def test_prf(test_fun, test_ws, test_ps, test_dists, test_vis, test_vs,
             test_rs, test_lens, test_entry_exit_mask, TAGGING):
    accu = [0, 0, 0]
    for sample_j in range(test_ws.shape[1]):
        output = test_fun(test_ws[-test_lens[sample_j]:, sample_j, :],
                          test_ps[-test_lens[sample_j]:, sample_j, :],
                          test_dists[-test_lens[sample_j]:, sample_j],
                          test_vs[-test_lens[sample_j]:, sample_j],
                          test_entry_exit_mask[-test_lens[sample_j]:, sample_j, :],
                          test_vis[sample_j],
                          test_rs[-test_lens[sample_j]:, sample_j])
        accu = cal_f1(output, test_rs[-test_lens[sample_j]:, sample_j], accu, TAGGING)
    precision = 0
    recall = 0
    f1 = 0
    if accu[1] != 0:
        precision = accu[0] / accu[1]
    if accu[2] != 0:
        recall = accu[0] / accu[2]
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def test_BasicNet(cfg):
    worddim = 50
    window = 3
    max_dist = 500
    datasets, vecs = CpbDatasets(cfg.data_name, worddim, window, max_dist,
                                 cfg.POSS, cfg.TAGGING, cfg.train_f, cfg.dev_f, cfg.test_f).get()
    train_set, dev_set, test_set = datasets
    test_ws, test_ps, test_dists, test_vis, test_vs, test_rs, test_lens = test_set
    test_entry_exit_mask = gen_masks(test_ws.shape[0], test_ws.shape[1], len(cfg.TAGGING), test_vis, test_lens,
                                     cfg.NOT_ENTRY_IDXS, cfg.NOT_EXIT_IDXS)

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
    net = BasicNet('basic', len(vecs), worddim, len(cfg.POSS), p_n_out, max_dist + 2, dist_n_out,
                   lin1_n_out, rnn_n_out, lin2_n_out, len(cfg.TAGGING),
                   window, cfg.TRANS2, cfg.TRANS0, inputs,
                   W=load_model(cfg))
    test_fun = theano.function(inputs=inputs, outputs=net.y_pred, on_unused_input='ignore')
    start_time = timeit.default_timer()
    precision, recall, f1 = test_prf(test_fun, test_ws, test_ps, test_dists, test_vis, test_vs, test_rs, test_lens,
                                     test_entry_exit_mask, cfg.TAGGING)
    end_time = timeit.default_timer()
    print(precision, recall, f1)
    print('%.1f s' % ((end_time - start_time) / 60.))
