from models import *
from dataset_api import *
import timeit
import pickle
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


def load_model(dump_name):
    sys.stdout.write('loading model ... ')
    with open(dump_name + '.pkl', 'rb') as f:
        params = pickle.load(f)
        try:
            [e_i, sample_i, precision, recall, f1] = pickle.load(f)
        except:
            [e_i, sample_i, precision, recall, f1] = [0, 0, 0.0, 0.0, 0.0]
    print(e_i, sample_i, precision, recall, f1)
    return params, e_i, sample_i, precision, recall, f1


def test_prf(test_fun, net1_data, test_ws, test_ps, test_dists, test_vis, test_vs,
             test_rs, test_lens, test_entry_exit_mask, TAGGING):
    accu = [0, 0, 0]
    for sample_j in range(test_ws.shape[1]):
        inputs_data1 = [d[-test_lens[sample_j]:, sample_j] for d in net1_data if d is not None]
        inputs_data = inputs_data1 + [test_ws[-test_lens[sample_j]:, sample_j, :],
                                      test_ps[-test_lens[sample_j]:, sample_j, :],
                                      test_dists[-test_lens[sample_j]:, sample_j],
                                      test_vs[-test_lens[sample_j]:, sample_j],
                                      test_entry_exit_mask[-test_lens[sample_j]:, sample_j, :],
                                      test_vis[sample_j],
                                      test_rs[-test_lens[sample_j]:, sample_j]]
        output = test_fun(*(inputs_data))
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


def train_model(cfg):
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
    l1_lr = numpy_floatX(0.0)
    l2_lr = numpy_floatX(0.0001)
    lr = numpy_floatX(cfg.learning_rate)
    use_noise = theano.shared(numpy.asarray(1., dtype=theano.config.floatX))
    # TODO: relu 代替 tanh
    if cfg.training == 2 or cfg.training == 0:
        loaded_W, pre_e_i, pre_sample_i, pre_precision, pre_recall, pre_f1 = load_model(cfg.dump_name)
        print([w.eval().shape for w in loaded_W])
    else:
        loaded_W, pre_e_i, pre_sample_i, pre_precision, pre_recall, pre_f1 = None, 0, 0, 0.0, 0.0, 0.0
    if cfg.model_type == 'basic':
        print('basic net building ...')
        inputs = [input_w, input_p, input_dist, input_v, input_entry_exit_mask, input_vi, input_y]
        net = BasicNet('basic', use_noise, len(vecs), worddim if cfg.use_vecs else cfg.w_n_out, len(cfg.POSS),
                       cfg.p_n_out, max_dist + 2, cfg.dist_n_out,
                       cfg.lin1_n_out, cfg.rnn_n_out, cfg.lin2_n_out, len(cfg.TAGGING),
                       window, cfg.TRANS2, cfg.TRANS0, inputs,
                       W=([vecs if cfg.use_vecs else None] +
                          ([None] if cfg.p_n_out else []) + [None, None, None, None, None, None, None,
                                                             None, None, None, None]) if loaded_W is None else loaded_W)
        ws1, vs1, test_ws1, test_vs1 = None, None, None, None
    else:
        print('progressive net building ...')
        input_w1 = T.matrix('in_w1', dtype='int32')
        input_v1 = T.vector('in_v1', dtype='int32')
        inputs1 = [input_w1, input_p, input_dist, input_v1, input_entry_exit_mask, input_vi, input_y]
        use_noise1 = theano.shared(numpy.asarray(0., dtype=theano.config.floatX))
        loaded_W1, _, _, _, _, _ = load_model(cfg.dump_name1)

        target = CpbDatasets('cpb', 50, 3, 500, CPB_POSS, CPB_TAGGING, cpb_train_f, cpb_dev_f, cpb_test_f)
        if cfg.data_name == 'cpb_pkupos':
            src = CpbDatasets('pku', 50, 3, 500, PKU_POSS, PKU_TAGGING, pku_train_f, pku_dev_f, pku_test_f)
        else:  # if cfg.data_name == 'cpb_cpbpos':
            src = CpbDatasets('pku_cpbpos', 50, 3, 500, CPB_POSS, PKU_TAGGING, pku_cpbpos_train_f, pku_cpbpos_dev_f,
                              pku_cpbpos_test_f)
        target_data, target_vecs = target.get()  # this line is to init cpb.word2idx
        src_data, src_vecs = src.get()
        net1 = BasicNet('basic', use_noise1, len(src_vecs), worddim, len(cfg.POSS), cfg.p_n_out, max_dist + 2,
                        cfg.dist_n_out,
                        cfg.lin1_n_out, cfg.rnn_n_out, cfg.lin2_n_out, len(cfg.TAGGING1),
                        window, cfg.TRANS2, cfg.TRANS0, inputs1,
                        W=loaded_W1)
        inputs2 = [input_w, input_p, input_dist, input_v, input_entry_exit_mask, input_vi, input_y]
        net = Progressive('progressive', net1, use_noise, len(vecs), worddim if cfg.use_vecs else cfg.w_n_out_new,
                          len(cfg.POSS), cfg.p_n_out_new, max_dist + 2, cfg.dist_n_out_new,
                          cfg.lin1_n_out_new, cfg.rnn_n_out_new, cfg.lin2_n_out_new, len(cfg.TAGGING),
                          cfg.is_concat, cfg.concat_n_out_ad, cfg.lin1_n_out_ad, cfg.rnn_n_out_ad,
                          window, cfg.TRANS2, cfg.TRANS0, inputs2,
                          W=(([None, None] if cfg.concat_n_out_ad else []) +
                             ([None, None] if cfg.lin1_n_out_ad else []) +
                             ([None, None] if cfg.rnn_n_out_ad else []) +  # None, None, None, None, None, None,
                             [vecs if cfg.use_vecs else None, None, None, None, None, None, None, None,
                              None, None, None, None, None]) if loaded_W is None else loaded_W)
        inputs = [input_w1, input_v1, input_w, input_p, input_dist, input_v, input_entry_exit_mask, input_vi, input_y]
        mapper = cpb2pku_mapper(target, src)
        ws1, vs1 = map_cpb2pku(mapper, ws, vs, lens)
        test_ws1, test_vs1 = map_cpb2pku(mapper, test_ws, test_vs, test_lens)
    nll_cost = net.cal_cost(input_y, l1_lr, l2_lr)
    gparams = [T.grad(nll_cost, param) for param in net.params]
    updates = [(param, param - lr * gparam) for param, gparam in zip(net.params, gparams)]
    train_fun = theano.function(inputs=inputs, outputs=nll_cost, updates=updates, on_unused_input='warn')
    test_fun = theano.function(inputs=inputs, outputs=net.y_pred, on_unused_input='ignore')
    if cfg.training == 2 and pre_f1 == .0:
        use_noise.set_value(0.0)
        pre_precision, pre_recall, pre_f1 = test_prf(test_fun, [test_ws1, test_vs1], test_ws, test_ps, test_dists,
                                                     test_vis, test_vs, test_rs, test_lens,
                                                     test_entry_exit_mask, cfg.TAGGING)
        print(pre_precision, pre_recall, pre_f1)
        use_noise.set_value(1.0)
    max_f1 = pre_f1
    iter = 0
    print('training ...')
    start_time = timeit.default_timer()
    for e_i in range(200):
        if e_i >= cfg.end_epoch: break
        if e_i < pre_e_i: continue
        for sample_i in range(ws.shape[1]):
            if e_i == pre_e_i and sample_i <= pre_sample_i: continue
            if cfg.model_type == 'progressive':
                inputs_data1 = [ws1[-lens[sample_i]:, sample_i, :], vs1[-lens[sample_i]:, sample_i]]
            else:
                inputs_data1 = []
            inputs_data = inputs_data1 + [ws[-lens[sample_i]:, sample_i, :],
                                          ps[-lens[sample_i]:, sample_i, :],
                                          dists[-lens[sample_i]:, sample_i],
                                          vs[-lens[sample_i]:, sample_i],
                                          entry_exit_mask[-lens[sample_i]:, sample_i],
                                          vis[sample_i],
                                          rs[-lens[sample_i]:, sample_i]]
            cost = train_fun(*(inputs_data))
            if (e_i >= cfg.test_from_epoch and sample_i % cfg.test_freq == 0) or sample_i == ws.shape[1] - 1:
                use_noise.set_value(0.0)
                precision, recall, f1 = test_prf(test_fun, [test_ws1, test_vs1], test_ws, test_ps, test_dists,
                                                 test_vis, test_vs, test_rs, test_lens,
                                                 test_entry_exit_mask, cfg.TAGGING)
                print(e_i, sample_i, precision, recall, f1, '  ', cost, 'max hit!' if f1 > max_f1 else '')
                if f1 > max_f1:
                    with open(cfg.dump_name + '.pkl', 'wb') as f:
                        pickle.dump(net.params, f)
                        pickle.dump([e_i, sample_i, precision, recall, f1], f)
                    print('dumped to %s.pkl' % cfg.dump_name)
                max_f1 = max(f1, max_f1)
                use_noise.set_value(1.0)
            iter += 1
        end_time = timeit.default_timer()
        print('%.1f min' % ((end_time - start_time) / 60.))
        start_time = end_time
