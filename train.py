from models import *
import timeit
import pickle
import random
import sys
import theano
import theano.tensor as T
from evaluate import *
from config import Config
import os


def get_weight(filename):
    print('get_weight', end=' ')
    try:
        allparams = pickle.load(open(filename, 'rb'))
    except:
        allparams = pickle._Unpickler(open(filename, 'rb'), encoding='latin1').load()
    try:
        pre_epoch, pre_i, pre_p, pre_r, pre_f = pickle.load(open(filename + '.ei', 'rb'))
    except:
        pre_epoch, pre_i, pre_p, pre_r, pre_f = 0, -1, 0, 0, 0
    rval = []
    for i in range(len(allparams)):
        v = numpy.asarray(allparams[i], dtype=theano.config.floatX)
        rval.append(theano.shared(value=v, borrow=True))
    return rval, pre_epoch, pre_i, pre_p, pre_r, pre_f


def eval_test(ts_t, validate_model, TAGGING):
    answer = [0, 0, 0]
    for index in range(len(ts_t[0])):
        y_pred2 = validate_model(ts_t[0][index], ts_t[1][index], ts_t[3][index], ts_t[4][index], ts_t[5][index])
        s = cal_f1(y_pred2, ts_t[6][index], TAGGING)
        answer[0] += s[0]
        answer[1] += s[1]
        answer[2] += s[2]

    precision = 0
    recall = 0
    f1 = 0
    if answer[1] != 0:
        precision = answer[0] / answer[1]
    if answer[2] != 0:
        recall = answer[0] / answer[2]
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def test_dp(cfg, window=3, L1_reg=0.00, L2_reg=0.0001, n_in_w=100000, n_in_vdx=500):
    print('... building the model')
    n_out_w = cfg.w_n_out
    n_in_p = len(cfg.POSS)
    n_out_p = cfg.p_n_out
    n_out_vdx = cfg.dist_n_out
    n1 = cfg.lin1_n_out
    n2 = cfg.lin2_n_out
    n3 = len(cfg.TAGGING)
    x_w = T.imatrix('x_w')
    x_p = T.imatrix('x_p')
    x_v = T.iscalar('x_v')
    x_s = T.iscalar('x_s')
    x_vdx = T.ivector('x_vdx')
    y = T.ivector('y')

    # W_w contains the pre-trained word embeddings
    # W_w_values = numpy.asarray(embeddings_final,dtype=theano.config.floatX)
    # W_w = theano.shared(value=W_w_values, name='w_w', borrow=True)
    # rval = [W_w, None, None, None, None, None, None, None, None, None, None, None]
    if cfg.training == 2 or cfg.training == 0:
        rval, pre_epoch, pre_i, pre_p, pre_r, pre_f = get_weight(cfg.dump_name)
    else:
        rval, pre_epoch, pre_i, pre_p, pre_r, pre_f = [None, None, None, None, None, None, None, None, None, None, None,
                                                       None, None, None, None, None, None, None], 0, -1, 0, 0, 0
    rng = numpy.random.RandomState(1234)
    if cfg.model_type == 'basic':
        deepnet = DeepNet(rng=rng, input_w=x_w, input_p=x_p, input_vdx=x_vdx, input_v=x_v, input_s=x_s, window=window,
                          n_in_w=n_in_w, n_out_w=n_out_w, n_in_p=n_in_p, n_out_p=n_out_p, n_in_vdx=n_in_vdx,
                          n_out_vdx=n_out_vdx, n1=n1, n2=n2, n3=n3,
                          TRANS0=cfg.TRANS0, TRANS2=cfg.TRANS2, NOT_ENTRY_IDXS=cfg.NOT_ENTRY_IDXS,
                          NOT_EXIT_IDXS=cfg.NOT_EXIT_IDXS, W=rval)
    else:
        n_out_w_new = cfg.w_n_out_new
        n_out_p_new = cfg.p_n_out_new
        n_out_vdx_new = cfg.dist_n_out_new
        n1_new = cfg.lin1_n_out_new
        n2_new = cfg.lin2_n_out_new
        n3 = len(cfg.TAGGING1)
        n3_new = len(cfg.TAGGING)
        net1 = DeepNet(rng=rng, input_w=x_w, input_p=x_p, input_vdx=x_vdx, input_v=x_v, input_s=x_s, window=window,
                       n_in_w=n_in_w, n_out_w=n_out_w, n_in_p=n_in_p, n_out_p=n_out_p,
                       n_in_vdx=n_in_vdx, n_out_vdx=n_out_vdx, n1=n1, n2=n2, n3=n3,
                       TRANS0=cfg.TRANS01, TRANS2=cfg.TRANS21, NOT_ENTRY_IDXS=cfg.NOT_ENTRY_IDXS1,
                       NOT_EXIT_IDXS=cfg.NOT_EXIT_IDXS1, W=get_weight(cfg.dump_name1)[0])
        deepnet = ProgNet(rng=rng, net1=net1, n_outs_ad=[cfg.concat_n_out_ad, cfg.lin1_n_out_ad, cfg.rnn_n_out_ad],
                          input_w=x_w, input_p=x_p,
                          input_vdx=x_vdx, input_v=x_v, input_s=x_s, window=window, n_in_w=n_in_w, n_out_w=n_out_w_new,
                          n_in_p=n_in_p, n_out_p=n_out_p_new, n_in_vdx=n_in_vdx, n_out_vdx=n_out_vdx_new, n1=n1_new,
                          n2=n2_new, n3=n3_new,
                          TRANS0=cfg.TRANS0, TRANS2=cfg.TRANS2, NOT_ENTRY_IDXS=cfg.NOT_ENTRY_IDXS,
                          NOT_EXIT_IDXS=cfg.NOT_EXIT_IDXS, W=rval)
    cost = deepnet.cal_cost(y) + L2_reg * deepnet.L2_sqr + L1_reg * deepnet.L1
    gparams = []
    for param in deepnet.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    updates = []
    for param, gparam in zip(deepnet.params, gparams):
        updates.append((param, param - cfg.learning_rate * gparam))

    if cfg.training != 0:
        train_model = theano.function(inputs=[x_w, x_p, x_vdx, x_v, x_s, y], outputs=cost, updates=updates)
    validate_model = theano.function(inputs=[x_w, x_p, x_vdx, x_v, x_s], outputs=deepnet.y_pred2)
    print([param.eval().shape for param in deepnet.params])

    # for the first time, should use get_dataset to get training and testing data
    ts = pickle.load(open(cfg.train_pkl, 'rb'))
    ts_t = pickle.load(open(cfg.test_pkl, 'rb'))
    for i in range(len(ts_t[0])):
        ts_t[0][i].pop(ts_t[5][i])
        ts_t[1][i].pop(ts_t[5][i])
        ts_t[3][i].pop(ts_t[5][i])
    for i in range(len(ts[0])):
        ts[0][i].pop(ts[5][i])
        ts[1][i].pop(ts[5][i])
        ts[3][i].pop(ts[5][i])
    if cfg.training != 1:
        pre_precision, pre_recall, pre_f = eval_test(ts_t, validate_model, cfg.TAGGING)
        print('f1 score %.3f%%' % (pre_f * 100))
        if cfg.training == 0: exit(0)
    print('... training')
    bs = [ii for ii in range(len(ts[0]))]
    n_samples = len(bs)
    best_f1 = pre_f
    epoch = 0
    show_freq = max(1, int(cfg.test_freq / 20))
    print(pre_epoch, pre_i, pre_p, pre_r, pre_f)
    while epoch <= cfg.end_epoch:
        if epoch < pre_epoch:
            epoch += 1
            continue
        start_time = timeit.default_timer()
        random.shuffle(bs)
        for i in range(len(bs)):
            sys.stdout.write((str(epoch) + ' ' + str(i) + ' \n') if i % show_freq == 0 else '')
            if epoch == pre_epoch and i <= pre_i: continue
            index = bs[i]
            train_model(ts[0][index], ts[1][index], ts[3][index], ts[4][index], ts[5][index], ts[6][index])
            if ((i + 1) % cfg.test_freq == 0 and epoch >= cfg.test_from_epoch) or i == n_samples - 1:
                print(epoch, i, end=' ')
                precision, recall, f1 = eval_test(ts_t, validate_model, cfg.TAGGING)
                print('f1 score %.3f%%' % (f1 * 100) + (' max hit!' if best_f1 < f1 else ''))
                if best_f1 < f1:
                    pickle.dump([param.get_value() for param in deepnet.params], open(cfg.dump_name, 'wb'))
                    pickle.dump([epoch, i, precision, recall, f1], open(cfg.dump_name + '.ei', 'wb'))
                    best_f1 = f1
                with open(cfg.dump_name + '.txt', 'a') as fileout:
                    fileout.write(str(epoch) + ' ' + str(i) + ' f1 score: ' + str(f1 * 100) + '\n')
        end_time = timeit.default_timer()
        print(' %d min' % ((end_time - start_time) / 60))
        epoch += 1


if __name__ == '__main__':
    cfg = Config('-t 1')
    cfg.dump_name = 'model/params_'
    cfg.dumpinit()
    test_dp(cfg)
