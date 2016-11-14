# coding: utf-8
import numpy
import theano
import theano.sandbox.rng_mrg as rng_mrg
import theano.tensor as T


class EmbedLayer:
    def __init__(self, rng, n_in, n_out,
                 name, input_e, W_e=None):
        if isinstance(W_e, list):  # init embedding by list
            W_e = theano.shared(value=numpy.asarray(W_e, dtype=theano.config.floatX), name=name + 'W_e', borrow=True)
        elif isinstance(W_e, numpy.ndarray):  # init embedding by ndarray
            W_e = theano.shared(value=W_e, name=name + 'W_e', borrow=True)
            print(name + 'W_e: init by numpy.ndarray' + str(W_e.get_value().shape))
        elif W_e is None:  # randomly init embedding
            W_e_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)),
                                                   high=numpy.sqrt(6. / (n_in + n_out)),
                                                   size=(n_in, n_out)), dtype=theano.config.floatX)
            W_e = theano.shared(value=W_e_values, name=name + 'W_e', borrow=True)

        self.input_e = input_e
        self.W_e = W_e
        self.params = [self.W_e]
        if self.input_e.ndim == 2:
            self.output_e = T.reshape(self.W_e[self.input_e.flatten()], (input_e.shape[0], input_e.shape[1], n_out))
        elif self.input_e.ndim == 3:
            self.output_e = T.reshape(self.W_e[self.input_e.flatten()],
                                      (input_e.shape[0], input_e.shape[1], n_out * input_e.shape[2]))
        else:
            raise NotImplementedError()
        # output dim: timesteps*n_samples*emb_dim

        self.l1 = abs(self.W_e).sum()
        self.l2 = (self.W_e ** 2).sum()
        # print('embedding layer params:%d' % (n_in * n_out))


class LinearLayer:
    def __init__(self, rng, n_in, n_out,
                 name, input_l, activation, W=None, b=None):
        if isinstance(W, list):  # init W
            W = theano.shared(value=numpy.asarray(W, dtype=theano.config.floatX), name=name + 'W', borrow=True)
        elif isinstance(W, numpy.ndarray):
            W = theano.shared(value=W, name=name + 'W', borrow=True)
            print(name + 'W: init by numpy.ndarray' + str(W.get_value().shape))

        if isinstance(b, list):  # init b
            b = theano.shared(value=numpy.asarray(b, dtype=theano.config.floatX), name=name + 'b', borrow=True)
        elif isinstance(b, numpy.ndarray):
            b = theano.shared(value=b, name=name + 'b', borrow=True)
            print(name + 'b: init by numpy.ndarray' + str(b.get_value().shape))

        if W is None:
            # for T.tanh
            if activation == 'sigmoid':  # theano.tensor.nnet.sigmoid:
                W_values = 4. * numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)),
                                                          high=numpy.sqrt(6. / (n_in + n_out)),
                                                          size=(n_in, n_out)), dtype=theano.config.floatX)
            else:
                W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)),
                                                     high=numpy.sqrt(6. / (n_in + n_out)),
                                                     size=(n_in, n_out)), dtype=theano.config.floatX)

            W = theano.shared(value=W_values, name=name + 'w', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            if activation == 'sigmoid':  # theano.tensor.nnet.sigmoid:
                b_values = -2 * numpy.ones((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=name + 'b', borrow=True)
        # print('linear layer params:%d' % (n_in * n_out + n_out))

        self.input_l = input_l
        self.W = W
        self.b = b

        self.params = [self.W, self.b]
        # linear output

        lin_output = T.dot(self.input_l, self.W) + self.b

        if activation == 'hardtanh':
            self.output_l = T.clip(lin_output, -1, 1)
        elif activation == 'softmax':  # not correct for 3D tensors
            if lin_output.ndim == 2:
                self.output_l = T.nnet.softmax(lin_output)
            elif lin_output.ndim == 3:
                self.output_l = T.reshape(T.nnet.softmax(T.reshape(lin_output,
                                                                   [input_l.shape[0] * input_l.shape[1], n_out])),
                                          [input_l.shape[0], input_l.shape[1], n_out])
        elif activation == 'tanh':
            self.output_l = T.tanh(lin_output)
        elif activation == 'relu':
            self.output_l = T.nnet.relu(lin_output)
        elif activation == 'sigmoid':
            self.output_l = T.nnet.sigmoid(lin_output)  # TODO:Test this and its initialization
        elif activation == 'none':
            self.output_l = lin_output

        self.l1 = abs(self.W).sum()
        self.l2 = (self.W ** 2).sum()


class DropoutLayer:
    def __init__(self, p, input_d, use_noise):
        self.input_d = input_d
        self.trng = rng_mrg.MRG_RandomStreams(394857)
        self.output_d = T.switch(use_noise, (input_d * self.trng.binomial(input_d.shape, p=p, n=1, )), input_d)


class LSTMLayer:
    def __init__(self, rng, n_in, n_out,
                 input_lstm, input_mask,
                 bidirect, name, activation, Ws=None, Us=None, bs=None):

        if bidirect:
            n_dim = n_out / 2
            if int(n_out / 2) * 2 != n_out:
                print('bidirectional model must have an even n_dim in LSTM layer!')
                input()
        else:
            n_dim = n_out
        # init by input ###############
        # init Ws
        if isinstance(Ws, list):
            Ws = theano.shared(value=numpy.asarray(Ws, dtype=theano.config.floatX), name=name + 'Ws', borrow=True)
        elif isinstance(Ws, numpy.ndarray):
            Ws = theano.shared(value=Ws, name=name + 'Ws', borrow=True)
            print(name + 'Ws: init by numpy.ndarray' + str(Ws.get_value().shape))
        # init Us
        if isinstance(Us, list):
            Us = theano.shared(value=numpy.asarray(Us, dtype=theano.config.floatX), name=name + 'Us', borrow=True)
        elif isinstance(Us, numpy.ndarray):
            Us = theano.shared(value=Us, name=name + 'Us', borrow=True)
            print(name + 'Us: init by numpy.ndarray' + str(Us.get_value().shape))
        # init bs
        if isinstance(bs, list):
            bs = theano.shared(value=numpy.asarray(bs, dtype=theano.config.floatX), name=name + 'bs', borrow=True)
        elif isinstance(bs, numpy.ndarray):
            bs = theano.shared(value=bs, name=name + 'bs', borrow=True)
            print(name + 'bs: init by numpy.ndarray' + str(bs.get_value().shape))
        # randomly init ###############
        # randomly init Ws: 2 for bidirectional, 4 for 4 gates in LSTM
        if Ws is None:
            W_v = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_dim + n_in)),
                                            high=numpy.sqrt(6. / (n_dim + n_in)),
                                            size=(2 if bidirect else 1, 4, n_in, n_dim)), dtype=theano.config.floatX)
            Ws = theano.shared(value=W_v, name=name + 'ws', borrow=True)
        # randomly init Us: 2 for bidirectional, 4 for 4 gates in LSTM
        if Us is None:
            U_v = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_dim + n_dim)),
                                            high=numpy.sqrt(6. / (n_dim + n_dim)),
                                            size=(2 if bidirect else 1, 4, n_dim, n_dim)), dtype=theano.config.floatX)
            Us = theano.shared(value=U_v, name=name + 'us', borrow=True)
        # randomly init bs: 2 for bidirectional, 4 for 4 gates in LSTM
        if bs is None:
            b_v = numpy.zeros((2 if bidirect else 1, 4, n_dim))
            bs = theano.shared(value=b_v, name=name + 'bs', borrow=True)

        # print('linear layer params:%d' % ((2 if bidirect else 1) * 4 * (n_in + n_dim + 1) * n_dim))

        self.input_lstm = input_lstm
        self.input_mask = input_mask
        self.Ws = Ws
        self.Us = Us
        self.bs = bs
        self.params = [self.Ws, self.Us, self.bs]

        self.l1 = abs(self.Us).sum()
        self.l2 = (self.Us ** 2).sum()

        def gates_ct_ht(x, mask, h, c, W, U, b):
            gate0 = T.dot(x, W[0]) + T.dot(h, U[0]) + b[0]
            gate1 = T.dot(x, W[1]) + T.dot(h, U[1]) + b[1]
            gate2 = T.dot(x, W[2]) + T.dot(h, U[2]) + b[2]
            gate3 = T.dot(x, W[3]) + T.dot(h, U[3]) + b[3]
            ct = T.nnet.sigmoid(gate0) * T.tanh(gate1) + T.nnet.sigmoid(gate2) * c  # dim = n_f
            ct = T.cast(mask[:, None] * ct + (1. - mask)[:, None] * c, theano.config.floatX)
            ht = T.nnet.sigmoid(gate3) * T.tanh(ct)  # dim = n_f
            ht = T.cast(mask[:, None] * ht + (1. - mask)[:, None] * h, theano.config.floatX)
            return [ht, ct]

        # forward LSTM
        hids1, updates = theano.scan(fn=gates_ct_ht, sequences=[self.input_lstm, self.input_mask],
                                     outputs_info=[T.zeros((input_lstm.shape[1], n_dim,), dtype=theano.config.floatX),
                                                   T.zeros((input_lstm.shape[1], n_dim,), dtype=theano.config.floatX)],
                                     non_sequences=[self.Ws[0], self.Us[0], self.bs[0]])
        # backward LSTM
        if not bidirect:
            lin_output = hids1[0]
        else:
            hids2, updates = theano.scan(fn=gates_ct_ht, sequences=[self.input_lstm[::-1], self.input_mask][::-1],
                                         outputs_info=[
                                             T.zeros((input_lstm.shape[1], n_dim,), dtype=theano.config.floatX),
                                             T.zeros((input_lstm.shape[1], n_dim,), dtype=theano.config.floatX)],
                                         non_sequences=[self.Ws[1], self.Us[1], self.bs[1]])
            lin_output = T.concatenate([hids1[0], hids2[0][::-1]], axis=2)

        if activation == 'hardtanh':
            self.output_lstm = T.clip(lin_output, -1, 1)
        elif activation == 'softmax':
            self.output_lstm = T.nnet.softmax(lin_output)
        elif activation == 'tanh':
            self.output_lstm = T.tanh(lin_output)
        elif activation == 'sigmoid':
            self.output_lstm = T.nnet.sigmoid(lin_output)  # TODO:Test this and its initialization
        elif activation == 'none':
            self.output_lstm = lin_output
        else:
            raise NotImplementedError('please specify you activation func')


# ref: https://github.com/dennybritz/rnn-tutorial-gru-lstm/blob/master/gru_theano.py
# http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
class GRULayer:  # TODO: test GRU

    def __init__(self, rng, n_in, n_out,
                 input_gru, input_mask,
                 bidirect, name, activation, Ws=None, Us=None, bs=None):

        if bidirect:
            n_dim = n_out / 2
            if int(n_out / 2) * 2 != n_out:
                print('bidirectional model must have an even n_dim in LSTM layer!')
                input()
        else:
            n_dim = n_out
        # ############### init by input
        # init Ws
        if isinstance(Ws, list):
            Ws = theano.shared(value=numpy.asarray(Ws, dtype=theano.config.floatX), name=name + 'Ws', borrow=True)
        elif isinstance(Ws, numpy.ndarray):
            Ws = theano.shared(value=Ws, name=name + 'Ws', borrow=True)
            print(name + 'Ws: init by numpy.ndarray' + str(Ws.get_value().shape))
        # init Us
        if isinstance(Us, list):
            Us = theano.shared(value=numpy.asarray(Us, dtype=theano.config.floatX), name=name + 'Us', borrow=True)
        elif isinstance(Us, numpy.ndarray):
            Us = theano.shared(value=Us, name=name + 'Us', borrow=True)
            print(name + 'Us: init by numpy.ndarray' + str(Us.get_value().shape))
        # init bs
        if isinstance(bs, list):
            bs = theano.shared(value=numpy.asarray(bs, dtype=theano.config.floatX), name=name + 'bs', borrow=True)
        elif isinstance(bs, numpy.ndarray):
            bs = theano.shared(value=bs, name=name + 'bs', borrow=True)
            print(name + 'bs: init by numpy.ndarray' + str(bs.get_value().shape))
        # ############### randomly init
        # randomly init Ws: 2 for bidirectional, 3 for 3 gates in GRU
        if Ws is None:
            W_v = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_dim + n_in)),
                                            high=numpy.sqrt(6. / (n_dim + n_in)),
                                            size=(2 if bidirect else 1, 3, n_in, n_dim)), dtype=theano.config.floatX)
            Ws = theano.shared(value=W_v, name=name + 'ws', borrow=True)
        # randomly init Us: 2 for bidirectional, 3 for 3 gates in GRU
        if Us is None:
            U_v = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_dim + n_dim)),
                                            high=numpy.sqrt(6. / (n_dim + n_dim)),
                                            size=(2 if bidirect else 1, 3, n_dim, n_dim)), dtype=theano.config.floatX)
            Us = theano.shared(value=U_v, name=name + 'us', borrow=True)
        # randomly init bs: 2 for bidirectional, 3 for 3 gates in GRU
        if bs is None:
            b_v = numpy.zeros((2 if bidirect else 1, 3, n_dim))
            bs = theano.shared(value=b_v, name=name + 'bs', borrow=True)
        # print('linear layer params:%d' % ((2 if bidirect else 1) * 3 * (n_in + n_dim + 1) * n_dim))

        self.input_gru = input_gru
        self.input_mask = input_mask
        self.Ws = Ws
        self.Us = Us
        self.bs = bs
        self.params = [self.Ws, self.Us, self.bs]
        self.l1 = abs(self.Us).sum()
        self.l2 = (self.Us ** 2).sum()

        def gates_ct_ht(x, mask, h, c, W, U, b):
            zt = T.dot(x, W[0]) + T.dot(h, U[0]) + b[0]  # update
            rt = T.dot(x, W[1]) + T.dot(h, U[1]) + b[1]  # reset
            ct = T.tanh(T.dot(x, W[2]) + T.dot(h * rt, U[2]) + b[2])  # dim = n_f
            ct = T.cast(mask[:, None] * ct + (1. - mask)[:, None] * c, theano.config.floatX)  # dim = n_f
            ht = (T.ones_like(zt) - zt) * ct + zt * h
            ht = T.cast(mask[:, None] * ht + (1. - mask)[:, None] * h, theano.config.floatX)
            return [ht, ct]

        # forward GRU
        hids1, updates = theano.scan(fn=gates_ct_ht, sequences=[self.input_gru, self.input_mask],
                                     outputs_info=[T.zeros((input_gru.shape[1], n_dim,), dtype=theano.config.floatX),
                                                   T.zeros((input_gru.shape[1], n_dim,), dtype=theano.config.floatX)],
                                     non_sequences=[self.Ws[0], self.Us[0], self.bs[0]])
        # backward GRU
        if not bidirect:
            lin_output = hids1[0]
        else:
            hids2, updates = theano.scan(fn=gates_ct_ht, sequences=[self.input_gru[::-1], self.input_mask][::-1],
                                         outputs_info=[
                                             T.zeros((input_gru.shape[1], n_dim,), dtype=theano.config.floatX),
                                             T.zeros((input_gru.shape[1], n_dim,), dtype=theano.config.floatX)],
                                         non_sequences=[self.Ws[1], self.Us[1], self.bs[1]])
            lin_output = T.concatenate([hids1[0], hids2[0][::-1]], axis=2)

        if activation == 'hardtanh':
            self.output_gru = T.clip(lin_output, -1, 1)
        elif activation == 'softmax':
            self.output_gru = T.nnet.softmax(lin_output)
        elif activation == 'tanh':
            self.output_gru = T.tanh(lin_output)
        elif activation == 'sigmoid':
            self.output_gru = T.nnet.sigmoid(lin_output)  # TODO:Test this and its initialization
        elif activation == 'none':
            self.output_gru = lin_output
