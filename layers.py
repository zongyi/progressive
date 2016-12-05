import theano
import theano.tensor as T
import numpy


class LookuptableLayer(object):
    def __init__(self, rng, name, input_w, input_p, input_vdx, input_v, n_in_w, n_out_w, n_in_p, n_out_p, n_in_vdx,
                 n_out_vdx, W_w=None, W_p=None, W_vdx=None):
        self.input_w = input_w
        self.input_p = input_p
        self.input_vdx = input_vdx
        self.input_v = input_v
        # word lookup table
        if W_w is None:
            W_w_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in_w + 3 + n_out_w)),
                high=numpy.sqrt(6. / (n_in_w + 3 + n_out_w)),
                size=(n_in_w + 3, n_out_w)), dtype=theano.config.floatX)
            W_w = theano.shared(value=W_w_values, name=name + 'w_w', borrow=True)

        # pos tag lookup table
        if W_p is None:
            W_p_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in_p + 2 + n_out_p)),
                high=numpy.sqrt(6. / (n_in_p + 2 + n_out_p)),
                size=(n_in_p + 2, n_out_p)), dtype=theano.config.floatX)
            W_p = theano.shared(value=W_p_values, name=name + 'w_p', borrow=True)

        # distance to verb lookuptable
        if W_vdx is None:
            W_vdx_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in_vdx + 2 + n_out_vdx)),
                high=numpy.sqrt(6. / (n_in_vdx + 2 + n_out_vdx)),
                size=(n_in_vdx + 2, n_out_vdx)), dtype=theano.config.floatX)
            W_vdx = theano.shared(value=W_vdx_values, name=name + 'w_vdx', borrow=True)

        self.W_w = W_w
        self.W_p = W_p
        self.W_vdx = W_vdx
        self.params = [self.W_w, self.W_p, self.W_vdx]

        # word feature, pos tag feature, verb feature, distance to verb feature
        lin_output_w = T.flatten(self.W_w[self.input_w], 2)
        lin_output_p = T.flatten(self.W_p[self.input_p], 2)
        lin_output_v = self.W_w[self.input_v]
        lin_output_vdx = self.W_vdx[self.input_vdx]

        temp = T.concatenate([lin_output_w, lin_output_p, lin_output_vdx], axis=1)
        lin_output_sen, updates = theano.scan(lambda t, v: T.concatenate([t, v], axis=0), sequences=temp,
                                              non_sequences=lin_output_v)

        # concatenate all above features
        self.output = lin_output_sen


class LinearLayer(object):
    def __init__(self, rng, name, input, n_in, n_out, W=None, b=None, activation=None):
        self.input = input
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)

            W = theano.shared(value=W_values, name=name + 'w', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=name + 'b', borrow=True)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        lin_output = T.dot(self.input, self.W) + self.b
        if activation == 'hardtanh':
            self.output = T.clip(lin_output, -1, 1)
        elif activation == 'softmax':
            self.output = T.nnet.softmax(lin_output)
        else:
            self.output = lin_output


# bidirectional rnn layer with LSTM (you can refer to deeplearning.net for LSTM)
class RNNlayer(object):
    def __init__(self, rng, name, input, n_in, n_out, n_f, Ws=None, Us=None, bs=None, W=None, b=None, activation=None):
        self.input = input

        # 2 for bidirectional, 4 for 4 gates in LSTM
        if Ws is None:
            W_v = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_f + n_in)),
                high=numpy.sqrt(6. / (n_f + n_in)),
                size=(2, 4, n_in, n_f)), dtype=theano.config.floatX)
            Ws = theano.shared(value=W_v, name=name + 'ws', borrow=True)

        if Us is None:
            W_v = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_f + n_f)),
                high=numpy.sqrt(6. / (n_f + n_f)),
                size=(2, 4, n_f, n_f)), dtype=theano.config.floatX)
            Us = theano.shared(value=W_v, name=name + 'us', borrow=True)

        if bs is None:
            b_v = numpy.zeros((2, 4, n_f), dtype=theano.config.floatX)
            bs = theano.shared(value=b_v, name=name + 'bs', borrow=True)

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (2 * n_f + n_out)),
                high=numpy.sqrt(6. / (2 * n_f + n_out)),
                size=(2 * n_f, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name=name + 'w', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=name + 'b', borrow=True)

        self.Ws = Ws
        self.Us = Us
        self.bs = bs
        self.W = W
        self.b = b
        self.params = [self.Ws, self.Us, self.bs, self.W, self.b]

        def f(x, h, c, W, U, b):
            info = T.dot(x, W) + T.dot(h, U) + b
            ct = T.nnet.sigmoid(info[0]) * T.tanh(info[1]) + T.nnet.sigmoid(info[2]) * c
            ht = T.nnet.sigmoid(info[3]) * T.tanh(ct)
            return [ht, ct]

        # forward LSTM
        hids1, updates = theano.scan(fn=f, sequences=self.input,
                                     outputs_info=[numpy.zeros((n_f,), dtype=theano.config.floatX),
                                                   numpy.zeros((n_f,), dtype=theano.config.floatX)],
                                     non_sequences=[self.Ws[0], self.Us[0], self.bs[0]])
        # backward LSTM
        hids2, updates = theano.scan(fn=f, sequences=self.input[::-1],
                                     outputs_info=[numpy.zeros((n_f,), dtype=theano.config.floatX),
                                                   numpy.zeros((n_f,), dtype=theano.config.floatX)],
                                     non_sequences=[self.Ws[1], self.Us[1], self.bs[1]])

        # concatenate the features from both direction and apply a linear transformation
        lin_output = T.dot(T.concatenate([hids1[0], hids2[0][::-1]], axis=1), self.W) + self.b

        if activation == 'hardtanh':
            self.output = T.clip(lin_output, -1, 1)
        elif activation == 'softmax':
            self.output = T.nnet.softmax(lin_output)
        elif activation == 'tanh':
            self.output = T.tanh(lin_output)
        else:
            self.output = lin_output


class AdapterLayer:
    def __init__(self, rng, name, n_in, n_out,
                 input, A=None, U=None):
        if isinstance(A, float) or isinstance(A, numpy.float32):
            A = theano.shared(value=numpy.float32(A), name=name + 'A', borrow=True)
            print(name + ' A: init by numpy.float32')

        if isinstance(A, list) or isinstance(A, numpy.ndarray):
            U = theano.shared(value=numpy.asarray(U, dtype=theano.config.floatX), name=name + 'U', borrow=True)
            print(name + ' U: init by numpy.ndarray' + str(U.get_value().shape))

        if A is None:
            A_values = numpy.float32(4. * (numpy.random.rand() * 0.0002 - 0.0001))
            A = theano.shared(value=A_values, name=name + 'A', borrow=True)

        if U is None:
            U_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out) * 0.0001),
                                                 high=numpy.sqrt(6. / (n_in + n_out) * 0.0001),
                                                 size=(n_in, n_out)), dtype=theano.config.floatX)
            U = theano.shared(value=U_values, name=name + 'U', borrow=True)
        # print('linear layer params:%d' % (n_in * n_out + n_out))

        self.input = input
        self.A = A
        self.U = U

        self.params = [self.A, self.U]
        # linear output

        self.output = T.dot(T.nnet.sigmoid(self.A * self.input), self.U)

        self.l1 = abs(self.U).sum()
        self.l2 = (self.U ** 2).sum()



