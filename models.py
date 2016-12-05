from layers import *
from const import *


class DeepNet(object):
    def __init__(self, rng, input_w, input_p, input_vdx, input_v, input_s, window, n_in_w, n_out_w, n_in_p, n_out_p,
                 n_in_vdx, n_out_vdx, n1, n2, n3,
                 TRANS0, TRANS2, NOT_ENTRY_IDXS, NOT_EXIT_IDXS,
                 W=[None, None, None, None, None, None, None, None, None, None, None, None]):
        self.lookup_n_out = window * (n_out_w + n_out_p) + n_out_vdx + n_out_w
        self.lin1_n_out = n1
        self.rnn_n_out = n2
        self.lin3_n_out = n3
        self.lookuptablelayer = LookuptableLayer(rng=rng, name='dp_lk', input_w=input_w, input_p=input_p,
                                                 input_vdx=input_vdx, input_v=input_v, n_in_w=n_in_w, n_out_w=n_out_w,
                                                 n_in_p=n_in_p, n_out_p=n_out_p, n_in_vdx=n_in_vdx, n_out_vdx=n_out_vdx,
                                                 W_w=W[0], W_p=W[1], W_vdx=W[2])
        self.linearlayer1 = LinearLayer(rng=rng, name='dp_l1', input=self.lookuptablelayer.output,
                                        n_in=self.lookup_n_out, n_out=n1, W=W[3], b=W[4])
        self.linearlayer2 = RNNlayer(rng=rng, name='dp_l2', input=self.linearlayer1.output, n_in=n1, n_out=n2, n_f=n2,
                                     Ws=W[5], Us=W[6], bs=W[7], W=W[8], b=W[9], activation='tanh')
        self.linearlayer3 = LinearLayer(rng=rng, name='dp_l3', input=self.linearlayer2.output, n_in=n2, n_out=n3,
                                        W=W[10], b=W[11], activation=None)
        self.params = self.lookuptablelayer.params + self.linearlayer1.params + self.linearlayer2.params + self.linearlayer3.params

        # decode
        ###################################
        def mask(output, i, s1, s2):
            return output + T.switch(T.eq(i, 0), NOT_ENTRY_IDXS, 0) + T.switch(T.eq(i, s1), NOT_ENTRY_IDXS,
                                                                               0) + T.switch(T.eq(i + 1, s1),
                                                                                             NOT_EXIT_IDXS,
                                                                                             0) + T.switch(
                T.eq(i + 1, s2), NOT_EXIT_IDXS, 0)

        self.p_y_given_x, updates = theano.scan(fn=mask, sequences=[self.linearlayer3.output,
                                                                    T.arange(self.linearlayer3.output.shape[0])],
                                                non_sequences=[input_s, self.linearlayer3.output.shape[0]])

        self.y_pred1 = T.argmax(self.p_y_given_x, axis=1)

        def decode(p_y_given_x, label):
            return [T.max(label + TRANS2, axis=1) + p_y_given_x, T.argmax(label + TRANS2, axis=1)]

        def jump(arrange, tag, tags):
            return tags[-arrange - 1][tag]

        score, updates = theano.scan(fn=decode, sequences=self.p_y_given_x[1:],
                                     outputs_info=[self.p_y_given_x[0], None])
        max_tag = T.argmax(score[0][-1])
        seq, updates = theano.scan(fn=jump, sequences=T.arange(score[1].shape[0]), outputs_info=max_tag,
                                   non_sequences=score[1])
        seq_reverse = seq[::-1]
        final_seq = T.concatenate([seq_reverse, [max_tag]])
        self.y_pred2 = final_seq
        ####################################

        self.score = score[0][-1][max_tag]

        # partition function
        ####################################
        def logadd(p_y_given_x, i, label, s):
            maxnum = T.max(label)
            return p_y_given_x + maxnum + T.log(
                T.sum(T.exp(label + T.switch(T.eq(i + 1, s), TRANS0, TRANS2) - maxnum), axis=1))

        paths, updates = theano.scan(fn=logadd,
                                     sequences=[self.p_y_given_x[1:], T.arange(self.p_y_given_x[1:].shape[0])],
                                     outputs_info=self.p_y_given_x[0], non_sequences=input_s)
        maxnum = T.max(paths[-1])
        self.allscore = T.log(T.sum(T.exp(paths[-1] - maxnum))) + maxnum
        ####################################

        # regularization
        self.L1 = abs(self.lookuptablelayer.W_w).sum() \
                  + abs(self.lookuptablelayer.W_vdx).sum() \
                  + abs(self.lookuptablelayer.W_p).sum() \
                  + abs(self.linearlayer1.W).sum() \
                  + abs(self.linearlayer2.W).sum() \
                  + abs(self.linearlayer3.W).sum()

        self.L2_sqr = (self.lookuptablelayer.W_w ** 2).sum() \
                      + (self.lookuptablelayer.W_vdx ** 2).sum() \
                      + (self.lookuptablelayer.W_p ** 2).sum() \
                      + (self.linearlayer1.W ** 2).sum() \
                      + (self.linearlayer2.W ** 2).sum() \
                      + (self.linearlayer3.W ** 2).sum()

    # cost function
    def cal_cost(self, y):
        y_score = T.sum(self.p_y_given_x[T.arange(y.shape[0]), y])
        return -(y_score - self.allscore)


class ProgNet(object):
    def __init__(self, net1, n_outs_ad, rng, input_w, input_p, input_vdx, input_v, input_s, window, n_in_w, n_out_w,
                 n_in_p, n_out_p, n_in_vdx, n_out_vdx, n1, n2, n3,
                 TRANS0, TRANS2, NOT_ENTRY_IDXS, NOT_EXIT_IDXS,
                 W=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                    None, None]):
        concat_n_out_ad, lin1_n_out_ad, rnn_n_out_ad = n_outs_ad
        if concat_n_out_ad: self.adapter0 = AdapterLayer(rng=rng, name='pr_ad0', input=net1.lookuptablelayer.output,
                                                         n_in=net1.lookup_n_out, n_out=concat_n_out_ad, A=W[12],
                                                         U=W[13])
        if lin1_n_out_ad: self.adapter1 = AdapterLayer(rng=rng, name='pr_ad1', input=net1.linearlayer1.output,
                                                       n_in=net1.lin1_n_out, n_out=lin1_n_out_ad, A=W[14], U=W[15])
        if rnn_n_out_ad: self.adapter2 = AdapterLayer(rng=rng, name='pr_ad2', input=net1.linearlayer2.output,
                                                      n_in=net1.rnn_n_out, n_out=rnn_n_out_ad, A=W[16], U=W[17])
        self.lookup_n_out = window * (n_out_w + n_out_p) + n_out_vdx + n_out_w
        self.lookuptablelayer = LookuptableLayer(rng=rng, name='pr_lk', input_w=input_w, input_p=input_p,
                                                 input_vdx=input_vdx, input_v=input_v, n_in_w=n_in_w, n_out_w=n_out_w,
                                                 n_in_p=n_in_p, n_out_p=n_out_p, n_in_vdx=n_in_vdx, n_out_vdx=n_out_vdx,
                                                 W_w=W[0], W_p=W[1], W_vdx=W[2])
        self.linearlayer1 = LinearLayer(rng=rng, name='pr_l1',
                                        input=T.concatenate([self.lookuptablelayer.output, self.adapter0.output],
                                                            axis=1) if concat_n_out_ad else self.lookuptablelayer.output,
                                        n_in=concat_n_out_ad + self.lookup_n_out, n_out=n1, W=W[3], b=W[4])
        self.linearlayer2 = RNNlayer(rng=rng, name='pr_l2',
                                     input=T.concatenate([self.linearlayer1.output, self.adapter1.output],
                                                         axis=1) if lin1_n_out_ad else self.linearlayer1.output,
                                     n_in=lin1_n_out_ad + n1, n_out=n2, n_f=n2, Ws=W[5], Us=W[6], bs=W[7], W=W[8],
                                     b=W[9], activation='tanh')
        self.linearlayer3 = LinearLayer(rng=rng, name='pr_l3',
                                        input=T.concatenate([self.linearlayer2.output, self.adapter2.output],
                                                            axis=1) if rnn_n_out_ad else self.linearlayer2.output,
                                        n_in=rnn_n_out_ad + n2, n_out=n3, W=W[10], b=W[11], activation=None)
        self.params = self.lookuptablelayer.params + self.linearlayer1.params + self.linearlayer2.params + self.linearlayer3.params + \
                      (self.adapter0.params if concat_n_out_ad else []) + (
                          self.adapter1.params if lin1_n_out_ad else []) + (
                          self.adapter2.params if rnn_n_out_ad else [])

        # decode
        ###################################
        def mask(output, i, s1, s2):
            return output + T.switch(T.eq(i, 0), NOT_ENTRY_IDXS, 0) + T.switch(T.eq(i, s1), NOT_ENTRY_IDXS,
                                                                               0) + T.switch(T.eq(i + 1, s1),
                                                                                             NOT_EXIT_IDXS,
                                                                                             0) + T.switch(
                T.eq(i + 1, s2), NOT_EXIT_IDXS, 0)

        self.p_y_given_x, updates = theano.scan(fn=mask, sequences=[self.linearlayer3.output,
                                                                    T.arange(self.linearlayer3.output.shape[0])],
                                                non_sequences=[input_s, self.linearlayer3.output.shape[0]])

        self.y_pred1 = T.argmax(self.p_y_given_x, axis=1)

        def decode(p_y_given_x, label):
            return [T.max(label + TRANS2, axis=1) + p_y_given_x, T.argmax(label + TRANS2, axis=1)]

        def jump(arrange, tag, tags):
            return tags[-arrange - 1][tag]

        score, updates = theano.scan(fn=decode, sequences=self.p_y_given_x[1:],
                                     outputs_info=[self.p_y_given_x[0], None])
        max_tag = T.argmax(score[0][-1])
        seq, updates = theano.scan(fn=jump, sequences=T.arange(score[1].shape[0]), outputs_info=max_tag,
                                   non_sequences=score[1])
        seq_reverse = seq[::-1]
        final_seq = T.concatenate([seq_reverse, [max_tag]])
        self.y_pred2 = final_seq
        ####################################

        self.score = score[0][-1][max_tag]

        # partition function
        ####################################
        def logadd(p_y_given_x, i, label, s):
            maxnum = T.max(label)
            return p_y_given_x + maxnum + T.log(
                T.sum(T.exp(label + T.switch(T.eq(i + 1, s), TRANS0, TRANS2) - maxnum), axis=1))

        paths, updates = theano.scan(fn=logadd,
                                     sequences=[self.p_y_given_x[1:], T.arange(self.p_y_given_x[1:].shape[0])],
                                     outputs_info=self.p_y_given_x[0], non_sequences=input_s)
        maxnum = T.max(paths[-1])
        self.allscore = T.log(T.sum(T.exp(paths[-1] - maxnum))) + maxnum
        ####################################

        # regularization
        self.L1 = abs(self.lookuptablelayer.W_w).sum() \
                  + abs(self.lookuptablelayer.W_vdx).sum() \
                  + abs(self.lookuptablelayer.W_p).sum() \
                  + abs(self.linearlayer1.W).sum() \
                  + abs(self.linearlayer2.W).sum() \
                  + abs(self.linearlayer3.W).sum() + \
                  (self.adapter0.l1 if concat_n_out_ad else numpy.float32(0)) + (
                      self.adapter1.l1 if lin1_n_out_ad else numpy.float32(0)) + (
                      self.adapter2.l1 if rnn_n_out_ad else numpy.float32(0))

        self.L2_sqr = (self.lookuptablelayer.W_w ** 2).sum() \
                      + (self.lookuptablelayer.W_vdx ** 2).sum() \
                      + (self.lookuptablelayer.W_p ** 2).sum() \
                      + (self.linearlayer1.W ** 2).sum() \
                      + (self.linearlayer2.W ** 2).sum() \
                      + (self.linearlayer3.W ** 2).sum() + \
                      (self.adapter0.l2 if concat_n_out_ad else numpy.float32(0)) + (
                          self.adapter1.l2 if lin1_n_out_ad else numpy.float32(0)) + (
                          self.adapter2.l2 if rnn_n_out_ad else numpy.float32(0))

    # cost function
    def cal_cost(self, y):
        y_score = T.sum(self.p_y_given_x[T.arange(y.shape[0]), y])
        return -(y_score - self.allscore)
