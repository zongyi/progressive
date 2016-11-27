from layers import *
from const import *


class BasicNet(object):
    def __init__(self, name, use_noise, w_emb_n_in, w_emb_n_out, p_emb_n_in, p_emb_n_out, dist_emb_n_in, dist_emb_n_out,
                 lin1_n_out, rnn_n_out, lin2_n_out, lin3_n_out,
                 window, TRANS2, TRANS0,
                 inputs, W):
        input_w, input_p, input_dist, input_v, input_entry_exit_mask, input_vi, input_y = inputs
        rng = numpy.random.RandomState(39287)
        wi = 0
        self.w_emb_layer = EmbedLayer(rng, w_emb_n_in, w_emb_n_out, window, name + 'w_emb', input_w, W_e=W[wi])
        wi += 1
        if p_emb_n_out:
            self.p_emb_layer = EmbedLayer(rng, p_emb_n_in, p_emb_n_out, window, name + 'p_emb', input_p, W_e=W[wi])
            wi += 1
        self.dist_emb_layer = EmbedLayer(rng, dist_emb_n_in, dist_emb_n_out, 1, name + 'dist_emb', input_dist,
                                         W_e=W[wi])
        wi += 1
        self.v_emb_layer = EmbedLayer(rng, w_emb_n_in, w_emb_n_out, 1, name + 'v_emb', input_v,
                                      W_e=self.w_emb_layer.W_e)
        if p_emb_n_out:
            self.concat = T.concatenate([self.w_emb_layer.output_e, self.p_emb_layer.output_e,
                                         self.dist_emb_layer.output_e, self.v_emb_layer.output_e],
                                        axis=self.w_emb_layer.output_e.ndim - 1)
        else:
            self.concat = T.concatenate(
                [self.w_emb_layer.output_e, self.dist_emb_layer.output_e, self.v_emb_layer.output_e],
                axis=self.w_emb_layer.output_e.ndim - 1)

        concat_n_out = window * (w_emb_n_out + p_emb_n_out) + dist_emb_n_out + w_emb_n_out
        self.dropout_layer = DropoutLayer(0.5, self.concat, use_noise)  # when use_noise.get_value() == 1, use dropout
        self.lin1_layer = LinearLayer(rng, concat_n_out, lin1_n_out,
                                      name + 'lin1', self.dropout_layer.output_d, 'none', W=W[wi], b=W[wi + 1])
        wi += 2
        self.rnn_layer = LSTMLayer(rng, lin1_n_out, rnn_n_out, self.lin1_layer.output_l, None, True,
                                   name + 'rnn', 'none', W[wi], W[wi + 1], W[wi + 2])
        wi += 3
        self.lin2_layer = LinearLayer(rng, rnn_n_out, lin2_n_out, name + 'lin2', self.rnn_layer.output_lstm, 'tanh',
                                      W[wi],
                                      W[wi + 1])
        wi += 2
        self.lin3_layer = LinearLayer(rng, lin2_n_out, lin3_n_out, name + 'lin3', self.lin2_layer.output_l, 'none',
                                      W[wi],
                                      W[wi + 1])
        wi += 2
        self.params = self.w_emb_layer.params + (
            self.p_emb_layer.params if p_emb_n_out else []) + self.dist_emb_layer.params + \
                      self.lin1_layer.params + self.rnn_layer.params + self.lin2_layer.params + self.lin3_layer.params
        self.l1 = self.w_emb_layer.l1 + (
            self.p_emb_layer.l1 if p_emb_n_out else numpy.float32(0.0)) + self.dist_emb_layer.l1 + \
                  self.lin1_layer.l1 + self.rnn_layer.l1 + self.lin2_layer.l1 + self.lin3_layer.l1
        self.l2 = self.w_emb_layer.l2 + (
            self.p_emb_layer.l2 if p_emb_n_out else numpy.float32(0.0)) + self.dist_emb_layer.l2 + \
                  self.lin1_layer.l2 + self.rnn_layer.l2 + self.lin2_layer.l2 + self.lin3_layer.l2

        self.p_y_given_x = self.lin3_layer.output_l + input_entry_exit_mask

        def Viterbi_decode(p_y_given_x, prev_p_path):
            p_path = T.max(prev_p_path + TRANS2, axis=1) + p_y_given_x
            argmax_prev_tag = T.argmax(prev_p_path + TRANS2, axis=1)
            return [p_path, argmax_prev_tag]

        score, updates = theano.scan(fn=Viterbi_decode,
                                     sequences=self.p_y_given_x[1:],
                                     outputs_info=[self.p_y_given_x[0], None])
        p_paths = score[0]
        argmax_prev_tags = score[1]
        max_tag = T.argmax(p_paths[-1])

        def get_prev_tag(idx, next_tag, _argmax_prev_tags):
            return _argmax_prev_tags[-idx - 1][next_tag]

        seq, updates = theano.scan(fn=get_prev_tag,
                                   sequences=T.arange(argmax_prev_tags.shape[0]),
                                   outputs_info=max_tag,
                                   non_sequences=argmax_prev_tags)
        seq_reverse = seq[::-1]
        final_seq = T.concatenate([seq_reverse, [max_tag]])
        self.y_pred = final_seq
        self.score = p_paths[-1][max_tag]

        def logadd(p_y_given_x, i, prev_p_path, s):
            max_p = T.max(prev_p_path)  # just to avoid overflow
            p = prev_p_path + T.switch(T.eq(i + 1, s), TRANS0, TRANS2) - max_p  # all the valid paths to current tags
            log_sum_exp = T.log(T.sum(T.exp(p), axis=1))
            return p_y_given_x + max_p + log_sum_exp

        paths, updates = theano.scan(fn=logadd,
                                     sequences=[self.p_y_given_x[1:],
                                                T.arange(self.p_y_given_x[1:].shape[0])],
                                     outputs_info=self.p_y_given_x[0],
                                     non_sequences=input_vi)
        # all paths for the whole sentence
        max_p = T.max(paths[-1])
        self.allscore = T.log(T.sum(T.exp(paths[-1] - max_p))) + max_p
        self.outputs_n_out = [concat_n_out, lin1_n_out, rnn_n_out, lin2_n_out, lin3_n_out]

    def cal_cost(self, input_y, l1_lr, l2_lr):
        y_score = T.sum(self.p_y_given_x[T.arange(input_y.shape[0]), input_y])
        return -(y_score - self.allscore) + l1_lr * self.l1 + l2_lr * self.l2


class Progressive:
    def __init__(self, name, net1,
                 use_noise, w_emb_n_in, w_emb_n_out, p_emb_n_in, p_emb_n_out, dist_emb_n_in, dist_emb_n_out,
                 lin1_n_out, rnn_n_out, lin2_n_out, lin3_n_out,
                 is_concat, concat_n_out_ad, lin1_n_out_ad, rnn_n_out_ad,
                 window, TRANS2, TRANS0, inputs, W):
        net1_concat_n_out, net1_lin1_n_out, net1_rnn_n_out, net1_lin2_n_out, net1_lin3_n_out = net1.outputs_n_out
        rng = numpy.random.RandomState(39287)
        # adapter
        input_w, input_p, input_dist, input_v, input_entry_exit_mask, input_vi, input_y = inputs
        wi = 0
        # self.concat_ad_layer = AdapterLayer(rng, net1_concat_n_out, lin1_n_out, name + 'concat_ad', net1.concat,
        #                               W[wi], W[wi + 1])
        # wi += 2
        if concat_n_out_ad:
            print('add concat_ad_layer')
            self.concat_ad_layer = AdapterLayer(rng, net1_lin1_n_out, concat_n_out_ad if is_concat else lin1_n_out,
                                                name + 'concat_ad', net1.concat,
                                                W[wi], W[wi + 1])
            wi += 2
        if lin1_n_out_ad:
            print('add lin1_ad_layer')
            self.lin1_ad_layer = AdapterLayer(rng, net1_lin1_n_out, lin1_n_out_ad if is_concat else rnn_n_out,
                                              name + 'lin1_ad', net1.lin1_layer.output_l,
                                              W[wi], W[wi + 1])
            wi += 2
        if rnn_n_out_ad:
            print('add rnn_ad_layer')
            self.rnn_ad_layer = AdapterLayer(rng, net1_rnn_n_out, rnn_n_out_ad if is_concat else lin2_n_out,
                                             name + 'rnn_ad', net1.rnn_layer.output_lstm,
                                             W[wi], W[wi + 1])
            wi += 2
        # self.lin2_ad_layer = AdapterLayer(rng, net1_lin2_n_out, lin3_n_out, name + 'lin2_ad', net1.lin2_layer.output_l,
        #                              W[wi], W[wi + 1])
        # wi += 2
        # self.lin3_ad_layer = AdapterLayer(rng, net1_lin3_n_out, lin3_n_out, name + 'lin3_ad', net1.lin3_layer.output_l,
        #                              W[wi], W[wi + 1])
        # wi += 2
        self.w_emb_layer = EmbedLayer(rng, w_emb_n_in, w_emb_n_out, window, name + 'w_emb', input_w, W_e=W[wi])
        wi += 1
        self.p_emb_layer = EmbedLayer(rng, p_emb_n_in, p_emb_n_out, window, name + 'p_emb', input_p, W_e=W[wi])
        wi += 1
        self.dist_emb_layer = EmbedLayer(rng, dist_emb_n_in, dist_emb_n_out, 1, name + 'dist_emb', input_dist,
                                         W_e=W[wi])
        wi += 1
        self.v_emb_layer = EmbedLayer(rng, w_emb_n_in, w_emb_n_out, 1, name + 'v_emb', input_v,
                                      W_e=self.w_emb_layer.W_e)

        self.concat = T.concatenate([self.w_emb_layer.output_e, self.p_emb_layer.output_e,
                                     self.dist_emb_layer.output_e, self.v_emb_layer.output_e],
                                    axis=self.w_emb_layer.output_e.ndim - 1)
        self.dropout_layer = DropoutLayer(0.5, self.concat, use_noise)  # when use_noise.get_value() == 1, use dropout
        concat_n_out = window * (w_emb_n_out + p_emb_n_out) + dist_emb_n_out + w_emb_n_out

        if is_concat:
            if concat_n_out_ad:
                lin1_input = T.concatenate([self.dropout_layer.output_d, self.concat_ad_layer.output_a], axis=1)
            else:
                lin1_input = self.concat
            self.lin1_layer = LinearLayer(rng, concat_n_out + concat_n_out_ad, lin1_n_out, name + 'lin1',
                                          lin1_input,  # self.concat_ad_layer.output_a,
                                          'none', W[wi], W[wi + 1])
        else:
            self.lin1_layer = LinearAdLayer(rng, concat_n_out, lin1_n_out, name + 'lin1',
                                            self.concat,
                                            self.concat_ad_layer.output_a if concat_n_out_ad else numpy.float32(0.0),
                                            'none', W[wi], W[wi + 1])
        wi += 2
        if is_concat:
            if lin1_n_out_ad:
                rnn_input = T.concatenate([self.lin1_layer.output_l, self.lin1_ad_layer.output_a], axis=1)
            else:
                rnn_input = self.lin1_layer.output_l
            self.rnn_layer = LSTMLayer(rng, lin1_n_out + lin1_n_out_ad, rnn_n_out,
                                       rnn_input,  # self.lin1_ad_layer.output_a,
                              None, True, name + 'rnn', 'none', W[wi], W[wi + 1], W[wi + 2])
        else:
            self.rnn_layer = LSTMAdLayer(rng, lin1_n_out, rnn_n_out,
                                         self.lin1_layer.output_l,
                                         self.lin1_ad_layer.output_a if lin1_n_out_ad else numpy.float32(0.0),
                                         None, True, name + 'rnn', 'none', W[wi], W[wi + 1], W[wi + 2])
        wi += 3
        if is_concat:
            if rnn_n_out_ad:
                lin2_input = T.concatenate([self.rnn_layer.output_lstm, self.rnn_ad_layer.output_a], axis=1)
            else:
                lin2_input = self.rnn_layer.output_lstm
            self.lin2_layer = LinearLayer(rng, rnn_n_out + rnn_n_out_ad, lin2_n_out, name + 'lin2',
                                          lin2_input,
                                          'tanh', W[wi], W[wi + 1])
        else:
            self.lin2_layer = LinearAdLayer(rng, rnn_n_out, lin2_n_out, name + 'lin2',
                                            self.rnn_layer.output_lstm,
                                            self.rnn_ad_layer.output_a if rnn_n_out_ad else numpy.float32(0.0),
                                 'tanh', W[wi], W[wi + 1])
        wi += 2
        self.lin3_layer = LinearLayer(rng, lin2_n_out, lin3_n_out, name + 'lin3',
                                      self.lin2_layer.output_l,
                                      # self.lin2_ad_layer.output_a+self.lin3_ad_layer.output_a,
                                 'none', W[wi], W[wi + 1])
        wi += 2
        self.params = (self.concat_ad_layer.params if concat_n_out_ad else []) + \
                      (self.lin1_ad_layer.params if lin1_n_out_ad else []) + \
                      (self.rnn_ad_layer.params if rnn_n_out_ad else []) + \
                      self.w_emb_layer.params + self.p_emb_layer.params + self.dist_emb_layer.params + \
                      self.lin1_layer.params + self.rnn_layer.params + self.lin2_layer.params + self.lin3_layer.params
        self.l1 = (self.concat_ad_layer.l1 if concat_n_out_ad else numpy.float32(0.0)) + \
                  (self.lin1_ad_layer.l1 if lin1_n_out_ad else numpy.float32(0.0)) + \
                  (self.rnn_ad_layer.l1 if rnn_n_out_ad else numpy.float32(0.0)) + \
                  self.w_emb_layer.l1 + self.p_emb_layer.l1 + \
                  self.dist_emb_layer.l1 + self.lin1_layer.l1 + self.rnn_layer.l1 + self.lin2_layer.l1 + self.lin3_layer.l1
        self.l2 = (self.concat_ad_layer.l2 if concat_n_out_ad else numpy.float32(0.0)) + \
                  (self.lin1_ad_layer.l2 if lin1_n_out_ad else numpy.float32(0.0)) + \
                  (self.rnn_ad_layer.l2 if rnn_n_out_ad else numpy.float32(0.0)) + \
                  self.w_emb_layer.l2 + self.p_emb_layer.l2 + \
                  self.dist_emb_layer.l2 + self.lin1_layer.l2 + self.rnn_layer.l2 + self.lin2_layer.l2 + self.lin3_layer.l2

        self.p_y_given_x = self.lin3_layer.output_l + input_entry_exit_mask

        def Viterbi_decode(p_y_given_x, prev_p_path):
            p_path = T.max(prev_p_path + TRANS2, axis=1) + p_y_given_x
            argmax_prev_tag = T.argmax(prev_p_path + TRANS2, axis=1)
            return [p_path, argmax_prev_tag]

        score, updates = theano.scan(fn=Viterbi_decode,
                                     sequences=self.p_y_given_x[1:],
                                     outputs_info=[self.p_y_given_x[0], None])
        p_paths = score[0]
        argmax_prev_tags = score[1]
        max_tag = T.argmax(p_paths[-1])

        def get_prev_tag(idx, next_tag, _argmax_prev_tags):
            return _argmax_prev_tags[-idx - 1][next_tag]

        seq, updates = theano.scan(fn=get_prev_tag,
                                   sequences=T.arange(argmax_prev_tags.shape[0]),
                                   outputs_info=max_tag,
                                   non_sequences=argmax_prev_tags)
        seq_reverse = seq[::-1]
        final_seq = T.concatenate([seq_reverse, [max_tag]])
        self.y_pred = final_seq
        self.score = p_paths[-1][max_tag]

        def logadd(p_y_given_x, i, prev_p_path, s):
            max_p = T.max(prev_p_path)  # just to avoid overflow
            p = prev_p_path + T.switch(T.eq(i + 1, s), TRANS0, TRANS2) - max_p  # all the valid paths to current tags
            log_sum_exp = T.log(T.sum(T.exp(p), axis=1))
            return p_y_given_x + max_p + log_sum_exp

        paths, updates = theano.scan(fn=logadd,
                                     sequences=[self.p_y_given_x[1:],
                                                T.arange(self.p_y_given_x[1:].shape[0])],
                                     outputs_info=self.p_y_given_x[0],
                                     non_sequences=input_vi)
        # all paths for the whole sentence
        max_p = T.max(paths[-1])
        self.allscore = T.log(T.sum(T.exp(paths[-1] - max_p))) + max_p

    def cal_cost(self, input_y, l1_lr, l2_lr):
        y_score = T.sum(self.p_y_given_x[T.arange(input_y.shape[0]), input_y])
        return -(y_score - self.allscore) + l1_lr * self.l1 + l2_lr * self.l2
