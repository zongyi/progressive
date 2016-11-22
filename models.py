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
        w_emb_layer = EmbedLayer(rng, w_emb_n_in, w_emb_n_out, window, name + 'w_emb', input_w, W_e=W[wi])
        wi += 1
        p_emb_layer = EmbedLayer(rng, p_emb_n_in, p_emb_n_out, window, name + 'p_emb', input_p, W_e=W[wi])
        wi += 1
        dist_emb_layer = EmbedLayer(rng, dist_emb_n_in, dist_emb_n_out, 1, name + 'dist_emb', input_dist, W_e=W[wi])
        wi += 1
        v_emb_layer = EmbedLayer(rng, w_emb_n_in, w_emb_n_out, 1, name + 'v_emb', input_v, W_e=w_emb_layer.W_e)
        concat = T.concatenate([w_emb_layer.output_e, p_emb_layer.output_e,
                                dist_emb_layer.output_e, v_emb_layer.output_e], axis=w_emb_layer.output_e.ndim - 1)
        self.concat_n_out = window * (w_emb_n_out + p_emb_n_out) + dist_emb_n_out + w_emb_n_out
        dropout_layer = DropoutLayer(0.5, concat, use_noise)  # when use_noise.get_value() == 1, use dropout
        lin1_layer = LinearLayer(rng, self.concat_n_out, lin1_n_out,
                                 name + 'lin1', dropout_layer.output_d, 'none', W=W[wi], b=W[wi + 1])
        wi += 2
        rnn_layer = LSTMLayer(rng, lin1_n_out, rnn_n_out, lin1_layer.output_l, None, True,
                              name + 'rnn', 'none', W[wi], W[wi + 1], W[wi + 2])
        wi += 3
        lin2_layer = LinearLayer(rng, rnn_n_out, lin2_n_out, name + 'lin2', rnn_layer.output_lstm, 'tanh', W[wi],
                                 W[wi + 1])
        wi += 2
        lin3_layer = LinearLayer(rng, lin2_n_out, lin3_n_out, name + 'lin3', lin2_layer.output_l, 'none', W[wi],
                                 W[wi + 1])
        wi += 2
        self.params = w_emb_layer.params + p_emb_layer.params + dist_emb_layer.params + \
                      lin1_layer.params + rnn_layer.params + lin2_layer.params + lin3_layer.params
        self.l1 = w_emb_layer.l1 + p_emb_layer.l1 + dist_emb_layer.l1 + \
                  lin1_layer.l1 + rnn_layer.l1 + lin2_layer.l1 + lin3_layer.l1
        self.l2 = w_emb_layer.l2 + p_emb_layer.l2 + dist_emb_layer.l2 + \
                  lin1_layer.l2 + rnn_layer.l2 + lin2_layer.l2 + lin3_layer.l2

        self.p_y_given_x = lin3_layer.output_l + input_entry_exit_mask

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
        self.outputs = [concat, lin1_layer.output_l, rnn_layer.output_lstm, lin2_layer.output_l, lin3_layer.output_l,
                        self.y_pred]

    def cal_cost(self, input_y, l1_lr, l2_lr):
        y_score = T.sum(self.p_y_given_x[T.arange(input_y.shape[0]), input_y])
        return -(y_score - self.allscore) + l1_lr * self.l1 + l2_lr * self.l2


class Progressive:
    def __init__(self, name, net1,
                 use_noise, w_emb_n_in, w_emb_n_out, p_emb_n_in, p_emb_n_out, dist_emb_n_in, dist_emb_n_out,
                 lin1_n_out, rnn_n_out, lin2_n_out, lin3_n_out,
                 concat_ad_n_out, lin1_ad_n_out, rnn_ad_n_out, lin2_ad_n_out, lin3_ad_n_out,
                 window, TRANS2, TRANS0, inputs, W):
        net1_concat, net1_lin1_out, net1_rnn_out, net1_lin2_out, net1_lin3_out, net1_y_pred = net1.outputs

        rng = numpy.random.RandomState(39287)
        # adapter
        input_w, input_p, input_dist, input_v, input_entry_exit_mask, input_vi, input_y = inputs
        wi = 0
        concat_ad_layer = LinearLayer(rng, net1.concat_n_out, concat_ad_n_out, name + 'concat_ad', net1_concat, 'none',
                                      W[wi], W[wi + 1])
        wi += 2
        lin1_ad_layer = LinearLayer(rng, net1.concat_n_out, lin1_ad_n_out, name + 'lin1_ad', net1_concat, 'none', W[wi],
                                    W[wi + 1])
        wi += 2
        rnn_ad_layer = LinearLayer(rng, net1.concat_n_out, rnn_ad_n_out, name + 'rnn_ad', net1_concat, 'none', W[wi],
                                   W[wi + 1])
        wi += 2
        lin2_ad_layer = LinearLayer(rng, net1.concat_n_out, lin2_ad_n_out, name + 'lin2_ad', net1_concat, 'none', W[wi],
                                    W[wi + 1])
        wi += 2
        lin3_ad_layer = LinearLayer(rng, net1.concat_n_out, lin3_ad_n_out, name + 'lin3_ad', net1_concat, 'none', W[wi],
                                    W[wi + 1])
        wi += 2
        w_emb_layer = EmbedLayer(rng, w_emb_n_in, w_emb_n_out, window, name + 'w_emb', input_w, W_e=W[wi])
        wi += 1
        p_emb_layer = EmbedLayer(rng, p_emb_n_in, p_emb_n_out, window, name + 'p_emb', input_p, W_e=W[wi])
        wi += 1
        dist_emb_layer = EmbedLayer(rng, dist_emb_n_in, dist_emb_n_out, 1, name + 'dist_emb', input_dist, W_e=W[wi])
        wi += 1
        v_emb_layer = EmbedLayer(rng, w_emb_n_in, w_emb_n_out, 1, name + 'v_emb', input_v, W_e=w_emb_layer.W_e)
        wi += 1

        concat = T.concatenate([w_emb_layer.output_e, p_emb_layer.output_e,
                                dist_emb_layer.output_e, v_emb_layer.output_e], axis=w_emb_layer.output_e.ndim - 1)
        self.concat_n_out = window * (w_emb_n_out + p_emb_n_out) + dist_emb_n_out + w_emb_n_out
        dropout_layer = DropoutLayer(0.5, concat, use_noise)  # when use_noise.get_value() == 1, use dropout
        lin1_layer = LinearLayer(rng, self.concat_n_out + concat_ad_n_out, lin1_n_out, name + 'lin1',
                                 T.concatenate([dropout_layer.output_d, concat_ad_layer.output_l], axis=1),
                                 'none', W=W[wi], b=W[wi + 1])
        wi += 2
        rnn_layer = LSTMLayer(rng, lin1_n_out + lin1_ad_n_out, rnn_n_out,
                              T.concatenate([lin1_layer.output_l, lin1_ad_layer.output_l], axis=1),
                              None, True, name + 'rnn', 'none', W[wi], W[wi + 1], W[wi + 2])
        wi += 3
        lin2_layer = LinearLayer(rng, rnn_n_out + rnn_ad_n_out, lin2_n_out, name + 'lin2',
                                 T.concatenate([rnn_layer.output_lstm, rnn_ad_layer.output_l], axis=1),
                                 'tanh', W[wi], W[wi + 1])
        wi += 2
        lin3_layer = LinearLayer(rng, lin2_n_out + lin2_ad_n_out + lin3_ad_n_out, lin3_n_out, name + 'lin3',
                                 T.concatenate([lin2_layer.output_l, lin2_ad_layer.output_l, lin3_ad_layer.output_l],
                                               axis=1),
                                 'none', W[wi], W[wi + 1])
        wi += 2
        self.params = concat_ad_layer.params + lin1_ad_layer.params + rnn_ad_layer.params + lin2_ad_layer.params + lin3_ad_layer.params + \
                      w_emb_layer.params + p_emb_layer.params + dist_emb_layer.params + \
                      lin1_layer.params + rnn_layer.params + lin2_layer.params + lin3_layer.params
        self.l1 = w_emb_layer.l1 + p_emb_layer.l1 + dist_emb_layer.l1 + \
                  lin1_layer.l1 + rnn_layer.l1 + lin2_layer.l1 + lin3_layer.l1
        self.l2 = w_emb_layer.l2 + p_emb_layer.l2 + dist_emb_layer.l2 + \
                  lin1_layer.l2 + rnn_layer.l2 + lin2_layer.l2 + lin3_layer.l2

        self.p_y_given_x = lin3_layer.output_l + input_entry_exit_mask

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
        self.outputs = [concat, lin1_layer.output_l, rnn_layer.output_lstm, lin2_layer.output_l, lin3_layer.output_l]

    def cal_cost(self, input_y, l1_lr, l2_lr):
        y_score = T.sum(self.p_y_given_x[T.arange(input_y.shape[0]), input_y])
        return -(y_score - self.allscore) + l1_lr * self.l1 + l2_lr * self.l2
