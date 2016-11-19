from layers import *
from const import *


class BasicNet(object):
    def __init__(self, w_emb_n_in, w_emb_n_out, p_emb_n_in, p_emb_n_out, dist_emb_n_in, dist_emb_n_out,
                 lin1_n_out, rnn_n_out, lin2_n_out, lin3_n_out,
                 window, TRANS2, TRANS0,
                 inputs, W):
        input_w, input_p, input_dist, input_v, input_entry_exit_mask, input_vi, input_y = inputs
        rng = numpy.random.RandomState(39287)
        wi = 0
        w_emb_layer = EmbedLayer(rng, w_emb_n_in, w_emb_n_out, window, 'w_emb', input_w, W_e=W[wi])
        wi += 1
        p_emb_layer = EmbedLayer(rng, p_emb_n_in, p_emb_n_out, window, 'p_emb', input_p, W_e=W[wi])
        wi += 1
        dist_emb_layer = EmbedLayer(rng, dist_emb_n_in, dist_emb_n_out, 1, 'dist_emb', input_dist, W_e=W[wi])
        wi += 1
        v_emb_layer = EmbedLayer(rng, w_emb_n_in, w_emb_n_out, 1, 'v_emb', input_v, W_e=w_emb_layer.W_e)
        wi += 1
        concat = T.concatenate([w_emb_layer.output_e, p_emb_layer.output_e,
                                dist_emb_layer.output_e, v_emb_layer.output_e], axis=w_emb_layer.output_e.ndim - 1)
        # TODO: add dropout
        lin1_layer = LinearLayer(rng, window * (w_emb_n_out + p_emb_n_out) + dist_emb_n_out + w_emb_n_out, lin1_n_out,
                                 'lin1', concat, 'none', W=W[wi], b=W[wi + 1])
        wi += 2
        rnn_layer = LSTMLayer(rng, lin1_n_out, rnn_n_out, lin1_layer.output_l, None, True,
                              'rnn', 'none', W[wi], W[wi + 1], W[wi + 2])
        wi += 3
        lin2_layer = LinearLayer(rng, rnn_n_out, lin2_n_out, 'lin2', rnn_layer.output_lstm, 'tanh', W[wi], W[wi + 1])
        wi += 2
        lin3_layer = LinearLayer(rng, lin2_n_out, lin3_n_out, 'lin3', lin2_layer.output_l, 'none', W[wi], W[wi + 1])
        wi += 2
        self.w_emb_layer = w_emb_layer
        self.p_emb_layer = p_emb_layer
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

    def cal_cost(self, input_y, l1, l2):
        y_score = T.sum(self.p_y_given_x[T.arange(input_y.shape[0]), input_y])
        return -(y_score - self.allscore) + l1 * self.l1 + l2 * self.l2
