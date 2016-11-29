import pickle
from const import *
from path import *


class Config:
    def __init__(self, argvs):
        self.data_name = 'cpb'
        self.model_type = 'basic'
        self.training = 0
        self.test_from_epoch = 0
        self.end_epoch = 30
        self.test_freq = 1000
        self.learning_rate = 0.005
        self.use_vecs = True
        self.is_concat = False  # sum or concat adapter outputs
        self.w_n_out = 50
        self.p_n_out = 20
        self.dist_n_out = 20
        self.lin1_n_out = 200
        self.rnn_n_out = 200
        self.lin2_n_out = 100
        self.w_n_out_new = 30
        self.p_n_out_new = 10
        self.dist_n_out_new = 10
        self.lin1_n_out_new = 100
        self.rnn_n_out_new = 100
        self.lin2_n_out_new = 80
        self.concat_n_out_ad = 0
        self.lin1_n_out_ad = 0
        self.rnn_n_out_ad = 0
        self.use_crf1 = False
        self.use_crf = False
        if '-t' in argvs:
            if argvs[argvs.index('-t') + 1] != '1':
                self.dump_name = argvs[argvs.index('-dn') + 1]
                self.loadinit()
        i = 0
        while i < len(argvs):
            argv = argvs[i]
            if argv == '-n':
                self.data_name = argvs[i + 1]
                i += 1
            elif argv == '-m':
                self.model_type = argvs[i + 1]
                i += 1
            elif argv == '-dn1':
                self.dump_name1 = argvs[i + 1]
                i += 1
            elif argv == '-dn':
                self.dump_name = argvs[i + 1]
                i += 1
            elif argv == '-t':
                self.training = int(argvs[i + 1])
                i += 1
            i += 1

    def myinit(self):
        self.TAGGING1 = PKU_TAGGING
        if self.data_name == 'cpb':  # for basic
            self.POSS = CPB_POSS
            self.TAGGING = CPB_TAGGING
            self.train_f = cpb_train_f
            self.dev_f = cpb_dev_f
            self.test_f = cpb_test_f
            self.TRANS2 = CPB_TRANS2
            self.TRANS0 = CPB_TRANS0
            self.NOT_ENTRY_IDXS = CPB_NOT_ENTRY_IDXS
            self.NOT_EXIT_IDXS = CPB_NOT_EXIT_IDXS
        elif self.data_name == 'pku':  # for basic
            self.POSS = PKU_POSS
            self.TAGGING = PKU_TAGGING
            self.train_f = pku_train_f
            self.dev_f = pku_dev_f
            self.test_f = pku_test_f
            self.TRANS2 = PKU_TRANS2
            self.TRANS0 = PKU_TRANS0
            self.NOT_ENTRY_IDXS = PKU_NOT_ENTRY_IDXS
            self.NOT_EXIT_IDXS = PKU_NOT_EXIT_IDXS
        elif self.data_name == 'cpb_pkupos':  # for progressive
            self.POSS = PKU_POSS
            self.TAGGING = CPB_TAGGING
            self.TAGGING1 = PKU_TAGGING
            self.train_f = cpb_pkupos_train_f
            self.dev_f = cpb_pkupos_dev_f
            self.test_f = cpb_pkupos_test_f
            self.TRANS2 = CPB_TRANS2
            self.TRANS0 = CPB_TRANS0
            self.NOT_ENTRY_IDXS = CPB_NOT_ENTRY_IDXS
            self.NOT_EXIT_IDXS = CPB_NOT_EXIT_IDXS
        elif self.data_name == 'pku_cpbpos':  # for basic
            self.POSS = CPB_POSS
            self.TAGGING = PKU_TAGGING
            self.train_f = pku_cpbpos_train_f
            self.dev_f = pku_cpbpos_dev_f
            self.test_f = pku_cpbpos_test_f
            self.TRANS2 = PKU_TRANS2
            self.TRANS0 = PKU_TRANS0
            self.NOT_ENTRY_IDXS = PKU_NOT_ENTRY_IDXS
            self.NOT_EXIT_IDXS = PKU_NOT_EXIT_IDXS
        elif self.data_name == 'cpb_cpbpos':  # for progressive
            self.POSS = CPB_POSS
            self.TAGGING = CPB_TAGGING
            self.TAGGING1 = PKU_TAGGING
            self.train_f = cpb_train_f
            self.dev_f = cpb_dev_f
            self.test_f = cpb_test_f
            self.TRANS2 = CPB_TRANS2
            self.TRANS0 = CPB_TRANS0
            self.NOT_ENTRY_IDXS = CPB_NOT_ENTRY_IDXS
            self.NOT_EXIT_IDXS = CPB_NOT_EXIT_IDXS

    def dumpinit(self):
        with open(self.dump_name + '.cfg', 'wb') as f:
            pickle.dump(self, f)
        self.myinit()
        print('data_name = %s, model_type = %s, \n'
              'dump_name = %s, training = %d, \n'
              'test_from_epoch = %d, end_epoch = %d, test_freq = %d, \n'
              'learning_rate = %f, use_vecs = %s, is_concat = %s, \n'
              'w_n_out = %d, p_n_out = %d, dist_n_out = %d, \n'
              'lin1_n_out = %d, rnn_n_out = %d, lin2_n_out = %d, \n'
              'w_n_out_new = %d, p_n_out_new = %d, dist_n_out_new = %d, \n'
              'lin1_n_out_new = %d, rnn_n_out_new = %d, lin2_n_out_new = %d, \n'
              'concat_n_out_ad = %d, lin1_n_out_ad = %d, rnn_n_out_ad = %d, \n'
              'use_crf1 = %s, use_crf = %s'
              % (self.data_name, self.model_type, self.dump_name, self.training, self.test_from_epoch,
                 self.end_epoch, self.test_freq, self.learning_rate, self.use_vecs, self.is_concat,
                 self.w_n_out, self.p_n_out, self.dist_n_out, self.lin1_n_out, self.rnn_n_out, self.lin2_n_out,
                 self.w_n_out_new, self.p_n_out_new, self.dist_n_out_new, self.lin1_n_out_new, self.rnn_n_out_new,
                 self.lin2_n_out_new,
                 self.concat_n_out_ad, self.lin1_n_out_ad, self.rnn_n_out_ad,
                 self.use_crf1, self.use_crf))

    def loadinit(self):
        with open(self.dump_name + '.cfg', 'rb') as f:
            s = pickle.load(f)
            self.data_name = s.data_name
            self.model_type = s.model_type
            self.training = s.training
            self.test_from_epoch = s.test_from_epoch
            self.end_epoch = s.end_epoch
            self.test_freq = s.test_freq
            try:
                self.learning_rate = s.learning_rate
            except:
                pass
            try:
                self.use_vecs = s.use_vecs
            except:
                pass
            try:
                self.w_n_out = s.w_n_out
            except:
                pass
            try:
                self.p_n_out = s.p_n_out
            except:
                pass
            try:
                self.dist_n_out = s.dist_n_out
            except:
                pass
            try:
                self.lin1_n_out = s.lin1_n_out
            except:
                pass
            try:
                self.rnn_n_out = s.rnn_n_out
            except:
                pass
            try:
                self.lin2_n_out = s.lin2_n_out
            except:
                pass
            try:
                self.w_n_out_new = s.w_n_out_new
            except:
                pass
            try:
                self.p_n_out_new = s.p_n_out_new
            except:
                pass
            try:
                self.dist_n_out_new = s.dist_n_out_new
            except:
                pass
            try:
                self.lin1_n_out_new = s.lin1_n_out_new
            except:
                pass
            try:
                self.rnn_n_out_new = s.rnn_n_out_new
            except:
                pass
            try:
                self.lin2_n_out_new = s.lin2_n_out_new
            except:
                pass
            try:
                self.is_concat = s.is_concat
            except:
                pass
            try:
                self.concat_n_out_ad = s.concat_n_out_ad
            except:
                pass
            try:
                self.lin1_n_out_ad = s.lin1_n_out_ad
            except:
                pass
            try:
                self.rnn_n_out_ad = s.rnn_n_out_ad
            except:
                pass
            try:
                self.use_crf1 = s.use_crf1
            except:
                pass
            try:
                self.use_crf = s.use_crf
            except:
                pass
        self.myinit()
