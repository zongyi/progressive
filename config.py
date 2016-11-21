import pickle
from const import *
from path import *


class Config:
    def __init__(self, argvs):
        self.data_name = 'cpb'
        self.model_type = 'basic'
        self.test_from_epoch = 10
        self.end_iter = 3000000
        self.test_freq = 1000
        self.learning_rate = 0.003
        for i, argv in enumerate(argvs):
            if argv == '-n':
                self.data_name = argvs[i + 1]
                i += 1
            elif argv == '-m':
                self.model_type = argvs[i + 1]
                i += 1
            elif argv == '-dn':
                self.dump_name = argvs[i + 1]
                i += 1
            elif argv == '-t':
                self.training = int(argvs[i + 1])
                if self.training == 0:
                    self.loadinit()
                i += 1

    def dump(self):
        with open(self.dump_name + '.cfg', 'wb') as f:
            pickle.dump(self, f)

    def myinit(self):
        if self.data_name == 'cpb':
            self.POSS = CPB_POSS
            self.TAGGING = CPB_TAGGING
            self.train_f = cpb_train_f
            self.dev_f = cpb_dev_f
            self.test_f = cpb_test_f
            self.TRANS2 = CPB_TRANS2
            self.TRANS0 = CPB_TRANS0
            self.NOT_ENTRY_IDXS = CPB_NOT_ENTRY_IDXS
            self.NOT_EXIT_IDXS = CPB_NOT_EXIT_IDXS
        elif self.data_name == 'pku':
            self.POSS = PKU_POSS
            self.TAGGING = PKU_TAGGING
            self.train_f = pku_train_f
            self.dev_f = pku_dev_f
            self.test_f = pku_test_f
            self.TRANS2 = PKU_TRANS2
            self.TRANS0 = PKU_TRANS0
            self.NOT_ENTRY_IDXS = PKU_NOT_ENTRY_IDXS
            self.NOT_EXIT_IDXS = PKU_NOT_EXIT_IDXS
        elif self.data_name == 'cpb_pkupos':
            self.POSS = PKU_POSS
            self.TAGGING = CPB_TAGGING
            self.train_f = cpb_pkupos_train_f
            self.dev_f = cpb_pkupos_dev_f
            self.test_f = cpb_pkupos_test_f
            self.TRANS2 = CPB_TRANS2
            self.TRANS0 = CPB_TRANS0
            self.NOT_ENTRY_IDXS = CPB_NOT_ENTRY_IDXS
            self.NOT_EXIT_IDXS = CPB_NOT_EXIT_IDXS

    def dumpinit(self):
        self.dump()
        self.myinit()

    def loadinit(self):
        with open(self.dump_name + '.cfg', 'rb') as f:
            s = pickle.load(f)
            self.data_name = s.data_name
            self.model_type = s.model_type
        self.myinit()
