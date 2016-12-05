from train import *


def wangzhen_train():
    cfg = Config('-t 1')
    cfg.dump_name = 'model/params_'
    cfg.dumpinit()
    test_dp(cfg)


def wangzhen_test():
    cfg = Config('-t 0 -dn model/params_')
    cfg.dump_name = 'model/params'
    cfg.dumpinit()
    test_dp(cfg)


def pku_cpbpos():
    cfg = Config('-t 0 -dn model/pku_cpbpos -n pku_cpbpos')
    cfg.dumpinit()
    test_dp(cfg)


def cpb_cpb_prog():  # local
    # 22 22099 f1 score: 77.289
    cfg = Config('-t 2 -dn model/cpb_cpbpos -dn1 model/pku_cpbpos -n cpb_cpbpos -m prog')
    cfg.w_n_out_new = 50
    cfg.p_n_out_new = 20
    cfg.dist_n_out_new = 30
    cfg.lin1_n_out_new = 200
    cfg.lin2_n_out_new = 80
    cfg.concat_n_out_ad = 20
    cfg.test_from_epoch = 4
    cfg.test_freq = 10
    cfg.dumpinit()
    test_dp(cfg)


def cpb_cpb_prog_all():  # local
    # 23 11415 f1 score: 77.837%
    cfg = Config('-t 0 -dn model/cpb_cpbpos_all -dn1 model/pku_cpbpos -n cpb_cpbpos -m prog')
    cfg.w_n_out_new = 50
    cfg.p_n_out_new = 20
    cfg.dist_n_out_new = 30
    cfg.lin1_n_out_new = 200
    cfg.lin2_n_out_new = 80
    cfg.concat_n_out_ad = 40
    cfg.lin1_n_out_ad = 20
    cfg.rnn_n_out_ad = 10
    cfg.test_from_epoch = 5
    cfg.test_freq = 2
    cfg.learning_rate = 0.0002
    cfg.dumpinit()
    test_dp(cfg)


def cpb_cpb_prog_all1():  # 221
    # [(100003, 50), (34, 20), (502, 30), (330, 200), (200,), (2, 4, 220, 80), (2, 4, 80, 80),
    # (2, 4, 80), (160, 80), (80,), (90, 73), (73,), (), (280, 40), (), (200, 20), (), (100, 10)]
    cfg = Config('-t 2 -dn model/cpb_cpbpos_all1 -dn1 model/pku_cpbpos -n cpb_cpbpos -m prog')
    cfg.w_n_out_new = 50
    cfg.p_n_out_new = 20
    cfg.dist_n_out_new = 30
    cfg.lin1_n_out_new = 200
    cfg.rnn_n_out_new = 200
    cfg.lin2_n_out_new = 80
    cfg.concat_n_out_ad = 40
    cfg.lin1_n_out_ad = 20
    cfg.rnn_n_out_ad = 10
    cfg.test_from_epoch = 10
    cfg.test_freq = 11000
    cfg.learning_rate = 0.0008
    cfg.dumpinit()
    test_dp(cfg)


def cpb_cpb_prog_all2():  # local
    # 16 23251 f1 score 77.985%
    # [(100003, 50), (34, 20), (502, 20), (320, 240), (240,), (2, 4, 270, 120), (2, 4, 120, 120),
    #  (2, 4, 120), (240, 120), (120,), (140, 73), (73,), (), (280, 40), (), (200, 30), (), (100, 20)]
    cfg = Config('-t 2 -dn model/cpb_cpbpos_all2 -dn1 model/pku_cpbpos -n cpb_cpbpos -m prog')
    cfg.w_n_out_new = 50
    cfg.p_n_out_new = 20
    cfg.dist_n_out_new = 20
    cfg.lin1_n_out_new = 240
    cfg.rnn_n_out_new = 240
    cfg.lin2_n_out_new = 120
    cfg.concat_n_out_ad = 40
    cfg.lin1_n_out_ad = 30
    cfg.rnn_n_out_ad = 20
    cfg.test_from_epoch = 10
    cfg.test_freq = 2
    cfg.learning_rate = 0.0002
    cfg.dumpinit()
    test_dp(cfg)


if __name__ == '__main__':
    # wangzhen_train()  # local
    # wangzhen_test()  # local
    # cpb_cpb_prog()  # local
    # pku_cpbpos()  # local
    cpb_cpb_prog_all()  # 221
    # cpb_cpb_prog_all1()  # 221
    # cpb_cpb_prog_all2()  # local
