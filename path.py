import socket as skt

if skt.gethostname() in ['acl', 'acl221']:
    vec_bin_f = '/home/xql/Data/wangzhen/bilstm/vector/vectors.bin'
    cpb_train_f = '/home/xql/Data/wangzhen/bilstm/text/cpbtrain'
    cpb_test_f = '/home/xql/Data/wangzhen/bilstm/text/cpbtest'
    cpb_dev_f = '/home/xql/Data/wangzhen/bilstm/text/cpbdev'
else:
    vec_bin_f = '/media/xql/新加卷/Data/wangzhen/bilstm/vector/vectors.bin'
    cpb_train_f = '/media/xql/新加卷/Data/wangzhen/bilstm/text/cpbtrain'
    cpb_test_f = '/media/xql/新加卷/Data/wangzhen/bilstm/text/cpbtest'
    cpb_dev_f = '/media/xql/新加卷/Data/wangzhen/bilstm/text/cpbdev'
