import socket as skt

if skt.gethostname() in ['acl', 'acl221']:
    vec_bin_f = '/home/xql/Data/wangzhen/bilstm/vector/vectors.bin'
    cpb_train_f = '/home/xql/Data/wangzhen/bilstm/text/cpbtrain'
    cpb_test_f = '/home/xql/Data/wangzhen/bilstm/text/cpbtest'
    cpb_dev_f = '/home/xql/Data/wangzhen/bilstm/text/cpbdev'
    pku_train_f = '/home/xql/Data/wangzhen/bilstm/text/pkutest'
    pku_test_f = '/home/xql/Data/wangzhen/bilstm/text/pkutest'
    pku_dev_f = '/home/xql/Data/wangzhen/bilstm/text/pkudev'
    cpb_pkupos_train_f = '/home/xql/Data/wangzhen/bilstm/text/cpb_pkupos_train'
    cpb_pkupos_test_f = '/home/xql/Data/wangzhen/bilstm/text/cpb_pkupos_test'
    cpb_pkupos_dev_f = '/home/xql/Data/wangzhen/bilstm/text/cpb_pkupos_dev'
else:
    vec_bin_f = '/media/xql/新加卷/Data/wangzhen/bilstm/vector/vectors.bin'
    cpb_train_f = '/media/xql/新加卷/Data/wangzhen/bilstm/text/cpbtrain'
    cpb_test_f = '/media/xql/新加卷/Data/wangzhen/bilstm/text/cpbtest'
    cpb_dev_f = '/media/xql/新加卷/Data/wangzhen/bilstm/text/cpbdev'
    pku_train_f = '/media/xql/新加卷/Data/wangzhen/bilstm/text/pkutrain'
    pku_test_f = '/media/xql/新加卷/Data/wangzhen/bilstm/text/pkutest'
    pku_dev_f = '/media/xql/新加卷/Data/wangzhen/bilstm/text/pkudev'
    cpb_pkupos_train_f = '/home/xql/Data/wangzhen/bilstm/text/cpb_pkupos_train'
    cpb_pkupos_test_f = '/home/xql/Data/wangzhen/bilstm/text/cpb_pkupos_test'
    cpb_pkupos_dev_f = '/home/xql/Data/wangzhen/bilstm/text/cpb_pkupos_dev'

pku_text_f = '/media/xql/新加卷/Source/VSO/PKU/Lab/POSTagger-Train_TestV5/data/news/text.txt'
