import numpy
OOV_IDX = -3
BOS_IDX = -2
EOS_IDX = -1
CPB_POSS = ['SP', 'BA', 'FW', 'DER', 'DEV', 'MSP', 'ETC', 'JJ', 'DT', 'DEC', 'DEG', 'LB', 'LC', 'NN', 'PU', 'NP',
            'NR', 'NT', 'VA', 'VC', 'AD', 'CC', 'VE', 'M', 'CD', 'P', 'AS', 'VV', 'CS', 'PN', 'OD', 'SB']
CPB_POSS = [p.encode('utf-8') for p in CPB_POSS]
CPB_POSS += ['BOS', 'EOS']
CPB_TAGS = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ADV', 'BNF', 'CND', 'DIR', 'DIS', 'DGR', 'EXT', 'FRQ', 'LOC',
            'MNR', 'PRP', 'TMP', 'TPC']
CPB_TAGGING = []
for i in range(len(CPB_TAGS)):
    if 'ARG' not in CPB_TAGS[i]:
        CPB_TAGS[i] = 'ARGM-' + CPB_TAGS[i]
    CPB_TAGGING.append('S-' + CPB_TAGS[i])
    CPB_TAGGING.append('B-' + CPB_TAGS[i])
    CPB_TAGGING.append('I-' + CPB_TAGS[i])
    CPB_TAGGING.append('E-' + CPB_TAGS[i])
CPB_TAGGING.append('O')
CPB_TAGS = [p.encode('utf-8') for p in CPB_TAGS]
CPB_TAGGING = [p.encode('utf-8') for p in CPB_TAGGING]
CPB_NOT_ENTRY_IDXS = [i for i in range(len(CPB_TAGGING)) if i % 4 != 0 and i % 4 != 1 and i != len(CPB_TAGGING) - 1]
CPB_NOT_EXIT_IDXS = [i for i in range(len(CPB_TAGGING)) if i % 4 != 0 and i % 4 != 3 and i != len(CPB_TAGGING) - 1]
CPB_TRANS = numpy.zeros((len(CPB_TAGGING), len(CPB_TAGGING)), dtype='float32')
CPB_TRANS2 = numpy.zeros((len(CPB_TAGGING), len(CPB_TAGGING)), dtype='float32')
for j in range(len(CPB_TAGS)):
    for i in range(len(CPB_TAGS)):
        CPB_TRANS[j * 4][i * 4] = 1  # S<-S
        CPB_TRANS[j * 4][i * 4 + 3] = 1  # S<-E
        CPB_TRANS[j * 4][len(CPB_TAGGING) - 1] = 1  # S<-O
        CPB_TRANS[j * 4 + 1][i * 4] = 1  # B<-S
        CPB_TRANS[j * 4 + 1][i * 4 + 3] = 1  # B<-E
        CPB_TRANS[j * 4 + 1][len(CPB_TAGGING) - 1] = 1  # B<-O
for i in range(len(CPB_TAGS)):
    CPB_TRANS[i * 4 + 2][i * 4 + 1] = 1  # I<-B
    CPB_TRANS[i * 4 + 2][i * 4 + 2] = 1  # I<-I
    CPB_TRANS[i * 4 + 3][i * 4 + 1] = 1  # E<-B
    CPB_TRANS[i * 4 + 3][i * 4 + 2] = 1  # E<-I
for j in range(len(CPB_TAGS)):
    CPB_TRANS[len(CPB_TAGGING) - 1][len(CPB_TAGGING) - 1] = 1  # O<-O
    CPB_TRANS[len(CPB_TAGGING) - 1][j * 4] = 1  # O<-S
    CPB_TRANS[len(CPB_TAGGING) - 1][j * 4 + 3] = 1  # O<-E
for i in range(len(CPB_TAGGING)):
    for j in range(len(CPB_TAGGING)):
        if CPB_TRANS[i][j] == 0:
            CPB_TRANS2[i][j] = numpy.float32('-inf')
CPB_TRANS0 = numpy.zeros((len(CPB_TAGGING), len(CPB_TAGGING)), dtype='float32')

PKU_POSS = ['l', 'Bg', 'k', 'd', 'q', 'h', 'u', 'p', 't', 'Mg', 'r', 'v', 'w', 'j', 'e', 'c', 'o', 's', 'Ng', 'nr', 'i',
            'b', 'f', 'n', 'm', 'z', 'vd', 'ad', 'vn', 'an', 'Tg', '%', 'Dg', 'nz', 'ns', 'nt', 'Qg', 'a', 'Ag', 'Vg',
            'y', 'Ug']
PKU_POSS = [p.encode('utf-8') for p in PKU_POSS]
PKU_POSS2to1 = {b'ld': b'l', b'Bg': b'Bg', b'lb': b'l', b'la': b'l', b'ln': b'l', b'lm': b'l', b'k1': b'k', b'lv': b'l',
                b'df': b'd',
                b'dc': b'd', b'du': b'd', b'qt': b'q', b'qr': b'q', b'd': b'd', b'qv': b'q', b'h': b'h', b'uz': b'u',
                b'l': b'l',
                b'p': b'p', b'qc': b'q', b'qb': b'q', b'qe': b'q', b't': b't', b'qj': b'q', b'ql': b'q', b'qz': b'q',
                b'uv': b'u',
                b'Mg': b'Mg', b'rr': b'r', b'rz': b'r', b'vt2': b'v', b'ui': b'u', b'qd': b'q', b'wf': b'w',
                b'vx': b'v', b'wd': b'w',
                b'wm': b'w', b'wj': b'w', b'ww': b'w', b'jn': b'j', b'wt': b'w', b'ws': b'w', b'wp': b'w', b'wzz': b'w',
                b'e1': b'e',
                b'wy': b'w', b'c': b'c', b'k': b'k', b'wu': b'w', b'o': b'o', b'ud': b'u', b's': b's', b'w': b'w',
                b'Ng': b'Ng',
                b'dfu': b'd', b'u1': b'u', b'nrg': b'nr', b'nrf': b'nr', b'iv': b'i', b'vi_a': b'v', b'b': b'b',
                b'f': b'f', b'j': b'j',
                b'us': b'u', b'n': b'n', b'ul': b'u', b'uo': b'u', b'r': b'r', b'mq': b'm', b'v': b'v', b'ue': b'u',
                b'vt': b'v',
                b'z': b'z', b'vd': b'vd', b'ad': b'ad', b'vi': b'v', b'vl': b'v', b'vn': b'vn', b'an': b'an',
                b'wky': b'w', b'vq': b'v',
                b'wkz': b'w', b'im': b'i', b'vu': b'v', b'in': b'i', b'ia': b'i', b'Tg': b'Tg', b'ic': b'i',
                b'ib': b'i', b'id': b'i',
                b'nh': b'n', b'Tg1': b'Tg', b'%': b'%', b'Dg': b'Dg', b'nx': b'n', b'nz': b'nz', b'ryw': b'r',
                b'nr': b'nr', b'ns': b'ns',
                b'nt': b'nt', b'Qg': b'Qg', b'a': b'a', b'e': b'e', b'Ag': b'Ag', b'Vg': b'Vg', b'i': b'i', b'm': b'm',
                b'wyz': b'w',
                b'wyy': b'w', b'q': b'q', b'n]nt': b'n', b'vt_a': b'v', b'u': b'u', b'y': b'y', b'tt': b't',
                b'jb': b'j', b'jd': b'j',
                b'jv': b'j', b'rzw': b'r', b'Ug': b'Ug'}
assert set(PKU_POSS) == set(PKU_POSS2to1.values())
PKU_POSS += [b'BOS', b'EOS']
PKU_TAGS = ['原因', '时间', '比较主体', '路径', '接事', '终点', '范围', '物量', '起始', '比较对象', '处所',
            '施事', '材料', '方式', '与事', '起点', '方向', '受事', '目的', '当事', '比较项', '结束', '比较客体', '对象',
            '工具', '结果', '内容', '比较结果', '系事', '同事', '时段']
PKU_TAGGING = []
for i in range(len(PKU_TAGS)):
    PKU_TAGGING.append('S-' + PKU_TAGS[i])
    PKU_TAGGING.append('B-' + PKU_TAGS[i])
    PKU_TAGGING.append('I-' + PKU_TAGS[i])
    PKU_TAGGING.append('E-' + PKU_TAGS[i])
PKU_TAGGING.append('O')
PKU_TAGS = [p.encode('utf-8') for p in PKU_TAGS]
PKU_TAGGING = [p.encode('utf-8') for p in PKU_TAGGING]
PKU_NOT_ENTRY_IDXS = [i for i in range(len(PKU_TAGGING)) if i % 4 != 0 and i % 4 != 1 and i != len(PKU_TAGGING) - 1]
PKU_NOT_EXIT_IDXS = [i for i in range(len(PKU_TAGGING)) if i % 4 != 0 and i % 4 != 3 and i != len(PKU_TAGGING) - 1]
PKU_TRANS = numpy.zeros((len(PKU_TAGGING), len(PKU_TAGGING)), dtype='float32')
PKU_TRANS2 = numpy.zeros((len(PKU_TAGGING), len(PKU_TAGGING)), dtype='float32')
for j in range(len(PKU_TAGS)):
    for i in range(len(PKU_TAGS)):
        PKU_TRANS[j * 4][i * 4] = 1  # S<-S
        PKU_TRANS[j * 4][i * 4 + 3] = 1  # S<-E
        PKU_TRANS[j * 4][len(PKU_TAGGING) - 1] = 1  # S<-O
        PKU_TRANS[j * 4 + 1][i * 4] = 1  # B<-S
        PKU_TRANS[j * 4 + 1][i * 4 + 3] = 1  # B<-E
        PKU_TRANS[j * 4 + 1][len(PKU_TAGGING) - 1] = 1  # B<-O
for i in range(len(PKU_TAGS)):
    PKU_TRANS[i * 4 + 2][i * 4 + 1] = 1  # I<-B
    PKU_TRANS[i * 4 + 2][i * 4 + 2] = 1  # I<-I
    PKU_TRANS[i * 4 + 3][i * 4 + 1] = 1  # E<-B
    PKU_TRANS[i * 4 + 3][i * 4 + 2] = 1  # E<-I
for j in range(len(PKU_TAGS)):
    PKU_TRANS[len(PKU_TAGGING) - 1][len(PKU_TAGGING) - 1] = 1  # O<-O
    PKU_TRANS[len(PKU_TAGGING) - 1][j * 4] = 1  # O<-S
    PKU_TRANS[len(PKU_TAGGING) - 1][j * 4 + 3] = 1  # O<-E
for i in range(len(PKU_TAGGING)):
    for j in range(len(PKU_TAGGING)):
        if PKU_TRANS[i][j] == 0:
            PKU_TRANS2[i][j] = numpy.float32('-inf')
PKU_TRANS0 = numpy.zeros((len(PKU_TAGGING), len(PKU_TAGGING)), dtype='float32')
