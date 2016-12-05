import numpy
from path import *

CPB_TAGS = ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ADV", "BNF", "CND", "DIR", "DIS", "DGR", "EXT", "FRQ", "LOC",
            "MNR", "PRP", "TMP", "TPC"]
CPB_POSS = ['SP', 'BA', 'FW', 'DER', 'DEV', 'MSP', 'ETC', 'JJ', 'DT', 'DEC', 'DEG', 'LB', 'LC', 'NN', 'PU', 'NP', 'NR',
            'NT', 'VA', 'VC', 'AD', 'CC', 'VE', 'M', 'CD', 'P', 'AS', 'VV', 'CS', 'PN', 'OD', 'SB']

# CPB_TAGGING in IBOES schema
CPB_TAGGING = []
for i in range(len(CPB_TAGS)):
    if 'ARG' not in CPB_TAGS[i]:
        CPB_TAGS[i] = 'ARGM-' + CPB_TAGS[i]
    CPB_TAGGING.append("S-" + CPB_TAGS[i])
    CPB_TAGGING.append("B-" + CPB_TAGS[i])
    CPB_TAGGING.append("I-" + CPB_TAGS[i])
    CPB_TAGGING.append("E-" + CPB_TAGS[i])
CPB_TAGGING.append("O")

# for valid tag paths used in calculating partition function
CPB_TRANS = numpy.zeros((len(CPB_TAGGING), len(CPB_TAGGING)))
CPB_TRANS2 = numpy.zeros((len(CPB_TAGGING), len(CPB_TAGGING)))
CPB_TRANS0 = numpy.zeros((len(CPB_TAGGING), len(CPB_TAGGING)))
CPB_NOT_ENTRY_IDXS = numpy.zeros(len(CPB_TAGGING))
CPB_NOT_EXIT_IDXS = numpy.zeros(len(CPB_TAGGING))
for j in range(len(CPB_TAGS)):
    for i in range(len(CPB_TAGS)):
        CPB_TRANS[j * 4][i * 4] = 1
        CPB_TRANS[j * 4][i * 4 + 3] = 1
        CPB_TRANS[j * 4][len(CPB_TAGGING) - 1] = 1
        CPB_TRANS[j * 4 + 1][i * 4] = 1
        CPB_TRANS[j * 4 + 1][i * 4 + 3] = 1
        CPB_TRANS[j * 4 + 1][len(CPB_TAGGING) - 1] = 1

for i in range(len(CPB_TAGS)):
    CPB_TRANS[i * 4 + 2][i * 4 + 1] = 1
    CPB_TRANS[i * 4 + 2][i * 4 + 2] = 1
    CPB_TRANS[i * 4 + 3][i * 4 + 1] = 1
    CPB_TRANS[i * 4 + 3][i * 4 + 2] = 1

for j in range(len(CPB_TAGS)):
    CPB_TRANS[len(CPB_TAGGING) - 1][len(CPB_TAGGING) - 1] = 1
    CPB_TRANS[len(CPB_TAGGING) - 1][j * 4] = 1
    CPB_TRANS[len(CPB_TAGGING) - 1][j * 4 + 3] = 1

for i in range(len(CPB_TAGGING)):
    for j in range(len(CPB_TAGGING)):
        if CPB_TRANS[i][j] == 0:
            CPB_TRANS2[i][j] = -float("inf")

for i in range(len(CPB_TAGGING)):
    if i % 4 != 0 and i % 4 != 1 and i != len(CPB_TAGGING) - 1:
        CPB_NOT_ENTRY_IDXS[i] = - float("inf")

for i in range(len(CPB_TAGGING)):
    if i % 4 != 0 and i % 4 != 3 and i != len(CPB_TAGGING) - 1:
        CPB_NOT_EXIT_IDXS[i] = - float("inf")

PKU_POSS = ['l', 'Bg', 'k', 'd', 'q', 'h', 'u', 'p', 't', 'Mg', 'r', 'v', 'w', 'j', 'e', 'c', 'o', 's', 'Ng', 'nr', 'i',
            'b', 'f', 'n', 'm', 'z', 'vd', 'ad', 'vn', 'an', 'Tg', '%', 'Dg', 'nz', 'ns', 'nt', 'Qg', 'a', 'Ag', 'Vg',
            'y', 'Ug']
PKU_POSS2to1 = {'ld': 'l', 'Bg': 'Bg', 'l': 'l', 'la': 'l', 'ln': 'l', 'lm': 'l', 'k1': 'k', 'lv': 'l',
                'df': 'd',
                'dc': 'd', 'du': 'd', 'qt': 'q', 'qr': 'q', 'd': 'd', 'qv': 'q', 'h': 'h', 'uz': 'u',
                'p': 'p', 'qc': 'q', 'q': 'q', 'qe': 'q', 't': 't', 'qj': 'q', 'ql': 'q', 'qz': 'q',
                'uv': 'u',
                'Mg': 'Mg', 'rr': 'r', 'rz': 'r', 'vt2': 'v', 'ui': 'u', 'qd': 'q', 'wf': 'w',
                'vx': 'v', 'wd': 'w',
                'wm': 'w', 'wj': 'w', 'ww': 'w', 'jn': 'j', 'wt': 'w', 'ws': 'w', 'wp': 'w', 'wzz': 'w',
                'e1': 'e',
                'wy': 'w', 'c': 'c', 'k': 'k', 'wu': 'w', 'o': 'o', 'ud': 'u', 's': 's', 'w': 'w',
                'Ng': 'Ng',
                'dfu': 'd', 'u1': 'u', 'nrg': 'nr', 'nrf': 'nr', 'iv': 'i', 'vi_a': 'v', '': '',
                'f': 'f',
                'us': 'u', 'n': 'n', 'ul': 'u', 'uo': 'u', 'r': 'r', 'mq': 'm', 'v': 'v', 'ue': 'u',
                'vt': 'v',
                'z': 'z', 'vd': 'vd', 'ad': 'ad', 'vi': 'v', 'vl': 'v', 'vn': 'vn', 'an': 'an',
                'wky': 'w', 'vq': 'v',
                'wkz': 'w', 'im': 'i', 'vu': 'v', 'in': 'i', 'ia': 'i', 'Tg': 'Tg', 'ic': 'i',
                'id': 'i',
                'nh': 'n', 'Tg1': 'Tg', '%': '%', 'Dg': 'Dg', 'nx': 'n', 'nz': 'nz', 'ryw': 'r',
                'nr': 'nr', 'ns': 'ns',
                'nt': 'nt', 'Qg': 'Qg', 'a': 'a', 'e': 'e', 'Ag': 'Ag', 'Vg': 'Vg', 'i': 'i', 'm': 'm',
                'wyz': 'w',
                'wyy': 'w', 'n]nt': 'n', 'vt_a': 'v', 'u': 'u', 'y': 'y', 'tt': 't',
                'j': 'j', 'jd': 'j',
                'jv': 'j', 'rzw': 'r', 'Ug': 'Ug'}
PKU_TAGS = ['原因', '时间', '比较主体', '路径', '接事', '终点', '范围', '物量', '起始', '比较对象', '处所',
            '施事', '材料', '方式', '与事', '起点', '方向', '受事', '目的', '当事', '比较项', '结束', '比较客体', '对象',
            '工具', '结果', '内容', '比较结果', '系事', '同事', '时段']

PKU_TAGGING = []
for i in range(len(PKU_TAGS)):
    PKU_TAGGING.append("S-" + PKU_TAGS[i])
    PKU_TAGGING.append("B-" + PKU_TAGS[i])
    PKU_TAGGING.append("I-" + PKU_TAGS[i])
    PKU_TAGGING.append("E-" + PKU_TAGS[i])
PKU_TAGGING.append("O")

# for valid tag paths used in calculating partition function
PKU_TRANS = numpy.zeros((len(PKU_TAGGING), len(PKU_TAGGING)))
PKU_TRANS2 = numpy.zeros((len(PKU_TAGGING), len(PKU_TAGGING)))
PKU_TRANS0 = numpy.zeros((len(PKU_TAGGING), len(PKU_TAGGING)))
PKU_NOT_ENTRY_IDXS = numpy.zeros(len(PKU_TAGGING))
PKU_NOT_EXIT_IDXS = numpy.zeros(len(PKU_TAGGING))
for j in range(len(PKU_TAGS)):
    for i in range(len(PKU_TAGS)):
        PKU_TRANS[j * 4][i * 4] = 1
        PKU_TRANS[j * 4][i * 4 + 3] = 1
        PKU_TRANS[j * 4][len(PKU_TAGGING) - 1] = 1
        PKU_TRANS[j * 4 + 1][i * 4] = 1
        PKU_TRANS[j * 4 + 1][i * 4 + 3] = 1
        PKU_TRANS[j * 4 + 1][len(PKU_TAGGING) - 1] = 1

for i in range(len(PKU_TAGS)):
    PKU_TRANS[i * 4 + 2][i * 4 + 1] = 1
    PKU_TRANS[i * 4 + 2][i * 4 + 2] = 1
    PKU_TRANS[i * 4 + 3][i * 4 + 1] = 1
    PKU_TRANS[i * 4 + 3][i * 4 + 2] = 1

for j in range(len(PKU_TAGS)):
    PKU_TRANS[len(PKU_TAGGING) - 1][len(PKU_TAGGING) - 1] = 1
    PKU_TRANS[len(PKU_TAGGING) - 1][j * 4] = 1
    PKU_TRANS[len(PKU_TAGGING) - 1][j * 4 + 3] = 1

for i in range(len(PKU_TAGGING)):
    for j in range(len(PKU_TAGGING)):
        if PKU_TRANS[i][j] == 0:
            PKU_TRANS2[i][j] = -float("inf")

for i in range(len(PKU_TAGGING)):
    if i % 4 != 0 and i % 4 != 1 and i != len(PKU_TAGGING) - 1:
        PKU_NOT_ENTRY_IDXS[i] = - float("inf")

for i in range(len(PKU_TAGGING)):
    if i % 4 != 0 and i % 4 != 3 and i != len(PKU_TAGGING) - 1:
        PKU_NOT_EXIT_IDXS[i] = - float("inf")
