import numpy
OOV_IDX = -3
BOS_IDX = -2
EOS_IDX = -1
POSS = ['SP', 'BA', 'FW', 'DER', 'DEV', 'MSP', 'ETC', 'JJ', 'DT', 'DEC', 'DEG', 'LB', 'LC', 'NN', 'PU', 'NP',
        'NR', 'NT', 'VA', 'VC', 'AD', 'CC', 'VE', 'M', 'CD', 'P', 'AS', 'VV', 'CS', 'PN', 'OD', 'SB']
POSS = [p.encode('utf-8') for p in POSS]
POSS += ['BOS', 'EOS']
TAGS = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ADV', 'BNF', 'CND', 'DIR', 'DIS', 'DGR', 'EXT', 'FRQ', 'LOC', 'MNR',
        'PRP', 'TMP', 'TPC']
TAGGING = []
for i in range(len(TAGS)):
    if 'ARG' not in TAGS[i]:
        TAGS[i] = 'ARGM-' + TAGS[i]
    TAGGING.append('S-' + TAGS[i])
    TAGGING.append('B-' + TAGS[i])
    TAGGING.append('I-' + TAGS[i])
    TAGGING.append('E-' + TAGS[i])
TAGGING.append('O')
TAGS = [p.encode('utf-8') for p in TAGS]

# ENTRY = numpy.zeros(len(TAGGING))
# EXIT = numpy.zeros(len(TAGGING))
# for i in range(len(TAGGING)):
#     if i % 4 != 0 and i % 4 != 1 and i != len(TAGGING) - 1:
#         ENTRY[i] = - float('inf')  # S-, B-, O- used in mask(output, i, s1, s2)
#     if i % 4 != 0 and i % 4 != 3 and i != len(TAGGING) - 1:
#         EXIT[i] = - float('inf')  # S-, E-, O-, used in mask(output, i, s1, s2)
NOT_ENTRY_IDXS = [i for i in range(len(TAGGING)) if i % 4 != 0 and i % 4 != 1 and i != len(TAGGING) - 1]
NOT_EXIT_IDXS = [i for i in range(len(TAGGING)) if i % 4 != 0 and i % 4 != 3 and i != len(TAGGING) - 1]

TRANS = numpy.zeros((len(TAGGING), len(TAGGING)), dtype='float32')
TRANS2 = numpy.zeros((len(TAGGING), len(TAGGING)), dtype='float32')
for j in range(len(TAGS)):
    for i in range(len(TAGS)):
        TRANS[j * 4][i * 4] = 1  # S<-S
        TRANS[j * 4][i * 4 + 3] = 1  # S<-E
        TRANS[j * 4][len(TAGGING) - 1] = 1  # S<-O
        TRANS[j * 4 + 1][i * 4] = 1  # B<-S
        TRANS[j * 4 + 1][i * 4 + 3] = 1  # B<-E
        TRANS[j * 4 + 1][len(TAGGING) - 1] = 1  # B<-O
for i in range(len(TAGS)):
    TRANS[i * 4 + 2][i * 4 + 1] = 1  # I<-B
    TRANS[i * 4 + 2][i * 4 + 2] = 1  # I<-I
    TRANS[i * 4 + 3][i * 4 + 1] = 1  # E<-B
    TRANS[i * 4 + 3][i * 4 + 2] = 1  # E<-I
for j in range(len(TAGS)):
    TRANS[len(TAGGING) - 1][len(TAGGING) - 1] = 1  # O<-O
    TRANS[len(TAGGING) - 1][j * 4] = 1  # O<-S
    TRANS[len(TAGGING) - 1][j * 4 + 3] = 1  # O<-E
for i in range(len(TAGGING)):
    for j in range(len(TAGGING)):
        if TRANS[i][j] == 0:
            TRANS2[i][j] = numpy.float32('-inf')
TRANS0 = numpy.zeros((len(TAGGING), len(TAGGING)), dtype='float32')
