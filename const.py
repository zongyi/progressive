OOV_IDX = -3
BOS_IDX = -2
EOS_IDX = -1
POSS = ['SP', 'BA', 'FW', 'DER', 'DEV', 'MSP', 'ETC', 'JJ', 'DT', 'DEC', 'DEG', 'LB', 'LC', 'NN', 'PU', 'NP',
        'NR', 'NT', 'VA', 'VC', 'AD', 'CC', 'VE', 'M', 'CD', 'P', 'AS', 'VV', 'CS', 'PN', 'OD', 'SB']
TAGS = ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ADV", "BNF", "CND", "DIR", "DIS", "DGR", "EXT", "FRQ", "LOC", "MNR",
        "PRP", "TMP", "TPC"]
TAGGING = []
for i in range(len(TAGS)):
    if 'ARG' not in TAGS[i]:
        TAGS[i] = 'ARGM-' + TAGS[i]
    TAGGING.append("S-" + TAGS[i])
    TAGGING.append("B-" + TAGS[i])
    TAGGING.append("I-" + TAGS[i])
    TAGGING.append("E-" + TAGS[i])
TAGGING.append("O")
POSS = [p.encode('utf-8') for p in POSS]
TAGS = [p.encode('utf-8') for p in TAGS]
TAGGING = [p.encode('utf-8') for p in TAGGING]
