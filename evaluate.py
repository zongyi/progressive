# compare a: result with b:answer
def cal_f1(a, b, TAGGING):
    a_seq = []
    b_seq = []
    for i in range(len(a)):
        a_seq.append(TAGGING[a[i]])
        b_seq.append(TAGGING[b[i]])

    s1 = 0
    s2 = 0

    for item in a_seq:
        if item.startswith('S-') or item.startswith('B-'):
            s1 += 1

    for item in b_seq:
        if item.startswith('S-') or item.startswith('B-'):
            s2 += 1

    s3 = 0
    i = 0
    while i < len(a_seq):
        if a_seq[i].startswith('S-'):
            if b_seq[i] == a_seq[i]:
                s3 += 1
        elif a_seq[i].startswith('B-'):
            isMatch = True
            while not a_seq[i].startswith('E-'):
                if a_seq[i] != b_seq[i]:
                    isMatch = False
                i += 1
            if a_seq[i] != b_seq[i]:
                isMatch = False
            if isMatch:
                s3 += 1
        i += 1

    return [s3, s1, s2]
