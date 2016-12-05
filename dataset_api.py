from const import *
import pickle
import sys


def find(word, wordlist, w_num):
    for i in range(w_num):
        if wordlist[i] == word:
            return i
    return w_num


def find_pos(pos, POSS):
    for i in range(len(POSS)):
        if POSS[i] == pos:
            return i
    print('stuck')
    input()
    return -1


def label(tag, TAGGING):
    for i in range(len(TAGGING)):
        if TAGGING[i] == tag:
            return i
    print(tag, 'stuck')
    input()
    return len(TAGGING) - 1


def dis_vector(i, tag, maxnum):
    pos = abs(i - tag) * 2 - (1 if i < tag else 0)
    if pos >= maxnum and i > tag:
        return -1
    if pos >= maxnum and i < tag:
        return -2
    return pos


def load_data(w_num=None, use_vecs=False):
    sys.stdout.write('load_data ')
    fp = open(vec_bin_f, 'r', newline='\n', errors='ignore')
    info = fp.readline()
    infostr = info.strip().split(' ')
    wordcount = int(infostr[0])
    worddim = int(infostr[1])

    wordlist = []
    embeddings = []
    if w_num is None or w_num > wordcount:
        w_num = wordcount
    while 1:
        tline = fp.readline()
        if not tline:
            break
        str = tline[:-1].split(' ')
        word = str[0]
        vectors = []
        for i in range(worddim):
            vectors.append(float(str[i + 1]))
        wordlist.append(word)
        embeddings.append(vectors)
    fp.close()
    embeddings_final = []
    if use_vecs:
        j = 0
        while j < w_num:
            embeddings_final.append(embeddings[j])
            j += 1

        embeddings_unknown = [0 for i in range(worddim)]
        while j < wordcount:
            for i in range(worddim):
                embeddings_unknown[i] += embeddings[j][i]
            j += 1
        if w_num != wordcount:
            for i in range(worddim):
                embeddings_unknown[i] /= (wordcount - w_num)
        embeddings_final.append(embeddings_unknown)
        embeddings_final.append([0 for i in range(worddim)])
        embeddings_final.append([0 for i in range(worddim)])
    print('done.')
    wordinfo = [wordlist, embeddings_final]
    return wordinfo


def get_dataset(window, v_num, w_num, wordlist, datafile, pklfile, POSS, TAGGING):
    r = (window - 1) / 2
    r = int(r)
    allwords = []
    allposes = []
    allverbs = []
    allidxes = []
    allvdxes = []
    allss = []
    alltags = []
    sensnum = 0
    fp = open(datafile, 'r')
    sss = 0
    nouncount = 0
    allcount = 0
    senlength = 0
    while 1:
        tline = fp.readline()
        if not tline:
            break
        sensnum += 1
        st = tline.strip().split(' ')
        sen = []
        senpos = []
        sentag = []
        verb_pos = -1
        for i in range(len(st)):
            pairs = st[i].split('/')
            if len(pairs) != 3:
                print('stuck')
                input()
            word = find(pairs[0], wordlist, w_num)
            pos = find_pos(pairs[1], POSS)
            if word == w_num:
                nouncount += 1
            allcount += 1
            sen.append(word)
            senpos.append(pos)
            if pairs[-1] == 'rel':
                if verb_pos == -1:
                    verb_pos = i
                else:
                    print('verb_pos!=-1')
                    verb_pos = -2
                    break
            else:
                sentag.append(label(pairs[-1], TAGGING))

        if verb_pos == -1 or verb_pos == -2 or len(sentag) <= 1:
            sss += 1
            continue

        alltags.append(sentag)
        allverbs.append(sen[verb_pos])
        allss.append(verb_pos)
        if len(sentag) > senlength:
            senlength = len(sentag)

        allword = []
        allpos = []
        allidx = []
        allvdx = []
        for i in range(len(st)):
            words = []
            poses = []
            j = 0
            while i - r + j < 0:
                words.append(-2)
                poses.append(-2)
                j += 1
            while j < window and i - r + j < len(st):
                words.append(sen[i - r + j])
                poses.append(senpos[i - r + j])
                j += 1
            while j < window:
                words.append(-1)
                poses.append(-1)
                j += 1
            allword.append(words)
            allpos.append(poses)
            allvdx.append(dis_vector(i, verb_pos, v_num))

        for i in range(len(st)):
            pairs = st[i].split('/')
            if pairs[-1] == 'rel':
                continue
            idx = []
            for j in range(len(st)):
                idx.append(dis_vector(j, i, v_num))

            allidx.append(idx)

        allwords.append(allword)
        allposes.append(allpos)
        allidxes.append(allidx)
        allvdxes.append(allvdx)

        print(sss)
        sss += 1
    fp.close()
    print('sss: %d' % sss)
    trainset = [allwords, allposes, allidxes, allvdxes, allverbs, allss, alltags, sensnum]
    pickle.dump(trainset, open(pklfile, 'wb'))
    return trainset


if __name__ == '__main__':
    n_in_w = 100000
    wordinfo = load_data(n_in_w)
    wordlist = wordinfo[0]
    get_dataset(3, 500, n_in_w, wordlist, pku_cpbpos_train_f, pku_cpbpos_train_pkl, CPB_POSS, PKU_TAGGING)
    get_dataset(3, 500, n_in_w, wordlist, pku_cpbpos_test_f, pku_cpbpos_test_pkl, CPB_POSS, PKU_TAGGING)
