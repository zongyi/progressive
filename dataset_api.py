# coding: utf-8
from path import *
from const import *
import numpy
import h5py
import os
import sys
from collections import OrderedDict


# This part should be modified before publish because not everyone has Chinese Tree Bank copyright.
# all process(self) (related to dataset splitting) should be a "pass".
class Datasets(object):
    def __init__(self, name):
        self.name = name

    def process(self):
        return None

    def get(self):
        exist, datasets = self.load()
        if not exist:
            datasets = self.process()
            self.dump()
        return datasets

    def dump(self):
        pass

    def load(self):
        return False, None


def load_embed_words(worddim):
    print('reading embed words ...')
    with open(vec_bin_f, 'r', newline='\n', errors='ignore') as fp:
        word2vecidx = dict()
        num = 0
        first_line = fp.readline().strip().split(' ')
        try:
            wordnum, _worddim = int(first_line[0]), int(first_line[1])
            assert worddim == _worddim
        except ValueError:
            word2vecidx[first_line[0]] = 0
            num += 1
        while 1:
            line = fp.readline()  # read all lines to get wordnum
            if not line: break
            word2vecidx[line[:line.find(' ')].encode('utf-8')] = num
            num += 1
        return word2vecidx


def load_vecs(vecs_idx):
    print('reading embed vecs ...')
    with open(vec_bin_f, 'r', newline='\n', errors='ignore') as fp:
        vecs = []
        first_line = fp.readline().strip().split(' ')
        num = 0
        try:
            wordnum = int(first_line[0])
        except ValueError:
            if num in vecs_idx:
                vecs.append([float(item) for item in first_line[1:]])
            num += 1
        while 1:
            line = fp.readline().strip()  # read all lines to get wordnum
            if not line: break
            if num in vecs_idx:
                vecs.append(numpy.asarray([float(item) for item in line.split(' ')[1:]], dtype='float32'))
            num += 1
    vecs.append(numpy.mean(vecs, 0, dtype='float32'))
    vecs.append(numpy.zeros_like(vecs[0], dtype='float32'))
    vecs.append(numpy.zeros_like(vecs[0], dtype='float32'))
    return vecs


def build_idx(words):
    word2idx = dict([(word, i) for i, word in enumerate(words)])
    idx2word = dict([(i, word) for i, word in enumerate(words)])
    return word2idx, idx2word


def process_one(window, max_dist, all_sents, max_len):
    def _distance(_i, _j, _max):
        """e.g. _i:   0 1 2 3 4 5 6 7
                _j:   - - - - + - - -
                temp: 7 5 3 1 0 2 4 6"""
        temp = abs(_i - _j) * 2 - (1 if _i < _j else 0)
        return temp if temp < _max else (EOS_IDX if _i > _j else BOS_IDX)

    ws_idx, ps_idx, vs_i, vs_idx, rs_idx = all_sents
    half_win = int((window - 1) / 2)
    w_sents, p_sents, dist_sents = [], [], []
    sent_lens = []  # for debug
    for w_idx, p_idx, v_i in zip(ws_idx, ps_idx, vs_i):
        w_sent, p_sent, dist_sent = [], [], []
        sent_len = len(w_idx)
        sent_lens.append(sent_len)
        for i in range(sent_len):
            if i == v_i: continue
            j = 0
            w_concat, p_concat = [], []
            while i - half_win + j < 0:
                w_concat.append(BOS_IDX)
                p_concat.append(BOS_IDX)
                j += 1
            while j < window and i - half_win + j < sent_len:
                w_concat.append(w_idx[i - half_win + j])
                p_concat.append(p_idx[i - half_win + j])
                j += 1
            while j < window:
                w_concat.append(EOS_IDX)
                p_concat.append(EOS_IDX)
                j += 1
            w_sent.append(w_concat)
            p_sent.append(p_concat)
            dist_sent.append(_distance(i, v_i, max_dist))
        w_sents.append(w_sent)
        p_sents.append(p_sent)
        dist_sents.append(dist_sent)
    dataset = [w_sents, p_sents, dist_sents, vs_i, vs_idx, rs_idx]
    max_len = max(max(sent_lens) - 1, max_len)
    print('Info:  max sent_len: %d, avg sent_len: %d,  sents: %d' % (
        max(sent_lens) - 1, sum(sent_lens) / len(sent_lens), len(vs_i)))
    return dataset, max_len


def pad(max_len, window, datasets):
    ret_datasets = []
    for dataset in datasets:
        w_sents, p_sents, dist_sents, vs_i, vs_idx, rs_idx = dataset
        n_samples = len(vs_i)
        ws = numpy.zeros((max_len, n_samples, window), dtype='int32')
        ps = numpy.zeros((max_len, n_samples, window), dtype='int32')
        dists = numpy.zeros((max_len, n_samples), dtype='int32')
        vis = numpy.zeros((n_samples), dtype='int32')
        vs = numpy.zeros((max_len, n_samples), dtype='int32')
        rs = numpy.zeros((max_len, n_samples), dtype='int32')
        lens = numpy.zeros((n_samples), dtype='int32')
        for i, (w, p, dist, v_i, v, r) in enumerate(zip(w_sents, p_sents, dist_sents, vs_i, vs_idx, rs_idx)):
            sent_len = len(w)
            ws[-sent_len:, i, :] = numpy.asarray(w, dtype='int32')
            ps[-sent_len:, i, :] = numpy.asarray(p, dtype='int32')
            dists[-sent_len:, i] = numpy.asarray(dist, dtype='int32')
            vis[i] = numpy.int32(v_i)
            vs[-sent_len:, i] = v * numpy.ones((sent_len), dtype='int32')
            rs[-sent_len:, i] = numpy.asarray(r, dtype='int32')
            lens[i] = numpy.int32(sent_len)
        ret_datasets.append([ws, ps, dists, vis, vs, rs, lens])
    return ret_datasets


class CpbDatasets(Datasets):  # 31235 2064 3604
    def __init__(self, name, worddim, window, max_dist, POSS, TAGGING, train_f, dev_f, test_f):
        super(CpbDatasets, self).__init__(name)
        self.worddim = worddim
        self.window = window
        self.max_dist = max_dist
        self.word2vecidx, self.vecs_idx, self.max_len = None, None, 0  # only for processing
        self.word2idx = OrderedDict()
        self.pos2idx, self.idx2pos = build_idx(POSS)
        self.role2idx = dict([(word, i) for i, word in enumerate(TAGGING)])
        self.datasets = []
        self.train_f = train_f
        self.dev_f = dev_f
        self.test_f = test_f
        self.name = name + '.%d.%d.%d.h5' % (worddim, window, max_dist)

    def read_sent__add_word(self, filename):
        line_num = 0
        zeroverb, multiverb = 0, 0
        ws_idx, ps_idx, rs_idx, vs_i, vs_idx = [], [], [], [], []
        num_seenwords = len(self.word2idx)
        with open(filename, 'rb') as fp:
            while 1:
                line = fp.readline()
                if not line:
                    break
                line_num += 1
                segs = line.strip().split(b' ')
                w_idx = []  # sen    len
                p_idx = []  # senpos len
                r_idx = []  # sentag len-1
                v_i = -1  # verb_i
                for i, seg in enumerate(segs):
                    role = seg[seg.rfind(b'/') + 1:]
                    tok_pos = seg[:seg.rfind(b'/')]
                    tok = tok_pos[:tok_pos.rfind(b'/')]
                    pos = tok_pos[tok_pos.rfind(b'/') + 1:]
                    items = [tok, pos, role]
                    if items[0] not in self.word2idx.keys():
                        if items[0] in self.word2vecidx.keys():
                            self.word2idx[items[0]] = num_seenwords
                            self.vecs_idx.append(self.word2vecidx[items[0]])
                            num_seenwords += 1
                    w_idx.append(self.word2idx.get(items[0], OOV_IDX))
                    p_idx.append(self.pos2idx[items[1]])
                    if items[-1] == b'rel':
                        if v_i == -1:
                            v_i = i
                        else:
                            v_i = -2
                            break
                    else:
                        r_idx.append(self.role2idx[items[-1]])
                if v_i == -1 or len(r_idx) <= 1:
                    zeroverb += 1
                    continue
                elif v_i == -2:
                    multiverb += 1
                    continue
                ws_idx.append(w_idx)
                ps_idx.append(p_idx)
                rs_idx.append(r_idx)
                vs_i.append(v_i)
                vs_idx.append(w_idx[v_i])
        print('lines: %d, zero verb sents: %d, multi verb sents: %d' % (line_num, zeroverb, multiverb))
        return ws_idx, ps_idx, vs_i, vs_idx, rs_idx

    def process(self):
        train_dataset, self.max_len = process_one(self.window, self.max_dist, self.read_sent__add_word(self.train_f),
                                                  self.max_len)
        dev_dataset, self.max_len = process_one(self.window, self.max_dist, self.read_sent__add_word(self.dev_f),
                                                self.max_len)
        test_dataset, self.max_len = process_one(self.window, self.max_dist, self.read_sent__add_word(self.test_f),
                                                 self.max_len)
        self.vecs = load_vecs(self.vecs_idx)
        self.datasets = pad(self.max_len, self.window, [train_dataset, dev_dataset, test_dataset])
        return self.datasets, self.vecs

    def dump(self):
        sys.stdout.write('dumping %s ... ' % self.name)
        with h5py.File('data/' + self.name, 'w') as f:
            for i, dataset in enumerate(self.datasets):
                f.create_dataset('%d_len' % (i), data=[len(dataset)])
                for j, d in enumerate(dataset):
                    f.create_dataset('%d_%d' % (i, j), data=d)
            f.create_dataset('vecs', data=self.vecs)
            f.create_dataset('word2idx', data=list(self.word2idx.keys()))
            print('all done.')

    def load(self):
        if not os.path.exists('data/' + self.name):
            self.word2vecidx = load_embed_words(self.worddim)
            self.vecs_idx = []
            return False, None
        sys.stdout.write('loading %s ... ' % self.name)
        with h5py.File('data/' + self.name, 'r') as f:
            for i in range(3):
                self.datasets.append([f['%d_%d' % (i, j)][:] for j in range(f['%d_len' % (i)][0])])
            self.vecs = f['vecs'][:]
            self.word2idx = OrderedDict([(word, i) for i, word in enumerate(f['word2idx'][:])])
            print('all done.')
        return True, (self.datasets, self.vecs)


def PkuDatasets_preprocess():
    # collect POSS, TAGGING
    # split datasets
    PKU_POSS = set()
    PKU_TAGS = set()
    lines = []
    with open(pku_text_f, 'rb') as fin:
        while 1:
            line = fin.readline().strip()
            if not line:
                break
            segs = line.split(b' ')
            new_segs, old_roles = [], []
            for i, seg in enumerate(segs):
                items = seg.split(b'/')
                PKU_POSS.add(items[-3].decode('utf-8'))
                PKU_TAGS.add(items[-1].decode('utf-8'))
                tok = seg[:seg.rfind(b'/')]
                tok = tok[:tok.rfind(b'/')]
                tok = tok[:tok.rfind(b'/')]
                new_segs.append([tok, items[-3]])
                old_roles.append(items[-1])
            sent_len = len(segs)
            for i, (tok, pos) in enumerate(new_segs):
                role = old_roles[i]
                if role in [b'rel', b'O']:
                    new_role = role
                else:
                    if i != 0 and old_roles[i - 1] == role:
                        if i == sent_len - 1 or old_roles[i + 1] != role:
                            new_role = b'E-' + role
                        else:
                            new_role = b'I-' + role
                    else:
                        if i == sent_len - 1 or old_roles[i + 1] != role:
                            new_role = b'S-' + role
                        else:
                            new_role = b'B-' + role
                new_segs[i] = tok.replace(b'/', b'*') + b'/' + pos + b'/' + new_role
            lines.append(b' '.join(new_segs))
    print(list(PKU_POSS))
    print(list(PKU_TAGS))
    print(len(lines))
    test_num = 1125
    dev_num = 973
    train_num = len(lines) - test_num - dev_num
    numpy.random.shuffle(lines)
    train_fout = open(pku_train_f, 'wb')
    dev_fout = open(pku_dev_f, 'wb')
    test_fout = open(pku_test_f, 'wb')
    for i, line in enumerate(lines):
        if i < train_num:
            train_fout.write(line + b'\n')
        elif i < train_num + dev_num:
            dev_fout.write(line + b'\n')
        else:
            test_fout.write(line + b'\n')
    train_fout.close()
    dev_fout.close()
    test_fout.close()


def pos_tag_cpb():
    in_files = [cpb_train_f, cpb_dev_f, cpb_test_f]
    out_files = [cpb_train_f_seg, cpb_dev_f_seg, cpb_test_f_seg]
    for f1, f2 in zip(in_files, out_files):
        if os.path.exists(f2): continue
        with open(f1, 'rb') as fin:
            with open(f2, 'wb') as fout:
                while 1:
                    line = fin.readline().strip()
                    if not line:
                        break
                    segs = line.split(b' ')
                    toks = []
                    for seg in segs:
                        items = seg.split(b'/')
                        assert len(items) == 3
                        toks.append(items[0])
                    fout.write(b' '.join(toks) + b'\n')
    print('cd ~/Data/wangzhen/bilsstm/text')
    print('. run_test.sh cpbtrain.seg cpb_pkupos_train.pos')
    print('. run_test.sh cpbdev.seg cpb_pkupos_dev.pos')
    print('. run_test.sh cpbtest.seg cpb_pkupos_test.pos')
    input('Are you sure all text has been tagged by pos-tagger?')
    pos_in_files = [cpb_pkupos_train_f_pre, cpb_pkupos_dev_f_pre, cpb_pkupos_test_f_pre]
    role_in_files = [cpb_train_f, cpb_dev_f, cpb_test_f]
    out_files = [cpb_pkupos_train_f, cpb_pkupos_dev_f, cpb_pkupos_test_f]
    for f1, f2, f3 in zip(pos_in_files, role_in_files, out_files):
        with open(f1, 'rb') as pos_fin:
            with open(f2, 'rb') as role_fin:
                with open(f3, 'wb') as fout:
                    while 1:
                        pos_line = pos_fin.readline().strip()
                        role_line = role_fin.readline().strip()
                        if not role_line:
                            break
                        pos_segs = pos_line.split(b' ')
                        pos_segs = pos_line.split(b' ')
                        role_segs = role_line.split(b' ')
                        new_segs = []
                        for pos_seg, role_seg in zip(pos_segs, role_segs):
                            pos_tok = pos_seg[:pos_seg.rfind(b'/') + 1]
                            pos2 = pos_seg[pos_seg.rfind(b'/') + 1:]
                            if pos2 in PKU_POSS2to1.keys():
                                pos = PKU_POSS2to1[pos2]
                            else:
                                pos = str.encode(pos2.decode('utf-8')[0])
                                print(pos2, pos)
                            assert role_seg.startswith(pos_tok)
                            new_segs.append(pos_tok + pos + role_seg[role_seg.rfind(b'/'):])
                        fout.write(b' '.join(new_segs) + b'\n')


def cpb2pku_mapper(cpb, pku):
    mapper = dict([(-3, -3), (-2, -2), (-1, -1)])
    for word in cpb.word2idx.keys():
        mapper[cpb.word2idx[word]] = pku.word2idx.get(word, OOV_IDX)
    print('mapper ready.')
    return mapper


def map_cpb2pku(mapper, ws, vs, lens):
    ws1 = numpy.zeros_like(ws, dtype='int32')
    vs1 = numpy.zeros_like(vs, dtype='int32')
    for sample_i, sent_len in enumerate(lens):
        for i in range(1, sent_len + 1):
            vs1[-i, sample_i] = mapper[vs[-i, sample_i]]
            for win_i in range(ws.shape[2]):
                ws1[-i, sample_i, win_i] = mapper[ws[-i, sample_i, win_i]]
    print('mapping done.')
    return ws1, vs1

if __name__ == '__main__':
    # data = CpbDatasets('cpb', 50, 3, 500, CPB_POSS, CPB_TAGGING, cpb_train_f, cpb_dev_f, cpb_test_f).get()
    # print(data[0][0][0].shape[1],data[0][1][0].shape[1],data[0][2][0].shape[1])
    # PkuDatasets_preprocess()
    data = CpbDatasets('pku', 50, 3, 500, PKU_POSS, PKU_TAGGING, pku_train_f, pku_dev_f, pku_test_f).get()
    # pos_tag_cpb()
    data = CpbDatasets('cpb_pkupos', 50, 3, 500, PKU_POSS, CPB_TAGGING, cpb_pkupos_train_f, cpb_pkupos_dev_f,
                       cpb_pkupos_test_f).get()
    #wordmap()
