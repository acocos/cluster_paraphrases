import sys
import numpy as np
from copy import deepcopy
from collections import namedtuple, defaultdict, OrderedDict
from scipy import sparse
from sklearn.cluster import SpectralClustering
from sklearn import preprocessing
from scipy.spatial.distance import cosine
import networkx as nx
import cPickle as pickle
import cluster_rotate as cr
import hgfc
import entropy
import sem_clust
from nltk.stem.lancaster import LancasterStemmer
from sklearn.metrics import silhouette_score


word_type = namedtuple("word_type", "word, type")

class Paraphrase:
    def __init__(self, wt):
        '''
        Initialize Paraphrase object.
        :param wt: word_type object
        '''
        self.word = wt.word
        self.pos = wt.type
        self.word_type = wt
        self.vector = []

    def load_vec(self, vec):
        '''
        Load embedding from vector
        :param vec: np array
        '''
        self.vector = vec

    def as_string(self):
        outstring = self.word + ';'
        return outstring

    def jdefault(self):
        return self.__dict__

class ParaphraseSet:
    def __init__(self, tgt_wt, pp_dict):
        '''
        Initialize ParaphraseSet from target word_type and a dictionary of
        Paraphrase objects, where the key is the Paraphrase.word and value is
        Paraphrase object
        :param tgt_wt: word_type
        :param pp_dict: {word -> Paraphrase}
        '''
        self.target_word = tgt_wt.word
        self.pos = tgt_wt.type
        self.word_type = tgt_wt
        self.pp_dict = OrderedDict(pp_dict)
        self.sense_clustering = defaultdict(set)
        self.cluster_count = 0
        self.own_vector = []

    def zmp_cluster(self, w2p, entail_scores=None):
        '''
        Perform unsupervised spectral clustering with local scaling

        Uses PPDB2.0Score directly to calculate similarities, and uses
        word2vec distributional similarities to choose number of clusters.

        [Zelnik-Manor and Perona. Self-Tuning Spectral Clustering. NIPS 2004.]

        :param: w2p - 2-layer dict of {word -> word -> score}
        :param: entail_scores - 2-layer dict of {word -> word -> score}
        :return: (updated sense clustering)
        '''
        self.reset_sense_clustering()
        wordlist = sorted(self.pp_dict.keys())
        oov = [w for w in wordlist if w not in w2p]
        if len(oov) > 0:
            sys.stderr.write('WARNING: Paraphrases %s are OOV. Removing from ppset.\n' % str(oov))
            wordlist = list(set(wordlist) - set(oov))

        ## Consolidate same-stem paraphrases, and check whether result has 2 or less terms
        wordlist_map = self.consolidate_stemmed_wordlist(wordlist, w2p)
        consol_wordlist = sorted(wordlist_map.keys())
        labeldict = self.check_tooeasy(consol_wordlist, w2p)
        singletons = set([])

        ## If not, use spectral clustering
        done = False
        if len(labeldict) == 0:

            ## Step 1: Form similarities matrix using PPDB2.0Score directly
            N = len(wordlist)
            sims = get_sims_matrix('direct', w2p, consol_wordlist)

            ## Step 1b. Check for singletons
            singind = [i for i,row in enumerate(sims*(1-np.identity(len(consol_wordlist)))) if np.linalg.norm(row,0)==0]
            if len(singind) > 0:
                singletons |= set([consol_wordlist[i] for i in singind])
                consol_wordlist = sorted(list(set(consol_wordlist) - set(singletons)))
                sims = get_sims_matrix('direct', w2p, consol_wordlist)
                labeldict = self.check_tooeasy(consol_wordlist, w2p)
                if len(labeldict) > 0:  # if removing singletons gets wordlist down to size 2 or less
                    done = True
                if len(sims) == 0:
                    labeldict = {r: [w] for r,w in enumerate(singletons)}
                    done = True

            if not done:
                ## Step 2: Compute sils matrix using distributional similarity of word2vec vectors
                wlst, x = self.vec_matrix()
                distrib_sims = dict(zip(wlst,x))
                sils = get_sils_matrix('vec_cosine', distrib_sims, consol_wordlist)

                ## Step 3: Incorporate entailments
                if entail_scores is not None:
                    entailments = np.array([[1-entail_scores.get(i,{}).get(j,1.0) for j in consol_wordlist] for i in consol_wordlist])
                    sims *= entailments

                ## Step 4: Compute Laplacian
                sigmas = cr.local_scaling_sims(sims)
                A = cr.affinities_sims(sims,sigmas)
                L = cr.laplacian(A)

                ## Step 5: Determine range of clustering sizes (k) to try
                maxC = 20
                evals, V = cr.evecs(L, maxC)
                thresh = 0.94
                group_num = []
                while len(group_num) == 0:
                    thresh += 0.01
                    if thresh >= 1.0:
                        group_num = range(2,min(20, len(consol_wordlist)))
                        sys.stderr.write("ERROR: Cannot find acceptable number of clusters to try. Defaulting to [2,%d]\n" % max(group_num))
                        break
                    minC = max(2, len([e for e in evals if e > thresh]))
                    group_num = range(minC, min(maxC, len(consol_wordlist)))


                ## Step 6: Cluster for each size k in range, and choose best one
                clusterings = []
                scores = []
                for k in group_num:
                    sc = SpectralClustering(n_clusters=k, affinity='precomputed')
                    sc.fit(L)
                    labels = sc.fit_predict(L)
                    ld = {l: [pp for (ll,pp) in zip(labels, consol_wordlist) if ll==l]
                          for l in set(labels)}
                    clusterings.append(ld)
                    if len(set(labels)) > 2 and len(set(labels)) < len(consol_wordlist)-1:  ## TODO: Fix this arbitrary size limit imposed by silhouette_score
                        scores.append(silhouette_score(sils, labels, metric='precomputed'))
                    else:
                        scores.append(0.0)
                labeldict = clusterings[np.argmax(scores)]

        sol = self.expand_solution(labeldict, wordlist_map, singletons)

        for k, lst in sol.iteritems():
            if len(lst) > 0:
                self.add_sense_cluster(lst)


    def hgfc_cluster(self, w2p, entail_scores=None):
        '''
        Perform Hierarchical Graph Factorization Clustering (HGFC):

        Kai Yu, Shipeng Yu, and Volker Tresp. Soft clustering on graphs. NIPS 18:1553,2006.

        Uses word2vec cosine sim as distributional similarity measure, and uses
        PPDB2.0Score to choose number of clusters.

        :param: w2p - dict of word -> paraphrase dict {str -> {str -> float}}
        :return: (updated sense clustering)
        '''
        self.reset_sense_clustering()
        wordlist = sorted(self.pp_dict.keys())
        oov = [w for w in wordlist if w not in w2p]
        if len(oov) > 0:
            sys.stderr.write('WARNING: Paraphrases %s are OOV. Removing from ppset.\n' % str(oov))
            wordlist = list(set(wordlist) - set(oov))

        ## Consolidate same-stem paraphrases, and check whether result has 2 or less terms
        wordlist_map = self.consolidate_stemmed_wordlist(wordlist, w2p)
        consol_wordlist = sorted(wordlist_map.keys())
        labeldict = self.check_tooeasy(consol_wordlist, w2p)
        singletons = set([])

        ## If not, use HGFC
        clusterings = labeldict
        if len(labeldict) == 0:

            ## Step 1: Form similarities matrix using w2vec cosine similarity
            wlst,x = self.vec_matrix()
            distrib_sims = dict(zip(wlst,x))
            sims = get_sims_matrix('vec_cosine', distrib_sims, consol_wordlist)

            ## Step 2: Compute sils matrix using PPDB2.0Score
            sils = get_sils_matrix('direct', w2p, consol_wordlist)

            ## Step 3: Incorporate entailments
            if entail_scores is not None:
                entailments = np.array([[1-entail_scores.get(i,{}).get(j,1.0)
                                         for j in consol_wordlist]
                                        for i in consol_wordlist])
                sims *= entailments

            ## Step 4: Normalize sims matrix
            sims = preprocessing.normalize(np.matrix(sims), norm='l2')

            ## Step 4: Run HGFC Clustering
            maxsc = =1.
            iter = 0
            scores = None
            while maxsc <= 0.:
            try:
                clusterings, scores = hgfc.h_cluster(wordlist, sims, sils)
                maxsc = max(scores)
                iter += 1
                if iter >= 10:
                    break
            except KeyError:
                continue

            labeldict = clusterings[np.argmax(scores)]

        sol = self.expand_solution(labeldict, wordlist_map, singletons)

        for k, lst in sol.iteritems():
            if len(lst) > 0:
                self.add_sense_cluster(lst)

        return clusterings


    def sem_clust(self, w2p, simsdict):
        ''' Baseline SEMCLUST method (dynamic thresholding), based on:

        Marianna Apidianaki, Emilia Verzeni, and Diana McCarthy. Semantic
        Clustering of Pivot Paraphrases. In LREC 2014.

        Builds a graph where nodes are words, and edges connect words that
        have a connection in <w2p>. Weights edges by the values given in
        <simsdict>.
        :param w2p: word -> {paraphrase: score} dictionary, used to decide which nodes to connect with edges
        :param simsdict: word -> {paraphrase: score} OR word -> vector, used for edge weights
        :return:
        '''
        self.reset_sense_clustering()
        wordlist = self.pp_dict.keys()

        oov = [w for w in wordlist if w not in w2p or w not in simsdict]
        if len(oov) > 0:
            sys.stderr.write('WARNING: Paraphrases %s are OOV. '
                             'Removing from ppset.\n' % str(oov))
            wordlist = list(set(wordlist) - set(oov))

        if len(wordlist) == 1:
            self.add_sense_cluster([wordlist[0]])
            return

        # Using cosine similarity of word-paraphrase vectors:
        if type(simsdict.values()[0]) != dict:
            similarities = np.array([[1-cosine(simsdict[i], simsdict[j])
                                      for j in wordlist] for i in wordlist])
        else:
            similarities = np.array([[(1-dict_cosine_dist(simsdict[i], simsdict[j]))
                                      for j in wordlist] for i in wordlist])

        gr = sem_clust.toGraph(similarities, wordlist, self.target_word, w2p)

        for c in nx.connected_components(gr):
            self.add_sense_cluster(c)

    def reset_sense_clustering(self):
        self.sense_clustering = {}
        self.cluster_count = 0

    def add_sense_cluster(self, clus):
        '''
        Add sense cluster to dictionary of
        {int -> set}
        where int indicates the cluster number and the set contains strings
        :clus: list of strings
        :return:
        '''
        self.cluster_count += 1
        self.sense_clustering[self.cluster_count] = set(clus)

    def load_vecs(self, vec_dict):
        '''
        Load paraphrase vectors for all paraphrases in vec_dict and the target
        word

        If paraphrase is not a key in vec_dict, try the following:
        1) Split by '-' and '_' and take the average of resulting words
        2) Americanize and take the American version
        :param vec_dict: {word -> np.array}
        :return:
        '''
        veckeys = set(vec_dict.keys())
        for p in self.pp_dict.itervalues():
            try:
                p.load_vec(vec_dict[p.word])
            except KeyError:
                # Check to see whether pieces of the composition are in the dict
                pieces = flatten([ww.split('-') for ww in p.word.split('_')])
                v = np.zeros(vec_dict.values()[0].shape[0])
                n = 0
                for wrd in pieces:
                    if wrd in veckeys:
                        v += vec_dict[wrd]
                        n += 1
                    elif try_americanize(wrd, veckeys) is not None:
                        proxy = try_americanize(wrd, veckeys)
                        v += vec_dict[proxy]
                        n += 1
                    elif len(wrd) > 1 and wrd[-1]=='s' and wrd[:-1] in veckeys:
                        proxy = wrd[:-1]
                        v += vec_dict[proxy]
                        n += 1
                    else:
                        pass
                v /= max(n, 1.0)
                p.load_vec(v)

        try:
            self.own_vector = vec_dict[self.target_word]
        except KeyError:
            # Check to see whether pieces of the composition are in the dict
            pieces = flatten([ww.split('-') for ww in self.target_word.split('_')])
            v = np.zeros(vec_dict.values()[0].shape[0])
            n = 0
            for wrd in pieces:
                if wrd in veckeys:
                    v += vec_dict[wrd]
                    n += 1
                elif try_americanize(wrd, veckeys) is not None:
                    proxy = try_americanize(wrd, veckeys)
                    v += vec_dict[proxy]
                    n += 1
                elif len(wrd) > 1 and wrd[-1]=='s' and wrd[:-1] in veckeys:
                    proxy = wrd[:-1]
                    v += vec_dict[proxy]
                    n += 1
                else:
                    pass
            v /= max(n, 1.0)
            self.own_vector = v



    def vec_matrix(self):
        '''
        Return a matrix of vectors for all paraphrases in this set and a list
        that gives the words corresponding to each row
        :return: list, np.array (len(ppdict) x len(vector))
        '''
        zippedlist = [(w, p.vector) for w,p in self.pp_dict.items()]
        words, vecs = zip(*zippedlist)
        return words, np.array(vecs)

    def sparse_vec_matrix(self, include_self=True):
        '''
        Returns a sparse matrix of sparse vectors for all paraphrases in this
        set and a list that gives the words corresponding to each row
        :return: list, np.csr_matrix (len(ppdict) x len(vector))
        '''
        zippedlist = [(w, p.vector) for w,p in self.pp_dict.items() if len(p.vector.shape) > 1]
        if include_self:
            zippedlist.append((self.target_word, self.own_vector))
        words, vecs = zip(*zippedlist)
        return words, sparse.vstack(vecs).todense()

    def as_str(self):
        outline = self.target_word + '.' + self.pos + ' :: '
        for pp in self.pp_dict:
            outline += self.pp_dict[pp].as_string() + ' '
        return outline.strip()

    def filter_ppset_by_gold(self, goldfile):
        '''
        Filter paraphrase sets in ppdict to include only words that appear in gold classes
        '''
        pp_sets = deepcopy(self.pp_dict)
        try:
            goldsoln = read_gold(goldfile)[self.word_type]  # ParaphraseSet object
            filtered = set([w.word for w in goldsoln.get_paraphrase_wtypes()])
        except KeyError:
            sys.stderr.write("No gold paraphrase set found for target %s..."
                             "No gold filtering performed.\n"
                             % self.target_word)
            filtered = set(pp_sets.keys())

        pp_sets = {w: pp_sets[w] for w in set(pp_sets.keys()) & filtered}
        self.pp_dict = pp_sets

    def filter_sense_clustering(self, otherppset):
        '''
        Filter sense clustering to include only terms that appear as
        paraphrases of otherppset
        '''
        if type(otherppset) == set:  # can pass just a set instead
            filtered = otherppset
        else:
            filtered = set([w.word for w in otherppset.get_paraphrase_wtypes()])
        self.sense_clustering = \
            {num: (clus & filtered) for num, clus in self.sense_clustering.iteritems()}
        self.sense_clustering = {k: v for k,v in self.sense_clustering.iteritems() if len(v) > 0}
        self.cluster_count = len(self.sense_clustering)

    def check_tooeasy(self, wordlist, w2p):
        '''
        Check if Paraphrase Set has only one or two elements, and
        cluster those simply if so.
        :return:
        '''
        labeldict = {}
        if len(wordlist) == 1:
            labeldict = {1: wordlist}
        elif len(wordlist) == 2:
            w1,w2 = wordlist
            if w2p[w1].get(w2,None) is not None:
                labeldict = {1: wordlist}
            else:
                labeldict = {1: [w1], 2: [w2]}
        return labeldict

    def consolidate_stemmed_wordlist(self, wl, w2p):
        '''
        Consolidates wordlist into its unique stemmed terms.
        For each set of terms with same stem, assign to term with largest number
        of paraphrases
        :param wl: list of str
        :return: {str -> [str,str...]}
        '''
        st = LancasterStemmer()
        stems = [st.stem(w.decode('utf8')).encode('utf8') for w in wl]
        groups = {s: [ww for ss,ww in zip(stems,wl) if ss==s] for s in set(stems)}

        # assign each stem to term with most paraphrases
        map = {s: lst[np.argmax([len(w2p.get(w,[])) for w in lst])] for s,lst in groups.iteritems()}
        return {map[s]: lst for s,lst in groups.iteritems()}


    def expand_solution(self, ld, cwl, sings):
        '''
        Expand a dictionary of solution labels using the consolidated wordlist dict,
        and add singletons back in
        :param ld: dict {int -> [word, word, ...]}
        :param cwl: {word -> [word, word, ...]}
        :param sings: set
        :return: dict {int -> [word, word, word, word...]}
        '''
        eld = {}
        maxind = max(ld.keys())
        # add singletons back in
        for s in sings:
            if [s] not in ld.values():
                maxind += 1
                ld[maxind] = [s]
        # expand consolidated (shared stem)
        for i, lst in ld.iteritems():
            wrdset = set(lst)
            wrdsetcopy = set(lst)
            for w in wrdsetcopy:
                if w in cwl:
                    wrdset |= set(cwl[w])
            eld[i] = list(wrdset)
        return eld

    def get_paraphrase_wtypes(self):
        return [p.word_type for p in self.pp_dict.itervalues()]

    def jdefault(self):
        return self.__dict__

def read_gold(infile):
    '''
    Read gold standard clustering solution from infile

    Gold file should be in the following format:

    <TARGET1>.<POS> :: <PP1_1>  <PP1_2>  ...
    <TARGET1>.<POS> :: <PP1_3>
    <TARGET2>.<POS> :: <PP2_1> <PP2_2>  ...

    Each target word has a number of lines corresponding to the number
    of gold sense clusters. Paraphrases belonging to a single gold sense
    cluster are listed together in a single line.

    :param infile:
    :return: dict (word_type -> {class -> set})
    :return: dict of {word_type -> ParaphraseSet}
    '''
    classes = {}

    for line in open(infile, 'rU'):
        entry = line.strip().split(' :: ')
        if len(entry) > 1:
            wtype = word_type(entry[0].split('.')[0], entry[0].split('.')[1])
            if wtype not in classes:
                classes[wtype] = ParaphraseSet(wtype, {})
                classes[wtype].cluster_count = 0
            poss_class = set([w for w in entry[1].split() if len(w)>0])
            if len(poss_class) > 0:
                classes[wtype].add_sense_cluster(poss_class)
    for wtype in classes:
        ppset = set().union(*[s for c,s in classes[wtype].sense_clustering.iteritems()])
        classes[wtype].pp_dict = {w: Paraphrase(word_type(w, wtype.type)) for w in ppset}
    return classes

def read_pps(infile):
    '''
    Read paraphrase lists from infile.

    Infile should be in format:

    <TARGET>.<POS> :: <PP1> <PP2> ...

    Were TARGET is a word and POS its part of speech, and PPX are paraphrases
    of the TARGET from PPDB.

    :param infile: str
    :return: dict {word_type -> ParaphraseSet}
    '''
    ppsets = {}
    with open(infile, 'rU') as fin:
        for line in fin:
            try:
                tgt, pps = line.split(' :: ')
            except ValueError:
                continue
            wtype = word_type(tgt.split('.')[0], tgt.split('.')[1])
            ppdict = {w: Paraphrase(word_type(w, wtype.type)) for w in pps.split()}
            ppsets[wtype] = ParaphraseSet(wtype, ppdict)
    return ppsets


def load_bin_vecs(filename):
    '''
    TODO: Add ability to filter by word list
    Loads 300x1 word vecs from Google word2vec in .bin format
    Thanks to word2vec google groups for this script
    :param filename: string
    :return: dict, int
    '''
    word_vecs = {}
    sys.stderr.write('Reading word2vec .bin file')
    cnt = 0
    with open(filename, 'rb') as fin:
        header = fin.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = fin.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word_vecs[word] = np.fromstring(fin.read(binary_len), dtype='float32')
            if cnt % 100000 == 0:
                sys.stderr.write('.')
            cnt += 1
    sys.stderr.write('\n')
    return word_vecs, layer1_size

def load_pickled_vecs(filename, returnpp=False):
    '''
    Load word vecs from word-paraphrase matrix
    :param filename: str
    :return: dict, list, dict
    '''
    with open(filename, 'rb') as fin:
        word2ind, ordered_vocab, w2p = pickle.load(fin)

    word_vecs = {}
    N = len(word2ind.keys())

    for w, d in w2p.iteritems():
        lil_v = sparse.lil_matrix((1,N), dtype='float')
        for p, sc in d.iteritems():
            try:
                lil_v[0,word2ind[p]] = sc
            except KeyError:
                print 'Error loading vector:', w, p, sc
        word_vecs[w] = sparse.csr_matrix(lil_v)  # is this slow?
    if returnpp:
        return word_vecs, N, w2p
    else:
        return word_vecs, N


def dict_cosine_dist(u,v):
    features = list(set(u.keys()) | set(v.keys()))
    features.sort()
    uvec = np.array([u[f] if f in u else 0.0 for f in features])
    vvec = np.array([v[f] if f in v else 0.0 for f in features])
    return cosine(uvec,vvec)

def dict_js_divergence(u,v,normalize=True):
    features = list(set(u.keys()) | set(v.keys()))
    features.sort()
    uvec = np.array([u[f] if f in u else 0.0 for f in features])
    vvec = np.array([v[f] if f in v else 0.0 for f in features])
    if normalize:
        uvec = uvec/sum(uvec)
        vvec = vvec/sum(vvec)
    return entropy.jensen_shannon_divergence(np.vstack([uvec,vvec]))


def try_americanize(w,st):
    '''
    Try to americanize the spelling of w to see if it can be found in set st
    '''
    if w.replace('isa','iza') in st:
        return w.replace('isa','iza')
    elif w.replace('ise','ize') in st:
        return w.replace('ise','ize')
    elif w.replace('ise','ice') in st:
        return w.replace('ise','ice')
    elif w.replace('our','or') in st:
        return w.replace('our','or')
    elif w.replace('re','er') in st:
        return w.replace('re','er')
    elif w.replace('nce','nse') in st:
        return w.replace('nce','nse')
    elif w.replace('yse','yze') in st:
        return w.replace('yse','yze')
    elif w.replace('mme','m') in st:
        return w.replace('mme','m')
    elif w.replace("'","") in st:
        return w.replace("'","")
    else:
        return None

def flatten(l):
    return [item for sublist in l for item in sublist]

def get_sims_matrix(method, scores, wordlist):
    '''
    Return a similarity matrix.

    Can use one of four methods to calculate similarity matrix for words in
    the <wordlist>:
    1. direct: calculates a direct pairwise similarity
    2. second_order_cosine: calculates the cosine similarity between word pairs,
        where we assume <scores> entries give the names and weights of features
        in a vector
    3. second_order_JS: calculates (1 - Jensen-Shannon divergence) between word
        pairs, where we assume <scores> entries give the names and weights of
        feaures in a vector. Normalizes each vector so that its sum is 1.
    4. vec_cosine: calculates the cosine similarity between word pairs, where
        we assume the <scores> values are np.array vectors

    The type of <scores> depends on the <method>. For the direct and
    second-order methods, the <scores> dict should be of the format:

    { tgt : { paraphrase: score } }

    For the vec_cosine method, the <scores> dict should be of the format:

    { tgt: np.ndarray }

    :param method: str, one of {direct, second_order_cosine, second_order_JS,
                   vec_cosine}
    :param scores: dict
    :param wordlist: list of str
    :return: 2-dimensional np.ndarray of size len(wordlist) x len(wordlist)
    '''
    if method == 'direct':
        sims = np.array([[scores.get(i,{}).get(j,0.0) if i in scores else 0.0 for j in wordlist] for i in wordlist])
    elif method == 'second_order_cosine': # cosine dist of word-PPDB2.0Score matrix
        sims = np.array([[(1-dict_cosine_dist(scores.get(i,{}),scores.get(j,{}))) for j in wordlist] for i in wordlist])
    elif method == 'second_order_JS': # JS divergence of word-PPDB2.0Score matrix
        sims = np.array([[(1-dict_js_divergence(scores.get(i,{}),scores.get(j,{}))[0]) for j in wordlist] for i in wordlist])
    elif method == 'vec_cosine':
        d = scores.values()[0].shape[0]
        sims = np.array([[(1-cosine(scores.get(i,np.zeros(d)),scores.get(j,np.zeros(d)))) if i!=j else 1.0 for j in wordlist] for i in wordlist])
    else:
        sys.stderr.write('Unknown sim method: %s' % method)
        return None
    sims = np.nan_to_num(sims)
    return sims

def get_sils_matrix(method, scores, wordlist):
    ''' See get_sims_matrix for definitions, which are the same here. The
    difference is that the resulting matrix contains distances instead of
    similarities.

    :return: 2-dimensional np.ndarray of size len(wordlist) x len(wordlist)
    '''
    if method =='direct':
        sims = get_sims_matrix(method, scores, wordlist)
        sims = preprocessing.normalize(np.matrix(sims), norm='l2')
        sils = 1-sims
    elif method == 'dict_cosine': # cosine dist of word-PPDB2.0Score matrix
        sils = np.array([[dict_cosine_dist(scores.get(i,{}),scores.get(j,{})) for j in wordlist] for i in wordlist])
    elif method == 'dict_JS': # JS divergence of word-PPDB2.0Score matrix
        sils = np.array([[dict_js_divergence(scores.get(i,{}),scores.get(j,{}))[0] for j in wordlist] for i in wordlist])
    elif method == 'vec_cosine':
        d = scores.values()[0].shape[0]
        sils = np.array([[cosine(scores.get(i,np.zeros(d)), scores.get(j,np.zeros(d))) for j in wordlist] for i in wordlist])
    else:
        sys.stderr.write('Unknown sil method: %s' % method)
        return None
    sils = np.nan_to_num(sils)
    return sils
