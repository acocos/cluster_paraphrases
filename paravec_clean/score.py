import os
import sys
from collections import defaultdict
from subprocess import call, STDOUT
from sklearn import metrics


def score_semeval(resultskey, goldkey, output, semeval_root='../eval/semeval_unsup_eval'):
    """
    Score clustering results in resultskey file against gold standard classes in goldkey file, using
    script provided as part of SEMEVAL2010 scoring script
    :param resultskey: str
    :param goldkey: str
    :param output: str (output I/O stream)
    :param semeval_root: str (path to semeval_unsup_eval directory)
    :return:
    """
    ## Score F-Score
    print(['java', '-jar', os.path.join(semeval_root, 'fscore.jar'), resultskey, goldkey, 'all'])
    call(['java', '-jar', os.path.join(semeval_root, 'fscore.jar'), resultskey, goldkey, 'all'], stderr=STDOUT, stdout=output)

    ## Score V-Measure
    print(['java', '-jar', os.path.join(semeval_root, 'vmeasure.jar'), resultskey, goldkey, 'all'])
    call(['java', '-jar', os.path.join(semeval_root, 'vmeasure.jar'), resultskey, goldkey, 'all'], stderr=STDOUT, stdout=output)

    return


def write_key(keyfile, tgt, sol, mode='w'):
    with open(keyfile, mode) as fout:
        for i, l in sol.iteritems():
            for pp in l:
                print >> fout, ' '.join([tgt, tgt+'.'+pp, tgt+'.'+str(i)])


def read_scoring_soln(tempfile, tgt):
    '''
    Return scores for target written in tempfile
    :param tempfile: str
    :param tgt:
    :return: fscore, prec, rec, vmeas, hom, comp (all float)
    '''
    fscore = 0.0
    prec = 0.0
    rec = 0.0
    vmeas = 0.0
    hom = 0.0
    comp = 0.0
    f = True
    with open(tempfile, 'rU') as fin:
        for line in fin:
            if 'FScore' in line:
                f = True
            elif 'V-Measure' in line:
                f = False
            elif tgt in line:
                if f:
                    fscore, prec, rec = [float(e.strip()) for e in line.split()[1:]]
                if not f:
                    vmeas, hom, comp = [float(e.strip()) for e in line.split()[1:]]
    return fscore, prec, rec, vmeas, hom, comp


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_labels(gold, sol):
    allwords = list(set(flatten(gold.values())) & set(flatten(sol.values())))
    goldlab = []
    sollab = []
    words = []
    def get_assgn(wrd, sl):
        return sorted([k for k,v in sl.iteritems() if wrd in v])
    for w in allwords:
        gldassgn = get_assgn(w, gold)
        solassgn = get_assgn(w, sol)
        if len(gldassgn) == len(solassgn):
            goldlab.extend(gldassgn)
            sollab.extend(solassgn)
            words.extend([w]*len(gldassgn))
        elif len(gldassgn) > len(solassgn):
            extralen = len(gldassgn) - len(solassgn)
            goldlab.extend(gldassgn)
            solassgn = solassgn + [solassgn[-1]]*extralen
            sollab.extend(solassgn)
            words.extend([w] * len(gldassgn))
        elif len(solassgn) > len(gldassgn):
            extralen = len(solassgn) - len(gldassgn)
            sollab.extend(solassgn)
            gldassgn = gldassgn + [gldassgn[-1]]*extralen
            goldlab.extend(gldassgn)
            words.extend([w] * len(solassgn))

    return goldlab, sollab, words


def score_clustering_solution(tgt, sol, gold, tempdir='../eval/semeval_unsup_eval/keys', use_sklearn_vmeas=True, semeval_root='../eval/semeval_unsup_eval'):
    '''
    Score clustering solution sol against gold classes.
    Both the sol and gold are passed as dictionaries with integer keys (value
    is unimportant) and sets of paraphrases in each cluster as values.
    Returns (fscore, precision, recall, vmeasure, homogeneity, completeness)
    :param tgt: str (target word you're clustering)
    :param sol: dict {int -> set}
    :param gold: dict {int -> set}
    :param tempdir: stra (temporary directory to store scoring key files)
    :param use_sklearn_vmeas: boolean (setting true will use SKLearn version of V-Measure instead of semeval script)
    :param semeval_root: str (path to semeval root directory)
    :return: FScore, precision, recall, V-Measure, homogeneity, completeness (all floats)
    '''
    ## Verify set of paraphrases in gold and sol are the same
    assert set.union(*sol.values()) == set.union(*gold.values())

    ## Write temporary key files
    tempsolkey = os.path.join(tempdir, 'sol_temp.key')
    tempgoldkey = os.path.join(tempdir, 'gld_temp.key')
    write_key(tempsolkey, tgt, sol)
    write_key(tempgoldkey, tgt, gold)

    ## Call scoring script
    tempscorefile = os.path.join(tempdir, 'scorestemp')
    tempscores = open(tempscorefile, 'w')
    score_semeval(tempsolkey, tempgoldkey, tempscores, semeval_root=semeval_root)
    tempscores.close()
    fscore, prec, rec, vmeas, hom, comp = read_scoring_soln(tempscorefile, tgt)

    ## Delete temporary key files
    # os.remove(tempsolkey)
    # os.remove(tempgoldkey)
    # os.remove(tempscorefile)
    if use_sklearn_vmeas:
        goldlab, sollab, words = get_labels(gold, sol)
        vmeas = metrics.v_measure_score(goldlab, sollab)
        hom = metrics.homogeneity_score(goldlab, sollab)
        comp = metrics.completeness_score(goldlab, sollab)

    return fscore, prec, rec, vmeas, hom, comp

def score_clustering_batch(solppsets, goldppsets, outfile='results', tempdir='../eval/semeval_unsup_eval/keys'):
    '''
    Score multiple clustering solutions contained in solppsets against goldppsets
    :param solppsets: dict {word_type -> ParaphraseSet}
    :param goldppsets: dict {word_type -> ParaphraseSet}
    :param tempdir:str
    :return:
    '''
    ## Write temporary key file
    tempsolkey = os.path.join(tempdir, 'sol.key')
    tempgoldkey = os.path.join(tempdir, 'gld.key')
    for wt, ppset in solppsets.iteritems():
        tgtname = '_'.join([wt.word, wt.type])
        sol = ppset.sense_clustering
        gld = goldppsets[wt].sense_clustering

        tgtset = set([item for sublist in sol.values() for item in sublist])
        gldfilt = defaultdict(set,{n: l & tgtset for n,l in gld.iteritems()})

        write_key(tempsolkey, tgtname, sol, mode='a')
        write_key(tempgoldkey, tgtname, gldfilt, mode='a')

    ## Call scoring script
    tempscores = open(outfile, 'w')
    score_semeval(tempsolkey, tempgoldkey, tempscores)
    tempscores.close()

    ## Delete temporary key files
    # os.remove(tempsolkey)
    # os.remove(tempgoldkey)

