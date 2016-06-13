# paraphrase_clustering
This repo contains code for clustering paraphrases by word sense.

If you build upon this code or use it in your work, please cite this paper:

```python
@article{CocosAndCallisonBurch-2016:NAACL:ParaphraseClustering, 
  author =  {Anne Cocos and Chris Callison-Burch},
  title =   {Clustering Paraphrases by Word Sense},
  booktitle = {Proceedings of the 15th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2016)},
  month     = {June},
  year      = {2016},
  address   = {San Diego, California},
  publisher = {Association for Computational Linguistics}
}
```
## Getting Started
We have included most of the data used in the paper that will enable you to try paraphrase clustering. 
But to run the clustering script, you will need to download an additional set of PPDB data files first, and we've provided a script to do so.
Navigate to the `cluster_paraphrases/data/ppdb_data/` directory and run `./get_ppdb_data.sh`. See the included readme for more details.

Once that is done, you can try paraphrase clustering out of the box:

```bash
cd paravec
python cluster.py 
```
Running this will perform hierarchical clustering on PPDB paraphrases for a random sample of 78 target words.

Other options:

```bash
python cluster.py -p <ppfile> -s <goldfile> -b -f -m <method>
```
`-p` Specifies the file containing the target words and paraphrases that you want to cluster. These should be stored one per line, with the following format:
`target.pos :: pp1 pp2 pp3`

`-s` Optionally specifies a file containing gold standard clusters, against which you can score your clustering solution. It should have the format:

```python
target1.pos :: pp1 pp2
target1.pos :: pp3
target2.pos :: pp1 pp2 pp3 pp4
target2.pos :: pp3 pp5 pp6`
```
where a target word appears in as many lines as it has gold standard clusters.

`-b` Optionally tells the program to score the clustering solution against baseline methods

`-f` Filters your input targets' paraphrases by the gold file, so that you only cluster paraphrases that appear in a gold cluster. This simplifies grading and is recommended when using a gold file.

`-m` Lets you specify the clustering method to use, which can be one of `spectral`, `hgfc` (default), or `semclust`.

## Detailed Contents

| File/Directory        | Description |
| ------------- |:-------------| 
| `readme.md`   | This readme doc |
| `data/pp` | contains example paraphrase files |
| `data/gold` | contains example gold files |
| `data/ppdb` | contains pickle files with dictionaries that store PPDB2.0 Scores, entailment probabilities, and word vectors, for use in clustering. If you want to try a new metric, you can replace these - see `cluster.py` for more details |
| `eval` | contains grading scripts, from the SEMEVAL07 Lexical Substitution shared task |
| `paravec/cluster.py` | An example of how to read in a big file of ParaphraseSets, cluster them all, and score against some gold standard. |
| `paravec/paraphrase.py` | Contains the main classes, Paraphrase and ParaphraseSet, functions for reading them from files, and helper functions for clustering. |
| `paravec/cluster_rotate.py` | Main code for self-tuning spectral clustering ([Zelnik and Perona 2004](http://www.vision.caltech.edu/lihi/Publications/SelfTuningClustering.pdf)), ported from the authors' original Matlab and C [code](http://www.vision.caltech.edu/lihi/Demos/SelfTuningClustering.html). |
| `paravec/entropy.py` | Contains Jensen-Shannon divergence function that may be used for calculating similarity for clustering (instead of cosine similarity). Cloned from [https://github.com/viveksck/langchangetrack/](https://github.com/viveksck/langchangetrack/). |
| `paravec/hgfc.py` | Main code for Hierarchical Graph Factorization Clustering ([Yu et al. 2006](http://papers.nips.cc/paper/2948-soft-clustering-on-graphs.pdf); [Sun and Korhonen 2011](http://www.anthology.aclweb.org/D/D11/D11-1095.pdf)).  Adapted from Ozan Irsoy's 2013 [implementation](https://gist.github.com/oir/5216719) in R. |
| `paravec/score.py` | Code for scoring clusters against gold standard using VMeasure and FScore. Utilizes SemEval 2007 scoring scripts (stored in ../eval/semeval\_unsup\_eval). |
| `paravec/sem_clust.py` | Main code for Semantic Clustering of Pivot Paraphrases ([Apidianaki et al. 2014](http://www.lrec-conf.org/proceedings/lrec2014/pdf/475_Paper.pdf)), adapted from Marianna Apidianaki's original code, which we thank her for! |

### Contact
You can contact me at `acocos@seas.upenn.edu` with any questions.