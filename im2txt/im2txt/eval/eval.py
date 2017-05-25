# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import jieba
import argparse
import subprocess
from collections import defaultdict
# only on my machine
import sys
sys.path.insert(0, "/home/foxfi/homework/pattern_recognition/project/coco-caption")
reload(sys)
sys.setdefaultencoding('UTF8')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

curpath = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("caption_file")
    parser.add_argument("candidate_file")
    parser.add_argument("index_file")
    parser.add_argument("--no-cut", action="store_true", default=False)
    parser.add_argument("--all-ref", action="store_true", default=False)
    args = parser.parse_args()
    
    indexes = [int(x.strip()) for x in open(args.index_file, "r").readlines()]
    #candidates = [x.strip() for x in open(args.candidate_file, "r").readlines()]
    candidates = [(" ".join(x.strip().split())).decode("utf-8") for x in open(args.candidate_file, "r").readlines()]
    assert len(indexes) == len(candidates)
    seg_dict = defaultdict(lambda : [])    
    print("Start parse ground truth caption file.")
    caption_indexes = parse_caption_file(args.caption_file, seg_dict, not args.no_cut)
    print("Finish parse ground truth caption file.")
    ref_dct = {ind: seg_dict[caption_indexes[ind]] for ind in indexes}
    cand_dct = {ind: [candidates[i].decode("utf-8")] for i, ind in enumerate(indexes)}
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    for scorer, method in scorers:
        print("Computing {} score... ".format(method))
        score, _ = scorer.compute_score(ref_dct, cand_dct)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                print("{}: {}".format(m, sc))
        else:
            print("{}: {}".format(method, score))
        print("------")
    # if args.all_ref:
    #     # all refs
    #     refs, candidates = zip(*[(seg, candidates[i]) for i, ind in enumerate(indexes) for seg in seg_dict[caption_indexes[ind]]])
    # else:
    #     refs = [seg_dict[caption_indexes[ind]][0] for ind in indexes]

    # # write into tmp files
    # output_fname = os.path.join(curpath, "output")
    # ref_fname = os.path.join(curpath, "reference")
    # print("\n".join(candidates), file=open(output_fname, 'w'))
    # print("\n".join(refs).encode("utf-8"), file=open(ref_fname, "w"))
    # bin_fname= os.path.join(curpath, "multi-bleu.perl")
    # print("use perl script: {}".format(bin_fname))
    # subprocess.check_call("{} {} < {}".format(bin_fname, ref_fname, output_fname), shell=True)

def parse_caption_file(fname, seg_dict, cut):
    indexes = []
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                index = int(line)
                indexes.append(index)
            except ValueError:
                if cut:
                    seg_list = list(jieba.cut(line))
                else:
                    seg_list = list(line.decode("utf-8"))
                seg_dict[index].append(" ".join(seg_list))
    return indexes

if __name__ == "__main__":
    main()
