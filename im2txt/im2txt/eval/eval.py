# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import jieba
import argparse
import subprocess
from collections import defaultdict

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
    candidates = [x.strip() for x in open(args.candidate_file, "r").readlines()]
    assert len(indexes) == len(candidates)
    seg_dict = defaultdict(lambda : [])    
    caption_indexes = parse_caption_file(args.caption_file, seg_dict, not args.no_cut)
    # write into tmp files
    output_fname = os.path.join(curpath, "output")
    ref_fname = os.path.join(curpath, "reference")
    if args.all_ref:
        refs, candidates = zip(*[(seg, candidates[i]) for i, ind in enumerate(indexes) for seg in seg_dict[caption_indexes[ind]]])
    else:
        refs = [seg_dict[caption_indexes[ind]][0] for ind in indexes]
    print("\n".join(candidates), file=open(output_fname, 'w'))
    print("\n".join(refs).encode("utf-8"), file=open(ref_fname, "w"))
    bin_fname= os.path.join(curpath, "multi-bleu.perl")
    print("use perl script: {}".format(bin_fname))
    subprocess.check_call("{} {} < {}".format(bin_fname, ref_fname, output_fname), shell=True)

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
                    seg_list = list(line)
                seg_dict[index].append(" ".join(seg_list))
    return indexes

if __name__ == "__main__":
    main()
