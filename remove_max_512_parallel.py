#!/usr/bin/env python
# Author: Ona de Gibert
# This file is used to remove sentences longer than 512 tokens on tokenized parallel corpora
# It expects two parallel tokenized files
# It writes the output in a folder called "filtered"
# Usage:
# python3 remove_max_512.py --src tokenized/train.ca-oc.ca --tgt tokenized/train.ca-oc.oc
import argparse
import ntpath

def read_files(src,tgt):
    read_src = open(src, 'r').read().splitlines()
    read_tgt = open(tgt, 'r').read().splitlines()
    return read_src, read_tgt

def filter(src, tgt):
    index = 0
    indeces_to_remove = []
    for line in src:
        if len(line.split()) > 256: #if a line is longer than 256
            indeces_to_remove.append(index)
        index += 1
    print(len(indeces_to_remove))
    index = 0
    for line in tgt:
        if len(line.split()) > 256: #if a line is longer than 256
            indeces_to_remove.append(index)
        index += 1
    print(len(indeces_to_remove))
    indeces_to_remove = set(indeces_to_remove)
    print(len(indeces_to_remove))
    for file in src, tgt:
        for index in sorted(indeces_to_remove, reverse=True):
            del file[index]
    return(src, tgt)

def write_file(file, filename):
    out = open("filtered/"+filename,'w')
    for sentence in file:
        out.write("%s\n" % sentence)
    out.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", nargs="+", default=['-'],
                        help="source language file")
    parser.add_argument("--tgt", nargs="+", default=['-'],
                        help="target language file")
    
    args = parser.parse_args()

    src_filename = ntpath.basename(args.src[0])
    tgt_filename = ntpath.basename(args.tgt[0])

    src, tgt = read_files(args.src[0], args.tgt[0])
    src_raw_lines = len(src)
    filtered_src, filtered_tgt = filter(src, tgt)
    src_clean_lines = len(filtered_src)
   
    write_file(filtered_src, src_filename)
    write_file(filtered_tgt, tgt_filename)
       
    removed_lines = src_raw_lines - src_clean_lines
    print("Removed {} lines for {}".format(removed_lines,src_filename))


if __name__ == "__main__":
    main()

# Removed 91 lines for train.ca-oc.ca
# Removed 1287 lines for train.ca-ro.ca
# Removed 1742 lines for train.ca-it.ca
