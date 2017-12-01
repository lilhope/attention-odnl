#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 08:35:08 2017

@author: lilhope
"""
import re

def load_vocab_dict_from_file(dict_file,pad_at_first=True):
    with open(dict_file) as f:
        words = [w.strip() for w in f.readlines()]
    if pad_at_first and words[0] != '<pad>':
        raise Exception("The first word needs to be <pad> in the word list.")
    vocab_dict = {words[n]:n for n in range(len(words))}
    return vocab_dict
UNK_IDENTIFIER = '<unk>' # <unk> is the word used to identify unknown words
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
def sentence2vocab_indices(sentence, vocab_dict):
    if isinstance(sentence, bytes):
        sentence = sentence.decode()
    words = SENTENCE_SPLIT_REGEX.split(sentence.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0]
    # remove .
    if len(words) > 0 and (words[-1] == '.' or words[-1] == '?'):
        words = words[:-1]
    vocab_indices = [(vocab_dict[w] if w in vocab_dict else vocab_dict[UNK_IDENTIFIER])
        for w in words]
    return vocab_indices

PAD_IDENTIFIER = '<pad>'
def preprocess_vocab_indices(vocab_indices, vocab_dict, T):
    # Truncate long sentences
    if len(vocab_indices) > T:
        vocab_indices = vocab_indices[:T]
    # Pad short sentences at the beginning with the special symbol '<pad>'
    if len(vocab_indices) < T:
        vocab_indices = [vocab_dict[PAD_IDENTIFIER]] * (T - len(vocab_indices)) + vocab_indices
    return vocab_indices

def preprocess_sentence(sentence, vocab_dict, T=None):
    vocab_indices = sentence2vocab_indices(sentence, vocab_dict)
    return vocab_indices

if __name__  == "__main__":
    dict_file = '/home/lilhope/attentatio-odnl/data/wordembedding/vocabulary_72700.txt'
    vocab_dict = load_vocab_dict_from_file(dict_file)