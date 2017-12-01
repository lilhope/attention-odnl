#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:14:50 2017

@author: lilhope
"""

import json
import os
from collections import defaultdict
import numpy as np
import cPickle
from utils import load_vocab_dict_from_file,preprocess_sentence

class visual_genome():
    
    def __init__(self,data_root,vocab_file,dataset='region'):
        
        self.data_root = data_root
        self.image_file = os.path.join(data_root,'image_data.json')
        if dataset=='region':
            self.ref_file = os.path.join(data_root,'region_descriptions.json')
        else:
            raise ValueError('No supported dataset:{}'.format(dataset))
        self.data = {}
        self.data['dataset'] = dataset
        self.data['ref_anns'] = json.load(open(self.ref_file,'r'))
        self.data['images'] = json.load(open(self.image_file,'r'))
        #build vocab
        self.vocab = load_vocab_dict_from_file(vocab_file)
        self.build()
        
    def build(self):
        """build the dataset,vocab.load word2vec file"""
        Imgs = {}
        for img in self.data['images']:
            Imgs[img['image_id']] = img
        Anns = []
        for ref_ann in self.data['ref_anns']:
            regions = ref_ann['regions']
            for region in regions:
                #Anns[region['region_id']] = region
                ref = region['phrase']
                #get image information
                img = Imgs[region['image_id']]
                im_height = img['height']
                im_width = img['width']
                img_path = os.path.join(self.data_root,img['url'].replace('https://cs.stanford.edu/people/rak248',''))
                #convert to word id
                tokens = preprocess_sentence(ref,self.vocab)
                region['expression'] = tokens
                x1 = float(region['x']) / im_width
                y1 = float(region['y']) / im_height
                x2 = float(region['x'] + region['width'] + 1) / im_width
                y2 = float(region['y'] + region['height'] + 1) / im_height
                bbox = np.array([[0,x1,y1,x2,y2,0]])
                region['bbox'] = bbox
                region['img_path'] = img_path
                Anns.append(region)                
        self.Anns = Anns
        #write to cache file
        with open('/home/liuhaopeng/cache.pkl','w') as f:
            cPickle.dump(Anns,f,-1)
    def getimrefdb(self):
        """get the image-refexpression dataset"""
        pass
    
if __name__=='__main__':
    #data_root = '/home/lilhope/attentation-odnl/data'
    data_root = '/home/liuhaopeng/data_nova/visualgenome/'
    vocab_file = '/home/liuhaopeng/data_nova/wordembedding/vocabulary_72700.txt'
    dataset = visual_genome(data_root,vocab_file)
    Anns = dataset.Anns
    del dataset
    
            
                
            
        
