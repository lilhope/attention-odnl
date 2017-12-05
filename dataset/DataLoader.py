# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import mxnet as mx
import bisect
import numpy as np
import cPickle #for test
import os
import cv2


class ImageRefDetIter(mx.io.DataIter):
    """Data Iteration for object detection base on Natural language"""
    
    def __init__(self,imdb,image_root,batch_size,data_shape,buckets,invalid_label=0,layout='NT',
                 mean_pixels=[128,128,128],is_train=True,
                 rand_samplers=[],rand_mirror=False): #data augmentation params
        super(ImageRefDetIter,self).__init__()
        buckets.sort()
        self.data = [[] for _ in buckets]
        ndiscard = 0
        for i, rec in enumerate(imdb):
            sent = rec['expression']
            buck = bisect.bisect_left(buckets,len(sent))
	    
            if buck==len(buckets):
		ndiscard+=1
		continue
            buff = np.full((buckets[buck],),invalid_label,dtype='float32')
            buff[:len(sent)] = sent
            rec['expression'] = buff
            self.data[buck].append(rec)
	print('discard %d samples'%(ndiscard))
        self.batch_size = batch_size
        self.buckets = buckets
        self.invalid_label = invalid_label
        self.layout = layout
        self._mean_pixels = mx.nd.array(mean_pixels).reshape((3,1,1))
        self._data_shape = data_shape
        self.is_train = is_train
        #image dara augumentation params
        self._rand_samplers = rand_samplers
	self._rand_mirror = rand_mirror
	self.image_root = image_root

        #print(self.size)
        self.idx = []
	self.default_bucket_key = max(buckets)
        for i, buck in enumerate(self.data):
            self.idx.extend([i,j] for j in range(0,len(buck) - batch_size + 1,batch_size))
	#print(self.idx)
	self.size = len(self.idx)
	print(self.size)
        #self.curr_idx = 0
	self.provide_data=[mx.io.DataDesc(name='expression',shape=(self.batch_size,self.default_bucket_key),layout='NT'),\
			   mx.io.DataDesc(name='data',shape=(self.batch_size,3,self._data_shape[0],self._data_shape[1]))]
	self.provide_label=[mx.io.DataDesc(name='label',shape=(self.batch_size,3,6))]
        #print(self.provide_data)
	self._data=None
	self._label=None
	self.reset()
	self._get_batch()
    """
    @property
    def provide_data(self):
	return [(k,v.shape) for k,v in self._data.items()]
    @property
    def provide_label(self):
	if self.is_train:
	    return [(k,v.shape) for k,v in self._label.items()]
	else:
	    return []
    """
        
    def reset(self):
        self.curr_idx = 0
    def iter_next(self):
        return self.curr_idx < self.size
    def next(self):
        if self.iter_next():
            self._get_batch()
	    #print(self._data)
	    self.provide_data = [mx.io.DataDesc(k,v.shape) for k,v in self._data.items()]
	    self.provide_label = [mx.io.DataDesc(k,v.shape) for k,v in self._label.items()]
            data_batch = mx.io.DataBatch(self._data.values(),
                                         self._label.values(),
					 bucket_key=self.bucket_key,
					 provide_data=[mx.io.DataDesc(name=self._data.keys()[0],shape=(self.batch_size,self.bucket_key),layout='NT'),
							mx.io.DataDesc(name=self._data.keys()[1],shape=(self.batch_size,3,self._data_shape[0],self._data_shape[1]))],
					 provide_label = [mx.io.DataDesc(name='label',shape=(self.batch_size,3,6))],
                                         pad=self.getpad(),index=self.getindex())
            self.curr_idx += 1
            return data_batch
        else:
            raise StopIteration
    def getindex(self):
        return self.curr_idx // self.batch_size
    
    def getpad(self):
        pad = self.curr_idx + self.batch_size - self.size
        return 0 if pad < 0 else pad
        
        
    def _get_batch(self):
        """Return the next batch of data."""
        if self.curr_idx == len(self.idx):
            raise StopIteration
        i,j = self.idx[self.curr_idx]
        #self.curr_idx += 1
	self.bucket_key = self.buckets[i]
        batch_data = mx.nd.zeros((self.batch_size, 3, self._data_shape[0], self._data_shape[1]))
        batch_expression_data = mx.nd.zeros((self.batch_size,self.bucket_key))
	batch_label = mx.nd.ones((self.batch_size,3,6)) * -1.
	num_samples = len(self.data[i])
	samples = self.data[i][j:j+self.batch_size] if (j+self.batch_size) < num_samples else self.data[i][j:num_samples]
        for idx,sample in enumerate(samples):
            path_ = sample['img_path']
	    im_path = self.image_root + str(path_)
            gt_boxes = sample['bbox'].copy() if self.is_train else None
	    #print(gt_boxes)
            with open(im_path,'rb') as fp:
                img_content = fp.read()
            img = mx.img.imdecode(img_content)
            im_data,label = self._data_augmentation(img,gt_boxes)
            expression = mx.nd.array(sample['expression'])
            batch_data[idx] = im_data
            batch_expression_data[idx] = expression
            batch_label[idx] = mx.nd.array(label)
        self._data = {'data':batch_data,'expression':batch_expression_data}
        if self.is_train:
            self._label = {'label':batch_label}
        else:
            self._label = {'label':None}
        
        
            
    
    def _data_augmentation(self,data,label):
        """
        perform data augmentations: crop, mirror, resize, sub mean, swap channels...
        """
        if self.is_train and self._rand_samplers:
            rand_crops = []
            for rs in self._rand_samplers:
                rand_crops += rs.sample(label)
            num_rand_crops = len(rand_crops)
            # randomly pick up one as input data
            if num_rand_crops > 0:
                index = int(np.random.uniform(0, 1) * num_rand_crops)
                width = data.shape[1]
                height = data.shape[0]
                crop = rand_crops[index][0]
                xmin = int(crop[0] * width)
                ymin = int(crop[1] * height)
                xmax = int(crop[2] * width)
                ymax = int(crop[3] * height)
                if xmin >= 0 and ymin >= 0 and xmax <= width and ymax <= height:
                    data = mx.img.fixed_crop(data, xmin, ymin, xmax-xmin, ymax-ymin)
                else:
                    # padding mode
                    new_width = xmax - xmin
                    new_height = ymax - ymin
                    offset_x = 0 - xmin
                    offset_y = 0 - ymin
                    data_bak = data
                    data = mx.nd.full((new_height, new_width, 3), 128, dtype='uint8')
                    data[offset_y:offset_y+height, offset_x:offset_x + width, :] = data_bak
                label = rand_crops[index][1]
        if self.is_train:
            interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, \
                              cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        else:
            interp_methods = [cv2.INTER_LINEAR]
        interp_method = interp_methods[int(np.random.uniform(0, 1) * len(interp_methods))]
        data = mx.img.imresize(data, self._data_shape[1], self._data_shape[0], interp_method)
        if self.is_train and self._rand_mirror:
            if np.random.uniform(0, 1) > 0.5:
                data = mx.nd.flip(data, axis=1)
                valid_mask = np.where(label[:, 0] > -1)[0]
                tmp = 1.0 - label[valid_mask, 1]
                label[valid_mask, 1] = 1.0 - label[valid_mask, 3]
                label[valid_mask, 3] = tmp
        data = mx.nd.transpose(data, (2,0,1))
        data = data.astype('float32')
        data = data - self._mean_pixels
        return data, label

def test():
    rec_path = os.path.join(os.getcwd(),'..','data','val_cache.pkl')
    with open(rec_path) as f:
	imdb = cPickle.load(f)
    image_root = '/home/liuhaopeng/data_nova/visualgenome'
    itertor = ImageRefDetIter(imdb,image_root,batch_size=54,data_shape=(640,720),buckets=[5,10,15,20])
    loader = mx.io.PrefetchingIter(itertor)
    print(loader.provide_data)
    data_iter = iter(loader)
    x = []
    for i in range(20000):
	data_batch = next(data_iter)
	print(data_batch.label)
if __name__=='__main__':
	test()
