#!/bin/python3

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

fc_plan = [
    [5, 4],
    [4, 1] 
    ]

param_count = np.prod(fc_plan)

params = tf.random_normal(shape=[param_count], stddev=0.05)

weights = []
location = 0
for fc in fc_plan:
    count = np.prod(fc)
    weights.append(tf.reshape(params[location:location+count], shape=fc))
    location = location + count

net = tf.random_uniform(shape=[1, 5])     
for weight in weights:
    net = tf.matmul(net, weight)
    net = tf.nn.elu(net)
    
    
h = tf.hessians([net], [params])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
h = sess.run(h)
with sess.as_default():
	for el in h:
		print(el)