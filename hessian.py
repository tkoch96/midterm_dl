#Thomas Koch HW3 CGML

import tensorflow as tf
import numpy as np
from mnist import MNIST
import sys
import ast
import csv
import pickle
import math 
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt

#hyperparams
dropout_rate = .5#float(sys.argv[1])
lambduh = .01#float(sys.argv[2])
test = 0#int(sys.argv[3])


percent_val = .20 #percent of all training to be validation set
num_classes = 10

def unpickle(file):
	fo = open(file,'rb')
	dict = pickle.load(fo,encoding='bytes')
	fo.close()
	return dict



mndata = MNIST('/home/tom/Documents/DeepLearning/hw3_koch/mnist_data/')

if test == 1: #testing
	test_img, test_label = mndata.load_testing()
	test_label = np.eye(num_classes)[test_label] #1 hot vector

total_train_img, total_train_label = mndata.load_training()
num_instances = len(total_train_label)

upper = int(np.floor(percent_val * num_instances))
train_img = total_train_img[0:upper]
train_label = total_train_label[0:upper]
val_img = total_train_img[upper+1:]
val_label = total_train_label[upper+1:]


if test == 1:
	num_classes = max(test_label) + 1
	test_label = np.eye(num_classes)[test_label]
else:
	num_classes = max(val_label) + 1
	val_label = np.eye(num_classes)[val_label]
train_label = np.eye(num_classes)[train_label]

num_features = len(train_img[0])
num_instances = len(train_img)

#batch it up
size_batch = 50
num_epochs = 200
train_img = np.split(np.array(train_img),num_instances/size_batch)
train_label = np.split(np.array(train_label),num_instances/size_batch)

eta = .01


#function which creates parameter objects
def model_variable(shape, name,type='weight',stddev=.0001):
	if type == 'bias':
		variable = tf.get_variable(name=name,
									dtype=tf.float32,
									shape=shape,
									initializer=tf.constant_initializer(0.0))
	else:
		variable = tf.get_variable(name=name,
                               dtype=tf.float32,
                               shape=shape,
                               initializer=tf.random_normal_initializer(stddev=stddev))
		#tf.add_to_collection('l2', tf.reduce_sum(tf.pow(variable,2)))
	#tf.add_to_collection('model_variables', variable)
	
	return variable


x_ = tf.placeholder(tf.float32, shape=(None,num_features))
y_ = tf.placeholder(tf.float32, shape=(None,num_classes))



#variables to optimize over
#model_variables = tf.get_collection('model_variables')

#curros idea
fc_plan = [
	[3,3,1,4],
	[7*7*4,10],
	[4],
	[10]
]
param_count = 3*3*1*4+7*7*4*10+4+10
params = tf.Variable(tf.random_normal(shape=[param_count], stddev=0.05))
location = 0
model_variables = []
for i,fc in enumerate(fc_plan):
	count = np.prod(fc)
	name = 'var' + str(i)
	new_var = tf.reshape(params[location:location+count], shape=fc)
	model_variables.append(new_var)
	location = location + count
# calculate class
w1 = model_variables[0]
wd1 = model_variables[1]
b1 = model_variables[2]
bd1 = model_variables[3]
tmp = tf.reshape(x_, shape=[-1,28, 28,1])

#layer 1 
tmp = tf.nn.conv2d(tmp,w1,strides=[1,4,4,1], padding='SAME')
tmp = tf.nn.bias_add(tmp,b1)
#tmp = tf.nn.max_pool(tmp,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
tmp = tf.nn.relu(tmp)
#fully connected layer
fc1 = tf.reshape(tmp, [-1, wd1.get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1, wd1), bd1)
fc1 = tf.nn.relu(fc1)
#fc1 = tf.nn.dropout(fc1,dropout_rate)
y_hat = fc1

#l2 penalty
l2_penalty = tf.reduce_sum(tf.get_collection('l2'))
#Objective Function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y_)) #+ lambduh * l2_penalty

#Use gradient descent
global_step = tf.Variable(0,trainable=False)
optimizer = tf.train.AdamOptimizer(eta)
#Train it with respect to the model variables
train = optimizer.minimize(loss, global_step=global_step,var_list=[params])
#hessian
hessian = tf.hessians([loss],[params])
#accuracy calc functions
correct_pred = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())


if test == 0: #train the model, evaluate performance on the validation set
	for _ in range(num_epochs):
		for i in range(len(train_label)):
			example = np.reshape(train_img[i],[-1,num_features])
			label = np.reshape(train_label[i],[-1,num_classes])
			l , _ = sess.run([loss, train], feed_dict={x_ : example, y_ : label})
			if math.isnan(l):
				sys.exit("got a nan")
	acc = 0
	for i in range(len(val_label)):
		example = np.reshape(val_img[i],[-1,num_features])
		label = np.reshape(val_label[i],[-1,num_classes])
		acc = acc + sess.run(accuracy, feed_dict={x_: example,y_: label})
	acc = acc/len(val_label)
	print([acc])
	example = np.reshape(val_img[25],[-1,num_features])
	label = np.reshape(val_label[25],[-1,num_classes])
	h = sess.run(hessian, feed_dict={x_: example, y_: label})
	np.save('test.out', h)
else: #calculate accuracy on test set
	for _ in range(num_epochs):
		for i in range(len(train_label)):
			example = np.reshape(train_img[i],[-1,num_features])
			label = np.reshape(train_label[i],[-1,num_classes])
			l , _ = sess.run([loss, train], feed_dict={x_ : example, y_ : label})
		acc = 0
	for i in range(len(test_label)):
		example = np.reshape(test_img[i],[-1,num_features])
		label = np.reshape(test_label[i],[-1,num_classes])
		acc = acc + sess.run(accuracy, feed_dict={x_: example,y_: label})
	acc = acc/len(test_label)
	print([acc])

w,v = LA.eig(h[0]) #w = evals, v = evects
plt.hist(np.real(w))

plt.savefig('plot.pdf', format='pdf', bbox_inches='tight')
freq,val = np.histogram(np.real(w))
print("Freq:%s"%freq)
print("Val:%s"%val)
#no max pool gradient function