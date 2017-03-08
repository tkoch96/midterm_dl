import pickle, gzip
import tensorflow as tf
import numpy as np
import sys
import ast
import csv
import pickle
import math 
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt

#print the whole array
np.set_printoptions(threshold=np.nan)


test = 0
num_classes = 10


with gzip.open('mnist.pkl.gz','rb') as ff :
	u = pickle._Unpickler( ff )
	u.encoding = 'latin1'
	train_set, valid_set, test_set= u.load()


if test == 1:
    test_img = test_set[0]
    test_label = test_set[1]
    test_label = np.eye(num_classes)[test_label]
else:
    val_img = valid_set[0]
    val_label = valid_set[1]
    val_label = np.eye(num_classes)[val_label]
    
train_img = train_set[0]
train_label = train_set[1]
train_label = np.eye(num_classes)[train_label]

num_features = len(train_img[0])
num_instances = len(train_img)


#batch it up
size_batch = 200
num_epochs = 50
train_img = np.split(np.array(train_img),num_instances/size_batch)
train_label = np.split(np.array(train_label),num_instances/size_batch)




#hyperparams
eta = .001
scope = .001
eta_prime = .5
lambduh = .01
dropout_rate = .2
epsilon_noise = .001
alpha = .75
L = 9

#function which creates parameter objects
def model_variable(shape, name,type='weight',stddev=.0001):
	if type == 'bias':
		variable = tf.get_variable(name=name,
									dtype=tf.float32,
									shape=shape,
									initializer=tf.random_normal_initializer(stddev=.0000001))
	else:
		variable = tf.get_variable(name=name,
                               dtype=tf.float32,
                               shape=shape,
                               initializer=tf.random_normal_initializer(stddev=stddev))
		tf.add_to_collection('l2', tf.reduce_sum(tf.pow(variable,2)))
	tf.add_to_collection('model_variables', variable)
	
	return variable

x_ = tf.placeholder(tf.float32, shape=(None,num_features))
y_ = tf.placeholder(tf.float32, shape=(None,num_classes))

#weights
w1 = model_variable([5,5,1,32],'w1',stddev = np.sqrt(2/(5*5*1*32)))
w2 = model_variable([5,5,32,64],'w2',stddev = np.sqrt(2/(5*5*32*64)))
wd1 = model_variable([7*7*64,1024],'wd1')
b1 = model_variable([32],'b1','bias')
b2 = model_variable([64],'b2','bias')
bd1 = model_variable([1024],'bd1','bias')
wlast = model_variable([1024,num_classes],'wlast')
blast = model_variable([num_classes],'blast','bias')

#function which calculates y_hat
def convnn(ex):
	w1 = model_variables[0]
	w2 = model_variables[1]
	wd1 = model_variables[2]
	b1 = model_variables[3]
	b2 = model_variables[4]
	bd1 = model_variables[5]
	wlast = model_variables[6]
	blast = model_variables[7]

	input_ = tf.reshape(ex, shape=[-1,28, 28,1])

	#layer 1 
	tmp = tf.nn.conv2d(input_,w1,strides=[1,1,1,1], padding='SAME')
	tmp = tf.nn.bias_add(tmp,b1)
	tmp = tf.nn.max_pool(tmp,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	tmp = tf.nn.relu(tmp)
	#layer 2
	tmp = tf.nn.conv2d(tmp,w2,strides=[1,1,1,1], padding='SAME')
	tmp = tf.nn.bias_add(tmp,b2)
	tmp = tf.nn.max_pool(tmp,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	tmp = tf.nn.relu(tmp)
	#fully connected layer
	fc1 = tf.reshape(tmp, [-1, wd1.get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, wd1), bd1)
	fc1 = tf.nn.relu(fc1)
	fc1 = tf.nn.dropout(fc1,dropout_rate)
	y_hat = tf.add(tf.matmul(fc1,wlast),blast)
	return y_hat
model_variables = tf.get_collection('model_variables')


#l2 penalty
l2_penalty = tf.reduce_sum(tf.get_collection('l2'))
#Objective Function
def cust_loss(ex,why):
	y_hat = convnn(ex)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=why)) + lambduh * l2_penalty
	return loss

#ENTROPY SGD
def ent_sgd(x,L): #input is weights and number of iterations to do, L
	num_weights = 8
	x_prime = x
	mu = x
	size_sub_mini_batch = size_batch / 10;

	for _ in range(L):
		ind = np.random.randint(size_batch,size=int(size_sub_mini_batch))
		sub_mini_batch_x = tf.gather(x_,ind)
		sub_mini_batch_y = tf.gather(y_,ind)
		s = [None] * num_weights
		out = [None] * num_weights
		for i in range(num_weights): #to hold the weights
			s[i] = tf.zeros(x[i].get_shape())
		for i in range(int(size_sub_mini_batch)): #for each sample in the sampled minibatch
			tmpx = tf.gather(sub_mini_batch_x,[i]) 
			tmpy = tf.gather(sub_mini_batch_y,[i]) 
			for j in range(num_weights): #get gradient for each weight
				s[j] += tf.gradients([cust_loss(tmpx,tmpy)],x_prime[j])[0]
				s[j] -= scope * (x[j] - x_prime[j])
		dx_prime = [None] * num_weights
		for i in range(num_weights):
			dx_prime[i] = s[i] / size_sub_mini_batch
			x_prime[i] = x_prime[i] - eta_prime * dx_prime[i] + tf.sqrt(eta_prime) * epsilon_noise * tf.random_normal(x_prime[i].get_shape())
			mu[i] = (1 - alpha) * mu[i] + alpha * x_prime[i]
	curr_weights = tf.get_collection('model_variables')
	for i in range(num_weights):
		out[i] = x[i] - eta * scope * (x[i] - mu[i])
		tf.assign(curr_weights[i],out[i])
	return out

model_variables = tf.get_collection('model_variables')
mv = ent_sgd(model_variables,L)

#accuracy calc functions
y_guess_val = convnn(x_) 
correct_pred = tf.equal(tf.argmax(y_guess_val, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())



if test == 0: #train the model, evaluate performance on the validation set
	val_accs = []
	for _ in range(num_epochs):
		for i in range(len(train_label)):
			example = np.reshape(train_img[i],[size_batch,num_features])
			label = np.reshape(train_label[i],[size_batch,num_classes])
			_ = sess.run([mv],feed_dict={x_: example, y_:label})
			print(i)
		acc = 0
		# for i in range(len(val_label)):
		# 	example = np.reshape(val_img[i],[1,num_features])
		# 	label = np.reshape(val_label[i],[1,num_classes])
		# 	acc = acc + sess.run(accuracy, feed_dict={x_: example,y_: label})
		for i in range(len(train_label)): #val error not working, get train error
			example = np.reshape(train_img[i],[size_batch,num_features])
			label = np.reshape(train_label[i],[size_batch,num_classes])
			acc = acc + sess.run(accuracy, feed_dict={x_: example,y_: label})
		acc = acc/len(train_label)
		val_accs.append(acc)
		print(val_accs)
	np.save('val_accs.out',val_accs)
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