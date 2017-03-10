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
#np.set_printoptions(threshold=np.nan)


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
size_batch = 50
num_epochs = 6
train_img = np.split(np.array(train_img),num_instances/size_batch)
train_label = np.split(np.array(train_label),num_instances/size_batch)




#hyperparams
eta = 1.0
scope = .0001
eta_prime = .1
dropout_rate = .85
epsilon_noise = .001
alpha = .75
L = 20
size_sub_mini_batch = int(size_batch / 5)
num_weights = 8



#function which creates parameter objects
def model_variable(shape, name,type='weight',stddev=.01):
	if type == 'bias':
		variable = tf.get_variable(name=name,
									dtype=tf.float32,
									shape=shape,
									initializer=tf.random_normal_initializer(stddev=stddev))
	else:
		variable = tf.get_variable(name=name,
                               dtype=tf.float32,
                               initializer=tf.zeros(shape=shape))
	tf.add_to_collection('model_variables', variable)
	
	return variable

fc_plan = [
	[5,5,1,32],
	[5,5,32,64],
	[3136,1024], #3136 = 7*7*64
	[32],
	[64],
	[1024],
	[1024,num_classes],
	[num_classes]
]

x_ = tf.placeholder(tf.float32, shape=(None,num_features))
y_ = tf.placeholder(tf.float32, shape=(None,num_classes))

const_w1 = tf.placeholder(tf.float32, shape=fc_plan[0])
const_w2 =  tf.placeholder(tf.float32, shape=fc_plan[1])
const_wd1 = tf.placeholder(tf.float32, shape=fc_plan[2])
const_b1 = tf.placeholder(tf.float32, shape=fc_plan[3])
const_b2 = tf.placeholder(tf.float32, shape=fc_plan[4])
const_bd1 = tf.placeholder(tf.float32, shape=fc_plan[5])
const_wlast = tf.placeholder(tf.float32, shape=fc_plan[6])
const_blast = tf.placeholder(tf.float32, shape=fc_plan[7])
const_x = [const_w1, const_w2, const_wd1, const_b1, const_b2, const_bd1, const_wlast, const_blast]



#weights
w1 = model_variable(fc_plan[0],'w1')
w2 = model_variable(fc_plan[1],'w2')
wd1 = model_variable(fc_plan[2],'wd1')
b1 = model_variable(fc_plan[3],'b1','bias')
b2 = model_variable(fc_plan[4],'b2','bias')
bd1 = model_variable(fc_plan[5],'bd1','bias')
wlast = model_variable(fc_plan[6],'wlast')
blast = model_variable(fc_plan[7],'blast','bias')
model_variables = tf.get_collection('model_variables')

#Mu's
for i in range(num_weights):
	name = "mu_" + str(i)
	tmp = tf.get_variable(name=name,dtype=tf.float32,shape=fc_plan[i],initializer=tf.random_normal_initializer(stddev=.01))
	tf.add_to_collection('mus',tmp)
mus = tf.get_collection('mus')

#function which calculates y_hat
def convnn(ex):
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


#l2 penalty
l2_penalty = tf.reduce_sum(tf.get_collection('l2'))
#Objective Function
def cust_loss(ex,why):
	y_hat = convnn(ex)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=why)) #+ lambduh * l2_penalty
	return loss

def update_const_x(curr_const_x,mus): #last line of the algo
	new_const_x = []
	for i in range(len(curr_const_x)):
		new_const_x.append(curr_const_x[i] - eta * scope * (curr_const_x[i] - mus[i]))
	return new_const_x

def init_assign(ccx,mvs,mews): #first line of the algo
	for i in range(len(ccx)):
		tf.assign(mvs[i],ccx[i])
		tf.assign(mews[i],ccx[i])
	return ccx

#ENTROPY SGD
def ent_sgd(const_x,weights_x,mus,l):
#const_x is saved over lang iterations (x), weights x is updated (x_prime), mus is updated (mu)
	
	updates = []
	grads = tf.gradients(l, weights_x)

	i = 0
	for mu,x_prime, grad in zip(mus,weights_x,grads):	
		dx_prime = (grad - scope * (const_x[i] - x_prime)) / size_sub_mini_batch
		x_p_t = x_prime - eta_prime * dx_prime + tf.sqrt(eta_prime) * epsilon_noise * tf.random_normal(shape=x_prime.get_shape())
		mu_t = (1-alpha) * mu + alpha * x_p_t
		
		i += 1
		
		updates.append(x_prime.assign(x_p_t))
		updates.append(mu.assign(mu_t))
	return tf.group(*updates)


#funcs to call
loss = cust_loss(x_,y_)
e_sgd_iter = ent_sgd(const_x,model_variables,mus,loss)
new_const_x = update_const_x(const_x,mus)

assign_mu_x_p = init_assign([const_w1, const_w2, const_wd1, const_b1, const_b2, const_bd1, const_wlast, const_blast],model_variables,mus)

#accuracy calc functions
y_guess_val = convnn(x_) 
correct_pred = tf.equal(tf.argmax(y_guess_val, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())




val_accs = []
for _ in range(num_epochs):
	for i in range(len(train_label)):
		if i > 2:
			eta = .1
		#batch
		batch_example = np.reshape(train_img[i],[size_batch,num_features])
		batch_label = np.reshape(train_label[i],[size_batch,num_classes])
		if i == 0: #store these for the first iteration
			current_x = sess.run([model_variables])
		current_x = current_x[0]
		_,current_x = sess.run([model_variables,assign_mu_x_p],feed_dict={ #first line of algo
			const_w1 : current_x[0],
			const_w2 : current_x[1],
			const_wd1 : current_x[2],
			const_b1 : current_x[3],
			const_b2 : current_x[4],
			const_bd1 : current_x[5],
			const_wlast : current_x[6],
			const_blast : current_x[7]
		})
		for j in range(L):
			ind = np.random.choice(size_batch,size=int(size_sub_mini_batch))

			sub_mini_batch_x = np.reshape(batch_example[ind],[size_sub_mini_batch,num_features])
			sub_mini_batch_y = np.reshape(batch_label[ind],[size_sub_mini_batch,num_classes])
			c_w,l,_ = sess.run([model_variables,loss,e_sgd_iter],feed_dict={ #inner loop
				x_ : sub_mini_batch_x,
				y_ : sub_mini_batch_y,
				const_w1 : current_x[0],
				const_w2 : current_x[1],
				const_wd1 : current_x[2],
				const_b1 : current_x[3],
				const_b2 : current_x[4],
				const_bd1 : current_x[5],
				const_wlast : current_x[6],
				const_blast : current_x[7]
			})
		current_x = sess.run([new_const_x], feed_dict={ #last line of algo
				const_w1 : current_x[0],
				const_w2 : current_x[1],
				const_wd1 : current_x[2],
				const_b1 : current_x[3],
				const_b2 : current_x[4],
				const_bd1 : current_x[5],
				const_wlast : current_x[6],
				const_blast : current_x[7]
			})
		scope = scope * 1.001
	acc = 0
	for i in range(len(val_label)):
		example = np.reshape(val_img[i],[1,num_features])
		label = np.reshape(val_label[i],[1,num_classes])
		out = sess.run([model_variables,accuracy], feed_dict={x_: example,y_: label})
		acc += out[1]
	# acc = acc/len(train_label)
	acc = acc/len(val_label)
	val_accs.append(acc)
	print(val_accs)
np.save('val_accs_ent_sgd.out',val_accs)
