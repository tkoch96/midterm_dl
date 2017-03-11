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
num_epochs = 500
train_img = np.split(np.array(train_img),num_instances/size_batch)
train_label = np.split(np.array(train_label),num_instances/size_batch)

eta = .001

x_ = tf.placeholder(tf.float32, shape=(None,num_features))
y_ = tf.placeholder(tf.float32, shape=(None,num_classes))



#curros idea
fc_plan = [
	[1,1,1,4],
	[7*7*4,10],
	[4],
	[10]
]
param_count = 1*1*1*4 + 7*7*4*10 + 4 + 10
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
tmp = tf.nn.relu(tmp)
#fully connected layer
fc1 = tf.reshape(tmp, [-1, wd1.get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1, wd1), bd1)
fc1 = tf.nn.relu(fc1)
#fc1 = tf.nn.dropout(fc1,dropout_rate)
y_hat = fc1

#Objective Function
loss = -tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y_hat)))

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
example = np.reshape(train_img[0],[-1,num_features])
label = np.reshape(train_label[0],[-1,num_classes])
_,h = sess.run([loss,hessian], feed_dict={x_: example, y_: label})
np.save('test.out', h)


w,v = LA.eig(h) #w = evals, v = evects
plt.hist(np.real(w))
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.title('Histogram of Eigenvalues of Hessian Matrix of the Loss Function')

plt.savefig('hessian_plot.pdf', format='pdf', bbox_inches='tight')

#no max pool gradient function
#softmax gradient broke eventually for whatever reason

