import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
import os
import math

def add_layer(inputs, in_size, out_size, activation_function):
    Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=0.001))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.03,)
    
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs,Weights,biases

def compute_accuracy(v_xs, v_ys, sess):
    global linear_model
    y_pre = sess.run(linear_model, feed_dict={x_d: v_xs})  #feed the value xs into prediction to get the y_pre is a probobility
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1)) # turn a matrix(200,10) [[F F T ...T],[F F T..T],....,[T T F..T]]
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))   # cast:change boolean to 0/1   reduce_mean:cal the mean of matrix
    result = sess.run(accuracy, feed_dict={x_d: v_xs, y_d: v_ys})
    return result

def training_init(lumda_init,linear_model,y_d,Weights_1,biases_1,Weights_2,biases_2,Weights_3,biases_3):#(lumda_init,linear_model,y_d):
	#loss function
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_d * tf.log(linear_model), reduction_indices=[1]))
	regularizer = tf.nn.l2_loss(Weights_1) + tf.nn.l2_loss(Weights_2) + tf.nn.l2_loss(Weights_3)
	loss = cross_entropy + lumda_init * regularizer
	train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init) # reset values to wrong
	return train, sess, init


def data_split_sign(content,sign):           #sign need to be string
	return content.split(sign),len(content.split(sign))

def content_to_data_list(max_class_number,all_data,txt_feature_list,txt_feature,txt_gt_list,txt_gt):
	global len_feature_of_one_row
	len_feature_of_one_row = 0
	txt_feature_list = []
	txt_gt_list = []
	txt_gt = []
	txt_feature = []
	one_row_of_data ,len_one_row_of_data = data_split_sign(all_data,'\n')
	for data_number_counter in range(len_one_row_of_data-5):#i don't want last five data
		if((data_number_counter % data_divide_rate) == 0):  #get data stride = 1 
			feature_of_one_row ,len_feature_of_one_row = data_split_sign(one_row_of_data[data_number_counter],' ') 
			for feature_number in range(len_feature_of_one_row-1):
				if(feature_number == (len_feature_of_one_row-2)):
					txt_feature_list.append(txt_feature)	 #full list of txt_feature
					print txt_feature_list
					gt_int = int(feature_of_one_row[feature_number])
					for i in range (max_class_number):
						if(i == gt_int):
							txt_gt.append(int(1))
						else:
							txt_gt.append(int(0))

					txt_gt_list.append((txt_gt))
					txt_gt = []
					txt_feature = []
				else:
					txt_feature.append(float(feature_of_one_row[feature_number]))
	return txt_feature_list,txt_gt_list

#def cross_validation(status,dropout_value,mode,train,sess,init,train_set_cross_valid_forlumda_x,train_set_cross_valid_forlumda_y,valid_set_cross_valid_forlumda_x,valid_set_cross_valid_forlumda_y,how_many_fold,batch_size,epoch_number_max,step1_how_many_batch_in_one_epoch,result_all,random_batch_number_list,lumda_counter,fold_number):
def train_tf(train,sess,init,x_train,y_train,x_test,y_test,x_valid,y_valid,batch_size,epoch_number_max):
	len_of_train = len(x_train)
	len_of_train_minus_batch_size = len_of_train - batch_size - 1
	iteration_max_number = int(len_of_train_minus_batch_size/batch_size)
	for epoch_number_count in range(epoch_number_max):
		random_iteration_list = random.sample(range(iteration_max_number), iteration_max_number)	
		for iteration_count in range(iteration_max_number):	
			batch_number = random_iteration_list[iteration_count]
			batch_xs = x_train[(batch_number*batch_size):(batch_number*batch_size+batch_size)]
			batch_ys = y_train[(batch_number*batch_size):(batch_number*batch_size+batch_size)]
			sess.run(train, {x_d:batch_xs, y_d:batch_ys })#keep_prob: dropout_value})
		print "leave"
		training_loss = (compute_accuracy(x_train,y_train, sess))
		valid_loss = (compute_accuracy(x_test, y_test, sess))
		result = ("epoch_number_count:" + str(epoch_number_count) + "/" + "training_error:" + str(training_loss) + "test_error:" + str(valid_loss) + "\n")	
		print result
	return result,training_loss,valid_loss

def split_data(x_data,y_data,test_rate,valid_rate):
	x_train = []
	y_train = []
	x_valid = []
	y_valid = []
	x_test = []
	y_test = []
	len_of_data = len(x_data)
	random_batch_number_list = random.sample(range(len_of_data),len_of_data)
	print random_batch_number_list
	for i in range (len_of_data):
		if i <(len_of_data*test_rate):
			x_test.append(x_data[(random_batch_number_list[i])])
			y_test.append(y_data[(random_batch_number_list[i])])
		if i <(len_of_data*(test_rate+test_rate)):
			x_valid.append(x_data[(random_batch_number_list[i])])
			y_valid.append(y_data[(random_batch_number_list[i])])
		else:
			x_train.append(x_data[(random_batch_number_list[i])])
			y_train.append(y_data[(random_batch_number_list[i])])			
	return x_train,y_train,x_test,y_test,x_valid,y_valid

#1:mac 2:windows 3:linux
os_flag = 1
max_class_number = 3

##########################################################################
#############################get data from txt############################
##########################################################################
data_divide_rate = 1################################################
txt_feature = []
txt_feature_aggregation = []
txt_feature_aggregation_list = []
txt_feature_aggregation_number = 3
txt_feature_list = []
txt_gt = []
txt_gt_aggregation = []
txt_gt_aggregation_list = []
txt_gt_list = []
video_data_list = []
video_gt_list = []
video_amount = 0

f = open("/Users/Wiz/Documents/project/hackthon/normal1/1.txt", "r")
x_data,y_data = content_to_data_list(max_class_number,f.read(),txt_feature_list,txt_feature,txt_gt_list,txt_gt)
test_rate = 0.3
valid_rate = 0
x_train,y_train,x_test,y_test,x_valid,y_valid = split_data(x_data,y_data,test_rate,valid_rate)

len_of_input =  len(x_train[0])
len_of_out =  len(y_train[0])
##########################################################################
###################################set up#################################
##########################################################################
# Model input and output
print len_of_input
print len_of_out
x_d = tf.placeholder(tf.float32, [None, len_of_input])   # len_feature_of_one_row is the dimension of feature
y_d = tf.placeholder(tf.float32, [None, len_of_out])
#keep_prob = tf.placeholder(tf.float32)

# add hidden layer
l1,Weights1,biases1 = add_layer(x_d, len_of_input, 40, activation_function=tf.nn.relu)
l2,Weights2,biases2 = add_layer(l1, 40, 30, activation_function=tf.nn.relu)
linear_model,Weights3,biases3 = add_layer(l2, 30, len_of_out,  activation_function=tf.nn.softmax)
#linear_model,Weights1,biases1 = add_layer(x_d, len_of_input, len_of_out,  activation_function=tf.nn.softmax)


lumda = 0.001####################
epoch_number_max = 400############
batch_size = 3
train, sess, init = training_init(lumda,linear_model,y_d,Weights1,biases1,Weights2,biases2,Weights3,biases3)


##################################step 1##################################
################################best lumda################################


train_tf(train,sess,init,x_train,y_train,x_test,y_test,x_valid,y_valid,batch_size,epoch_number_max)
