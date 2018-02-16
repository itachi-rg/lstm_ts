from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt


iterations=10000
#unrolled through these many time steps
time_steps=50

#hidden LSTM units
num_units=128

# number of features - for now time series only 1
n_input=1

#learning rate for adam
learning_rate=0.001

# number of classes - for now only one real value 
n_classes=1

#size of batch for each iteration
batch_size=128


# In[3]:

# Read data appropriately

#series = read_csv('shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

series = read_csv('/home/rgangaraju/chaos/entropy_data/xalan-smallJikesRVM-both-100k-64.entropy')

# In[4]:

# Convert series to dataframe to perform shift and generate test set
series_dataframe = DataFrame(series)
shifted_dataframe = series_dataframe.shift(1)
shifted_dataframe.fillna(0, inplace=True)


series_dataframe_narray =  series_dataframe.as_matrix()
shifted_dataframe_narray = shifted_dataframe.as_matrix()

split_length = time_steps
num_samples = shifted_dataframe_narray.shape[0]-split_length

X_samples = np.zeros([num_samples,split_length])
y_samples = np.zeros([num_samples,1])

for i in range(num_samples) :
    X_samples[i] = shifted_dataframe_narray[i:i+split_length,0]
    y_samples[i] = series_dataframe_narray[i+split_length,0]

X_samples = np.reshape(X_samples,[X_samples.shape[0],X_samples.shape[1],1])
#print "Input samples shape ", X_samples.shape
#print "Output samples shape", y_samples.shape
#print y_samples


# In[5]:

split_percent = 0.8
split = int(split_percent*num_samples)
X_train_samples = X_samples[:split]
y_train_samples = y_samples[:split]
X_test_samples = X_samples[split:]
y_test_samples = y_samples[split:]

#print "X_train_samples shape", X_train_samples.shape
#print "X_test_samples shape", X_test_samples.shape


# In[6]:

# Create static graph

out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))

#input image placeholder
x=tf.placeholder("float",[None,time_steps,n_input])
#input label placeholder
y=tf.placeholder("float",[None,n_classes])

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
tfinput=tf.unstack(x ,time_steps,1)

#defining the network
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,tfinput,dtype="float32")

#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction=tf.matmul(outputs[-1],out_weights)+out_bias
#loss_function
#loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#loss = tf.reduce_mean(tf.pow(prediction - y,2))

loss_array = tf.pow(prediction-y,2)

loss = tf.reduce_mean(loss_array)

tf_mean_loss, tf_loss_variance = tf.nn.moments(loss_array,axes=[0])
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#model evaluation
#correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
#accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#accuracy = tf.reduce_mean(tf.pow(prediction - y,2))


# In[27]:
saver = tf.train.Saver()

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    latest_cp = tf.train.latest_checkpoint('/home/rgangaraju/lstm/cp1')
    if latest_cp is not None:
        print("Restoring from ", latest_cp)
        saver.restore(sess, latest_cp)
    iter=1
    while iter<iterations:
        #batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)

        #batch_x=batch_x.reshape((batch_size,time_steps,n_input))

        sess.run(opt, feed_dict={x: X_train_samples, y: y_train_samples})

        if iter %10==0:
            #acc=sess.run(accuracy,feed_dict={x: X_train_samples, y: y_train_samples})
            mean_loss,loss_variance=sess.run([tf_mean_loss,tf_loss_variance],feed_dict={x: X_train_samples, y: y_train_samples})
            print(iter, " : iteration , Mean Loss :", mean_loss, "Variance :", loss_variance, " MSE/Var :", (mean_loss/loss_variance) )
            
            saver.save(sess, "/home/rgangaraju/lstm/cp1/model.ckpt", global_step = iter)

        iter=iter+1
    
    train_preds, train_loss, train_variance=sess.run([prediction,tf_mean_loss,tf_loss_variance],feed_dict={x: X_train_samples, y: y_train_samples})
    test_preds, test_loss, test_variance=sess.run([prediction,tf_mean_loss,tf_loss_variance],feed_dict={x: X_test_samples, y: y_test_samples})
    #print train_preds
    #print test_preds
    

    


# In[28]:

#print y_test_samples, train_preds, test_preds, 


# In[29]:

#print train_preds.shape, test_preds.shape
#print X_train_samples[0].shape
#print shifted_dataframe_narray.shape
test_plot = np.concatenate([X_train_samples[0], train_preds, test_preds])


# In[31]:

print("Final test loss", test_loss, "Final test variance", test_variance)
print("MSE/Variance ratio ", (test_loss/test_variance))
#plt.plot(shifted_dataframe_narray)
#plt.plot(test_plot)
#plt.show()
#plt.savefig('plot.png')

out = DataFrame(shifted_dataframe_narray)
out['1'] = test_plot
out.to_csv('orig_prediction.csv',sep=',',encoding='utf-8')
#print("Output", out)


