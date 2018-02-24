from os import listdir
from os.path import isfile, join

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib import rnn

from pandas import DataFrame, Series, concat, read_csv, datetime
from sklearn.utils import shuffle

import operator
import gc

def getbin(value, width):
    return format(value, "0"+str(width)+"b")

def getFileVector(absolute_path):
    cpu_arch = {'4096':0, '64':1}
    block_size = {'1m':0, '100k':1}
    data_inst =  {'inst':0, 'both':1, 'data':2}
    benchmark = {'refequake': 0, 'refmcf': 1, 'refbzip23': 2, 'refapsi': 3, 'refcc15': 4, 'defaultJ9': 5, 'refparser': 6, 'refgalgel': 7, 'refapplu': 8, 'refperlbmk2': 9, 'tinyJikesRVM': 10, 'refbzip22': 11, 'defaultJikesRVM': 12, 'refcrafty': 13, 'defaultHotSpot': 14, 'refvpr2': 15, 'smallJikesRVM': 16, 'refcc12': 17, 'refperlbmk1': 18, 'smallHotSpot': 19, 'smallJ9': 20, 'small6trace': 21, 'refammp': 22, 'refvpr1': 23, 'refart': 24, 'small1trace': 25, 'small0trace': 26, 'tinyHotSpot': 27, 'refvortex': 28, 'refperlbmk5': 29, 'refvortex3': 30, 'refmesa': 31, 'refbzip21': 32, 'refperlbmk6': 33, 'refperlbmk3': 34, 'small3trace': 35}
    prog = {'parser': 0, 'luindex': 1, 'lusearch': 2, 'fop': 3, 'batik': 4, 'gcc': 5, 'equake': 6, 'galgel': 7, 'mesa': 8, 'pmd': 9, 'vortex': 10, 'apsi': 11, 'tomcat': 12, 'crafty': 13, 'vpr': 14, 'h2': 15, 'xalan': 16, 'avrora': 17, 'applu': 18, 'art': 19, 'sunflow': 20, 'bzip2': 21, 'ammp': 22, 'perlbmk': 23, 'jython': 24, 'eclipse': 25, 'mcf': 26}
    filename_w_extn = absolute_path.split('/')[-1]
    filename = filename_w_extn.split('.')[0]
    splitnames = filename.split('-')
    vector_string = ""
    cpu_arch_vector = splitnames[-1]
    block_size_vector = splitnames[-2]
    data_inst_vector = splitnames[-3]
    benchmark_vector = ''.join(splitnames[1:-3])
    prog_vector = splitnames[0]
    #print("filename ", filename)
    #print(prog_vector,benchmark_vector,data_inst_vector,block_size_vector,cpu_arch_vector)
    vector_string += getbin(prog[prog_vector], 5)
    vector_string += getbin(benchmark[benchmark_vector], 6)
    vector_string += getbin(data_inst[data_inst_vector],2)
    vector_string += getbin(block_size[block_size_vector],1)
    vector_string += getbin(cpu_arch[cpu_arch_vector],1)
    #print("vector string ",vector_string)
    vector = [int(bit) for bit in list(vector_string)]
    #print("vector ", vector)
    return vector

def getData(training_file, time_steps, split_percent):
    print("Train on file",training_file)

    series = read_csv(training_file)

    file_vector = getFileVector(training_file)
    #file_vector_size = len(file_vector)

    print("File vector ", file_vector)
    file_vector_size = len(file_vector)
    
    # Convert series to dataframe to perform shift and generate test set
    series_dataframe = DataFrame(series)
    shifted_dataframe = series_dataframe.shift(1)
    shifted_dataframe.fillna(0, inplace=True)

    # DataFrames to Numpy arrays
    series_dataframe_narray =  series_dataframe.as_matrix()
    shifted_dataframe_narray = shifted_dataframe.as_matrix()

    # Create small samples of time_steps size
    split_length = time_steps
    num_samples = shifted_dataframe_narray.shape[0]-split_length

    file_vector_narray = np.array(file_vector).reshape(1,1,-1)

    #print("File vector reshaped before np repeat ", file_vector_narray.shape)

    file_vector_repeat = np.repeat(np.repeat(file_vector_narray,num_samples,axis=0),split_length,axis=1)

    #print("X_samples shape repeated ", file_vector_repeat.shape)
    num_features = file_vector_size+1

    X_samples = np.zeros([num_samples, split_length, num_features])
    y_samples = np.zeros([num_samples, 1])

    #print("X_samples shape ", X_samples.shape)

    for i in range(num_samples) :
        #print("X samples shape ", X_samples[i].shape)
        #print("shifted ", shifted_dataframe_narray[i:i+split_length,0].reshape(-1,1).shape)
        X_samples[i] = np.append(file_vector_repeat[i],shifted_dataframe_narray[i:i+split_length,0].reshape(-1,1),axis=1)

        #print("X_samples ", X_samples[i])
        y_samples[i] = series_dataframe_narray[i+split_length,0]

    # Add additional dimension to feed it into tensorflow. It needs 3D tensor for LSTM
    #X_samples = np.reshape(X_samples,[X_samples.shape[0],X_samples.shape[1],1])


    # Split into training and test data
    split = int(split_percent*num_samples)
    X_train_samples = X_samples[:split]
    y_train_samples = y_samples[:split]
    X_test_samples = X_samples[split:]
    y_test_samples = y_samples[split:]

    return X_train_samples, y_train_samples, X_test_samples, y_test_samples


def train(training_files, iterations=5000, time_steps=50, num_lstm_hidden_units=128, num_classes=1, batch_size=1024, learning_rate=0.001, split_percent=0.8, print_iter=100):
    print(" Parameter values")
    print(" Iterations : ", iterations)
    print(" time_steps : ", time_steps)
    print("learning rat: ", learning_rate)

    file_vector_size = 15
    time_series_feature = 1
    num_features = file_vector_size + time_series_feature

    # Create static graph
    #input image placeholder
    x=tf.placeholder("float",[None,time_steps,num_features])
    #input label placeholder
    y=tf.placeholder("float",[None,num_classes])

    #processing the input tensor from [batch_size,n_steps,num_features] to "time_steps" number of [batch_size,num_features] tensors
    tfinput=tf.unstack(x ,time_steps,1)

    #defining the network
    lstm_layer=rnn.BasicLSTMCell(num_lstm_hidden_units,forget_bias=1)
    outputs,_=rnn.static_rnn(lstm_layer,tfinput,dtype="float32")

    out_weights=tf.Variable(tf.random_normal([num_lstm_hidden_units,num_classes]))
    out_bias=tf.Variable(tf.random_normal([num_classes]))

    #converting last output of dimension [batch_size,num_lstm_hidden_units] to [batch_size,num_classes] by out_weight multiplication
    prediction=tf.matmul(outputs[-1],out_weights)+out_bias
    
    loss_array = tf.pow(prediction-y,2)

    loss = tf.reduce_mean(loss_array)

    tf_mean_loss, tf_loss_variance = tf.nn.moments(loss_array,axes=[0])
    
    #optimization
    opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


    saver = tf.train.Saver()



    # Run created graph

    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        latest_cp = tf.train.latest_checkpoint('/home/rgangaraju/lstm/cp1')
        if latest_cp is not None:
            print("Restoring from ", latest_cp)
            saver.restore(sess, latest_cp)


        iter=1
        while iter<iterations:
            for data_file in training_files:
                X_train_samples, y_train_samples, X_test_samples, y_test_samples = getData(data_file, time_steps,split_percent)

                num_train_samples = X_train_samples.shape[0]
                print("num_train_samples ", num_train_samples)
                print("X_train_samples ", X_train_samples.shape)

                X_samples_shuffle, y_samples_shuffle = shuffle(X_train_samples, y_train_samples,random_state=iter)
                gc.collect()
                current_batch_length = 0

                if iter%print_iter==0:
                    mean_loss_list = []
                    loss_variance_list = []
                    print("current_batch+batch_size ", current_batch_length+batch_size)
                    while (current_batch_length)<num_train_samples:
                        if current_batch_length+batch_size<num_train_samples:
	                        X_batch_samples = X_samples_shuffle[current_batch_length:current_batch_length+batch_size]
	                        y_batch_samples = y_samples_shuffle[current_batch_length:current_batch_length+batch_size]
                        else:
                        	X_batch_samples = X_samples_shuffle[current_batch_length:]
	                        y_batch_samples = y_samples_shuffle[current_batch_length:]

                        current_batch_length += batch_size

                        batch_mean_loss, batch_loss_variance = sess.run([tf_mean_loss,tf_loss_variance],feed_dict={x: X_batch_samples, y: y_batch_samples})
                        print("batch_mean_loss ", batch_mean_loss, " batch_loss_variance ", batch_loss_variance)
                        mean_loss_list.append(batch_mean_loss)
                        loss_variance_list.append(batch_loss_variance)
                    
                    mean_loss = np.mean(mean_loss_list)
                    loss_variance = np.mean(loss_variance_list)
                    print(iter, " : iteration , Mean Loss :", mean_loss, "Variance :", loss_variance, " MSE/Var :", (mean_loss/loss_variance) )
                else:
                    while (current_batch_length)<num_train_samples:
                    	if current_batch_length+batch_size<num_train_samples:
	                        X_batch_samples = X_samples_shuffle[current_batch_length:current_batch_length+batch_size]
	                        y_batch_samples = y_samples_shuffle[current_batch_length:current_batch_length+batch_size]
                        else:
                        	X_batch_samples = X_samples_shuffle[current_batch_length:]
	                        y_batch_samples = y_samples_shuffle[current_batch_length:]
        
                        current_batch_length += batch_size

                        sess.run(opt, feed_dict={x: X_batch_samples, y: y_batch_samples})
     
                    
                    #saver.save(sess, "/home/rgangaraju/lstm/cp1/model.ckpt", global_step = iter)

                # Get the predictions from the model to construct the graph
                #train_preds, train_loss, train_variance=sess.run([prediction,tf_mean_loss,tf_loss_variance],feed_dict={x: X_train_samples, y: y_train_samples})
                test_preds, test_loss, test_variance=sess.run([prediction,tf_mean_loss,tf_loss_variance],feed_dict={x: X_test_samples, y: y_test_samples})
                print("Final test loss", test_loss, "Final test variance", test_variance)
                print("MSE/Variance ratio ", (test_loss/test_variance))

                #gc.collect()
            iter=iter+1


if __name__ == '__main__':
    #print("Start main")
    filename = '/home/rgangaraju/chaos/entropy_data/xalan-smallJikesRVM-both-100k-64.entropy'
    filedir = '/home/rgangaraju/chaos/selectedFiles_rishikesh/files'
    files = [join(filedir,f) for f in listdir(filedir) if isfile(join(filedir, f))]
    files.sort(reverse=True)
    files = ['/home/rgangaraju/chaos/selectedFiles_rishikesh/files/gcc-ref-cc12-data-1m-4096.entropy']

    print("File list ", files)
    train(files, iterations=1000, learning_rate=0.01,batch_size=16384,print_iter=1)
