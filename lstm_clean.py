from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt


def train(training_file, iterations=5000, time_steps=50, num_lstm_hidden_units=128, num_features=1, num_classes=1, batch_size=128, learning_rate=0.001, split_percent=0.8, print_iter=100):

    series = read_csv(training_file)
    
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

    X_samples = np.zeros([num_samples,split_length])
    y_samples = np.zeros([num_samples,1])

    for i in range(num_samples) :
        X_samples[i] = shifted_dataframe_narray[i:i+split_length,0]
        y_samples[i] = series_dataframe_narray[i+split_length,0]

    # Add additional dimension to feed it into tensorflow. It needs 3D tensor for LSTM
    X_samples = np.reshape(X_samples,[X_samples.shape[0],X_samples.shape[1],1])


    # Split into training and test data
    split = int(split_percent*num_samples)
    X_train_samples = X_samples[:split]
    y_train_samples = y_samples[:split]
    X_test_samples = X_samples[split:]
    y_test_samples = y_samples[split:]

    # Create static graph

    out_weights=tf.Variable(tf.random_normal([num_lstm_hidden_units,num_classes]))
    out_bias=tf.Variable(tf.random_normal([num_classes]))

    #input image placeholder
    x=tf.placeholder("float",[None,time_steps,num_features])
    #input label placeholder
    y=tf.placeholder("float",[None,num_classes])

    #processing the input tensor from [batch_size,n_steps,num_features] to "time_steps" number of [batch_size,num_features] tensors
    tfinput=tf.unstack(x ,time_steps,1)

    #defining the network
    lstm_layer=rnn.BasicLSTMCell(num_lstm_hidden_units,forget_bias=1)
    outputs,_=rnn.static_rnn(lstm_layer,tfinput,dtype="float32")

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
            sess.run(opt, feed_dict={x: X_train_samples, y: y_train_samples})

            if iter%print_iter==0:

                mean_loss, loss_variance=sess.run([tf_mean_loss,tf_loss_variance],feed_dict={x: X_train_samples, y: y_train_samples})
                print(iter, " : iteration , Mean Loss :", mean_loss, "Variance :", loss_variance, " MSE/Var :", (mean_loss/loss_variance) )
                
                saver.save(sess, "/home/rgangaraju/lstm/cp1/model.ckpt", global_step = iter)

            iter=iter+1
        
        # Get the predictions from the model to construct the graph
        train_preds, train_loss, train_variance=sess.run([prediction,tf_mean_loss,tf_loss_variance],feed_dict={x: X_train_samples, y: y_train_samples})
        test_preds, test_loss, test_variance=sess.run([prediction,tf_mean_loss,tf_loss_variance],feed_dict={x: X_test_samples, y: y_test_samples})


    test_plot = np.concatenate([X_train_samples[0], train_preds, test_preds])

    print("Final test loss", test_loss, "Final test variance", test_variance)
    print("MSE/Variance ratio ", (test_loss/test_variance))

    # Write the entire series to a file, which can be used to later construct the graph
    out = DataFrame(shifted_dataframe_narray)
    out['1'] = test_plot
    out.to_csv('orig_prediction.csv',sep=',',encoding='utf-8')


if __name__ == '__main__':
    filename = '/home/rgangaraju/chaos/entropy_data/xalan-smallJikesRVM-both-100k-64.entropy'
    train(filename)


