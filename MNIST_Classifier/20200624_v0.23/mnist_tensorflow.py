# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 00:50:26 2020

@author: 82103
"""

import tensorflow as tf
import numpy as np
#import cnn_module as cnn
from tensorflow.examples.tutorials.mnist import input_data

class CNN:
    def __init__(self,sess):
        self.sess = sess
        self.build_net()
 
    def build_net(self):
        self.X = tf.placeholder(tf.float32,[None,784])  # input, 784개의 값을 가지며 n개의 이미지이다.
        X_img = tf.reshape(self.X,[-1,28,28,1]) # input 을 이미지로 인식하기 위해 reshape을 해준다. 28*28의 이미지이며 단일색상, 개수는 n개이므로 -1
        self.Y = tf.placeholder(tf.float32,[None,10])  # output
 
        L1 = tf.layers.conv2d(X_img, 64, [3,3], padding='SAME', activation=tf.nn.relu)
        L1 = tf.layers.max_pooling2d(L1, [2,2], [2,2],padding='SAME')
        
        L2 = tf.layers.conv2d(L1, 64, [3,3], padding='SAME', activation=tf.nn.relu)
        L2 = tf.layers.max_pooling2d(L2, [2,2], [2,2], padding='SAME')
        
        L3 = tf.layers.flatten(L2)
        self.logits = tf.layers.dense(L3,10,activation=None)
 
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.Y))
        #self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(0.001).minimize(self.cost)
        
 
        self.predicted = tf.argmax(self.logits,1)
        is_correct = tf.equal(self.predicted, tf.argmax(self.Y,1))
        self.accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
 
    def train(self,x_data,y_data):
        return self.sess.run([self.cost,self.optimizer], feed_dict={self.X:x_data,self.Y:y_data})
 
    def get_accuracy(self, x_data,y_data):
        return self.sess.run(self.accuracy, feed_dict={self.X:x_data,self.Y:y_data})
 
    def prediction(self,x_data):
        return self.sesss.run(self.predicted, feed_diec={self.X:x_data})

 
class Class_Tensorflow:
    def __init__(self):
        return
        
    def train_model(self,models,dataset):  
        
        self.model = models
        self.dataset = dataset
        
        ##########################################################################
        # 데이터셋
        ##########################################################################
        if self.dataset == 'Tensor' :
            mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
            
        elif self.dataset == 'csv' :
            # mnist_skrean.py mnist_keras.py 참조
            pass # 코딩하고 이라인  지울것

           
        elif self.dataset == 'unicode' :
            # mnist_skrean.py mnist_keras.py 참조
            pass # 코딩하고 이라인  지울것

        else :
            pass      
        
        ##########################################################################
        # 모델 선텍
        ##########################################################################        
        saver = tf.train.Saver()
        sess = tf.Session()
        model = CNN(sess)
        sess.run(tf.global_variables_initializer())
        #sess.run(tf.compat.v1.global_variables_initializer())
         
        batch_size = 100
        training_epochs = 15
        
        total_batch = int(mnist.train.num_examples/batch_size)
        
        ##########################################################################
        # Taining
        ##########################################################################            
        # train
        print('Learning started. It takes sometimes.') 
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)
            
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                c,_ = model.train(batch_xs,batch_ys)
                avg_cost+=c/total_batch
            print("Epoch:","%04d"%(epoch + 1),"cost =","{:.9f}".format(avg_cost))
        print('Learning Finished!')
         
        acc_flag = True
        
        #================================
        # 결과 확인
        #================================ 
        if acc_flag:
            arr_c=[]
            for i in range(10):
                xs,ys = mnist.test.next_batch(1000)
                arr_c.append(model.get_accuracy(xs,ys))
            
            print('Acc:',np.mean(arr_c))     
                  
        ##########################################################################
        # 학습시킨 모델을  현재 경로에 저장
        ##########################################################################        
        ckpt_path = saver.save(sess, './savemodels/tensor_model')
        print("job`s ckpt files is save as : ", ckpt_path)
        ##########################################################################
        # 학습시킨 모델을  로드
        ##########################################################################
        #saver.restore(sess, '.saved/model')    
    ##########################################################################
    # GUI에서 사용하기 위한 테스팅 함수
    ##########################################################################
    def testing(self,models):
        import cv2    
        
        
        ##########################################################################
        # 학습시킨 모델을  로드
        ##########################################################################
        saver.restore(sess, './savemodels/tensor_model')    
        
        ##########################################################################
        # 이미지데이터 읽어오기
        ##########################################################################
        img=cv2.imread('image.png',0)
        img=cv2.bitwise_not(img)
    ##    cv2.imshow('img',img)
        img=cv2.resize(img,(28,28))
        img=img.reshape(1,28,28,1)  # CNN 인경우 이렇게
        #img=img.reshape(1,784)       # CNN 인 아닌경우 이렇게
        img=img.astype('float32')
        img=img/255.0
    
        print(img)
    
        ##########################################################################
        # 7. 모델 사용하기       
        ##########################################################################
        #prediction=Load_model.predict(img)
        #print (prediction)
        
        # predict results
        # select the indix with the maximum probability
        pred = np.argmax(prediction,axis = 1)        
        print (pred)
        return pred