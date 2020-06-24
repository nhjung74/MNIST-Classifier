# -*- coding: utf-8 -*-
"""
Created on Wed May 20 21:18:24 2020

@author: Jung Nak Hyun 정낙현
"""
################################################################
# 변경이력 
# 2020-05-20 Keras 를 사용함
# 2020-05-22 class 로 전환 
#
# 참고 URL
# https://tykimos.github.io/2017/01/27/Keras_Talk/
# 관련 모듈
# pip install tensorflow
# pip install keras
##########################################################################



##########################################################################
# 학습
##########################################################################
from keras.models import model_from_json
import numpy as np

# 0. 사용할 패키지 불러오기
from tensorflow import keras
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D ,MaxPool2D
from keras.callbacks import ModelCheckpoint, EarlyStopping 

from keras.optimizers import RMSprop, Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


TRAIN_RESULT_FILE = './Training_Result.txt'

##########################################################################
# Library_Keras 로  학습하는 클래스 
##########################################################################
class Class_Keras :    
    
    shape_x = 28
    shape_y = 28
    
    def __init__(self):
        return

    def train_model(self,models,dataset):   
        
        self.model = models
        self.dataset = dataset
        
        ##########################################################################
        # 데이터셋
        ##########################################################################
        if self.dataset == 'mnist' :
            self.shape_x = 28
            self.shape_y = 28         
            # Scikit learn 에 있는 dataset 사용
       
            # 1. 데이터셋 생성하기
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
            X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
            y_train = np_utils.to_categorical(y_train)
            y_test = np_utils.to_categorical(y_test)    
    
            # =============================================================================
            # Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
            # 11493376/11490434 [==============================] - 280s 24us/step
            # [0. 1.]     
            # =============================================================================
            
        elif self.dataset == 'csv' :
            self.shape_x = 28
            self.shape_y = 28            
            import numpy as np
            
            # 0~9 숫자 이미지가 784개의 숫자 (28X28) 로 구성되어 있는 training data 읽어옴
            data = np.loadtxt('./mnist_data/mnist_train.csv', delimiter=',', dtype=np.float32)
            
            # 0~9 숫자 이미지가 784개의 숫자 (28X28) 로 구성되어 있는 test data 읽어옴
            test_data = np.loadtxt('./mnist_data/mnist_test.csv', delimiter=',', dtype=np.float32)
            
            print("data.shape = ", data.shape, " ,  test_data.shape = ", test_data.shape)
            print("data[0,0] = ", data[0,0], ",  test_data[0,0] = ", test_data[0,0])
            print("len(data[0]) = ", len(data[0]), ",  len(test_data[0]) = ", len(test_data[0]))
        
            
            #Splitting the dataset into X and Y
            X_train = data[:,1:]
            y_train  = data[:,0]            
                
            X_test = test_data[:,1:]
            y_test = test_data[:,0]
            
            X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
            X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
            y_train = np_utils.to_categorical(y_train)
            y_test = np_utils.to_categorical(y_test)    
                
            print(X_train.shape, y_train.shape)
            # ((63000, 784), (63000,))
            print(X_test.shape, y_test.shape)
            # ((7000, 784), (7000,))
            
            # Garbage Collection 을 위해서 삭제함
            del data
            del test_data            

        elif self.dataset == 'unicsv_128' :
            self.shape_x = 128
            self.shape_y = 128
            import numpy as np
            
            # 한글 이미지가 4096개의 숫자로 (64X64) 로 구성되어 있는 training data 읽어옴
            data = np.loadtxt('./mnist_data/Ztrain_han_128X128.csv', delimiter=',', dtype=np.float32)
            
            # 한글 이미지가 4096개의 숫자로 (64X64) 로 구성되어 있는 training data 읽어옴
            test_data = np.loadtxt('./mnist_data/Ztest_han_128X128.csv', delimiter=',', dtype=np.float32)
            
            print("data.shape = ", data.shape, " ,  test_data.shape = ", test_data.shape)
            print("data[0,0] = ", data[0,0], ",  test_data[0,0] = ", test_data[0,0])
            print("len(data[0]) = ", len(data[0]), ",  len(test_data[0]) = ", len(test_data[0]))
        
            
            #Splitting the dataset into X and Y
            X_train = data[:,1:]
            y_train  = data[:,0]            
                
            X_test = test_data[:,1:]
            y_test = test_data[:,0]
            
            #X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
            #X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
            y_train = np_utils.to_categorical(y_train)
            y_test = np_utils.to_categorical(y_test)    
                
            print(X_train.shape, y_train.shape)
            # ((63000, 784), (63000,))
            print(X_test.shape, y_test.shape)
            # ((7000, 784), (7000,))
            
            # Garbage Collection 을 위해서 삭제함
            del data
            del test_data            
            
        elif self.dataset == 'unicsv_64' :
            self.shape_x = 64
            self.shape_y = 64
            import numpy as np
            
            # 한글 이미지가 4096개의 숫자로 (64X64) 로 구성되어 있는 training data 읽어옴
            data = np.loadtxt('./mnist_data/Ztrain_han_64X64.csv', delimiter=',', dtype=np.float32)
            
            # 한글 이미지가 4096개의 숫자로 (64X64) 로 구성되어 있는 training data 읽어옴
            test_data = np.loadtxt('./mnist_data/Ztest_han_64X64.csv', delimiter=',', dtype=np.float32)
            
            print("data.shape = ", data.shape, " ,  test_data.shape = ", test_data.shape)
            print("data[0,0] = ", data[0,0], ",  test_data[0,0] = ", test_data[0,0])
            print("len(data[0]) = ", len(data[0]), ",  len(test_data[0]) = ", len(test_data[0]))
        
            
            #Splitting the dataset into X and Y
            X_train = data[:,1:]
            y_train  = data[:,0]            
                
            X_test = test_data[:,1:]
            y_test = test_data[:,0]
            
            #X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
            #X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
            y_train = np_utils.to_categorical(y_train)
            y_test = np_utils.to_categorical(y_test)    
                
            print(X_train.shape, y_train.shape)
            # ((63000, 784), (63000,))
            print(X_test.shape, y_test.shape)
            # ((7000, 784), (7000,))
            
            # Garbage Collection 을 위해서 삭제함
            del data
            del test_data            
            
        elif self.dataset == 'unicsv_46' :
            self.shape_x = 46
            self.shape_y = 46
            import numpy as np
            
            # 한글 이미지가 4096개의 숫자로 (64X64) 로 구성되어 있는 training data 읽어옴
            data = np.loadtxt('./mnist_data/Ztrain_han_46X46.csv', delimiter=',', dtype=np.float32)
            
            # 한글 이미지가 4096개의 숫자로 (64X64) 로 구성되어 있는 training data 읽어옴
            test_data = np.loadtxt('./mnist_data/Ztest_han_46X46.csv', delimiter=',', dtype=np.float32)
            
            print("data.shape = ", data.shape, " ,  test_data.shape = ", test_data.shape)
            print("data[0,0] = ", data[0,0], ",  test_data[0,0] = ", test_data[0,0])
            print("len(data[0]) = ", len(data[0]), ",  len(test_data[0]) = ", len(test_data[0]))
        
            
            #Splitting the dataset into X and Y
            X_train = data[:,1:]
            y_train  = data[:,0]            
                
            X_test = test_data[:,1:]
            y_test = test_data[:,0]
            
            #X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
            #X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
            y_train = np_utils.to_categorical(y_train)
            y_test = np_utils.to_categorical(y_test)    
                
            print(X_train.shape, y_train.shape)
            # ((63000, 784), (63000,))
            print(X_test.shape, y_test.shape)
            # ((7000, 784), (7000,))
            
            # Garbage Collection 을 위해서 삭제함
            del data
            del test_data            
            
        elif self.dataset == 'unicsv_28' :
            self.shape_x = 28
            self.shape_y = 28
            import numpy as np
            
            # 한글 이미지가 784개의 숫자로 (28x28) 로 구성되어 있는 training data 읽어옴
            data = np.loadtxt('./mnist_data/Ztrain_han_28X28.csv', delimiter=',', dtype=np.float32)
            
            # 한글 이미지가 784개의 숫자로 (64X64) 로 구성되어 있는 training data 읽어옴
            test_data = np.loadtxt('./mnist_data/Ztest_han_28X28.csv', delimiter=',', dtype=np.float32)
            
            print("data.shape = ", data.shape, " ,  test_data.shape = ", test_data.shape)
            print("data[0,0] = ", data[0,0], ",  test_data[0,0] = ", test_data[0,0])
            print("len(data[0]) = ", len(data[0]), ",  len(test_data[0]) = ", len(test_data[0]))
        
            
            #Splitting the dataset into X and Y
            X_train = data[:,1:]
            y_train  = data[:,0]            
                
            X_test = test_data[:,1:]
            y_test = test_data[:,0]
            
            #X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
            #X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
            y_train = np_utils.to_categorical(y_train)
            y_test = np_utils.to_categorical(y_test)    
                            
            print(X_train.shape, y_train.shape)
            # ((63000, 784), (63000,))
            print(X_test.shape, y_test.shape)
            # ((7000, 784), (7000,))
            
            # Garbage Collection 을 위해서 삭제함
            del data
            del test_data            
            
        elif self.dataset == 'unicode' :
            self.shape_x = 64
            self.shape_y = 64
            import numpy as np
            
            # 0~9 숫자 이미지가 784개의 숫자 (28X28) 로 구성되어 있는 training data 읽어옴
            Xdata = np.loadtxt('./mnist_data/Xtrain_han.csv',delimiter=',',dtype=int)
            ydata = np.loadtxt('./mnist_data/ytrain_han.csv',delimiter=',',dtype=str)
            
            # 0~9 숫자 이미지가 784개의 숫자 (28X28) 로 구성되어 있는 test data 읽어옴
            Xtest_data = np.loadtxt('./mnist_data/Xtest_han.csv', delimiter=',',dtype=int)
            ytest_data = np.loadtxt('./mnist_data/ytest_han.csv', delimiter=',',dtype=str)
            
            print("Xdata.shape = ", Xdata.shape, "  ,  Xtest_data.shape = ", Xtest_data.shape)
            print("ydata.shape = ", ydata.shape, "  ,  ytest_data.shape = ", ytest_data.shape)
            #print("data[0,0] = " , Xdata[0,0], "   ,  test_data[0,0] = ", Xtest_data[0,0])
            print("len(Xdata[0]) = ", len(Xdata[0]), ",  len(Xtest_data[0]) = ", len(Xtest_data[0]))
            print("len(ydata[0]) = ", len(ydata[0]), ",  len(ytest_data[0]) = ", len(ytest_data[0]))
        
            
            #Splitting the dataset into X and Y
            X_train = Xdata[:]
            y_train = ydata[:]
    
            print(X_train.shape, y_train.shape)
            # ((63000, 784), (63000,))
            
            X_test = Xtest_data[:]
            y_test = ytest_data[:]
            
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
            
            y_train = np_utils.to_categorical(y_train)
            y_test = np_utils.to_categorical(y_test)              
            
            print(X_test.shape, y_test.shape)
            # ((7000, 784), (7000,))
            
            # Garbage Collection 을 위해서 삭제함
            del Xdata
            del ydata            
            
        else :
            pass        
        
        # Checking uniqueness of the target
        import numpy as np
        print(np.unique(y_train))
        
        if ( self.dataset == 'mnist' ) or ( self.dataset == 'csv' ) :
            num_classes = len(np.unique(y_train))         
            # 급하게 수정 (20200622)
            num_classes  = 10
        else :
            ylabel_data = np.loadtxt('./mnist_data/ylabel_han.csv', delimiter=',',dtype=str)
            num_classes = int(len(ylabel_data)/2)
            
        print("num_classes=",num_classes)

        with open(TRAIN_RESULT_FILE, 'a') as result_file:
            result_file.write('\nX_train.shape='+str(X_train.shape) + 'y_train.shape='+str(y_train.shape))
            result_file.write('\nX_test.shape ='+str(X_test.shape) + 'y_test.shape  ='+str(y_test.shape))
       
        
        # array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype=object)
        
        # import numpy as np
        from datetime import datetime      # datetime.now() 를 이용하여 학습 경과 시간 측정
        
        #------------------------------------------------------------------------------
        # 여길 참고하자
        # https://tykimos.github.io/2017/01/27/Keras_Talk/
        #------------------------------------------------------------------------------
        
        start_time = datetime.now()
        
        # 2. 모델 구성하기
        #Load_model = Sequential()
        #Load_model.add(Dense(units=64, input_dim=28*28, activation='relu'))
        #Load_model.add(Dense(units=10, activation='softmax'))
        # 3. 모델 학습과정 설정하기
        #Load_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        
        
        #[출처] kaggle, Introduction to CNN Keras - 0.997(top 6%) 리뷰|작성자 kbsdr11
        # https://blog.naver.com/kbsdr11/221636236185
        # CNN을 적용하기 위해 일자로 나열되어있던 X_train과 test의 데이터를 (28, 28, 1)의 차원을 가진 행렬이 42000개 있는 형태로 reshape한다
        # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
            
        print( "self.shape_x=",self.shape_x,"self.shape_y=",self.shape_y)
            
        
        X_train = X_train.reshape(-1,self.shape_x,self.shape_y,1)
        X_test  = X_test.reshape (-1,self.shape_x,self.shape_y,1)
        
        # Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
        # num_classes = 클래스의 수
        
        #elif self.dataset == 'csv_64' :
        #    y_train = np_utils.to_categorical(y_train)
        #    y_test = np_utils.to_categorical(y_test)  
        #else:
        #    Y_train = np_utils.to_categorical(y_train, num_classes = 10)
        
        #Load_model = Sequential()
        # Set the CNN model 
        # my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
        Load_model = Sequential()
        
 
            
        Load_model = Sequential()
        Load_model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                         activation='relu',
                         input_shape=(self.shape_x,self.shape_y,1)))
        Load_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        Load_model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
        Load_model.add(MaxPooling2D(pool_size=(2, 2)))
        Load_model.add(Dropout(0.25))
        Load_model.add(Flatten())
        Load_model.add(Dense(1000, activation='relu'))
        Load_model.add(Dropout(0.5))
        Load_model.add(Dense(num_classes, activation='softmax'))
        
        Load_model.summary()            
            
        
        # Define the optimizer
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)        
        
        # Set a learning rate annealer
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                                    patience=3, 
                                                    verbose=1, 
                                                    factor=0.5, 
                                                    min_lr=0.00001)
        
        # Compile the model, 모델을 설명(optimizer : RMSprop, loss : crossentropy)
        Load_model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
       
        print("models=",models)
        
    # =============================================================================
    #     모델은 더 공부해보고 하자
    #     loss function https://keras.io/api/losses/
    #     optimizer     https://keras.io/api/optimizers/
    #     성능평가방법  https://keras.io/api/metrics/
    # =============================================================================

      
        # 4. 모델 학습시키기
        hist = Load_model.fit(X_train, y_train, epochs=5, batch_size=32)                        
                                     
        # 5. 학습과정 살펴보기
        print('## training loss and acc ##')
        hist.history
        #print(hist.history['loss'])
        #print(hist.history['acc'])
          # 결과파일을 찍어보자
        with open(TRAIN_RESULT_FILE, 'a') as result_file:
            result_file.write("\n########################################################")
            result_file.write("\n\nTraining Mode = "+self.model+"Dataset = "+self.dataset)
            result_file.write('\n## training loss and acc ##')
            result_file.write('\n'+str(hist.history))
        
     
         # 6. 모델 평가하기
        loss_and_metrics = Load_model.evaluate(X_test, y_test, batch_size=32)
        print('## evaluation loss and_metrics ##')
        print(loss_and_metrics)    
        print('\nloss=',loss_and_metrics[0])
        print('\nAccuracy=',loss_and_metrics[1])  

        with open(TRAIN_RESULT_FILE, 'a') as result_file:
            result_file.write('\n## evaluation loss and_metrics ##')
            result_file.write('\n'+str(loss_and_metrics))
            result_file.write('\nloss='+str(loss_and_metrics[0]))
            result_file.write('\nAccuracy='+str(loss_and_metrics[1]))
        
                
        end_time = datetime.now() 
        print("\nelapsed time = ", end_time - start_time)
    
        with open(TRAIN_RESULT_FILE, 'a') as result_file:
            result_file.write("\nelapsed time = "+ str(end_time - start_time))
       
        
        ##########################################################################
        # 학습시킨 모델을  현재 경로에 저장
        ##########################################################################
        # Save the weights
        Load_model.save_weights('./savemodels/Keras_model_weights_'+self.dataset+'_.h5')
        
        # Save the model architecture
        with open('./savemodels/Keras_model_architecture_'+self.dataset+'_.json', 'w') as f:
            f.write(Load_model.to_json())
        
        ##########################################################################
        # 학습시킨 모델을  로드
        ##########################################################################
        
        # Model reconstruction from JSON file
        with open('./savemodels/Keras_model_architecture_'+self.dataset+'_.json', 'r') as f:
            Load_model = model_from_json(f.read())
        
        # Load weights into the new model
        Load_model.load_weights('./savemodels/Keras_model_weights_'+self.dataset+'_.h5')
        
        #(true_list_1, false_list_1, index_label_prediction_list) =loaded_model.accuracy(test_data)    # epochs == 1 인 경우

        ##########################################################################
        # Garbage Collection 을 위해서 리스트 삭제
        ##########################################################################
        del X_train
        del y_train        
        del X_test
        del y_test   
        del Load_model    
    
    ##########################################################################
    # GUI에서 사용하기 위한 테스팅 함수
    ##########################################################################
    def testing(self,models,dataset):
        import cv2    
        self.model = models        
        self.dataset = dataset        

        ##########################################################################
        # 이미지사이즈 
        ##########################################################################       
        if self.dataset == 'mnist' :
            self.shape_x = 28
            self.shape_y = 28         
            
        elif self.dataset == 'csv' :
            self.shape_x = 28
            self.shape_y = 28            
             
        elif self.dataset == 'unicsv_28' :
            self.shape_x = 28
            self.shape_y = 28
 
        elif self.dataset == 'unicsv_46' :
            self.shape_x = 46
            self.shape_y = 46         
  
        elif self.dataset == 'unicsv_64' :
            self.shape_x = 64
            self.shape_y = 64         
            
        elif self.dataset == 'unicsv_128' :
            self.shape_x = 128
            self.shape_y = 128                    
 
        elif self.dataset == 'unicode' :
            self.shape_x = 64
            self.shape_y = 64 
            
        else :
            pass  
        
        ##########################################################################
        # 학습시킨 모델을  로드
        ##########################################################################
        
        # Model reconstruction from JSON file
        with open('./savemodels/Keras_model_architecture_'+self.dataset+'_.json', 'r') as f:
            Load_model = model_from_json(f.read())
        
        # Load weights into the new model
        Load_model.load_weights('./savemodels/Keras_model_weights_'+self.dataset+'_.h5')       
        
        ##########################################################################
        # 이미지데이터 읽어오기
        ##########################################################################
        img=cv2.imread('image.png',0)
        img=cv2.bitwise_not(img)
    ##    cv2.imshow('img',img)
        
        
        
        img=cv2.resize(img,(self.shape_x,self.shape_y))
        img2=cv2.resize(img,(self.shape_x,self.shape_y))
        img=img.reshape(1,self.shape_x,self.shape_y,1)  # CNN 인경우 이렇게

        print("self.shape_x=",self.shape_x,"self.shape_y=",self.shape_y)
        print("img.shape=",img.shape)        

        img=img.astype('float32')
        img=img/255.0
    
        print(img)
        # 입력이미지를 보여주자 (디버깅)
        from matplotlib import pyplot as plt
        plt.grid()
        plt.imshow(img2)
        plt.show()
        
        ##########################################################################
        # 7. 모델 사용하기       
        ##########################################################################
        prediction=Load_model.predict(img)
        print (prediction)
        #pred = prediction.argmax(axis=1)
        
        # predict results
        # select the indix with the maximum probability
        pred = np.argmax(prediction,axis = 1)        
        print ("pred=",pred)
        print ("pred[0]=",pred[0])
        
        num_pred = pred[0]
        print (pred)
        print ('num_pred=',num_pred)
        
        # unicode 인경우 레이블파일을 읽어서 레이블-실제값 으로 변환해줌
        if (self.dataset == 'unicode') or (self.dataset == 'unicsv_128')  or (self.dataset == 'unicsv_64') or (self.dataset == 'unicsv_46') or (self.dataset == 'unicsv_28') :     
             # 레이블 데이터 data 읽어옴
             ylabel_data = np.loadtxt('./mnist_data/ylabel_han.csv', delimiter=',',dtype=str)
             print("ylabel_data.shape = ", ylabel_data.shape)
          
             # 파일을 읽어서 데이터 딕셔너리로 만들어준다.
             ylabel_data_dict=dict()
             i = 0            
             while i < len(ylabel_data) :
                 #print(ylabel_data[i],ylabel_data[i+1])
                 # {가:0,각:1} 로 만들어줌
                 ylabel_data_dict[ylabel_data[i]] = ylabel_data[i+1]
                 i+=2
             
             print ('ylabel_data_dict=',ylabel_data_dict)  
             num_pred = ylabel_data_dict[str(num_pred)]

        else :             
             num_pred = str(num_pred)
         
        return num_pred
    
    
