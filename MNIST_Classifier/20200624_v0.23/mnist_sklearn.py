# -*- coding: utf-8 -*-
"""
Library_Sklearn 클래스
Created on Mon May 18 08:20:36 2020

@author: Jung Nak Hyun 정낙현
"""
##########################################################################
# 변경이력 
# 2020-05-20 Skit-learn 을 사용함
# 2020-05-22 class 로 전환 
#
# 참고 URL
# https://teddylee777.github.io/scikit-learn/sklearn%EC%9C%BC%EB%A1%9C-mnist-%EC%86%90%EA%B8%80%EC%94%A8%EB%B6%84%EB%A5%98%ED%95%98%EA%B8%B0 
#
##########################################################################
# Import joblib Package
import joblib
import numpy as np
from sklearn.linear_model import SGDRegressor
from tqdm import tqdm

TRAIN_RESULT_FILE = './Training_Result.txt'

                            

# Scikt Learn으로  학습하는 클래스 
class Class_Sklearn :
  
    shape_x = 28
    shape_y = 28

    def __init__(self):
        self.shape_x = 28
        self.shape_y = 28
        
        return
    
   
    def train_model(self,model,dataset):         
        
        self.model = model
        self.dataset = dataset
        ##########################################################################
        # 데이터셋
        ##########################################################################
        
        from sklearn.datasets import fetch_openml
        
        if self.dataset == 'mnist' :
            self.shape_x = 28
            self.shape_y = 28
            # Scikit learn 에 있는 dataset 사용
            mnist = fetch_openml('mnist_784', version=1,data_home="./")
            mnist.data.shape, mnist.target.shape
            # (70000, 784)
            
            #------------------------------------------------------------------------------
            #Dataset을 train data와 test data로 split하기
            #dataset을 split 하는 방법은 직접 구현할 수 도 있고 sklearn에서 제공하는 라이브러리를 사용해도 됩니다.
            #------------------------------------------------------------------------------
            
            split_ratio = 0.9
            n_train = int(mnist.data.shape[0] * split_ratio)
            print(n_train)
            # 63000
            
            n_test = mnist.data.shape[0] - n_train
            print(n_test)
            #7000
            
            X_train = mnist.data[:n_train]
            y_train = mnist.target[:n_train]
            
            X_test = mnist.data[n_train:]
            y_test = mnist.target[n_train:]
            print(X_test.shape, y_test.shape)
            # ((7000, 784), (7000,))
            
            # Garbage Collection 을 위해서 삭제함
            del mnist
            del n_train
            del n_test
        
            
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
    
            print(X_train.shape, y_train.shape)
            # ((63000, 784), (63000,))
            
            X_test = test_data[:,1:]
            y_test = test_data[:,0]
            
            print(X_test.shape, y_test.shape)
            # ((7000, 784), (7000,))
            
            # Garbage Collection 을 위해서 삭제함
            del data
            del test_data
            
        elif self.dataset == 'unicsv_28' :
            self.shape_x = 28
            self.shape_y = 28
            import numpy as np
            
            # 한글 이미지가 2116개의 숫자로 (46x46) 로 구성되어 있는 training data 읽어옴
            data = np.loadtxt('./mnist_data/Ztrain_han_28X28.csv', delimiter=',', dtype=np.float32)
            
            # 한글 이미지가 2116개의 숫자로 (46x46) 로 구성되어 있는 test data 읽어옴
            test_data = np.loadtxt('./mnist_data/Ztest_han_28X28.csv', delimiter=',', dtype=np.float32)
            
            print("data.shape = ", data.shape, " ,  test_data.shape = ", test_data.shape)
            print("data[0,0] = ", data[0,0], ",  test_data[0,0] = ", test_data[0,0])
            print("len(data[0]) = ", len(data[0]), ",  len(test_data[0]) = ", len(test_data[0]))
        
            
            #Splitting the dataset into X and Y
            X_train = data[:,1:]
            y_train  = data[:,0]
    
            print(X_train.shape, y_train.shape)
            # ((63000, 784), (63000,))
            
            X_test = test_data[:,1:]
            y_test = test_data[:,0]
            
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
           
            print(X_test.shape, y_test.shape)
            # ((7000, 784), (7000,))         
            
            # Garbage Collection 을 위해서 삭제함
            del data
            del test_data
 
        elif self.dataset == 'unicsv_46' :
            self.shape_x = 46
            self.shape_y = 46         
            import numpy as np
            
            # 한글 이미지가 2116개의 숫자로 (46x46) 로 구성되어 있는 training data 읽어옴
            data = np.loadtxt('./mnist_data/Ztrain_han_46X46.csv', delimiter=',', dtype=np.float32)
            
            # 한글 이미지가 2116개의 숫자로 (46x46) 로 구성되어 있는 test data 읽어옴
            test_data = np.loadtxt('./mnist_data/Ztest_han_46X46.csv', delimiter=',', dtype=np.float32)
            
            print("data.shape = ", data.shape, " ,  test_data.shape = ", test_data.shape)
            print("data[0,0] = ", data[0,0], ",  test_data[0,0] = ", test_data[0,0])
            print("len(data[0]) = ", len(data[0]), ",  len(test_data[0]) = ", len(test_data[0]))
        
            
            #Splitting the dataset into X and Y
            X_train = data[:,1:]
            y_train  = data[:,0]
    
            print(X_train.shape, y_train.shape)
            # ((63000, 784), (63000,))
            
            X_test = test_data[:,1:]
            y_test = test_data[:,0]
            
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
           
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
            
            # 한글 이미지가 4096개의 숫자로 (64X64) 로 구성되어 있는 test data 읽어옴
            test_data = np.loadtxt('./mnist_data/Ztest_han_64X64.csv', delimiter=',', dtype=np.float32)
            
            print("data.shape = ", data.shape, " ,  test_data.shape = ", test_data.shape)
            print("data[0,0] = ", data[0,0], ",  test_data[0,0] = ", test_data[0,0])
            print("len(data[0]) = ", len(data[0]), ",  len(test_data[0]) = ", len(test_data[0]))
        
            
            #Splitting the dataset into X and Y
            X_train = data[:,1:]
            y_train  = data[:,0]
    
            print(X_train.shape, y_train.shape)
            # ((63000, 784), (63000,))
            
            X_test = test_data[:,1:]
            y_test = test_data[:,0]
            
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
           
            print(X_test.shape, y_test.shape)
            # ((7000, 784), (7000,))   
            
            # Garbage Collection 을 위해서 삭제함
            del data
            del test_data
            
        elif self.dataset == 'unicsv_128' :
            self.shape_x = 128
            self.shape_y = 128        
            import numpy as np
            
            # 한글 이미지가 2116개의 숫자로 (128*128) 로 구성되어 있는 training data 읽어옴
            data = np.loadtxt('./mnist_data/Ztrain_han_128X128.csv', delimiter=',', dtype=np.float32)
            
            # 한글 이미지가 2116개의 숫자로 (128*128) 로 구성되어 있는 test data 읽어옴
            test_data = np.loadtxt('./mnist_data/Ztest_han_128X128.csv', delimiter=',', dtype=np.float32)
            
            print("data.shape = ", data.shape, " ,  test_data.shape = ", test_data.shape)
            print("data[0,0] = ", data[0,0], ",  test_data[0,0] = ", test_data[0,0])
            print("len(data[0]) = ", len(data[0]), ",  len(test_data[0]) = ", len(test_data[0]))
        
            
            #Splitting the dataset into X and Y
            X_train = data[:,1:]
            y_train  = data[:,0]
    
            print(X_train.shape, y_train.shape)
            # ((63000, 784), (63000,))
            
            X_test = test_data[:,1:]
            y_test = test_data[:,0]
            
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
           
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
            
            print(X_test.shape, y_test.shape)
            # ((7000, 784), (7000,)) 
            
            # Garbage Collection 을 위해서 삭제함
            del Xdata
            del ydata
            
            
        else :
            pass
        
        
        with open(TRAIN_RESULT_FILE, 'a') as result_file:
            result_file.write('\nX_train.shape='+str(X_train.shape) + 'y_train.shape='+str(y_train.shape))
            result_file.write('\nX_test.shape ='+str(X_test.shape) + 'y_test.shape  ='+str(y_test.shape))
        
        # Checking uniqueness of the target
        import numpy as np
        #print(np.unique(y_train))
        # array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype=object)
        
        # import numpy as np
        from datetime import datetime      # datetime.now() 를 이용하여 학습 경과 시간 측정
        
        #------------------------------------------------------------------------------
        # 여길 참고하자
        # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
        #------------------------------------------------------------------------------
        
        from sklearn.linear_model   import SGDClassifier
        from sklearn.ensemble       import RandomForestClassifier
        from sklearn.ensemble       import GradientBoostingClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics        import accuracy_score
        from sklearn.metrics        import precision_score
        from sklearn.metrics        import recall_score
        from sklearn.metrics        import f1_score
        from sklearn.metrics        import classification_report
        from sklearn.metrics        import confusion_matrix
        from sklearn.svm            import SVC
        from sklearn.naive_bayes    import BernoulliNB
        
        start_time = datetime.now()
        
        print("model=",self.model)
        
        if self.model == 'RFC' :
            # module loading
            print("Training RandomForestClassifier")
            clf = RandomForestClassifier()
        elif self.model == 'RFC-MINI' :
            # module loading
            print("Training RandomForestClassifier-MINI BATCH")
            #clf = RandomForestClassifier(warm_start=True)
            # 이건 아래에서 모델선택
        elif self.model == 'NAIV' :    
            # module loading
            print("Training Naive Bayes Classifier")
            #clf = BernoulliNB(alpha=0)
            clf = BernoulliNB()
        elif self.model == 'GBC' :    
            # module loading
            print("Training GradientBoostingClassifier")
            clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train,y_train)
        elif  self.model == 'SGD' :    
            # module loading
            print("Training SGDClassifier SGD")
            #clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(10,), random_state=1)
            #clf =MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
            #           solver='adam', verbose=0, tol=1e-8, random_state=1,
            #           learning_rate_init=.01)
            # SGD 모형은 가중치를 계속 업데이트하므로 일부 데이터를 사용하여 구한 가중치를 
            # 다음 단계에서 초기 가중치로 사용할 수 있다. 이렇게 하려면 모형을 생성할 때 warm_start 인수를 True로 하고 학습
            clf = SGDClassifier(warm_start=True)            
        elif  self.model == 'SGD-MINI' :    
            # module loading
            print("Training SGDClassifier SGD-MINI")
            # SGD 모형은 가중치를 계속 업데이트하므로 일부 데이터를 사용하여 구한 가중치를 
            # 다음 단계에서 초기 가중치로 사용할 수 있다. 이렇게 하려면 모형을 생성할 때 warm_start 인수를 True로 하고 학습
            #clf = SGDClassifier(warm_start=True)
            clf = SGDClassifier(alpha=.0001, loss='log', penalty='l2', n_jobs=-1, shuffle=True, max_iter=100, verbose=0, tol=0.001,warm_start=True)
            
            
        elif self.model == 'LBF' :    
            # module loading
            print("Training MLPClassifier lbfgs")
            clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,), random_state=1)
        elif self.model == 'SVM' :
            # module loading
            print("Training SVM model ")
            clf = SVC(C=200,kernel='rbf',gamma=0.01,cache_size=8000,probability=False)
        elif self.model == 'LRG' :
            # module loading
            print("Training LogisticRegression model ")
            clf = LogisticRegression(C=50. / 5000, penalty='l1', solver='saga', tol=0.1)
    
        
        # 미니 배치 학습
        if self.model == 'RFC-MINI' :
            # https://datascienceschool.net/view-notebook/cabff291dfb046dfaac6e8f14ee50dac/
            from sklearn.model_selection import KFold
            from sklearn.metrics import accuracy_score
            
            
            # 데이터를 400개의 조각으로 나누어 읽는다.
            n_split = 400
            n_epoch = 1
            
            num_tree_ini = 10
            num_tree_step = 10

            clf = RandomForestClassifier(n_estimators=num_tree_ini, warm_start=True)
            
            # ytain Data 가 n_split보다 작으면 ytrain 크기만큼으로 조정해준다.
            if len(y_train) < n_split :
               n_split = 2
               n_epoch = 5
                
                
            n_X = len(y_train)// n_split
            
            accuracy_train = []
            accuracy_test = []
            N_CLASSES = np.unique(y_train)
            #N_CLASSES = np.array(y_train)
            
            print('n_split=',n_split)
            print('n_epoch=',n_epoch)
            print('len(y_train)=',len(y_train))
            print('N_CLASSES=',len(N_CLASSES))   
            
            # Create the vectorizer and limit the number of features to a reasonable
            # maximum
            
            for epoch in range(n_epoch):
                print("epoch=",epoch)  
                for n in tqdm(range(n_split)):
                    idx = list(range(n * n_X, min(len(y_train) - 1, (n + 1) * n_X))) 
                    
                    X = X_train[idx,:]
                    y = y_train[idx]
                    #print("X=",X)
                    
                    # 미니배치학습                    
                    clf.fit(X, y)
                    
# =============================================================================
# 20200618 잘되다가 안되서 막았음
#                     accuracy_train.append(accuracy_score(y_train, clf.predict(X_train)))
#                     accuracy_test.append(accuracy_score(y_test, clf.predict(X_test)))
#                     clf.n_estimators += num_tree_step
#     
#                 print ("accuracy_train=",accuracy_train)
#                 print ("accuracy_test=",accuracy_test)
#             
#             
#             # 정확도를 한번 봐보자
#             from matplotlib import pyplot as plt
#             plt.rcParams["font.family"] = 'Batang'
#             plt.plot(accuracy_train, "g:", label="학습 성능")
#             plt.plot(accuracy_test, "r-", alpha=0.5, label="검증 성능")
#             plt.legend()
#             plt.savefig('accuracy.png')
#             plt.show()           
# =============================================================================
            
        elif self.model == 'SGD-MINI' :
            # https://datascienceschool.net/view-notebook/cabff291dfb046dfaac6e8f14ee50dac/
            from sklearn.model_selection import KFold
            from sklearn.metrics import accuracy_score
            
            
            # 데이터를 400개의 조각으로 나누어 읽는다.
            n_split = 400
            n_epoch = 2
            
            # ytain Data 가 n_split보다 작으면 ytrain 크기만큼으로 조정해준다.
            if len(y_train) < n_split :
               # n_split = int (len(y_train)/2)+1
               # n_split = int (len(y_train))+1
               #n_split   = len(y_train)+1
               n_split = 2
               n_epoch = 5
                
                
            n_X = len(y_train) // n_split
            
            accuracy_train = []
            accuracy_test = []
            N_CLASSES = np.unique(y_train)
            #N_CLASSES = np.array(y_train)
            
            print('n_split=',n_split)
            print('n_epoch=',n_epoch)
            print('len(y_train)=',len(y_train))
            print('N_CLASSES=',len(N_CLASSES))   
            
            # Create the vectorizer and limit the number of features to a reasonable
            # maximum
            
            for epoch in range(n_epoch):
                print("epoch=",epoch)  
                for n in tqdm(range(n_split)):
                    idx = list(range(n * n_X, min(len(y_train) - 1, (n + 1) * n_X))) 
                    #print("idx=",idx)
                    #print("range=",range(n * n_X, min(len(y_train) - 1, (n + 1) * n_X)))
                    #print("n * n_X=",n * n_X ,"(n + 1) * n_X)=",(n + 1) * n_X )
                    
                    X = X_train[idx,:]
                    y = y_train[idx]
                    #print("X=",X)
                    
                    # 미니배치학습                    
                    clf.partial_fit(X, y, classes=N_CLASSES)
                    
                    accuracy_train.append(accuracy_score(y_train, clf.predict(X_train)))
                    accuracy_test.append(accuracy_score(y_test, clf.predict(X_test)))
    
                print ("accuracy_train=",accuracy_train)
                print ("accuracy_test=",accuracy_test)
            
            
            # 정확도를 한번 봐보자
            from matplotlib import pyplot as plt
            plt.rcParams["font.family"] = 'Batang'
            plt.plot(accuracy_train, "g:", label="학습 성능")
            plt.plot(accuracy_test, "r-", alpha=0.5, label="검증 성능")
            plt.legend()
            plt.savefig('accuracy.png')
            plt.show()           
                
        else :
            # train data!
            clf.fit(X_train, y_train)     
            
        
        # make predicition
        prediction = clf.predict(X_test)
        print(prediction.shape)
        # 7000
        
        # accuracy
        result = (prediction == y_test).mean()
        print(result)
        # 0.9617142857142857
                
        end_time = datetime.now() 
        print("\nelapsed time = ", end_time - start_time)
        print(accuracy_score(y_test, prediction))
        
        # Evaluation Metrics (성능평가지표)
        print('accuracy = ', accuracy_score(y_test,prediction) )
        print('precision= ', precision_score(y_test,prediction,average='micro') )
        print('recall   = ', recall_score(y_test,prediction, average='micro') )
        print('f1       = ', f1_score(y_test,prediction, average='micro') )
        print(classification_report(y_test,prediction))
        print(confusion_matrix(y_test,prediction))        
    
        # 결과파일을 찍어보자
        with open(TRAIN_RESULT_FILE, 'a') as result_file:
            result_file.write("\nTraining Mode = "+self.model+"Dataset = "+self.dataset)
            result_file.write("\nEelapsed time = "+str(end_time - start_time))
            result_file.write("\n"+str(accuracy_score(y_test, prediction)))
            result_file.write('\naccuracy = '+ str(accuracy_score(y_test,prediction) ))
            result_file.write('\nprecision= '+ str(precision_score(y_test,prediction,average='micro') ))
            result_file.write('\nrecall   = '+ str(recall_score(y_test,prediction, average='micro') ))
            result_file.write('\nf1       = '+ str(f1_score(y_test,prediction, average='micro') ))
            result_file.write("\n"+str(classification_report(y_test,prediction)))
            result_file.write("\n"+str(confusion_matrix(y_test,prediction))        )
    
        
        ##########################################################################
        # 학습시킨 모델을  현재 경로에 저장
        ##########################################################################
        joblib.dump(clf,'./savemodels/SKlearn_model'+ self.model +'_' + self.dataset +'.pkl')
        
        ##########################################################################
        # 학습시킨 모델을  로드
        ##########################################################################
        #loaded_model = joblib.load('./savemodels/SKlearn_model'+ self.model +'_' + self.dataset+'.pkl')
        
        #(true_list_1, false_list_1, index_label_prediction_list) =loaded_model.accuracy(test_data)    # epochs == 1 인 경우
    
        ##########################################################################
        # Garbage Collection 을 위해서 리스트 삭제
        ##########################################################################
        del X_train
        del y_train        
        del X_test
        del y_test   
        del clf
    
    ##########################################################################
    # GUI에서 사용하기 위한 테스팅 함수
    ##########################################################################
    def testing(self,model,dataset):

        self.model = model        
        self.dataset = dataset
        
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
        
        import cv2    

        loaded_model = joblib.load('./savemodels/SKlearn_model'+ self.model +'_' +self.dataset +'.pkl')
        
        img=cv2.imread('image.png',0)
        # 검은색 하얀색 반전
        img=cv2.bitwise_not(img)
        ##    cv2.imshow('img',img)
        #
        #img=img.reshape(1,28,28,1)
        #

        
        img=cv2.resize(img,(self.shape_x,self.shape_y))   
        img2=cv2.resize(img,(self.shape_x,self.shape_y))  
        img=img.reshape(1,self.shape_x*self.shape_y) 
        
        print("self.shape_x=",self.shape_x,"self.shape_y=",self.shape_y)
        print("img.shape=",img.shape)
        
        
        img=img.astype('float32')
        img=img/255.0

        print(img)
        pred=loaded_model.predict(img)
        
        # 입력이미지를 보여주자 (디버깅)
        from matplotlib import pyplot as plt
        plt.grid()
        plt.imshow(img2)
        plt.show()
      
        print ("pred=",pred)
        print ("pred[0]=",pred[0])
        
        num_pred = int(pred[0])
        print (pred)
        print ('num_pred=',num_pred)        
        
        # unicode 인경우 레이블파일을 읽어서 레이블-실제값 으로 변환해줌
        if (self.dataset == 'unicode')  :        
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
             pred = ylabel_data_dict[pred[0]]
        
        elif self.dataset == 'unicsv_64'  or self.dataset == 'unicsv_46'  or self.dataset == 'unicsv_28'  or self.dataset == 'unicsv_128'   :        
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
             pred = ylabel_data_dict[str(int(pred[0]))]             
        else :
             pred = str(num_pred)
        
        
        return pred
