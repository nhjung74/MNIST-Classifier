# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:38:25 2020

@author: 정낙현
"""
# 논문작성용 일괄작업 배치

# 인공지능 학습/예측 라이브러리 가져오기
from mnist_sklearn      import Class_Sklearn
from mnist_keras        import Class_Keras
from mnist_tensorflow   import Class_Tensorflow
from datetime import datetime      # datetime.now() 를 이용하여 학습 경과 시간 측정
        

# 학습모델 라이브러리 클래스 
lib_sklearn = Class_Sklearn() 
lib_keras   = Class_Keras() 
lib_tensor  = Class_Tensorflow() 

TRAIN_RESULT_FILE = './Training_Result.txt'


#traing_model_list = ['RFC','NAIV','GBC','SGD','LBF','SVM','RFC-MINI','SGD-MINI']
#traing_model_list = ['RFC','NAIV','GBC','SGD','LBF','SVM']
traing_model_list = []
#combo_list_list = ['unicsv_128','unicsv_64','unicsv_46','unicsv_28']
#combo_list_list = ['unicsv_28','unicsv_46']
combo_list_list = ['csv']



start_time = datetime.now()
print ("Taining Start!!!!!!!!!!!")
# 결과파일을 찍어보자(기존꺼 지우고)
with open(TRAIN_RESULT_FILE, 'w') as result_file:
    result_file.write('start_time='+str(start_time))    

for traing_model in traing_model_list:
    for Combo in combo_list_list :
        print ('traing_model=',traing_model,'Combo=',Combo)    
        print ('Sklearn Train Start..')  
        with open(TRAIN_RESULT_FILE, 'a') as result_file:
            result_file.write('\n\nSklearn Train Start..' + 'traing_model='+str(traing_model)+'Combo='+str(Combo))          
        # 학습을 한다     
        pred=lib_sklearn.train_model(model=traing_model,dataset= Combo)
        del pred
        print ('Sklearn Train End..')  


traing_model = 'CNN'
for Combo in combo_list_list :
    print ('traing_model=',traing_model,'Combo=',Combo)    
    print ('Keras Train start..')
    with open(TRAIN_RESULT_FILE, 'a') as result_file:
        result_file.write('\n\Keras Train Start..' + 'traing_model='+str(traing_model)+'Combo='+str(Combo))          
    # 학습을 한다     
    pred=lib_keras.train_model(models=traing_model,dataset= Combo)
    del pred
    print ('Keras Train End..')

print ("Taining End!!!!!!!!!!!")
end_time = datetime.now() 
print("\nelapsed time = ", end_time - start_time)



# 결과파일을 찍어보자
with open(TRAIN_RESULT_FILE, 'a') as result_file:
    result_file.write("\nTaining End!!!!!!!!!!!")
    result_file.write("\nelapsed time = "+ str((end_time - start_time)))
    