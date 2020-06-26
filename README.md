## MNIST-Classifier
MNIST Classifier

DFC605-2020S: Final Project Report 



1) 필요라이브러리
   pip install -r requirements.txt
   
2) 라이브러리 수동설치   
   - pip install openCV-python
   - pip install tensorflow
   - pip install keras
   - pip install scikit-learn
   - pip install Pillow
   - 텐서플로 강제 1.14.0 버전설치
   - pip install "numpy<1.17"  넘파이는 1.17 버전이하로
   - 텐서플로설치,케라스설치 with 아나콘다
      -- 첫번째로 콘다, 파이썬 패키지 업데이트부터해준다.
      ---  conda update -n base conda
      ---  conda update --all
      --  이후에
        --- conda install tensorflow
        ---conda install keras
   
2) 한글데이터셋 생성
      - mnist_fonts_to_dataset_csv.py : 한글폰트--> Mnist datasets (csv) , 클래스로 변경 ,레이블 파일 생성
      - ./fonts_train : Train용 font   
      - ./fonts_test  : Test용 font    
      - 테스트용 폰트 : https://hangeul.naver.com/2017/nanum
      - ./Hangul_Syllables_train : Train 폰트를 읽어와서 Syllable 단위로 image 를 생성되는 폴더
      - ./Hangul_Syllables_test  : Test  폰트를 읽어와서 Syllable 단위로 image 를 생성되는 폴더
      - ./Xtrain.csv 학습용 한글 mnist파일 (이미지정보)
      - ./ytrain.csv 학습용 한글 mnist파일 (폰트)
      - ./Xtest.csv 테스트용 한글 mnist파일 (이미지정보)
      - ./ytest.csv 테스트용 한글 mnist파일 (폰트)
      - ./ylabel.csv 한글 label 파일 (레이블코드값, 음절 로 구성됨)
      - ./Ztrain_han_64X64.csv     mnist_train.csv 와 유사한 형태 64*64 ( 첫번째가 ylabel , 나머지 64*64 (4096) 개의 X이미지로 구성됨 )
      - ./Ztest_han_64X64.csv     mnist_test.csv 와 유사한 형태 64*64  ( 첫번째가 ylabel , 나머지 64*64 (4096) 개의 X이미지로 구성됨 )
      - ./Ztrain_han_46X46.csv     mnist_train.csv 와 유사한 형태 46*46 ( 첫번째가 ylabel , 나머지 64*64 (4096) 개의 X이미지로 구성됨 )
      - ./Ztest_han_46X46.csv     mnist_test.csv 와 유사한 형태 46*46  ( 첫번째가 ylabel , 나머지 64*64 (4096) 개의 X이미지로 구성됨 )
      - ./Ztrain_han_28X28.csv     mnist_train.csv 와 유사한 형태 28*28 ( 첫번째가 ylabel , 나머지 64*64 (4096) 개의 X이미지로 구성됨 )
      - ./Ztest_han_28X28.csv     mnist_test.csv 와 유사한 형태 28*28  ( 첫번째가 ylabel , 나머지 64*64 (4096) 개의 X이미지로 구성됨 )
            
      - 한글나눔폰트  https://hangeul.naver.com/2017/nanum
      

3) 파일설명
     - MNIST_Classifier.py     실행 main 파일
     - MNIST_Classifier.ui     QT Designer 로 만든 화면
     - mnist_sklearn.py        SKlearn  모델 학습 , 예측
     - mnist_keras.py          Keras    모델 학습 , 예측
     - mnist_tensorflow.py     텐서플로우 모델 학습,예측
     - mnist_fonts_to_dataset_csv.py  : 폰트 -> 이미지, Xtrain_han.csv , ytrain_han.csv , Xtest_han.csv , ytest_han.csv, ylabel_han.csv
      
4) Datasets 
     - sklearn 을 이용하여 데이터셋 저장  ( sklearn 의 버그가 있어서. 비추천)
     - Keras   을 이용하여 데이터셋 저장  (Keras.mnist.load()를 써서 사용 잘됨)
     - tensorflow 을 이용하여 데이터셋 저장 (input_data.read_data_sets 사용 )
     - mnist_train.csv , mnist_test.csv 를 직접읽어서 처리가 가능토록 (이게 잘됨)
     
5) 학습 후 이미지 예측 (손글씨)
![mnist_classifier_CNN_hand_ga](https://user-images.githubusercontent.com/59309187/85803723-c3648e00-b782-11ea-8d11-48c7e3655347.png)

5) 학습 후 이미지 예측 (폰트)
![mnist_classifier_CNN_128_ga](https://user-images.githubusercontent.com/59309187/85803736-cc555f80-b782-11ea-914c-35e8a5639e62.png)
