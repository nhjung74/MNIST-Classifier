## MNIST-Classifier
MNIST Classifier  

* DFC605-2020S: Final Project Report 

* 메인화면

![mnist_classifier_01](https://user-images.githubusercontent.com/59309187/85819989-3cc6a580-b7b0-11ea-89fa-6a4e12e744fd.png)


* 환경 및 프로그램 설명 
1) 개발 및 테스트환경
    - Intel(R) Core(TM) i5-4570S CPU @ 2.90GHz , RAM8GB, MS Windows 10
   
2) 필요라이브러리
    - pip install -r requirements.txt
   
3) 라이브러리 수동설치   
   - pip install openCV-python
   - pip install tensorflow
   - pip install keras
   - pip install scikit-learn
   - pip install Pillow
   - 텐서플로 강제 1.14.0 버전설치
   - pip install "numpy<1.17"  넘파이는 1.17 버전이하로
  
   - 텐서플로설치,케라스설치 with 아나콘다
   - 첫번째로 콘다, 파이썬 패키지 업데이트부터해준다.
   -  conda update -n base conda
   -  conda update --all
   - 이후에
   -  conda install tensorflow
   -  conda install keras

4) 파일/폴더 설명
     - MNIST_Classifier.py           : main program
     - MNIST_Classifier.ui           : QT Designer 로 만든 화면
     - mnist_sklearn.py              : SKlearn  모델 학습 , 예측
     - mnist_keras.py                : Keras    모델 학습 , 예측
     - mnist_tensorflow.py           : 텐서플로우 모델 학습,예측
     - mnist_fonts_to_dataset_csv.py : 폰트 -> 이미지 -> Xtrain_han.csv , ytrain_han.csv , Xtest_han.csv , ytest_han.csv, ylabel_han.csv
     - ./savemodels                  : 모델별 학습결과 저장폴더
     - ./mnist_data                  : mnist 관련데이터 저장폴더 , csv 파일이 생정또는 저장됨
     - ./mnist_data/mnist_train.csv  : 학습시킬 숫자 mnist csv 파일
     - ./mnist_data/mnist_test.csv   : 테스트시킬 숫자 mnist csv 파일
     - ./mnist_data/fonts_train      : 학습시킬 unicode font 를 보관
     - ./mnist_data/fonts_test       : 테스트시킬 unicode font 를 보관
     - ./mnist_data/labels                      
     - ./mnist_data/labels/common-hangul.csv   : unicode font 중 학습시킬 글자정의: 가 ~ 하
     - ./mnist_data/Hangul_Syllables_test  : 데이터셋 생성후 생성 (테스트용 폰트이미지 저장)
     - ./mnist_data/Hangul_Syllables_train : 데이터셋 생성후 생성 (학습용 폰트이미지 저장)

5) 한글데이터셋 생성
      - mnist_fonts_to_dataset_csv.py : 한글폰트--> Mnist datasets (csv) , 클래스로 변경 ,레이블 파일 생성
      - ./fonts_train : Train용 font   
      - ./fonts_test  : Test용 font    
      - ./Hangul_Syllables_train : Train 폰트를 읽어와서 Syllable 단위로 image 를 생성되는 폴더
      - ./Hangul_Syllables_test  : Test  폰트를 읽어와서 Syllable 단위로 image 를 생성되는 폴더
      - ./Xtrain.csv 학습용 한글 mnist파일 (이미지정보)
      - ./ytrain.csv 학습용 한글 mnist파일 (폰트)
      - ./Xtest.csv 테스트용 한글 mnist파일 (이미지정보)
      - ./ytest.csv 테스트용 한글 mnist파일 (폰트)
      - ./ylabel.csv 한글 label 파일 (레이블코드값, 음절 로 구성됨)
      - 아래 파일들은 mnist_train.csv와 유사한 형태의 파일들이다.
      - ./Ztrain_han_128X128.csv     첫번째가 ylabel , 나머지 128x128 (16384) 개의 X이미지로 구성됨 
      - ./Ztest_han_128X128.csv      첫번째가 ylabel , 나머지 128x128 (16384) 개의 X이미지로 구성됨 
      - ./Ztrain_han_64X64.csv      첫번째가 ylabel , 나머지 64x64 (4096) 개의 X이미지로 구성됨 
      - ./Ztest_han_64X64.csv       첫번째가 ylabel , 나머지 64x64 (4096) 개의 X이미지로 구성됨 
      - ./Ztrain_han_46X46.csv     첫번째가 ylabel , 나머지 46x46 (2116) 개의 X이미지로 구성됨 
      - ./Ztest_han_46X46.csv      첫번째가 ylabel , 나머지 46x46 (2116) 개의 X이미지로 구성됨 
      - ./Ztrain_han_28X28.csv      첫번째가 ylabel , 나머지 28x28 (784) 개의 X이미지로 구성됨 
      - ./Ztest_han_28X28.csv      첫번째가 ylabel , 나머지 28x28 (784) 개의 X이미지로 구성됨 
      
      
6) Datasets 
     - sklearn 을 이용하여 데이터셋 저장  ( sklearn 의 버그가 있어서. 비추천)
     - Keras   을 이용하여 데이터셋 저장  (Keras.mnist.load()를 써서 사용 잘됨)
     - tensorflow 을 이용하여 데이터셋 저장 (input_data.read_data_sets 사용 )
     - mnist_train.csv , mnist_test.csv 를 직접읽어서 처리가 가능토록 (이게 잘됨)
     
7) 학습 후 이미지 손글씨 예측 
![mnist_classifier_CNN_hand_ga](https://user-images.githubusercontent.com/59309187/85803723-c3648e00-b782-11ea-8d11-48c7e3655347.png)

8) 학습 후 이미지 폰트이미지 예측
![mnist_classifier_CNN_128_ga](https://user-images.githubusercontent.com/59309187/85803736-cc555f80-b782-11ea-914c-35e8a5639e62.png)


9) 참고사이트
   -  https://hangeul.naver.com/2017/nanum  한글나눔폰트
   -  http://www.unicode.org/charts/PDF/UAC00.pdf 한글Unicode
   -  https://pjreddie.com/media/files/mnist_train.csv  MNIST in CSV 숫자(0~9)필기체 train set 
   -  https://pjreddie.com/media/files/mnist_test.csv   MNIST in CSV 숫자(0~9)필기체 test set
