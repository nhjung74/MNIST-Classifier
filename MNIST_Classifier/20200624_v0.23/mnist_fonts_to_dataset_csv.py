# -*- coding: utf-8 -*-
"""
Created on Mon May 25 07:34:26 2020

@author: Jung Nak Hyun 정낙현
"""
# 폰트를 이미지로 변환
import io
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2   
import random
import numpy
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
   

# Number of random distortion images to generate per font and character.
DISTORTION_COUNT = 5

# Width and height of the resulting image.
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

# Default data paths.
DEFAULT_LABEL_FILE = './mnist_data/labels/common-hangul.csv'


def elastic_distort(image, alpha, sigma):
    """Perform elastic distortion on an image.
    Here, alpha refers to the scaling factor that controls the intensity of the
    deformation. The sigma variable refers to the Gaussian filter standard
    deviation.
    """
    random_state = numpy.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha

    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)



class ttf_to_png():        
    
    # OpenCV2 가 한글경로를 인식 못하여 함수를 만들어 우회함
    def imread(self, filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8): 
        try:
            n = np.fromfile(filename, dtype) 
            img = cv2.imdecode(n, flags) 
            return img 
        except Exception as e:
            print(e) 
            return None
    
    
    def imwrite(self,filename, img, params=None):
        try: 
            ext = os.path.splitext(filename)[1] 
            result, n = cv2.imencode(ext, img, params) 
            
            if result: 
                with open(filename, mode='w+b') as f: 
                    n.tofile(f) 
                return True 
            else: return False 
        except Exception as e: 
            print(e) 
            return False
    

    def process(self,gubun):         
        
        self.gubun = gubun
        
        if gubun == 'train' :
            font_path         = "./mnist_data/fonts_train/"
            xcvs_file         = "./mnist_data/Xtrain_han.csv"
            ycvs_file         = "./mnist_data/ytrain_han.csv"
            zcvs_file128      = "./mnist_data/Ztrain_han_128X128.csv"
            zcvs_file64       = "./mnist_data/Ztrain_han_64X64.csv"
            zcvs_file46       = "./mnist_data/Ztrain_han_46X46.csv"
            zcvs_file28       = "./mnist_data/Ztrain_han_28X28.csv"
            Syllables_path    = "./mnist_data/Hangul_Syllables_train/"
        elif gubun == 'test' :
            font_path         = "./mnist_data/fonts_test/"
            xcvs_file         = "./mnist_data/Xtest_han.csv"
            ycvs_file         = "./mnist_data/ytest_han.csv"
            zcvs_file128      = "./mnist_data/Ztest_han_128X128.csv"
            zcvs_file64      = "./mnist_data/Ztest_han_64X64.csv"
            zcvs_file46       = "./mnist_data/Ztest_han_46X46.csv"
            zcvs_file28       = "./mnist_data/Ztest_han_28X28.csv"
            Syllables_path = "./mnist_data/Hangul_Syllables_test/"
        elif gubun == 'label' :
            ylabel_file    = "./mnist_data/ylabel_han.csv"
        else :
            print ("gubun error")
            return 
        
        # 일단지워버리고 시작하자
        if gubun == 'label':
            with open(ylabel_file, 'w') as h: 
                h.write('')           
        else:
            with open(xcvs_file, 'w') as f:
                f.write('') 
            with open(ycvs_file, 'w') as g:
                g.write('') 
            with open(zcvs_file128, 'w') as h128:
                h128.write('') 
            with open(zcvs_file64, 'w') as h64:
                h64.write('') 
            with open(zcvs_file46, 'w') as h46:
                h46.write('')                
            with open(zcvs_file28, 'w') as h28:
                h28.write('')                
            
        # 폰트파일 가져오기
        if gubun != 'label':
            fonts = os.listdir(font_path)
            #print(fonts)
        
            #print ('font_len=',len(fonts))
            #print ('font End=',fonts[len(fonts)-1])  # 마지막 폰트 파일
            end_font =  fonts[len(fonts)-1]
        
        ###################################################################################
        ###################################################################################
        
# =============================================================================
#         co = "0 1 2 3 4 5 6 7 8 9 A B C D E F"
#         start = "AC00" # 가
#         end = "D7A3"   # 힣
#         #end = "AC00"   #    // 학습량을 줄여서 해보자
#         
#         co = co.split(" ")
#         
#         Hangul_Syllables = [a+b+c+d 
#                             for a in co 
#                             for b in co 
#                             for c in co 
#                             for d in co]
#         
#         Hangul_Syllables = np.array(Hangul_Syllables)
#         
#         s = np.where(start == Hangul_Syllables)[0][0]
#         e = np.where(end == Hangul_Syllables)[0][0]
#         
#         Hangul_Syllables = Hangul_Syllables[s : e + 1]
#         
#         print(Hangul_Syllables)
#         #array(['AC00', 'AC01', 'AC02', ..., 'D7AD', 'D7AE', 'D7AF'], dtype='<U4')
#         
#         len(Hangul_Syllables)
#         
#         print(chr(int('AC00', 16)), chr(int("D7A3", 16)))
#         
#         print(Hangul_Syllables)
#         #array(['AC00', 'AC01', 'AC02', ..., 'D7AD', 'D7AE', 'D7AF'], dtype='<U4')
# 
#         # Hangul_Syllables 를 한글char로 변환시킴
#         for i in range (0, len(Hangul_Syllables)):
#             uni_code = Hangul_Syllables[i]
#             Hangul_Syllables[i] = chr(int(uni_code, 16))
#             #print("Convert  Hangul_Syllables[",str(i),"]=",Hangul_Syllables[i])
#             
#         print("Convert  Hangul_Syllables[",str(i),"]=",Hangul_Syllables[i])            
#             
#         #print(Hangul_Syllables)
# =============================================================================

        ###################################################################################
        # DEFAULT_LABEL_FILE 파일을 가져와서 Hangul_Syllables 를 만들자
        ###################################################################################
         
        # 학습시킬 한글 글자가 저장되어 있음 
        # 인코딩 타입이 유니코드(unicode) 또는 UTF-8인 문서를 읽을 때 파일의 처음에 \ufeff 가 추가되는데, utf-8-sig 로해결한다.
        # 출처: https://redcarrot.tistory.com/216 [빨간당무 마을]
        with io.open(DEFAULT_LABEL_FILE, 'r', encoding='utf-8-sig') as f:
            char_labels = f.read().splitlines()
            
                    
        
        
        print("char_labels=",char_labels)
        print("len(char_labels)=",len(char_labels))
        print("char_labels[len(char_labels)]=",char_labels[len(char_labels)-1])

        end_char = char_labels[len(char_labels)-1]       
        
         # set으로 중복제거후 정렬 -> 딕셔너리 만듬
        Hangul_dict       = {i:v  for i, v in  enumerate(sorted(set(char_labels)))} # key - valuse        
        Hangul_dict_value = {v:i  for i, v in  enumerate(sorted(set(char_labels)))} # valuse-key 
        #print (Hangul_dict_value)        
              
        ###################################################################################
        ###################################################################################

                                
        # 레이블 데이터 만들기
        if gubun == 'label' :
            
       
            k = 0
            #print (Hangul_dict)
            for value in tqdm(char_labels) :
                with open(ylabel_file, 'a') as h:
                    if end_char == value  :
                        han_code = value
                        print ('k=',k,'value=',value,'han_code =',han_code)
                        #h.write('%d'%k+","+han_code)
                        h.write('%d'%k+","+'%s'%han_code)
                    else :
                        han_code = value
                        print ('k=',k,'value=',value,'han_code =',han_code)
                        #h.write('%d'%k+","+han_code+",")
                        h.write('%d'%k+","+'%s'%han_code+",")
                k += 1
                
            
        # Trian set , Test set 만들기
        else :
            # gubun 이 'train' , 'test' 때만 파일생성
            char_cnt = 0
            # label 폴더에 있는 파일만 학습함    
            for character in tqdm(char_labels):
                
                char_cnt += 1
                path = Syllables_path + character
                
                os.makedirs(path, exist_ok = True)
                    
                for ttf in fonts:
                     
                    #######################################
                    # size 48,60 자리 두사이즈의 폰트를 만듬(다양한 사이즈 가능하게)
                    #######################################
                    ListFontSize =[48,60]
                    #ListFontSize =[48]
                    
                    for font_size in ListFontSize:
                        # mnist 형태의 csv파일을 만들어보자
                        from PIL import Image
                
                        font = font_path + ttf
                        #print(font)                   
                        
                        #print ("FontSize =",font_size)                        
                        image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color='black')
                        font = ImageFont.truetype(font, font_size)
                        drawing = ImageDraw.Draw(image)
                        w, h = drawing.textsize(character, font=font)
                        drawing.text(
                            ((IMAGE_WIDTH-w)/2, (IMAGE_HEIGHT-h)/2),
                            character,
                            fill='white',
                            font=font
                        )
                         
                        # 이미지 저장
                        msg = path + "/" + ttf[:-4] + "_" + str(font_size) +"_" + character
                        
                        image.save('{}.png'.format(msg))
                        
                        # 이미지를 왜곡(distortion) 시켜서 이미지를 다시 저장하자
                        for i in range(DISTORTION_COUNT):
                            arr = numpy.array(image)
    
                            distorted_array = elastic_distort(
                                arr, alpha=random.randint(30, 36),
                                sigma=random.randint(5, 6)
                            )
                            distorted_image = Image.fromarray(distorted_array)
                           
                            msg = path + "/" + ttf[:-4] + "_"+str(i)+"_" + str(font_size) +"_" + character                     
                            distorted_image.save('{}.png'.format(msg)) 
                            
    
    
                            # 이미지를 읽어와서 mnist csv 파일형태로 만들어줌 
                            # 20200612 Mnist처럼 검은색바탕에  흰색으로 하자
                            #img=cv2.bitwise_not(img)  # 검은색바탕흰색 이미지를 -> 흰색바탕 검은색 글씨로 인식 
                            
                            ##  cv2.imshow('img',img)
                            #img=cv2.resize(img,(28,28)) # mnist 예제는 24*24
                            #img=img.reshape(1,28,28,1)
                            #img=img.reshape(1,784)      # mnist 예제는 24*24
                            #img=img/255.0

                            img=self.imread('{}.png'.format(msg),0)
                            img=cv2.resize(img,(64,64))  # 한글은 나중에 64*64 로 바꾸자
                            img=img.reshape(1,64*64)     # 한글은 나중에 64*64 로 바꾸자 
                            img=img.astype('int')
                        
                            #print(img)                      
                            # 128*128 이미지
                            img128=self.imread('{}.png'.format(msg),0)
                            img128=cv2.resize(img128,(128,128))  # 128 * 128
                            img128=img128.reshape(1,128*128)     # 128 * 128
                            img128=img128.astype('int') 

                           # 46*46 이미지
                            img46=self.imread('{}.png'.format(msg),0)
                            img46=cv2.resize(img46,(46,46))  # 46 *46
                            img46=img46.reshape(1,46*46)     # 46 *46
                            img46=img46.astype('int')
                            
                            # 28*28 이미지
                            img28=self.imread('{}.png'.format(msg),0)
                            img28=cv2.resize(img28,(28,28))  # 28 * 28
                            img28=img28.reshape(1,28*28)     # 28 * 28 
                            img28=img28.astype('int')
                            
                            # Xtrain Data 저장
                            with open(xcvs_file, 'ab') as f:
                                np.savetxt(f, img, fmt='%3d',delimiter=",") 
                    
                            # yTrain Data
                            # unicode 값을 한글로 변환시켜서 봄    
                            #han_code = chr(int(uni, 16))  
                            # Dictinary 에 있는 label index값으로 변환시켜 저장
                       
                            
                            han_index= Hangul_dict_value[character]
                            #print ('char_cnt =',char_cnt , 'character=',character,'ttf=',ttf, 'end_font=',end_font)             
                            with open(ycvs_file, 'a') as g:
                                # 마지막인경우
                                if (char_cnt == len(char_labels)) &  (ttf == end_font) & (i ==  DISTORTION_COUNT -1) & ( ListFontSize[len(ListFontSize)-1] == font_size ):
                                    #g.write(han_index)
                                    g.write('%d'%han_index)
                                else :
                                    #g.write(han_index+",")
                                    g.write('%d'%han_index+",")

                            # Ztrain Data 저장 : mnist_csv 형태로 변환해줌
                            with open(zcvs_file128, 'a') as h128:
                                h128.write('%d'%han_index+",")
                                
                            with open(zcvs_file128, 'ab') as i128:
                                np.savetxt(i128, img128, fmt='%d',delimiter=",")                          
               
                            # Ztrain Data 저장 : mnist_csv 형태로 변환해줌
                            with open(zcvs_file64, 'a') as h64:
                                h64.write('%d'%han_index+",")
                                
                            with open(zcvs_file64, 'ab') as i64:
                                np.savetxt(i64, img, fmt='%d',delimiter=",") 
                                
                            # Ztrain Data 저장 : mnist_csv 형태로 변환해줌
                            with open(zcvs_file46, 'a') as h46:
                                h46.write('%d'%han_index+",")
                                
                            with open(zcvs_file46, 'ab') as i46:
                                np.savetxt(i46, img46, fmt='%d',delimiter=",")                                 
                                
                                
                            # Ztrain Data 저장 : mnist_csv 형태로 변환해줌
                            with open(zcvs_file28, 'a') as h28:
                                h28.write('%d'%han_index+",")
                                
                            with open(zcvs_file28, 'ab') as i28:
                                np.savetxt(i28, img28, fmt='%d',delimiter=",")     
                    
                            # yTrain Data
                            # unicode 값을 한글로 변환시켜서 봄    
                            #han_code = chr(int(uni, 16))  
                            # Dictinary 에 있는 label index값으로 변환시켜 저장

                         
                         
if __name__ == "__main__" :
    
    TTF = ttf_to_png()  
    
    print('Traing Set Data Create Start')
    TTF.process('train') 
    print('Traing Set Data Create End')
    
    print('Test Set Data Create Start')
    TTF.process('test')    
    print('Test Set Data Create End')
    
    print('Label Data Create Start')
    TTF.process('label')    
    print('Label  Data Create End')    
             
# #with open('train1.csv', 'w') as g:
#    g.write(chr(int(labels, 16)) 
 
       


# =============================================================================
# # 텍스트를 이미지로 만들어서 훈련 데이터로 사용하기 위한 목적
# from PIL import Image,ImageDraw,ImageFont
#  
# # 이미지로 출력할 글자 및 폰트 지정 (나눔고딕 트루타입폰트 파일)
# draw_text = '가'
# font = ImageFont.truetype("./NanumGothic.ttf", 25)
#  
# # 이미지 사이즈 지정
# text_width = 28
# text_height = 28
#  
# # 이미지 객체 생성 (배경 검정)
# canvas = Image.new('RGB', (text_width, text_height), "white")
#  
# # 가운데에 그리기 (폰트 색: 하양)
# draw = ImageDraw.Draw(canvas)
# w, h = font.getsize(draw_text)
# draw.text(((text_width-w)/2.0,(text_height-h)/2.0), draw_text, 'black', font)
#  
# # 가.png로 저장 및 출력해서 보기
# canvas.save(draw_text+'.png', "PNG")
# canvas.show()
# 
# =============================================================================
