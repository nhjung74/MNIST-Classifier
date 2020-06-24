# -*- coding: utf-8 -*-
"""
Created on Tue May 19 07:47:07 2020

@author: Jung Nak Hyun 정낙현
"""
##########################################################################
# 변경이력 
# 2020-05-19 최초버전 개발
# 2020-05-20 Skit-learn 을 사용함
# 2020-05-22 Class call  하는 형태로 전환
##########################################################################

import sys
 
# PYQT5 를 이용하기 위한 모듈갱신
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic

# 인공지능 학습/예측 라이브러리 가져오기
from mnist_sklearn      import Class_Sklearn
from mnist_keras        import Class_Keras
from mnist_tensorflow   import Class_Tensorflow

# 화면에서 PLOT 그래프를 출력하기 위해서 추가
#from matplotlibwidgetFile import matplotlibwidget
    

###############################################################################
# QT Designer 로 만든 UI 파일 QDialog 로 만듬
# 불러오고자 하는 .ui 파일
# .py 파일과 같은 위치에 있어야 한다 *****
###############################################################################
form_class = uic.loadUiType("MNIST_Classifier.ui")[0]


###############################################################################
# User Class 정의 
###############################################################################
# Colour3 색상 정의
class Colour3:
    R = 0
    G = 0
    B = 0
    #CONSTRUCTOR
    def __init__(self): 
        self.R = 0
        self.G = 0
        self.B = 0
    #CONSTRUCTOR - with the values to give it
    def __init__(self, nR, nG, nB):
        self.R = nR
        self.G = nG
        self.B = nB
        
## My Own Point Class, simple and light weight
class Point:
    #X Coordinate Value
    X = 0
    #Y Coordinate Value
    Y = 0
    #CONSTRUCTOR
    def __init__(self): 
        self.X = 0
        self.Y = 0
    #CONSTRUCTOR - with the values to give it
    def __init__(self, nX, nY):
        self.X = nX
        self.Y = nY
    #So we can set both values at the same time
    def Set(self,nX, nY):
        self.X = nX
        self.Y = nY 
## Shape class; holds data on the drawing point
class Shape:
    Location = Point(0,0)
    Width = 0.0
    Colour = Colour3(0,0,0)
    ShapeNumber = 0
    #CONSTRUCTOR - with the values to give it
    def __init__(self, L, W, C, S):
        self.Location = L
        self.Width = W
        self.Colour = C
        self.ShapeNumber = S


# Shapes Class 정의
class Shapes:
    #Stores all the shapes
    __Shapes = []
    def __init__(self):
        self.__Shapes = []
    #Returns the number of shapes being stored.
    def NumberOfShapes(self):
        return len(self.__Shapes)
    #Add a shape to the database, recording its position,
    #width, colour and shape relation information
    def NewShape(self,L,W,C,S):
        Sh = Shape(L,W,C,S)
        self.__Shapes.append(Sh)
    #returns a shape of the requested data.
    def GetShape(self, Index):
        return self.__Shapes[Index]
    #Removes any point data within a certain threshold of a point.
    def RemoveShape(self, L, threshold):
        #do while so we can change the size of the list and it wont come back to bite me in the ass!!
        i = 0
        while True:
            if(i==len(self.__Shapes)):
                break 
            #Finds if a point is within a certain distance of the point to remove.
            if((abs(L.X - self.__Shapes[i].Location.X) < threshold) and (abs(L.Y - self.__Shapes[i].Location.Y) < threshold)):
                #removes all data for that number
                del self.__Shapes[i]
                #goes through the rest of the data and adds an extra
                #1 to defined them as a seprate shape and shuffles on the effect.
                for n in range(len(self.__Shapes)-i):
                    self.__Shapes[n+i].ShapeNumber += 1
                #Go back a step so we dont miss a point.
                i -= 1
            i += 1

# Painter Class 정의                    
class Painter(QWidget):
    ParentLink = 0
    MouseLoc = Point(0,0)  
    LastPos = Point(0,0)  
    def __init__(self,parent):
        super(Painter, self).__init__()
        self.ParentLink = parent
        self.MouseLoc = Point(0,0)
        self.LastPos = Point(0,0) 
    #Mouse down event
    def mousePressEvent(self, event): 
        if(self.ParentLink.Brush == True):
            self.ParentLink.IsPainting = True
            self.ParentLink.ShapeNum += 1
            self.LastPos = Point(0,0)
        else:
            self.ParentLink.IsEraseing = True      
    #Mouse Move event        
    def mouseMoveEvent(self, event):
        if(self.ParentLink.IsPainting == True):
            self.MouseLoc = Point(event.x(),event.y())
            if((self.LastPos.X != self.MouseLoc.X) and (self.LastPos.Y != self.MouseLoc.Y)):
                self.LastPos =  Point(event.x(),event.y())
                self.ParentLink.DrawingShapes.NewShape(self.LastPos,self.ParentLink.CurrentWidth,self.ParentLink.CurrentColour,self.ParentLink.ShapeNum)
            self.repaint()
        if(self.ParentLink.IsEraseing == True):
            self.MouseLoc = Point(event.x(),event.y())
            self.ParentLink.DrawingShapes.RemoveShape(self.MouseLoc,10)     
            self.repaint()        
                
    #Mose Up Event         
    def mouseReleaseEvent(self, event):
        if(self.ParentLink.IsPainting == True):
            self.ParentLink.IsPainting = False
        if(self.ParentLink.IsEraseing == True):
            self.ParentLink.IsEraseing = False  
    
    def paintEvent(self,event):
        painter = QPainter()
        
        # 이미지를 저장하기위해서 v_paint 를 만들어줌                
        v_painter = QPainter(self.ParentLink.Vimage)
        
        painter.begin(self)
        self.drawLines(event, painter,v_painter)
        #painter.drawImage(self.rect(), self.image, self.image.rect()) #Qimage
        painter.end()

        
    def drawLines(self, event, painter,v_painter):
        painter.setRenderHint(QPainter.Antialiasing);
        
        for i in range(self.ParentLink.DrawingShapes.NumberOfShapes()-1):
            
            T = self.ParentLink.DrawingShapes.GetShape(i)
            T1 = self.ParentLink.DrawingShapes.GetShape(i+1)
        
            if(T.ShapeNumber == T1.ShapeNumber):
                pen = QPen(QColor(T.Colour.R,T.Colour.G,T.Colour.B), T.Width/2, Qt.SolidLine, Qt.RoundCap)
                painter.setPen(pen)
                painter.drawLine(T.Location.X,T.Location.Y,T1.Location.X,T1.Location.Y)
                # 이미지를 저장하기위해서 v_paint 에도 같이 그려준다
                v_painter.setPen(pen)
                v_painter.drawLine(T.Location.X,T.Location.Y,T1.Location.X,T1.Location.Y)
                
                
###############################################################################
# MyWindow 정의 
###############################################################################
class MyWindow(QDialog, form_class):
    Combo = 'Sklearn'
    Combo2 = 'mnist'
    Brush = True
    DrawingShapes = Shapes()
    IsPainting = False
    IsEraseing = False

    CurrentColour = Colour3(0,0,0)
    CurrentWidth = 60
    ShapeNum = 0
    IsMouseing = False
    PaintPanel = 0   
    traing_model = ""        
    
    Vimage = QImage(QSize(400, 400), QImage.Format_RGB32)
    #Vimage.fill(Qt.white)
    
    
    # 학습모델 라이브러리 클래스 
    lib_sklearn = Class_Sklearn() 
    lib_keras   = Class_Keras() 
    lib_tensor  = Class_Tensorflow() 
    
    # 초기 설정해주는 init
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # 시그널(이벤트루프에서 발생할 이벤트) 선언
        # self.객체명.객체함수.connect(self.슬롯명)

        self.setupUi(self)
        self.setWindowTitle('MNIST Classifier')
        
        # QStackedWidget 0 번에는 손글씨용 PaintPanel 을 만듬
        self.PaintPanel = Painter(self)
        self.PaintPanel.close()
        self.DrawingFrame.insertWidget(0,self.PaintPanel)
        self.DrawingFrame.setCurrentWidget(self.PaintPanel)   # 0번활성화

        # QStackedWidget 1 번에는 이미지용  QLabel 을 만듬
        self.Label = QLabel()
        self.DrawingFrame.insertWidget(1,self.Label)
        
        self.radioButton1.setChecked(True) 
        self.traing_model = "RFC"
        
        self.Vimage.fill(Qt.white)
        
        # 미사용
        #self.widget = matplotlibwidget(self.widget)



        #self.statusBar = QStatusBar(self)
        #self.setStatusBar(self.statusBar)
        
        #self.cb = QComboBox(self)
        self.comboBox.addItem('Sklearn')
        self.comboBox.addItem('Keras')
        self.comboBox.addItem('TensorFlow')
        #self.comboBox.addItem('Coding')

        self.comboBox2.addItem('mnist')
        self.comboBox2.addItem('csv')
        self.comboBox2.addItem('unicsv_28') # Ztrain_han_28X28.csv
        self.comboBox2.addItem('unicsv_46') # Ztrain_han_46X46.csv
        self.comboBox2.addItem('unicsv_64') # Ztrain_han_64X64.csv
        self.comboBox2.addItem('unicsv_128') # Ztrain_han_128X128.csv
        self.comboBox2.addItem('unicode')   # Xtrain_han.csv , ytrain_han.csv
        
        
        # 슬롯 연결
        self.Establish_Connections()       
        

    # 학습할 모델 선택
    def radioButtonClicked(self):        
        if self.radioButton1.isChecked():
            self.traing_model = "RFC"
        elif self.radioButton2.isChecked():
            self.traing_model = "RFC-MINI"
        elif self.radioButton3.isChecked():
            self.traing_model = "NAIV"
        elif self.radioButton4.isChecked():
            self.traing_model = "GBC"
        elif self.radioButton5.isChecked():
            self.traing_model = "SGD"
        elif self.radioButton6.isChecked():
            self.traing_model = "SGD-MINI"
        elif self.radioButton7.isChecked():
            self.traing_model = "LBF"
        elif self.radioButton8.isChecked():
            self.traing_model = "SVM"
        else:
            self.traing_model = ""
        print("Select Model=:",self.traing_model)
        #self.statusBar.showMessage(msg + "선택 됨")
      
    # 시그널을 처리할 슬롯
    def btn_click(self):
        self.textEdit.setText("hello world!")
        
    def SwitchBrush(self):
        if(self.Brush == True):
            self.Brush = False
        else:
            self.Brush = True
    
    def ChangeColour(self):
        col = QColorDialog.getColor()
        if col.isValid():
            self.CurrentColour = Colour3(col.red(),col.green(),col.blue())
   
    def ChangeThickness(self,num):
        self.CurrentWidth = num
            
    def ClearSlate(self):
        # 초기화하자
        self.DrawingShapes = Shapes()
        self.PaintPanel.repaint()  
        self.Label.clear() 
        self.Label2.clear()
        self.textEdit.setText("")
        
        
        # QStackedWidget 0번페이지를 활성화한다
        self.DrawingFrame.setCurrentWidget(self.PaintPanel) 
    
    # 이미지를 저장한다
    def SaveImage(self):
        self.Vimage.save('image.png')
        self.Vimage.fill(Qt.white)
 
    # 이미지를 반전시킨다
    def InvertImage(self):
        self.org_image = QPixmap()
        self.org_image = self.Label.pixmap()
        
        # 이미지 반전
        inv_image = self.org_image.toImage()
        inv_image.invertPixels()
        invert_pix = QPixmap.fromImage(inv_image)
        
        invert_pix = invert_pix.scaled(400,400) 
        self.Label.setPixmap(invert_pix)
        

    # 정확도 그래프를 불러온다
    def LoadAccuracy(self):        # 정확도를 그래프로 보여주자
        if self.Combo == 'Sklearn':
            self.pixmap = QPixmap('accuracy.png')
            self.pixmap = self.pixmap.scaled(140,140)
            self.DrawingFrame.setCurrentWidget(self.Label2)  
            self.Label2.setPixmap(self.pixmap)

            self.show()    

    # 이미지를 불러온다
    def LoadImage(self):
        
        # 일단 초기화부터 하자
        self.ClearSlate()
        
        selection = QMessageBox.question(self,'확인',"파일을 로드 하시겠습니까?",QMessageBox.Yes,QMessageBox.No)
        
        if selection == QMessageBox.Yes: # 파일로드
            try:
                fname = QFileDialog.getOpenFileName(self)
                print("fname=",fname)
                
            except Exception as e:
                self._stateexpression.setText(e)
                QMessageBox.about(self,"경고",e)
                
            else:
                # 보여주기
                self.pixmap = QPixmap(fname[0])
                self.pixmap = self.pixmap.scaled(400,400)
                self.DrawingFrame.setCurrentWidget(self.Label)  
                self.Label.setPixmap(self.pixmap)
                
               
                self.show()  
               
                
                print ("image...print")
    
    # 이미지를 예측함
    def PredictImage(self):
        self.pixmap.save("image.png")
        
        # 예측
        self.Predict()
        
    # 손글씨예측함
    def PredictHandWrite(self):
        self.Vimage.save('image.png')
        self.Vimage.fill(Qt.white)
        self.textEdit.clear() 
        
        # 예측        
        self.Predict()
        

    def Train(self): 
        print("Combo=",self.Combo)
        print("Combo2=",self.Combo2)
       
        if self.Combo == 'Sklearn':
            print ('Sklearn Train start..')
            pred=self.lib_sklearn.train_model(model=self.traing_model,dataset= self.Combo2)
            print ('Sklearn Train End..')  
        elif self.Combo == 'Keras':
            print ('Keras Train start..')
            pred=self.lib_keras.train_model(models=self.traing_model,dataset= self.Combo2)
            print ('Keras Train End..')
        elif self.Combo == 'TensorFlow':
            print ('Tensor Train start..')
            pred=self.lib_tensor.train_model(models=self.traing_model,dataset= self.Combo2)
            print ('Tensor Train End..')            
        else :
            pass
       
        # 정확도 이미지를 불러온다
        self.LoadAccuracy()
     
        
    def Predict(self):   
        
        print("Predict")        
        print("Combo=",self.Combo)        
        print("Combo2=",self.Combo2)
       
        if self.Combo == 'Sklearn':
            pred=self.lib_sklearn.testing(model=self.traing_model,dataset= self.Combo2)
        elif self.Combo == 'Keras':
            print ('Keras Predict start..')
            pred=self.lib_keras.testing(models=self.traing_model,dataset= self.Combo2)   
        elif self.Combo == 'TensorFlow':
            print ('Tensor Predict start..')
            pred=self.lib_tensor.testing(models=self.traing_model)              
        else :
            pass
        
        print(pred)
        self.textEdit.setText("Predict : {}".format(pred))  
        
        #if self.Combo2 =='unicode':
        self.Label2.setText(pred)                # 텍스트
        self.Label2.setFont(QFont("궁서",60))    # 폰트/크기조절     
        
    # 라이브러리 선택 콤보        
    def ComboOnActivated(self, text):
        self.Combo = text
        print(self.Combo)
        
        # SKLearn 일때만 라디오버튼 활성화 
        if self.Combo == 'Sklearn':
            self.radioButton1.setEnabled(True)
            self.radioButton2.setEnabled(True)
            self.radioButton3.setEnabled(True)
            self.radioButton4.setEnabled(True)
            self.radioButton5.setEnabled(True)
            self.radioButton6.setEnabled(True)
            self.radioButton7.setEnabled(True)
            self.radioButton8.setEnabled(True)
        else :
            self.radioButton1.setEnabled(False)
            self.radioButton2.setEnabled(False)
            self.radioButton3.setEnabled(False)
            self.radioButton4.setEnabled(False)
            self.radioButton5.setEnabled(False)
            self.radioButton6.setEnabled(False)
            self.radioButton7.setEnabled(False)
            self.radioButton8.setEnabled(False)
            
    
    # Dataset 종류선택 콤보
    def ComboOnActivated2(self, text):
        self.Combo2 = text
        print(self.Combo)

    def Establish_Connections(self):
        # combo box
        self.comboBox.activated[str].connect(self.ComboOnActivated)
        self.comboBox2.activated[str].connect(self.ComboOnActivated2)
        
        # push button
        self.pushButton1.clicked.connect(self.Train)
        self.pushButton2.clicked.connect(self.PredictHandWrite)      # Predict 
        self.pushButton3.clicked.connect(self.ClearSlate)
        self.pushButton4.clicked.connect(self.SaveImage)
        self.pushButton5.clicked.connect(self.LoadImage)
        self.pushButton6.clicked.connect(self.PredictImage)
        self.pushButton7.clicked.connect(self.InvertImage)
        # radio button 
        self.radioButton1.clicked.connect(self.radioButtonClicked)
        self.radioButton2.clicked.connect(self.radioButtonClicked)
        self.radioButton3.clicked.connect(self.radioButtonClicked)
        self.radioButton4.clicked.connect(self.radioButtonClicked)
        self.radioButton5.clicked.connect(self.radioButtonClicked)
        self.radioButton6.clicked.connect(self.radioButtonClicked)
        self.radioButton7.clicked.connect(self.radioButtonClicked)
        self.radioButton8.clicked.connect(self.radioButtonClicked)
        #QtCore.QObject.connect(self.BrushErase_Button, QtCore.SIGNAL("clicked()"),self.SwitchBrush)
        #QtCore.QObject.connect(self.ChangeColour_Button, QtCore.SIGNAL("clicked()"),self.ChangeColour)
        #QtCore.QObject.connect(self.Clear_Button, QtCore.SIGNAL("clicked()"),self.ClearSlate)
        #QtCore.QObject.connect(self.Thickness_Spinner, QtCore.SIGNAL("valueChanged(int)"),self.ChangeThickness)


###############################################################################
# QApplication 윈도우 띄우기
###############################################################################  
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # myWindow 라는 변수에 GUI 클래스 삽입
    myWindow = MyWindow()
    # GUI 창 보이기
    myWindow.show()
       
    #########################
    # 이벤트루프 진입전 작업할 부분
    ######################### 
    # 이벤트루프 진입
    app.exec_()
 