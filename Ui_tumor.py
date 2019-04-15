from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon,QFont
class Ui_tumor_seg(object):
    def setupUi(self, tumor_seg):
        tumor_seg.setObjectName("tumor_seg")
        tumor_seg.resize(713, 571)
        
        #tumor_seg.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        tumor_seg.setFixedSize(tumor_seg.width(), tumor_seg.height())
        
        self.centralWidget = QtWidgets.QWidget(tumor_seg)
        self.centralWidget.setObjectName("centralWidget")
        self.centralWidget.setWindowIcon(QIcon('../liver_icon.png'))
        
        self.show_area = QtWidgets.QLabel(self.centralWidget)
        self.show_area.setGeometry(QtCore.QRect(20, 20, 512, 512))
        self.show_area.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.show_area.setText("")
        self.show_area.setObjectName("show_area")
        
        self.choose_raw = QtWidgets.QLineEdit(self.centralWidget)
        self.choose_raw.setGeometry(QtCore.QRect(550, 30, 141, 21))
        self.choose_raw.setObjectName("choose_raw")
        
        self.state_show_label1 = QtWidgets.QLabel(self.centralWidget)
        self.state_show_label1.setGeometry(QtCore.QRect(560, 430, 131, 16))
        self.state_show_label1.setObjectName("state_show_label1")
        
        self.num = QtWidgets.QLineEdit(self.centralWidget)
        self.num.setGeometry(QtCore.QRect(190, 540, 71, 20))
        self.num.setObjectName("num")

 
        self.show_1 = QtWidgets.QPushButton(self.centralWidget)
        self.show_1.setGeometry(QtCore.QRect(270, 540, 75, 23))
        self.show_1.setObjectName("show_1")

        self.left = QtWidgets.QPushButton(self.centralWidget)
        self.left.setGeometry(QtCore.QRect(20, 540, 75, 23))
        self.left.setObjectName("left")
        
        self.right_R = QtWidgets.QPushButton(self.centralWidget)
        self.right_R.setGeometry(QtCore.QRect(460, 540, 71, 23))
        self.right_R.setObjectName("right_R")
        
        self.seg_run = QtWidgets.QPushButton(self.centralWidget)
        self.seg_run.setGeometry(QtCore.QRect(590, 150, 61, 23))
        self.seg_run.setObjectName("seg_run")
        
        self.line = QtWidgets.QFrame(self.centralWidget)
        self.line.setGeometry(QtCore.QRect(598, 82, 90, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        
        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setGeometry(QtCore.QRect(568, 82, 41, 16))
        
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        
        self.label.setFont(font)
        self.label.setScaledContents(False)
        self.label.setObjectName("label")
        self.line_2 = QtWidgets.QFrame(self.centralWidget)
        self.line_2.setGeometry(QtCore.QRect(550, 172, 140, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self.centralWidget)
        self.line_3.setGeometry(QtCore.QRect(683, 89, 16, 91))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_4 = QtWidgets.QFrame(self.centralWidget)
        self.line_4.setGeometry(QtCore.QRect(543, 88, 16, 91))
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(self.centralWidget)
        self.line_5.setGeometry(QtCore.QRect(550, 80, 16, 20))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        
        
        self.label_3 = QtWidgets.QLabel(self.centralWidget)
        self.label_3.setGeometry(QtCore.QRect(568, 212, 85, 16))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        
        
        self.label_3.setFont(font)
        self.label_3.setScaledContents(False)
        self.label_3.setObjectName("label_3")
       # self.label_3.setFont(QFont("Roman times",7,QFont.Bold))
        
        
        
        
        self.line_6 = QtWidgets.QFrame(self.centralWidget)
        self.line_6.setGeometry(QtCore.QRect(683, 219, 16, 71))
        self.line_6.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.line_7 = QtWidgets.QFrame(self.centralWidget)
        self.line_7.setGeometry(QtCore.QRect(543, 218, 16, 71))
        self.line_7.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.line_8 = QtWidgets.QFrame(self.centralWidget)
        self.line_8.setGeometry(QtCore.QRect(550, 282, 140, 16))
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.line_11 = QtWidgets.QFrame(self.centralWidget)
        self.line_11.setGeometry(QtCore.QRect(543, 416, 16, 61))
        self.line_11.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.label_4 = QtWidgets.QLabel(self.centralWidget)
        self.label_4.setGeometry(QtCore.QRect(568, 407, 85, 20))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setScaledContents(False)
        self.label_4.setObjectName("label_4")
        self.line_12 = QtWidgets.QFrame(self.centralWidget)
        self.line_12.setGeometry(QtCore.QRect(683, 417, 16, 61))
        self.line_12.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_12.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_12.setObjectName("line_12")
        self.tri_d_run = QtWidgets.QPushButton(self.centralWidget)
        self.tri_d_run.setGeometry(QtCore.QRect(590, 345, 61, 23))
        self.tri_d_run.setObjectName("tri_d_run")
        self.line_13 = QtWidgets.QFrame(self.centralWidget)
        self.line_13.setGeometry(QtCore.QRect(550, 470, 140, 16))
        self.line_13.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_13.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_13.setObjectName("line_13")
        self.line_14 = QtWidgets.QFrame(self.centralWidget)
        self.line_14.setGeometry(QtCore.QRect(550, 408, 16, 20))
        self.line_14.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_14.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_14.setObjectName("line_14")
        self.line_15 = QtWidgets.QFrame(self.centralWidget)
        self.line_15.setGeometry(QtCore.QRect(658, 410,30, 16))
        self.line_15.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_15.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_15.setObjectName("line_15")
        
        
        
        self.line_10 = QtWidgets.QFrame(self.centralWidget)
        self.line_10.setGeometry(QtCore.QRect(658, 212, 30, 16))
        self.line_10.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        
        
        
        
        self.line_10.setObjectName("line_10")
        self.line_16 = QtWidgets.QFrame(self.centralWidget)
        self.line_16.setGeometry(QtCore.QRect(550, 210, 16, 20))
        self.line_16.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_16.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_16.setObjectName("line_16")
        self.line_17 = QtWidgets.QFrame(self.centralWidget)
        self.line_17.setGeometry(QtCore.QRect(543, 328, 16, 51))
        self.line_17.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_17.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_17.setObjectName("line_17")
        self.line_18 = QtWidgets.QFrame(self.centralWidget)
        self.line_18.setGeometry(QtCore.QRect(683, 329, 16, 51))
        self.line_18.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_18.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_18.setObjectName("line_18")
        self.label_5 = QtWidgets.QLabel(self.centralWidget)
        self.label_5.setGeometry(QtCore.QRect(568, 320, 85, 20))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setScaledContents(False)
        self.label_5.setObjectName("label_5")
        self.line_19 = QtWidgets.QFrame(self.centralWidget)
        self.line_19.setGeometry(QtCore.QRect(658, 322, 30, 16))
        self.line_19.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_19.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_19.setObjectName("line_19")
        self.line_20 = QtWidgets.QFrame(self.centralWidget)
        self.line_20.setGeometry(QtCore.QRect(550, 372, 140, 16))
        self.line_20.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_20.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_20.setObjectName("line_20")
        self.line_21 = QtWidgets.QFrame(self.centralWidget)
        self.line_21.setGeometry(QtCore.QRect(550, 320, 16, 20))
        self.line_21.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_21.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_21.setObjectName("line_21")
        self.state_show_label2 = QtWidgets.QLabel(self.centralWidget)
        self.state_show_label2.setGeometry(QtCore.QRect(560, 450, 131, 16))
        self.state_show_label2.setText("")
        self.state_show_label2.setObjectName("state_show_label2")
        self.widget = QtWidgets.QWidget(self.centralWidget)
        self.widget.setGeometry(QtCore.QRect(568, 237, 110, 40))
        self.widget.setObjectName("widget")
        
 
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        
        
        self.liver_layer = QtWidgets.QCheckBox(self.widget)
        self.liver_layer.setBaseSize(QtCore.QSize(1, 1))
        self.liver_layer.setObjectName("liver_layer")
        
        self.verticalLayout.addWidget(self.liver_layer)
        
        self.tumor_layer = QtWidgets.QCheckBox(self.widget)
        self.tumor_layer.setTristate(False)
        self.tumor_layer.setObjectName("tumor_layer")
        
        
        self.verticalLayout.addWidget(self.tumor_layer)
        
        
        
        
        
        
        
        
        self.widget1 = QtWidgets.QWidget(self.centralWidget)
        self.widget1.setGeometry(QtCore.QRect(568, 110, 110, 40))
        self.widget1.setObjectName("widget1")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.liver_mark = QtWidgets.QCheckBox(self.widget1)
        self.liver_mark.setBaseSize(QtCore.QSize(1, 1))
        self.liver_mark.setObjectName("liver_mark")
        self.verticalLayout_2.addWidget(self.liver_mark)
        self.tumor_mark = QtWidgets.QCheckBox(self.widget1)
        self.tumor_mark.setTristate(False)
        self.tumor_mark.setObjectName("tumor_mark")
        
        self.verticalLayout_2.addWidget(self.tumor_mark)
        
        tumor_seg.setCentralWidget(self.centralWidget)

        self.retranslateUi(tumor_seg)
        QtCore.QMetaObject.connectSlotsByName(tumor_seg)
        
        self.liver_layer.setEnabled(False)
        self.tumor_layer.setEnabled(False)
        
        

    def retranslateUi(self, tumor_seg):
        _translate = QtCore.QCoreApplication.translate
        tumor_seg.setWindowTitle(_translate("tumor_seg", "LATS_0.2"))
        self.choose_raw.setText(_translate("tumor_seg", "选择目录"))
        self.state_show_label1.setText(_translate("tumor_seg", "无任务"))
        self.show_1.setText(_translate("tumor_seg", "显示"))
        self.left.setText(_translate("tumor_seg", "上一张"))
        self.right_R.setText(_translate("tumor_seg", "下一张"))
        self.seg_run.setText(_translate("tumor_seg", "执行"))
        self.label.setText(_translate("tumor_seg", "标记"))
        self.label_3.setText(_translate("tumor_seg", "图层显示"))
        self.label_4.setText(_translate("tumor_seg", "状态显示"))
        self.tri_d_run.setText(_translate("tumor_seg", "执行"))
        self.label_5.setText(_translate("tumor_seg", "三维重建"))
        
        
        self.tumor_layer.setText(_translate("tumor_seg", "肿瘤图层"))
        self.liver_layer.setText(_translate("tumor_seg", "肝脏图层"))
        
        
        self.liver_mark.setText(_translate("tumor_seg", "肝脏标记"))
        self.tumor_mark.setText(_translate("tumor_seg", "肿瘤标记"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    tumor_seg = QtWidgets.QMainWindow()
    ui = Ui_tumor_seg()
    ui.setupUi(tumor_seg)
    tumor_seg.show()
    sys.exit(app.exec_())

