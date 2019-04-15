from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QFileDialog
import cv2 as cv
import os
from Ui_tumor import Ui_tumor_seg
import imageio
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import time
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from unet import *
from WL import *
import pydicom


class Worker(QThread):
    sinOut = pyqtSignal(str, str)  # 自定义信号，执行run()函数时，从相关线程发射此信号
    sinOut2 = pyqtSignal(bytes)
    sinOut3 = pyqtSignal(bytes, bytes)
    sinOut4 = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super(Worker, self).__init__(parent)
        self.working = True
        self.num = 0

    def __del__(self):
        self.working = False
        self.wait()

    def run(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")            
     
        if torch.cuda.is_available():
            gpu_flag = '(on GPU)'
        else:
            gpu_flag = '(on CPU)'
        t_start = time.time()
 
        global t_elapsed
        if flag == 1:
            print('flag = 1')
            
            filenames = os.listdir(path)


            results_liver = np.zeros([len(filenames), 512, 512])
            results_tumor = np.zeros([len(filenames), 512, 512])

            slices_liver = []
            slices_tumor = []
            idx = []
            for i, name in enumerate(filenames):
                name = os.path.join(path, name)
                slice = pydicom.dcmread(name)
                idx.append(int(slice.InstanceNumber))
                slices_liver.append(WL(slice, 0, 2048))
                slices_tumor.append(WL(slice, 100, 150))

            idx_new = np.argsort(idx)

            slices_liver = np.stack(slices_liver)
            slices_tumor = np.stack(slices_tumor)

            slices_liver = slices_liver[idx_new]
            slices_tumor = slices_tumor[idx_new]

            slices_liver_tensor = torch.tensor(slices_liver)
            slices_liver_tensor = slices_liver_tensor.unsqueeze(1).float() / 255.

            slices_tumor_tensor = torch.tensor(slices_tumor)
            slices_tumor_tensor = slices_tumor_tensor.unsqueeze(1).float() / 255.
            
            
            model_path = 'liver_7WL.pth'
            model = torch.load(model_path, map_location=device)
            model = model.to(device)
            model = model.eval()
            sm = nn.Softmax(dim=1)
            
            for i in range(slices_liver_tensor.shape[0]):

                self.sinOut.emit("标记肝脏: " ,  str(i+1)+"/" +
                                 str(slices_liver_tensor.shape[0]) + gpu_flag)
                                 
                output = model(slices_liver_tensor[i, :].unsqueeze(0).to(device))
                output_sm = sm(output)
                _, result = torch.max(output_sm, dim=1)
                results_liver[i] = result[0, :].cpu().detach().numpy()
                
            print(results_liver.shape)
            
            a = results_liver.tostring()
 
            b = results_tumor.tostring()
            
            t_end = time.time()
            global t_elapsed
            t_elapsed = t_end - t_start         
            
            t_elapsed = t_end - t_start
            
            
            #print(str(round(t_elapsed, 4)))
            
            self.sinOut4.emit("耗时: " ,  str(round(t_elapsed, 4)))
            #self.sinOut.emit("耗时: " ,  's')
            
            self.sinOut3.emit(a, b)

            
           
        elif flag == 3 or flag == 2:

            filenames = os.listdir(path)


            results_liver = np.zeros([len(filenames), 512, 512])
            results_tumor = np.zeros([len(filenames), 512, 512])

            slices_liver = []
            slices_tumor = []
            idx = []
            for i, name in enumerate(filenames):
                name = os.path.join(path, name)
                slice = pydicom.dcmread(name)
                idx.append(int(slice.InstanceNumber))
                slices_liver.append(WL(slice, 0, 2048))
                slices_tumor.append(WL(slice, 100, 150))

            idx_new = np.argsort(idx)

            slices_liver = np.stack(slices_liver)
            slices_tumor = np.stack(slices_tumor)

            slices_liver = slices_liver[idx_new]
            slices_tumor = slices_tumor[idx_new]

            slices_liver_tensor = torch.tensor(slices_liver)
            slices_liver_tensor = slices_liver_tensor.unsqueeze(1).float() / 255.

            slices_tumor_tensor = torch.tensor(slices_tumor)
            slices_tumor_tensor = slices_tumor_tensor.unsqueeze(1).float() / 255.
            
            
            model_path = 'liver_7WL.pth'
            model = torch.load(model_path, map_location=device)
            model = model.to(device)
            model = model.eval()
            sm = nn.Softmax(dim=1)
            
            for i in range(slices_liver_tensor.shape[0]):

                self.sinOut.emit("标记肝脏: " ,  str(i+1)+"/" +
                                 str(slices_liver_tensor.shape[0]) + gpu_flag)
                                 
                output = model(slices_liver_tensor[i, :].unsqueeze(0).to(device))
                output_sm = sm(output)
                _, result = torch.max(output_sm, dim=1)
                results_liver[i] = result[0, :].cpu().detach().numpy()
            a = results_liver.tostring()


            del(model)
            del(output)
            del(output_sm)
            del(result)

            model_path_2 = './best_tumor.pth'
            model_2 = torch.load(model_path_2, map_location=device)
            model_2 = model_2.to(device)
            model_2 = model_2.eval()
            sm = nn.Softmax(dim=1)
            for i in range(slices_tumor_tensor.shape[0]):
          
                self.sinOut.emit("标记肿瘤: " ,  str(i+1)+"/" +
                                 str(slices_tumor_tensor.shape[0]) + gpu_flag)

                output_2 = model_2(slices_tumor_tensor[i, :].unsqueeze(0).to(device))
                output_sm_2 = sm(output_2)
                _, result_2 = torch.max(output_sm_2, dim=1)
                results_tumor[i] = result_2[0, :].cpu().detach().numpy()
            b = results_tumor.tostring()
            t_end = time.time()
      
            t_elapsed = t_end - t_start
         
            self.sinOut4.emit("耗时: " ,  str(round(t_elapsed, 4)))
        
            self.sinOut3.emit(a, b)
#sinOut3显示图用
            

class MainWindow(QMainWindow, Ui_tumor_seg):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.choose_raw.mousePressEvent = self.lineedit_clicked

    def show_img(self, image):
        img = Image.fromarray(image.astype('uint8')).convert('RGB')
        self.img_rgb = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
        self.QtImg = QtGui.QImage(
            self.img_rgb.data, self.img_rgb.shape[1], self.img_rgb.shape[0], QtGui.QImage.Format_RGB888)
        self.show_area.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

    def slotAdd3(self, a, b):
     


        filenames = os.listdir(path)

        self.slices_liver = []
        self.slices_tumor = []

        idx = []
        for i, name in enumerate(filenames):
            name = os.path.join(path, name)
            slice = pydicom.dcmread(name)
            idx.append(slice.InstanceNumber)
            self.slices_liver.append(WL(slice, 0, 2048))
            self.slices_tumor.append(WL(slice, 100, 150))

        idx_new = np.argsort(idx)
        
        self.slices_liver = np.stack(self.slices_liver)[idx_new]
        self.slices_tumor = np.stack(self.slices_tumor)[idx_new]



        self.liver = np.fromstring(a)
        self.liver = np.reshape(self.liver, (len(filenames), 512, 512))
        self.liver_backup = self.liver
        self.liver *= 255.

        self.tumor = np.fromstring(b)
        self.tumor = np.reshape(self.tumor, (len(filenames), 512, 512))
        self.tumor_backup = self.tumor
        self.tumor *= 255.


        global sum_number
        sum_number = filenames
        half_sample_num = sample_num/2
        self.num.setText(str(int(half_sample_num)))

        num_pic = self.num.text()
        num2 = int(num_pic)
        
        self.show_img(self.tumor[num2])
        
        a = self.liver[num2]#liver
        b = self.slices_tumor[num2]
        c = self.tumor[num2]
        overlay = b
      
        overlay = np.uint8(overlay)
        overlay = cv.cvtColor(overlay, cv.COLOR_GRAY2RGB)
        mask = np.uint8(c)
       
        _, binary_pred = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
        contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(overlay, contours_pred, -1, (255, 20, 147), 2) # pink contours stand for prediction
  
        mask_2 = np.uint8(a)
    
        _, binary_pred = cv.threshold(mask_2, 127, 255, cv.THRESH_BINARY)
        contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(overlay, contours_pred, -1, (0, 255, 0), 2) # pink contours stand for prediction  
        self.show_img(overlay)
        
        if flag == 1:
            self.liver_layer.setEnabled(True)
            self.liver_layer.setChecked(True)
       
        elif flag == 3 or flag == 2:
            self.liver_layer.setEnabled(True)
            self.tumor_layer.setEnabled(True)
            
            self.liver_layer.setChecked(True)
            self.tumor_layer.setChecked(True)
        
    def slotAdd2(self, file_inf):
        self.state_show_label1.setText(
            "Done. Time elapsed: " + str(format(t_elapsed, '0.2f') + 's.'))
        self.state_show_label2.setText( str(format(t_elapsed, '0.2f') + 's.')) 
        
        filen = os.listdir(path)
        self.results = np.fromstring(file_inf)
        self.results = np.reshape(self.results, (len(filen), 512, 512))
        self.results_backup = self.results
        self.results *= 255.
        half_sample_num = sample_num/2
        self.num.setText(str(int(half_sample_num)))
        num_pic = self.num.text()
        num2 = int(num_pic)
        array = self.results[num2]
        self.show_img( array)

    def slotAdd(self, file_inf, file_inf2):
        self.state_show_label1.setText(file_inf)
        self.state_show_label2.setText(file_inf2 )
    def slotAdd4(self, file_inf, file_inf2):
        self.state_show_label1.setText(file_inf)
        self.state_show_label2.setText(file_inf2 + '秒')

    def lineedit_clicked(self, e):
        

        self.liver_layer.setEnabled(True)
        self.tumor_layer.setEnabled(True)
        self.liver_layer.setChecked(False)
        self.tumor_layer.setChecked(False)
     
        
        self.choose_raw.setText('')
        my_file_path = QFileDialog.getExistingDirectory(self, "选择文件夹", ".")
        self.choose_raw.setText(my_file_path)

    def on_tumor_mark_clicked(self):
        self.liver_mark.setChecked(True)

    @pyqtSlot()
    def on_seg_run_clicked(self):
        test = self.choose_raw.text()
        if test == '' or test == '选择目录':
            QMessageBox.information(self, "Warning", "Plese choose dir" )
        else:
            global flag

            if self.liver_mark.isChecked() and self.tumor_mark.isChecked():
                flag = 3
            elif self.liver_mark.isChecked():
                self.tumor_layer.setEnabled(False)
                flag = 1
            elif self.tumor_mark.isChecked():
                self.liver_mark.setChecked(True)
                flag = 2
                self.liver_mark.setChecked(True)
            else:
                QMessageBox.information(self, "Warning", "Please choose .")
            control_path = self.choose_raw.text()
            global path
            path = control_path
    
      
    
            filenames = os.listdir(path)
            global sample_num
            sample_num = len(filenames)
    
    
            self.thread = Worker()
            self.thread.sinOut.connect(self.slotAdd)
            self.thread.sinOut2.connect(self.slotAdd2)
            self.thread.sinOut3.connect(self.slotAdd3)
            self.thread.sinOut4.connect(self.slotAdd4)
    
            self.thread.start()
            self.state_show_label1.setText("模型初始化中…")

    @pyqtSlot()
    def on_left_clicked(self):
        num_pic = self.num.text()
        new_num_pic = int(num_pic)-1
        new_num_str = str(new_num_pic)

        if new_num_pic>0:   
            self.num.setText(new_num_str)
            new2_num_pic = int(self.num.text())
            num2 = new2_num_pic
            b1 = self.slices_tumor[num2]
            b2 = self.slices_tumor[num2]
            b3 = self.slices_tumor[num2]
            a = self.liver[num2]
       
            c = self.tumor[num2]
            if self.liver_layer.isChecked() and self.tumor_layer.isChecked():
                b1 = np.uint8(b1)
                b1 = cv.cvtColor(b1, cv.COLOR_GRAY2RGB)
                mask = np.uint8(c)
          
                _, binary_pred = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
                contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(b1, contours_pred, -1, (255, 20, 147), 2) 
          
                mask_2 = np.uint8(a)
           
                _, binary_pred = cv.threshold(mask_2, 127, 255, cv.THRESH_BINARY)
                contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(b1, contours_pred, -1, (0, 255, 0), 2)      
                self.show_img(b1)
            elif self.liver_layer.isChecked():
                b2 = np.uint8(b2)
                b2 = cv.cvtColor(b2, cv.COLOR_GRAY2RGB)
                mask = np.uint8(a)
         
                _, binary_pred = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
                contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(b2, contours_pred, -1, (0, 255, 0), 2) 
                self.show_img(b2)
            elif self.tumor_layer.isChecked():
                b3 = np.uint8(b3)
                b3 = cv.cvtColor(b3, cv.COLOR_GRAY2RGB)
                mask = np.uint8(c)
       
                _, binary_pred = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
                contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(b3, contours_pred, -1, (255, 20, 147), 2) 
    
                self.show_img(b3)
            else:            
                self.show_img(self.slices_tumor[num2])
   
 
        else:
      
            QMessageBox.information(self, "Warning", "超出索引范围" )
        
        
    @pyqtSlot()
    def on_right_R_clicked(self):
        num_pic = self.num.text()
        new_num_pic = int(num_pic)+1
        new_num_str = str(new_num_pic)
        if  new_num_pic+1>sample_num:
     
            QMessageBox.information(self, "Warning", "超出索引范围" )
        else:

            
            self.num.setText(new_num_str)
            new2_num_pic = int(self.num.text())
            num2 = new2_num_pic
            b1 = self.slices_tumor[num2]
            b2 = self.slices_tumor[num2]
            b3 = self.slices_tumor[num2]
            a = self.liver[num2]
         
            c = self.tumor[num2]
            if self.liver_layer.isChecked() and self.tumor_layer.isChecked():
                b1 = np.uint8(b1)
                b1 = cv.cvtColor(b1, cv.COLOR_GRAY2RGB)
                mask = np.uint8(c)
                _, binary_pred = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
                contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(b1, contours_pred, -1, (255, 20, 147), 2) # pink contours stand for prediction
                mask_2 = np.uint8(a)
           
                _, binary_pred = cv.threshold(mask_2, 127, 255, cv.THRESH_BINARY)
                contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(b1, contours_pred, -1, (0, 255, 0), 2) # pink contours stand for prediction  
            
                self.show_img(b1)
            elif self.liver_layer.isChecked():
                b2 = np.uint8(b2)
                b2 = cv.cvtColor(b2, cv.COLOR_GRAY2RGB)
                mask = np.uint8(a)
               
                _, binary_pred = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
                contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(b2, contours_pred, -1, (0, 255, 0), 2) # pink contours stand for prediction
                self.show_img(b2)
            elif self.tumor_layer.isChecked():
                b3 = np.uint8(b3)
                b3 = cv.cvtColor(b3, cv.COLOR_GRAY2RGB)
                mask = np.uint8(c)
              
                _, binary_pred = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
                contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(b3, contours_pred, -1, (255, 20, 147), 2) # pink contours stand for prediction
                self.show_img(b3)
            else:

                self.show_img(self.slices_tumor[num2])

    @pyqtSlot()
    def on_show_1_clicked(self):
     
        num_pic = self.num.text()
        new_num_pic = int(num_pic)

        if new_num_pic<sample_num and new_num_pic>0 :
            num2 = new_num_pic
            b1 = self.slices_tumor[num2]
            b2 = self.slices_tumor[num2]
            b3 = self.slices_tumor[num2]
            a = self.liver[num2]
    
            c = self.tumor[num2]
            if self.liver_layer.isChecked() and self.tumor_layer.isChecked():
                b1 = np.uint8(b1)
                b1 = cv.cvtColor(b1, cv.COLOR_GRAY2RGB)
                mask = np.uint8(c)
                _, binary_pred = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
                contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(b1, contours_pred, -1, (255, 20, 147), 2) # pink contours stand for prediction
          
                mask_2 = np.uint8(a)
  
                _, binary_pred = cv.threshold(mask_2, 127, 255, cv.THRESH_BINARY)
                contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(b1, contours_pred, -1, (0, 255, 0), 2) # pink contours stand for prediction  
                self.show_img(b1)
            elif self.liver_layer.isChecked():
                b2 = np.uint8(b2)
                b2 = cv.cvtColor(b2, cv.COLOR_GRAY2RGB)
                mask = np.uint8(a)
               
                _, binary_pred = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
                contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(b2, contours_pred, -1, (0, 255, 0), 2) # pink contours stand for prediction
                self.show_img(b2)
            elif self.tumor_layer.isChecked():
                b3 = np.uint8(b3)
                b3 = cv.cvtColor(b3, cv.COLOR_GRAY2RGB)
                mask = np.uint8(c)
             
                _, binary_pred = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
                contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(b3, contours_pred, -1, (255, 20, 147), 2) # pink contours stand for prediction
           
                self.show_img(b3)
            else:
        
                self.show_img(self.slices_tumor[num2])
 
        else:
  
            QMessageBox.information(self, "Warning", "超出索引范围" )
            
    # 两个图层
    @pyqtSlot()
    def on_liver_layer_clicked(self):
        num_pic = self.num.text()
        num2 = int(num_pic)
        b1 = self.slices_tumor[num2]
        b2 = self.slices_tumor[num2]
        b3 = self.slices_tumor[num2]
        a = self.liver[num2]
     
        c = self.tumor[num2]
        if self.liver_layer.isChecked() and self.tumor_layer.isChecked():
            b1 = np.uint8(b1)
            b1 = cv.cvtColor(b1, cv.COLOR_GRAY2RGB)
            mask = np.uint8(c)
      
            _, binary_pred = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
            contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(b1, contours_pred, -1, (255, 20, 147), 2) # pink contours stand for prediction
      
            mask_2 = np.uint8(a)
      
            _, binary_pred = cv.threshold(mask_2, 127, 255, cv.THRESH_BINARY)
            contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(b1, contours_pred, -1, (0, 255, 0), 2) # pink contours stand for prediction  
        
            self.show_img(b1)
        elif self.liver_layer.isChecked():
            b2 = np.uint8(b2)
            b2 = cv.cvtColor(b2, cv.COLOR_GRAY2RGB)
            mask = np.uint8(a)
         
            _, binary_pred = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
            contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(b2, contours_pred, -1, (0, 255, 0), 2) # pink contours stand for prediction
            self.show_img(b2)
        elif self.tumor_layer.isChecked():
            b3 = np.uint8(b3)
            b3 = cv.cvtColor(b3, cv.COLOR_GRAY2RGB)
            mask = np.uint8(c)
      
            _, binary_pred = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
            contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(b3, contours_pred, -1, (255, 20, 147), 2) # pink contours stand for prediction
     
            self.show_img(b3)
        else:
    
            self.show_img(self.slices_tumor[num2])
  

    # 同上
    @pyqtSlot()
    def on_tumor_layer_clicked(self):
        num_pic = self.num.text()            
        num2 = int(num_pic)
        b1 = self.slices_tumor[num2]
        b2 = self.slices_tumor[num2]
        b3 = self.slices_tumor[num2]
        a = self.liver[num2]

    
        c = self.tumor[num2]
        if self.liver_layer.isChecked() and self.tumor_layer.isChecked():
            b1 = np.uint8(b1)
            b1 = cv.cvtColor(b1, cv.COLOR_GRAY2RGB)
            mask = np.uint8(c)
     
            _, binary_pred = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
            contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(b1, contours_pred, -1, (255, 20, 147), 2) # pink contours stand for prediction
      
            mask_2 = np.uint8(a)

            _, binary_pred = cv.threshold(mask_2, 127, 255, cv.THRESH_BINARY)
            contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(b1, contours_pred, -1, (0, 255, 0), 2) # pink contours stand for prediction  
 
            self.show_img(b1)
        elif self.liver_layer.isChecked():
            b2 = np.uint8(b2)
            b2 = cv.cvtColor(b2, cv.COLOR_GRAY2RGB)
            mask = np.uint8(a)
        
            _, binary_pred = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
            contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(b2, contours_pred, -1, (0, 255, 0), 2) # pink contours stand for prediction
            self.show_img(b2)
        elif self.tumor_layer.isChecked():
            b3 = np.uint8(b3)
            b3 = cv.cvtColor(b3, cv.COLOR_GRAY2RGB)
            mask = np.uint8(c)
      
            _, binary_pred = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
            contours_pred, _ = cv.findContours(binary_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(b3, contours_pred, -1, (255, 20, 147), 2) # pink contours stand for prediction
      
            self.show_img(b3)
        else:
     
            self.show_img(self.slices_tumor[num2])
 
            
            
    @pyqtSlot()
    def on_tri_d_run_clicked(self):


        # TODO: not implemented yet

        QMessageBox.information(self, "Warning", "三维模型导出成功" )






if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())
