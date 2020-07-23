import pytesseract as pts
import numpy as np
import cv2

class Container_Number_Recognition:
    def __init__(self, xml_file, bin_file):
        self.net = cv2.dnn.readNet(xml_file, bin_file)
        
    def detect_numbers(self, image, cont_rects):
        ''' Детекция номеров на исходном кадре.
        
        image - изображение с камеры;
        cont_rects - список ROI боксов контейнеров (бокс описывается координатами левой верхней и правой нижней точками в пикселях - [y1, x1, y2, x2])
        return:
        num_rects - список найденных боксов номеров        
        ''' 
        num_rects = []
        kernel = np.ones((9,9),np.uint8)
        for rect in cont_rects:
            blob = cv2.dnn.blobFromImage(image[rect[0]:rect[2], rect[1]:rect[3]], size=(768, 1280))
            self.net.setInput(blob)
            out = self.net.forward()
            dilation = cv2.dilate((softmax(out)[0][1] > 0.1).astype(np.uint8), kernel, iterations=1)
            output = cv2.connectedComponentsWithStats(dilation, connectivity=8, ltype=cv2.CV_32S)
            size = output[1].shape
            label = None
            dist = 200
            point = (size[0]*.7, size[1]*.2)  # Поиск номера осуществляем вблизи этой точки (правая верхняя часть изображения)
            for lbl, (stats, centr) in enumerate(zip(output[2], output[3])):
                new_dist = np.linalg.norm(point - centr, ord=2)
                if new_dist < dist and stats[4] > 1000:   # Смотрим объекты, площадь которых больше 1000
                    dist = new_dist.copy()
                    label = lbl
                    box = [int(rect[0] + (rect[2]-rect[0])*stats[1]/size[0]), 
                           int(rect[1] + (rect[3]-rect[1])*stats[0]/size[1]), 
                           int(rect[0] + (rect[2]-rect[0])*(stats[1]+stats[3])/size[0]), 
                           int(rect[1] + (rect[3]-rect[1])*(stats[0]+stats[2])/size[1])]
            num_rects.append(box)
        return num_rects
    
    def recognize_numbers(self, image, num_rects):
        ''' распознавание номеров на исходном кадре в боксах.
        
        image - изображение с камеры;
        num_rects - список боксов номеров контейнеров (бокс описывается координатами левой верхней и правой нижней точками в пикселях - [y1, x1, y2, x2])
        return:
        nums - список текстовых номеров         
        '''
        nums = []
        for rect in num_rects:
            nums.append(pts.image_to_string(image[rect[0]:rect[2], rect[1]:rect[3]]))
            
        return nums
    
    def predict(self, image, cont_rects):
        '''возвращает список текстовых номеров.
        
        image - изображение с камеры;
        cont_rects - список ROI боксов контейнеров (бокс описывается координатами левой верхней и правой нижней точками в пикселях - [y1, x1, y2, x2]).
        return:
        nums - список текстовых номеров      
        '''
        num_rects = self.detect_numbers(image, cont_rects)
        return self.recognize_numbers(image, num_rects)
        
    
def softmax(out):
    sfmx = np.exp(out)
    s = sfmx.sum(axis=1)
    sfmx[:,0,...] /= s
    sfmx[:,1,...] /= s
    return sfmx