import cv2
from text_recognition import Container_Number_Recognition

xml_file = 'text-detection.xml'
bin_file = 'text-detection.bin'
cont_num_rec = Container_Number_Recognition(xml_file, bin_file)

frame = cv2.imread('image.jpg')
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cont_rects = [[400, 470, 870, 920],
              [0, 0, 380, 400],
              [0, 470, 400, 920],
              [380, 0, 850, 420],
              [400, 920, 850, 1280]
             ]

nums = cont_num_rec.predict(frame, cont_rects)
print(nums)