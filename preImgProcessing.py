import cv2
import os

#Normalize and Catagorize Image into difference folder
imgPATH = "D:/1.airplane_detection_preTrainModel/planesnet/"
imgPATHplan = ""
count = 0
for filename in os.listdir(imgPATH):
    image = cv2.imread(imgPATH + filename, cv2.IMREAD_COLOR)  # uint8 image
    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    print("img: "+str(count+1))
    if(filename[0]=='0'):
        cv2.imwrite("D:/1.airplane_detection_preTrainModel/Data/NotAirplane/"+filename, norm_image)
    else:
        cv2.imwrite("D:/1.airplane_detection_preTrainModel/Data/Airplane/"+filename, norm_image)
    count = count+1