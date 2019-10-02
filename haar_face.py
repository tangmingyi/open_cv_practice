'''
人脸识别别，采用的是训练好的模型
模型架构为 haar-like 图像特征+adaboost树分类
'''
import cv2
image = cv2.imread("data//tmy.jpg")
greyImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(greyImage,scaleFactor=1.15,minNeighbors=5,minSize=None,maxSize=None)
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0))
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.imshow("image",image)
cv2.waitKey(0)