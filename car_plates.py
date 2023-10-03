# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 01:17:36 2021

@author: styka
"""
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import math
import imutils
import cv2 # if not found: pip install opencv-python
import requests
from io import BytesIO

# FUNKCJE
def rgbData2GrayData(imageDataRGB):
    r, g, b = imageDataRGB[:,:,0], imageDataRGB[:,:,1], imageDataRGB[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return np.uint8(gray)

def imageReadRGB(url):
    response = requests.get(url)
    inputImage = Image.open(BytesIO(response.content))
    width, height = inputImage.size
    inputArray = np.array(inputImage)
    inputArray = inputArray[:, :, 0:3]
    return inputArray, width, height

def showImageAndHistogram(imageData, typeOfData):
	outputImage = Image.fromarray(imageData, typeOfData)
	
	plt.subplot(2,1,1),plt.imshow(outputImage)
	plt.title('Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,1,2),plt.hist(imageData.ravel(),bins = 256, range = [0,256]) 
	plt.title('Histogram')
	plt.show()

def treshold(imageData, treshold):
    #imageData=rgbData2GrayData(imageData)
    return np.where( (imageData <= treshold), 0, 255).astype('uint8')

# pX pY to punkty referencyjne w szablonie
def morphologyErosion(imageData, template, pX, pY):
    imageData1 = 255*np.ones([imageData.shape[0], imageData.shape[1]])
    # py, px = np.where(template=='X')[0][0], np.where(template=='X')[1][0]
    for i in range(pY,imageData.shape[0]-template.shape[0]+pY+1):
        for j in range(pX,imageData.shape[1]-template.shape[1]+pX+1):
            if np.array_equal(np.where(template==0),np.where(imageData[(i-pY):(i+template.shape[0]-pY),(j-pX):(j+template.shape[1]-pX)]==0)):
                 imageData1[i,j]=0
    return imageData1.astype('uint8')

# def morphologyDilation(imageData, template, pX, pY):
#     imageData1 = np.zeros([imageData.shape[0], imageData.shape[1]])
#     for i in range(pY,imageData.shape[0]-template.shape[0]+pY+1):
#         for j in range(pX,imageData.shape[1]-template.shape[1]+pX+1):
#             if np.array_equal(np.where(template==0),np.where(imageData[(i-pY):(i+template.shape[0]-pY),(j-pX):(j+template.shape[1]-pX)]==0)):
#                  imageData1[i,j]=255
#     return imageData1.astype('uint8')

def cutPart(image, x, y, width, height):  
    #image=rgbData2GrayData(image)
    heightIm, widthIm = image.shape
    # image=image[(heightIm-y-height):(heightIm-y),x:(x+width)] -> od dolnego lewego w górę
    # od górnego lewego w dól:
    image=image[y:(height+y),x:(x+width)]
    return image.astype('uint8')

def edgeDetectorPrewitt(imageData):
    width = imageData.shape[1]
    height = imageData.shape[0]
    outputMagnitude=np.zeros([height-1,width-1])
    outputDirection=np.zeros([height-1,width-1])
    # itertools.product - iloczyn kartezjański, równoważny zagnieżdżonej pętli for
    for x in range(1, width-1):
        for y in range(1, height-1):
            mX,mY = 0.0, 0.0
            for c in range(-1, 2):
                mX = mX + float(imageData[y + c, x - 1]) - float(imageData[y + c, x + 1])
                mY = mY + float(imageData[y - 1, x + c]) - float(imageData[y + 1, x + c])
            outputMagnitude[y][x] = math.sqrt(mX * mX + mY * mY)
            outputDirection[y][x] = math.atan2(mY, mX)
    return outputMagnitude, outputDirection

def reverse(imageData):
	return 255-imageData

###############################################################
url = 'https://raw.githubusercontent.com/annaStykowska/Car-license-plate-recognition-python/main/rej7.jpg'
imageData, width, height = imageReadRGB(url)

# RGB
showImageAndHistogram(imageData, typeOfData='RGB')
# # odcienie szarosci
# imageData=rgbData2GrayData(imageData)
# showImageAndHistogram(imageData, typeOfData='P')
# # czarno-bialy
# imageData=treshold(imageData, 150)
# showImageAndHistogram(imageData, typeOfData='P')
gray = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv2.Canny(bfilter, 30, 200) #Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
location
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255,-1)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

# odwrocony kolor, treshold - zeby wyeliminowac niebieski prostokąt kolo literek
rev=treshold(reverse(gray),100)
plt.imshow(cv2.cvtColor(rev, cv2.COLOR_BGR2RGB))

new_image2 = cv2.bitwise_and(new_image, rev, mask=mask)

plt.imshow(cv2.cvtColor(new_image2, cv2.COLOR_BGR2RGB))
(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))

cropped_image = reverse(new_image2[x1:x2+1, y1:y2+1])
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

# mała templatka, żeby zredukowac poziome linie
cropped_image=morphologyErosion(cropped_image, np.zeros([2,2]), 0, 0)
showImageAndHistogram(cropped_image, typeOfData='P')
##############################################################################

# IZOLOWANIE LITER
# rozdzielone obszary połączonych pikseli

# croppedImage, width, height = imageReadRGB('C:/Users/styka/OneDrive/Pulpit/IFE/projekt/rejestracje/1.jpg')
# showImageAndHistogram(croppedImage, 'RGB')
# croppedImage=rgbData2GrayData(croppedImage)
# treshold(croppedImage,150)
# showImageAndHistogram(croppedImage, 'P')

from skimage.measure import label, regionprops, regionprops_table
from skimage.io import imshow
label_im = label(cropped_image,background=255)
regions = regionprops(label_im)
imshow(label_im)
properties = ['area','convex_area','bbox_area', 'extent',  
              'mean_intensity', 'solidity', 'eccentricity', 
              'orientation']
pd.DataFrame(regionprops_table(label_im, cropped_image, 
             properties=properties))

# prostokąty dookoła cyfr/liter
import matplotlib.patches as mpatches
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(cropped_image)
idxVec=[]
for region in regionprops(label_im):
    # take regions with large enough areas
    if region.area > 100:
        # draw rectangle around segmented
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)   

saved=0
characterIm=[]
for region in regionprops(label_im):
    # take regions with large enough areas
    if region.area > 100:
        # wycinanie litery
        minr, minc, maxr, maxc = region.bbox
        characterIm=cutPart(cropped_image,minc, minr, (maxc - minc), (maxr - minr))
        showImageAndHistogram(characterIm, typeOfData='P')
        # analiza głównych składowych
        # wspolrzedne gdzie jest litera
        y, x = np.where(characterIm==0)
        #liczymy srednie dla x i y
        yMean, xMean = np.mean(y), np.mean(x)
        # odejmujemy srednie od skladowych
        x, y = x-xMean, y-yMean
        # macierz kowariancji, wektory i wartosci wlasne tej macierzy
        mCov=np.cov(x,y)
        wartWlasne, wekWlasne = np.linalg.eig(mCov)
        # wektory wlasne zaznaczaja kierunki zmian danych
        # skladnik glowny - ten wektor własny, ktory ma najwieksza zmiennosc danych (ma najwieksza wartosc wlasna)-
        # im wartosc wlasna mniejsza, tym mniej istotna dana skladowa 
        # ogolnie: po obliczeniu wektorow i wlasnych i wartosci wlasnych porzadkujemy je wg malejacych 
        # wartosci wlasnych - ustawiamy skladowe wg ich istotnosci. wybierajac teraz tylko najwazniejsze 
        # skladowe mozemy oryginalne dane przeksztalcic w nowy zbior, zlozony z elementow o mniejszym
        # wymiarze, ktory bedzie opisywal te najwazniejsze naszym zdaniem cechy. Wybrane w ten sposob 
        # wektory wlasne tworzą macierz, tzw. macierz cech (wektor cech)
        # bede przeksztalcac dane nie odrzucajac zadnej ze skladowych
        # wzor na transformacje danych : D_fin=f^T x D^T, f - wybrany wektor cech, D - nasze x i y
        f=wekWlasne.T 
        D = np.array([x, y])
        # mnozenie macierzy
        new=f@D
        plt.scatter(new[1],new[0])
        plt.show()
        new[0]=10*(new[0]-min(new[0]))/max(new[0]) 
        new[1]=10*(new[1]-min(new[1]))/max(new[1])
        # zaokrąglam otrzymane współrzędne do int
        new = np.round(new,0).astype('int')
        if characterIm.shape[0]>characterIm.shape[1]:
            newImageData = 255*np.ones([characterIm.shape[0],characterIm.shape[0]])
        else:
            newImageData = 255*np.ones([characterIm.shape[1],characterIm.shape[1]])
        for i in range(new.shape[1]):
            newImageData[new[0][i],new[1][i]]=0
        plt.scatter(new[1],new[0])
        plt.savefig(str(saved)+'.png')
        saved=saved+1
        plt.show()