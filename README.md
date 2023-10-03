# Car-license-plate-recognition-python
Separation of letters and numbers of license plates using PCA
## Preparation
You may need to install python packages, such as opencv-python, scikit-image, requests. 
More about installing packages you can read here: https://packaging.python.org/en/latest/tutorials/installing-packages/
For changing image to process, change ulr in line 90 (uploaded images in my repository):
* url = 'https://raw.githubusercontent.com/annaStykowska/Car-license-plate-recognition-python/main/rej3.jpg'
* url = 'https://raw.githubusercontent.com/annaStykowska/Car-license-plate-recognition-python/main/rej7.jpg'
* url = 'https://raw.githubusercontent.com/annaStykowska/Car-license-plate-recognition-python/main/rej9.jpg'
* url = 'https://raw.githubusercontent.com/annaStykowska/Car-license-plate-recognition-python/main/rej14.jpg'
## Steps
##### 1. Separation area with one character (letter or digit). 
I assume that character consist of continuous (connected) series of pixels.
##### 2. Extraction of character.
##### 3. Normalization of character.
a. Using morphological filter to get skeleton of character.<br>
b. Using PCA to get new coordinates.<br>
c. Rotation of image to align new coordinate system with image edges.<br>
d. Scaling it to fill the entire frame.
