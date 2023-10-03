# Car-license-plate-recognition-python
Separation of letters and numbers of license plates using PCA
## Steps
##### 1. Separation area with one character (letter or digit). 
I assume that character consist of continuous (connected) series of pixels.
##### 2. Extraction of character.
##### 3. Normalization of character.
a. Using morphological filter to get skeleton of character.<br>
b. Using PCA to get new coordinates.<br>
c. Rotation of image to align new coordinate system with image edges.<br>
d. Scaling it to fill the entire frame.
##### 4. Find character descriptor.
