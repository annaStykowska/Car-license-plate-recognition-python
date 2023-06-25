# Car-license-plate-recognition-python
Recognition of letters and numbers of license plates using PCA
## Steps
##### 1. Separate area with one character (letter or digit). You can
assume that character consist of continuous (connected)
series of pixels.
##### 2. Extract character.
##### 3. Normalize character.
a. Use morphological filter to get skeleton of character.<br>
b. Use PCA to get new coordinates.<br>
c. Rotate image to align new coordinate system with
image edges.<br>
d. Scale it to fill the entire frame.
##### 4. Find character descriptor.
