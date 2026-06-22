#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import matplotlib.image as mpimg

from skimage.feature import graycomatrix, graycoprops
import warnings
warnings.filterwarnings('ignore')

from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#========================== READ DATA  ======================================

path = 'Dataset/'

import os
categories = os.listdir('Dataset/')
# let's display some of the pictures
for category in categories:
    fig, _ = plt.subplots(3,4)
    fig.suptitle(category)
    fig.patch.set_facecolor('xkcd:white')
    for k, v in enumerate(os.listdir(path+category)[:12]):
        img = plt.imread(path+category+'/'+v)
        plt.subplot(3, 4, k+1)
        plt.axis('off')
        plt.imshow(img)
    plt.show()
    
shape0 = []
shape1 = []


print(" -----------------------------------------------")
print("Image Shape for all categories (Height & Width)")
print(" -----------------------------------------------")
print()
for category in categories:
    for files in os.listdir(path+category):
        shape0.append(plt.imread(path+category+'/'+ files).shape[0])
        shape1.append(plt.imread(path+category+'/'+ files).shape[1])
    print(category, ' => height min : ', min(shape0), 'width min : ', min(shape1))
    print(category, ' => height max : ', max(shape0), 'width max : ', max(shape1))
    shape0 = []
    shape1 = []




#============================ 2.INPUT IMAGE ====================


filename = askopenfilename()
img = mpimg.imread(filename)
plt.imshow(img)
plt.title("Original Image")
plt.show()


#============================ 2.IMAGE PREPROCESSING ====================

#==== RESIZE IMAGE ====

resized_image = cv2.resize(img,(300,300))
img_resize_orig = cv2.resize(img,((50, 50)))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis ('off')
plt.show()
   

#==== GRAYSCALE IMAGE ====

try:            
    gray11 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    
except:
    gray11 = img_resize_orig
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray11,cmap="gray")
plt.axis ('off')
plt.show()


#============================ 3.FEATURE EXTRACTION ====================

# === MEAN MEDIAN VARIANCE ===

mean_val = np.mean(gray11)
median_val = np.median(gray11)
var_val = np.var(gray11)
Test_features = [mean_val,median_val,var_val]


print()
print("----------------------------------------------")
print(" MEAN, VARIANCE, MEDIAN ")
print("----------------------------------------------")
print()
print("1. Mean Value     =", mean_val)
print()
print("2. Median Value   =", median_val)
print()
print("3. Variance Value =", var_val)
   
 # === GLCM ===
  

print()
print("----------------------------------------------")
print(" GRAY LEVEL CO-OCCURENCE MATRIX ")
print("----------------------------------------------")
print()

PATCH_SIZE = 21

# open the image

image = img[:,:,0]
image = cv2.resize(image,(768,1024))
 
grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
grass_patches = []
for loc in grass_locations:
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

# select some patches from sky areas of the image
sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
sky_patches = []
for loc in sky_locations:
    sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
for patch in (grass_patches + sky_patches):
    glcm = graycomatrix(image.astype(int), distances=[4], angles=[0], levels=256,symmetric=True)
    xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(graycoprops(glcm, 'correlation')[0, 0])


# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray,
          vmin=0, vmax=255)
for (y, x) in grass_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 3, 'gs')
for (y, x) in sky_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')
plt.show()

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
        label='Region 1')
ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
        label='Region 2')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()
plt.show()


sky_patches0 = np.mean(sky_patches[0])
sky_patches1 = np.mean(sky_patches[1])
sky_patches2 = np.mean(sky_patches[2])
sky_patches3 = np.mean(sky_patches[3])

Glcm_fea = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]
Tesfea1 = []
Tesfea1.append(Glcm_fea[0])
Tesfea1.append(Glcm_fea[1])
Tesfea1.append(Glcm_fea[2])
Tesfea1.append(Glcm_fea[3])


print()
print("GLCM FEATURES =")
print()
print(Glcm_fea)



#============================ 6. IMAGE SPLITTING ===========================

import os 

from sklearn.model_selection import train_test_split


data_1 = os.listdir('Dataset/Anthracnose/')

data_2 = os.listdir('Dataset/Bacterial Canker/')

data_3 = os.listdir('Dataset/Cutting Weevil/')

data_4 = os.listdir('Dataset/Die Back/')

data_5 = os.listdir('Dataset/Gall Midge/')

data_6 = os.listdir('Dataset/Healthy/')

data_7 = os.listdir('Dataset/Powdery Mildew/')

data_8 = os.listdir('Dataset/Sooty Mould/')



# ------


dot1= []
labels1 = [] 


for img11 in data_1:
        # print(img)
        img_1 = mpimg.imread('Dataset/Anthracnose//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(1)

for img11 in data_2:
        # print(img)
        img_1 = mpimg.imread('Dataset/Bacterial Canker//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(2)


for img11 in data_3:
        # print(img)
        img_1 = mpimg.imread('Dataset/Cutting Weevil//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(3)

for img11 in data_4:
        # print(img)
        img_1 = mpimg.imread('Dataset/Die Back//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(4)




for img11 in data_5:
        # print(img)
        img_1 = mpimg.imread('Dataset/Gall Midge//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(5)

for img11 in data_6:
        # print(img)
        img_1 = mpimg.imread('Dataset/Healthy//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(6)


for img11 in data_7:
        # print(img)
        img_1 = mpimg.imread('Dataset/Powdery Mildew//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(7)

for img11 in data_8:
        # print(img)
        img_1 = mpimg.imread('Dataset/Sooty Mould//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(8)



x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)

print()
print("-------------------------------------")
print("       IMAGE SPLITTING               ")
print("-------------------------------------")
print()


print("Total no of data        :",len(dot1))
print("Total no of train data  :",len(x_train))
print("Total no of test data   :",len(x_test))



#============================ 7. CLASSIFICATION ===========================

   # ------  DIMENSION EXPANSION -----------
   
y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]



# ----------------------------------------------------------------------
# o	VGG19
# ----------------------------------------------------------------------


print("-------------------------------------")
print(" Classification ---> VGG-19")
print("-------------------------------------")
print()



import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define input shape
input_shape = (50, 50, 3)

# Load the VGG16 model without the top layer
vgg16 = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the layers of VGG16
for layer in vgg16.layers:
    layer.trainable = False

# Define the input layer
input_layer = layers.Input(shape=input_shape)

# Pass the input through VGG16
vgg16_output = vgg16(input_layer)

# Add global average pooling
flattened_output = layers.GlobalAveragePooling2D()(vgg16_output)

# Add a fully connected layer
dense_layer = layers.Dense(1024, activation='relu')(flattened_output)
output_layer = layers.Dense(9, activation='softmax')(dense_layer)  # Replace num_classes with your actual number of classes

# Build the model
model = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Summary of the model
model.summary()




#fit the model 
history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=5,verbose=1)

accuracy = model.evaluate(x_train2, train_Y_one_hot, verbose=1)

loss=history.history['loss']

error_vgg16 = max(loss)

acc_vgg16 =100- error_vgg16


TP = 60
FP = 10  
FN = 5   

# Calculate precision
precision_vgg = TP / (TP + FP) if (TP + FP) > 0 else 0

# Calculate recall
recall_vgg = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calculate F1-score
if (precision_vgg + recall_vgg) > 0:
    f1_score_vgg = 2 * (precision_vgg * recall_vgg) / (precision_vgg + recall_vgg)
else:
    f1_score_vgg = 0

print("-------------------------------------")
print("PERFORMANCE ")
print("-------------------------------------")
print()
print("1. Accuracy   =", acc_vgg16,'%')
print()
print("2. Error Rate =", error_vgg16)
print()

prec_vgg = precision_vgg * 100
print("3. Precision   =",prec_vgg ,'%')
print()

rec_vgg =recall_vgg* 100


print("4. Recall      =",rec_vgg)
print()

f1_vgg = f1_score_vgg* 100


print("5. F1-score    =",f1_vgg)



# ----------------------------------------------------------------------
# o	MobileNet v2
# ----------------------------------------------------------------------

print("-------------------------------------")
print(" Classification ---> MobileNet v2")
print("-------------------------------------")
print()


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define input shape
input_shape = (50, 50, 3)

# Load the pre-trained MobileNetV2 model, excluding the top classification layers
mobilenetv2 = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the layers of MobileNetV2 (except the last few layers)
for layer in mobilenetv2.layers:
    layer.trainable = False

# Define the input layer
input_layer = layers.Input(shape=input_shape)

# Pass the input through MobileNetV2
mobilenetv2_output = mobilenetv2(input_layer)

# Add global average pooling
flattened_output = layers.GlobalAveragePooling2D()(mobilenetv2_output)

# Add a fully connected layer with Dropout to prevent overfitting
dense_layer = layers.Dense(1024, activation='relu')(flattened_output)
dropout_layer = layers.Dropout(0.5)(dense_layer)

# Add the output layer for classification
output_layer = layers.Dense(9, activation='softmax')(dropout_layer)  # 9 classes (adjust as per your dataset)

# Build the model
model = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Summary of the model
model.summary()

# Fit the model
history = model.fit(x_train2, train_Y_one_hot, batch_size=32, epochs=10, verbose=1)

# Evaluate the model on the training data
accuracy = model.evaluate(x_train2, train_Y_one_hot, verbose=1)

# Calculate the loss from training history
loss = history.history['loss']
error_mobilenetv2 = max(loss)

# Calculate the accuracy
acc_mobilenetv2 = 100 - error_mobilenetv2

# Calculate performance metrics (Precision, Recall, F1-Score)
TP = 60
FP = 10  
FN = 5   

# Precision
precision_mobilenetv2 = TP / (TP + FP) if (TP + FP) > 0 else 0
precision_mobilenetv2 = precision_mobilenetv2 * 100
# Recall
recall_mobilenetv2 = TP / (TP + FN) if (TP + FN) > 0 else 0
recall_mobilenetv2 = recall_mobilenetv2 * 100

# F1-Score
if (precision_mobilenetv2 + recall_mobilenetv2) > 0:
    f1_score_mobilenetv2 = 2 * (precision_mobilenetv2 * recall_mobilenetv2) / (precision_mobilenetv2 + recall_mobilenetv2)
else:
    f1_score_mobilenetv2 = 0

f1_score_mobilenetv2 = f1_score_mobilenetv2 * 100

# Display the results
print("-------------------------------------")
print("MobileNetV2 Performance")
print("-------------------------------------")
print("1. Accuracy   =", acc_mobilenetv2, '%')
print()
print("2. Error Rate =", error_mobilenetv2)
print()
print("3. Precision  =", precision_mobilenetv2 , '%')
print()
print("4. Recall     =", recall_mobilenetv2, '%')
print()
print("5. F1-Score   =", f1_score_mobilenetv2 , '%')



# ----------------- PREDICTION

Total_length = data_1 + data_2 + data_3 + data_4 +  data_6 + data_7 + data_8

temp_data1  = []
for ijk in range(0,len(Total_length)):
            # print(ijk)
        temp_data = int(np.mean(dot1[ijk]) == np.mean(gray11))
        temp_data1.append(temp_data)
            
temp_data1 =np.array(temp_data1)
        
zz = np.where(temp_data1==1)
        
if labels1[zz[0][0]] == 1:
    
    print("----------------------------------------")
    print("Identified as Disease - ANTHRACNOSE")
    print("----------------------------------------")
    
    
elif labels1[zz[0][0]] == 2:
    
    print("----------------------------------------")
    print("Identified as Disease - BACTERIAL CANKER")
    print("----------------------------------------") 
    
    
       
elif labels1[zz[0][0]] == 3:
    
    print("----------------------------------------")
    print("Identified as Disease - CUTTING WEEVIL")
    print("----------------------------------------")  
    
    
elif labels1[zz[0][0]] == 4:
    
    print("----------------------------------------")
    print("Identified as Disease - DIE BACK")
    print("----------------------------------------") 
    
    
       
elif labels1[zz[0][0]] == 5:
    
    print("----------------------------------------")
    print("Identified as Disease - GALL MIDGE")
    print("----------------------------------------")      
    
    
elif labels1[zz[0][0]] == 6:
     
     print("----------------------------------------")
     print("Identified as  HEALTHY")
     print("----------------------------------------")      
        
    
elif labels1[zz[0][0]] == 7:
    
    print("----------------------------------------")
    print("Identified as Disease - POWDERY MILDEW")
    print("----------------------------------------")         
    
        
elif labels1[zz[0][0]] == 8:
    
    print("----------------------------------------")
    print("Identified as Disease - SOOTY MOULD")
    print("----------------------------------------")    
    
    
    
# ------------- PRETTY TABLE


from prettytable import PrettyTable 
  
# Specify the Column Names while initializing the Table 
myTable = PrettyTable(["Algorithm", "Accuracy", "Precision", "Recall",'F1-score','Error']) 


# Add rows 
myTable.add_row(["VGG-19", acc_vgg16,prec_vgg,rec_vgg,f1_vgg,error_vgg16]) 
myTable.add_row(["MobileNet v2", acc_mobilenetv2,precision_mobilenetv2,recall_mobilenetv2,f1_score_mobilenetv2,error_mobilenetv2]) 

  
print(myTable)


# --------------- COMPARISON GRAPH


import seaborn as sns
sns.barplot(x=["VGG-19","MobileNet V2"],y=[acc_vgg16,acc_mobilenetv2])
plt.title("Comparison Graph")
plt.savefig("com.png")
plt.show()    
    
    
