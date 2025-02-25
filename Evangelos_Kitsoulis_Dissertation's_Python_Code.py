#!/usr/bin/env python
# coding: utf-8
# General utilities

#oading necessary libraries
import os
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.combine import SMOTETomek
import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator, K2Score
import cv2
from skimage.segmentation import mark_boundaries
import shap
import lime
import lime.lime_image
import netron




#dataset loading
data_dir = r'C:\Users\vange\OneDrive\Desktop\Master_Thesis_Dataset\chest_xray'
paths = glob.glob(data_dir + '/*/*/*.jpeg')
df = pd.DataFrame(paths, columns=['path'])
for path in paths[:10]:
    print(path)
df['label'] = df['path'].apply(lambda x: x.split('\\')[8].strip())  # Adjust index based on your file structure

df.head(100)

#sampling random images from the batch
sampled_df = df.sample(n=8).reset_index(drop=True)
plt.figure(figsize=(7, 5))
for i in range(8):
    plt.subplot(3, 3, i + 1)
    img_path = sampled_df.iloc[i]['path']
    img = plt.imread(img_path)
    plt.imshow(img,cmap='gray')
    plt.title(sampled_df.iloc[i]['label'])
    plt.axis('off')

plt.tight_layout()
plt.show()


#creating training, validation, and testing splits and printing them
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.7, random_state=42)

print("Training set shape:", train_df.shape)
print("Validation set shape:", valid_df.shape)
print("Testing set shape:", test_df.shape)
#checking for imbalances in the training set
counts=train_df['label'].value_counts()
print(counts)
plt.figure(figsize=(8,8))
plt.bar(['PNEUMONIA','NORMAL'], counts, width=0.5,color='darkblue')
plt.xlabel('Type')
plt.ylabel('Sample Size')
plt.title("Number of images per type")
plt.xticks(rotation=45, ha='right') 
plt.tight_layout() 
plt.show()


# creating a function to read and resize images
def read_and_resize_image(path, target_size=(150, 150)):
    img = plt.imread(path)
    if img.ndim == 2:  
        img = tf.image.grayscale_to_rgb(tf.convert_to_tensor(img[..., np.newaxis]))
    else:
        img = tf.convert_to_tensor(img)
    img_resized = tf.image.resize(img, target_size)
    return img_resized.numpy()

# Prepare data for SMOTETomek
X_train = np.array([read_and_resize_image(path) for path in train_df['path']])
y_train = train_df['label'].values


X_train_flat = X_train.reshape(len(X_train), -1)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train_flat, y_train_encoded)
X_resampled = X_resampled.reshape(-1, 150, 150, 3)
balanced_train_paths = [f"resampled_image_{i}.jpeg" for i in range(len(X_resampled))]
balanced_train_labels = le.inverse_transform(y_resampled)
balanced_train_df = pd.DataFrame({
    'path': balanced_train_paths,
    'label': balanced_train_labels
})

# Save resampled images to disk
os.makedirs('resampled_images', exist_ok=True)
for i, img in enumerate(X_resampled):
    img_normalized = img / 255.0
    plt.imsave(f'resampled_images/resampled_image_{i}.jpeg', img_normalized)
balanced_train_df['path'] = balanced_train_df['path'].apply(lambda x: os.path.join('resampled_images', x))



counts=balanced_train_df['label'].value_counts()
print(counts)
plt.figure(figsize=(8,8))
plt.bar(['PNEUMONIA','NORMAL'], counts, width=0.5,color='darkblue')
plt.xlabel('Type')
plt.ylabel('Sample Size')
plt.title("Number of images")
plt.xticks(rotation=45, ha='right') 
plt.tight_layout() 
plt.show()


#saving the balanced training set, the validation set, and the test set.
balanced_train_df.to_csv('balanced_train_df.csv', index=False)
valid_df.to_csv('valid_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)


# importing the sets
balanced_train_df = pd.read_csv('balanced_train_df.csv')
valid_df = pd.read_csv('valid_df.csv')
test_df = pd.read_csv('test_df.csv')



# Data augmentation with validation split for the training set
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=.1
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=balanced_train_df,
    x_col='path',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

valid_generator = test_datagen.flow_from_dataframe(
    dataframe=valid_df,
    x_col='path',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',
    y_col='label',
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
)

#printing the generators to check again
print(f"Training generator: {len(train_generator)} batches, {train_generator.samples} samples")
print(f"Validation generator: {len(valid_generator)} batches, {valid_generator.samples} samples")
print(f"Test generator: {len(test_generator)} batches, {test_generator.samples} samples")

classes=['NORMAL','PNEUMONIA']
batch_size = 9
images, labels = next(train_generator)
plt.figure(figsize=(7, 5))
for i in range(min(len(images), 9)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i],cmap='gray')
    plt.title(f"{classes[int(labels[i])]}")
    plt.axis('off')

plt.tight_layout()
plt.show()


# Define the CNN model using the Functional API
inputs = Input(shape=(150, 150, 3))
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

model.summary()


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


weight_path = "cnn_weights.best.weights.h5"
checkpoint = ModelCheckpoint(weight_path, monitor='val_accuracy', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only=True)

early = EarlyStopping(monitor="val_accuracy", 
                      mode="max", 
                      patience=4)

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=1)

callbacks_list = [checkpoint, early,lr_scheduler]
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10,  
    callbacks=callbacks_list
)
model.save('cnn_full_model.keras')


loaded_model = load_model('cnn_full_model.keras')
print(loaded_model.summary())


#visualization of the CNN
netron.start('cnn_full_model.keras') 

evaluation_result_1=loaded_model.evaluate(test_generator)
print("Test Loss:", evaluation_result_1[0])
print("Test Accuracy:", evaluation_result_1[1]* 100, '%')


# Generate predictions for the test set
y_pred = loaded_model.predict(test_generator)
y_pred_binary = np.squeeze(np.round(y_pred))

predictions_df = pd.DataFrame({
    'True Label': test_generator.classes,
    'Predicted Label': y_pred_binary
})

print(predictions_df)
predictions_df.to_csv('predictions.csv', index=False)

# create and printing confusion matrix
confusion_Matrix = confusion_matrix(test_generator.classes, y_pred_binary)
print(confusion_Matrix)
sns.heatmap(confusion_Matrix, annot=True,fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()


tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]
Epochs = [i+1 for i in range(len(tr_acc))]
loss_label = f'best epoch= {str(index_loss + 1)}'
acc_label = f'best epoch= {str(index_acc + 1)}'

# Plot training history
plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')
plt.subplot(1, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout
plt.show()
# printing the classification report
print(classification_report(test_generator.classes, y_pred_binary))




#loading the model
loaded_model.summary()
model.summary()

print( loaded_model.input_shape)

#naminng the three convolutional layers
first_conv_layer='conv2d'
second_conv_layer='conv2d_1'
last_conv_layer='conv2d_2'


#sample image pre-processing from testing set. One Pneumonia and One Normal
im_path =r'C:\Users\vange\OneDrive\Desktop\Master_Thesis_Dataset\chest_xray\test\PNEUMONIA\person1_virus_6.jpeg' 
im_path_1 =r'C:\Users\vange\OneDrive\Desktop\Master_Thesis_Dataset\chest_xray\test\NORMAL\IM-0001-0001.jpeg' 

#functions of image pre-processing and division

# pre_process imagess
def preprocess_image(im_path, target_size=(150, 150)):
    img = Image.open(im_path)
    img = img.convert('RGB')
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)   
    return img_array
#image division
def divide_into_grid(image, grid_size=(3, 3)):
    """ Divide the image into a grid of regions. """
    height, width = image.shape[:2]
    grid_h, grid_w = height // grid_size[0], width // grid_size[1]
    regions = []

    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            start_row, start_col = row * grid_h, col * grid_w
            end_row, end_col = start_row + grid_h, start_col + grid_w
            region = image[start_row:end_row, start_col:end_col]
            regions.append(region)
    return regions

img_array = preprocess_image(im_path)
# Create a model that outputs features from the last convolutional layer
feature_extraction_model = Model(inputs=loaded_model.input, outputs=loaded_model.get_layer(last_conv_layer).output)

# Pass the sample image through the model
features = feature_extraction_model.predict(img_array)
print(features.shape)
# Print the shape of the extracted features
print("Shape of features before PCA:", features.shape)
#load the sample dataset used to create the BN and the Counterfactuals
sample_labels = np.load('sample_labels.npy')  
sample_images = np.load('sample_images.npy')  


sample_images_reduced, _, sample_labels_reduced, _ = train_test_split(
    sample_images, sample_labels, 
    train_size=0.15,  
    stratify=sample_labels, 
    random_state=42
)
print("Reduced sample size:", sample_images_reduced.shape[0])
unique_labels, counts = np.unique(sample_labels_reduced, return_counts=True)
print("Reduced label distribution:", dict(zip(unique_labels, counts)))
#saving the reduced sample dataset
np.save('reduced_sample_images.npy', sample_images_reduced)
np.save('reduced_sample_labels.npy', sample_labels_reduced)
#loading the reduce sample dataset
sample_images_reduced = np.load('reduced_sample_images.npy')
sample_labels_reduced = np.load('reduced_sample_labels.npy')

# re-check after loading
print("Loaded sample size:", sample_images_reduced.shape[0])
print("Loaded label distribution:", np.unique(sample_labels_reduced, return_counts=True))
img_array = preprocess_image(im_path)



#Grad-Cam creation.

#use for single result
img_array = preprocess_image(im_path)
img_array_1 = preprocess_image(im_path_1)

#always use

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = tf.gather(predictions[0], pred_index)

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def calculate_gradcam_importance(heatmap, grid_size=(3, 3)):
    heatmap_regions = divide_into_grid(heatmap, grid_size)
    total_important_intensity = np.sum(heatmap)
    importance_scores = []    
    for region in heatmap_regions:
        region_importance = np.sum(region)
        if total_important_intensity == 0:
            importance_percentage = 0
        else:
            importance_percentage = (region_importance / total_important_intensity) * 100     
        importance_scores.append(importance_percentage)    
    return importance_scores

#use when displaying on 3x3

def overlay_gradcam_on_image(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    superimposed_img = heatmap_resized * alpha + img
    superimposed_img = np.uint8(superimposed_img)
    return superimposed_img
def display_gradcam_3x3(img_path, heatmap):
    superimposed_img = overlay_gradcam_on_image(img_path, heatmap)
    regions = divide_into_grid(superimposed_img, grid_size=(3, 3))
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))    
    for idx, ax in enumerate(axs.flat):
        ax.imshow(cv2.cvtColor(regions[idx], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Region {idx}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#use when displaying GRAD CAM vs Original

def display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.uint8(superimposed_img)
    cv2.imwrite(cam_path, superimposed_img)
    
    # Display the original and Grad-CAM images
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM")
    plt.axis('off')
    plt.show()

#single Result use
heatmap = make_gradcam_heatmap(img_array_1,loaded_model, last_conv_layer)
gradcam_importance = calculate_gradcam_importance(heatmap, grid_size=(3, 3))
display_gradcam_3x3(im_path_1, heatmap)
display_gradcam(im_path, heatmap)
gradcam_df = pd.DataFrame({
    'Region Name': [f'region_{i}' for i in range(len(gradcam_importance))],
    'Grad-CAM Importance': gradcam_importance
})
print(gradcam_df)
gradcam_df.to_csv('gradcam_importance.csv', index=False)

#multiple Result use
num_regions = 9  
region_scores_sum = np.zeros(num_regions)
for img_array in sample_images_reduced:
    heatmap = make_gradcam_heatmap(np.expand_dims(img_array, axis=0), loaded_model, last_conv_layer)
    gradcam_importance = calculate_gradcam_importance(heatmap, grid_size=(3, 3))
    region_scores_sum += gradcam_importance
region_scores_avg = region_scores_sum / len(sample_images_reduced)
region_scores_avg = (region_scores_avg / np.sum(region_scores_avg)) * 100
gradcam_df = pd.DataFrame({
    'Region Name': [f'region_{i}' for i in range(len(region_scores_avg))],
    'Grad-CAM Importance (%)': region_scores_avg
})
print(gradcam_df)
gradcam_df.to_csv('gradcam_importance_avg.csv', index=False)
# end of Grad-Cam

#Lime Explainer Creation.
#use for single LIME result
img_array = preprocess_image(im_path)
explainer = lime.lime_image.LimeImageExplainer()
img_array_new = np.expand_dims(img_array, axis=0)
predictions = loaded_model.predict(img_array)

def calculate_lime_importance(mask, grid_size=(3, 3)):
    mask_regions = divide_into_grid(mask, grid_size)
    total_pixels_per_region = mask_regions[0].size
    total_important_pixels = np.sum(mask != 0)
    importance_scores = []
    
    for region in mask_regions:
       
        important_pixels_in_region = np.sum(region != 0)
        
        if total_important_pixels == 0:
            importance_percentage = 0
        else:
            importance_percentage = (important_pixels_in_region / total_important_pixels) * 100
        importance_scores.append(importance_percentage)
    
    return importance_scores

def lime_prediction(images):
    processed_images = np.array( images)  
    return loaded_model.predict(processed_images)
#use for multiple lime results
num_regions = 9  
region_scores_sum = np.zeros(num_regions)


for img_array in sample_images_reduced:
    img_array_expanded = np.expand_dims(img_array, axis=0)
    explainer = lime.lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image=img_array,  
        classifier_fn=lime_prediction, 
        top_labels=2,  
        hide_color=0,  
        num_samples=1000  
    )
    _, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],  
        positive_only=True, 
        num_features=5,  
        hide_rest=False  
    )
    lime_importance = calculate_lime_importance(mask, grid_size=(3, 3))
    region_scores_sum += lime_importance
region_scores_avg = region_scores_sum / len(sample_images_reduced)
total_avg_score = sum(region_scores_avg)
normalized_region_scores = [(score / total_avg_score) * 100 for score in region_scores_avg]
lime_df = pd.DataFrame({
    'Region Name': [f'region_{i}' for i in range(len(normalized_region_scores))],
    'LIME Importance (%)': normalized_region_scores
})

print(lime_df)
lime_df.to_csv('lime_importance_normalized.csv', index=False)


#use for single LIME 
explanation = explainer.explain_instance(
    image=img_array[0],  
    classifier_fn=lime_prediction, 
    top_labels=2,  
    hide_color=0,  
    num_samples=1000  
)

temp, mask = explanation.get_image_and_mask(
    label=explanation.top_labels[0],  
    positive_only=True, 
    num_features=5,  
    hide_rest=False  
)
lime_importance = calculate_lime_importance(mask, grid_size=(3, 3))
lime_df = pd.DataFrame({
    'Region Name': [f'region_{i}' for i in range(len(lime_importance))],
    'LIME Importance (%)': lime_importance
})
print(lime_df)
lime_df.to_csv('lime_importance.csv', index=False)

#for viewing
if img_array.shape[0] == 1:
    img_array = np.squeeze(img_array, axis=0)
img_with_boundaries = mark_boundaries(img_array, mask, color=(1, 1, 0))

# Only use when 3x3 is the goal
image_regions = divide_into_grid(img_array, grid_size=(3, 3))
mask_regions = divide_into_grid(mask, grid_size=(3, 3))

#visualization of LIME vs. Original
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img_array)
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(img_with_boundaries)
ax[1].set_title('LIME')
ax[1].axis('off')
plt.tight_layout()
plt.show()

# Visualization of LIME in Grids
fig, ax = plt.subplots(3, 3, figsize=(12, 12))

for i in range(3):
    for j in range(3):
        region_with_boundaries = mark_boundaries(image_regions[i * 3 + j], mask_regions[i * 3 + j], color=(1, 1, 0))
        ax[i, j].imshow(region_with_boundaries)
        ax[i, j].axis('off')
        ax[i, j].set_title(f"Region {i * 3 + j}")

plt.tight_layout()
plt.show()

#end lime explainer.

#SHAP Explainer creation.


def display_shap_3x3(img_path, shap_img, alpha = 0.5):
    img = cv2.imread(img_path)
    shap_img_resized = cv2.resize(shap_img, (img.shape[1], img.shape[0]))
    shap_img_colored = np.uint8(255 * shap_img_resized)  # Scale to [0, 255]
    shap_img_colored = cv2.applyColorMap(shap_img_colored, cv2.COLORMAP_JET)  # Apply colormap
    superimposed_img = cv2.addWeighted(img, 1 - alpha, shap_img_colored, alpha, 0)
    regions = divide_into_grid(superimposed_img, grid_size=(3, 3))
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for idx, ax in enumerate(axs.flat):
        ax.imshow(cv2.cvtColor(regions[idx], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Region {idx}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#always use
background_data = np.random.randn(1, 150, 150, 3)
explainer = shap.DeepExplainer(loaded_model, background_data)
def calculate_shap_importance(shap_values, grid_size=(3, 3)):
    shap_regions = divide_into_grid(shap_values, grid_size)
    importance_scores = []

    for region in shap_regions:
        positive_shap_values = np.sum(region[region > 0])
        negative_shap_values = np.sum(region[region < 0])
        total_shap = positive_shap_values + negative_shap_values
        importance_scores.append({
            'total_shap_sum': total_shap
        })
    
    return importance_scores

#use for single SHAP result

img_array = preprocess_image(im_path)
img_array_1 = preprocess_image(im_path_1)
shap_values = explainer.shap_values(img_array)
shap_img_abs = np.abs(shap_values[0])
shap_img = np.sum(shap_img_abs, axis=-1)
shap_img = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min())
shap_regions = divide_into_grid(shap_img, grid_size=(3, 3))

shap_importance = calculate_shap_importance(shap_img)
shap_df = pd.DataFrame(shap_importance)
print(shap_df)
shap_df.to_csv('shap_importance.csv', index=False)
shap_img = np.squeeze(shap_values[0])  
shap_img = np.sum(shap_img, axis=-1)   
shap_img = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min())  
display_shap_3x3(im_path_1, shap_img)
display_shap_3x3(im_path, shap_img)
#end of single results

#use for multiple 
num_regions = 9  
region_scores_sum = np.zeros(num_regions)

for img_array in sample_images_reduced:
    shap_values = explainer.shap_values(np.expand_dims(img_array, axis=0))
    shap_img_abs = np.abs(shap_values[0])
    shap_img = np.sum(shap_img_abs, axis=-1)  
    shap_img = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min())
    shap_importance = calculate_shap_importance(shap_img, grid_size=(3, 3))
    for idx, region_score in enumerate(shap_importance):
        region_scores_sum[idx] += region_score['total_shap_sum']
region_scores_avg = region_scores_sum / len(sample_images_reduced)


region_names = [f"region_{i}" for i in range(num_regions)]
shap_df = pd.DataFrame({
    'Region Name': region_names,
    'SHAP Importance (%)': [f"{round((score / np.sum(region_scores_avg)) * 100, 2)}%" for score in region_scores_avg]
})

shap_df.to_csv('shap_importance_avg.csv', index=False)
print(shap_df)




#end of multiple results

#simple visualization SHAP vs Original Image
shap_img = shap_values[0][0]     
shap_img = np.squeeze(shap_values[0])  
shap_img = np.sum(shap_img, axis=-1)  
shap_img = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min())
shap.image_plot([shap_img], img_array[0])
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_array[0], cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(img_array[0], cmap='gray', alpha=0.6)  
axes[1].imshow(shap_img, cmap='hot', alpha=0.4)  
axes[1].set_title('SHAP Explanation')
axes[1].axis('off')
plt.tight_layout()
plt.show()

#end of shap explainer



# optional feature maps craetion
img_array = preprocess_image(im_path)
layer_names = ['conv2d', 'conv2d_1', 'conv2d_2']
layer_outputs = [loaded_model.get_layer(name).output for name in layer_names]
activation_model = Model(inputs=loaded_model.input, outputs=layer_outputs)
activations = activation_model.predict(img_array)


def plot_feature_maps(activation, title, max_columns=4, top_filters=16):
    mean_activations = np.mean(activation[0], axis=(0, 1)) 
    top_indices = np.argsort(mean_activations)[-top_filters:][::-1] 
    n_columns = min(max_columns, top_filters)
    n_rows = np.ceil(top_filters / n_columns).astype(int)
    fig, axs = plt.subplots(n_rows, n_columns, figsize=(n_columns * 2, n_rows * 2))
    for i, idx in enumerate(top_indices):
        row = i // n_columns
        col = i % n_columns
        axs[row, col].imshow(activation[0, :, :, idx], cmap='viridis')
        axs[row, col].axis('off')
    for i in range(len(top_indices), n_rows * n_columns):
        axs.flat[i].axis('off')
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()
    
for layer_name, activation in zip(layer_names, activations):
    print(f"Visualizing feature maps for layer: {layer_name}")
    plot_feature_maps(activation, layer_name)
#end of feature maps



# example image visualised in grids
img_array = preprocess_image(im_path)

regions = divide_into_grid(img_array[0], grid_size=(3, 3))


fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for idx, region in enumerate(regions):
    row, col = divmod(idx, 3)
    axes[row, col].imshow(region)
    axes[row, col].axis('off')
    axes[row, col].set_title(f'Region_{idx}')

plt.tight_layout()
plt.show()


#function to extract feature for a specific region

def extract_features_for_region(region, intermediate_layer_model, target_size=(150,150)):
    region_resized = tf.image.resize(region, target_size)
    region_input = np.expand_dims(region_resized, axis=0)
    features = intermediate_layer_model(region_input)    
    return features

#function to extract features from a specific layer for every region of the image
def extract_features_for_regions(regions, model, feature_layer_name):
    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(feature_layer_name).output)
    region_features = {}
    for idx, region in enumerate(regions):
        features = extract_features_for_region(region, intermediate_layer_model)
        region_name = f'region_{idx}' 
        region_features[region_name] = features
    return region_features


def visualize_region_features(region_features, num_features=10):
    for region_name, features in region_features.items():
        feature_map = features[0].numpy()  
        fig, axes = plt.subplots(1, num_features, figsize=(15, 5))
        for i in range(min(num_features, feature_map.shape[-1])):  
            ax = axes[i]
            ax.imshow(feature_map[:, :, i], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Feature {i+1}')
        
        plt.tight_layout()
        plt.show()


region_features = extract_features_for_regions(regions, loaded_model, last_conv_layer)
visualize_region_features(region_features, num_features=16)


#creating a sample of 30 batches as this is the number of batches that memory isn't maxed out
num_batches = 30
sample_images = []
sample_labels = []

for batch_idx, (images, labels) in enumerate(train_generator):
    if batch_idx >= num_batches:
        break  

    sample_images.append(images)
    sample_labels.extend(labels)  


sample_images = np.concatenate(sample_images, axis=0)  
sample_labels = np.array(sample_labels)  

# Save the sample dataset for reuse in later steps
np.save('sample_images.npy', sample_images)
np.save('sample_labels.npy', sample_labels)

unique_labels, counts = np.unique(sample_labels, return_counts=True)


print("Label distribution in the sample dataset:")
for label, count in zip(unique_labels, counts):
    print(f"Label {label}: {count} samples")

#distribution of the sample
plt.bar(unique_labels, counts)
plt.xlabel('Labels')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Sample Dataset')
plt.xticks(unique_labels)
plt.show()

#load the sample batches
sample_labels = np.load('sample_labels.npy')  
sample_images = np.load('sample_images.npy')  
print(sample_images.shape)

#creating the PCA model for universal usage

def extract_features_for_pca(sample_images, model, feature_layer_name):
    feature_vectors = []
    for img in sample_images:
        regions = divide_into_grid(img, grid_size=(3, 3))
        region_features = extract_features_for_regions(regions, model, feature_layer_name)
        for region_name, features in region_features.items():
            pooled_features = tf.reduce_mean(features, axis=[1, 2])  
            feature_vectors.append(pooled_features.numpy().flatten())  
    return np.array(feature_vectors)
def create_pca_model(sample_images, model, feature_layer_name, num_components=100):
    sample_feature_vectors = extract_features_for_pca(sample_images, model, feature_layer_name)
    pca_model = PCA(n_components=num_components, random_state=42)
    pca_model.fit(sample_feature_vectors) 
    return pca_model
pca_model = create_pca_model(sample_images, loaded_model, last_conv_layer)

#saving and loading the PCA model
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca_model, f)
with open('pca_model.pkl', 'rb') as f:
    pca_model = pickle.load(f)

#checking dimenions 
sample_feature_vectors = extract_features_for_pca(sample_images, loaded_model, last_conv_layer)
transformed_features = pca_model.transform(sample_feature_vectors)
print(transformed_features.shape)




#prototype creation
 
def create_prototypes_with_kmeans(sample_images, model, feature_layer_name, pca_model, grid_size=(3, 3), num_clusters=1):
    region_features_accum = {f"region_{i}": [] for i in range(grid_size[0] * grid_size[1])}
    for img in sample_images:
        preprocessed_img = img / 255.0  
        regions = divide_into_grid(preprocessed_img, grid_size)  
        region_features = extract_features_for_regions(regions, model, feature_layer_name)
        for region_idx, (region_name, features) in enumerate(region_features.items()):
            if features.shape[-1] == 128: 
                pooled_features = tf.reduce_mean(features, axis=[1, 2])  
                features_np = pooled_features.numpy().flatten() 
                region_features_accum[region_name].append(features_np)
    prototypes = {}
    for region_name, region_features_list in region_features_accum.items():
        region_features_array = np.array(region_features_list)
        reduced_features = pca_model.transform(region_features_array)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(reduced_features)
        prototypes[region_name] = kmeans.cluster_centers_
        print(f"Prototype for {region_name} shape: {kmeans.cluster_centers_.shape}")
        print(f"Prototype for {region_name}: {kmeans.cluster_centers_}")
    return prototypes


prototypes = create_prototypes_with_kmeans(sample_images, loaded_model, last_conv_layer, pca_model)

#saving the prototypes

with open('prototypes.pkl', 'wb') as f:
    pickle.dump(prototypes, f)

#loading the prototypes
with open('prototypes.pkl', 'rb') as f:
    prototypes = pickle.load(f)
print(prototypes)


# match with cosine similarity
def match_to_prototypes(image, model, prototypes, feature_layer_name,pca_model, grid_size=(3, 3), target_size=(150, 150)):
    regions = divide_into_grid(image, grid_size)
    region_features = extract_features_for_regions(regions, model, feature_layer_name)
    matches = {}
    for region_name, features in region_features.items():
        pooled_features = tf.reduce_mean(features, axis=[1, 2])
        features_np = pooled_features.numpy().flatten().reshape(1, -1)
        features_reduced = pca_model.transform(features_np)
        prototype_np = prototypes[region_name].reshape(1, -1)
        similarity = cosine_similarity(features_reduced, prototype_np)
        matches[region_name] = similarity[0][0]   
    return matches

# Example 
new_image = img_array[0]
matches = match_to_prototypes(new_image, loaded_model, prototypes, last_conv_layer, pca_model)

print(matches)



#BN dataset creation

feature_model = Model(inputs=loaded_model.input, outputs=loaded_model.get_layer(last_conv_layer).output)
similarity_scores_list = []
for img in sample_images:
    scores = match_to_prototypes(img, loaded_model, prototypes, last_conv_layer, pca_model)
    similarity_scores_list.append(scores)
bn_df = pd.DataFrame(similarity_scores_list)
print(bn_df)


predictions = loaded_model.predict(sample_images)
print(predictions.shape)
predicted_labels = (predictions >= 0.5).astype(int).flatten()
print(predicted_labels)
print(predicted_labels.shape)



bn_df['label'] = predicted_labels
bn_df.to_csv('bn_df.csv', index=False)
bn_df = pd.read_csv('bn_df.csv')
print("BN DataFrame with Similarity Scores:\n", bn_df.head())


# Check the DataFrame's dimensions
print(f"DataFrame shape: {bn_df.shape}")

similarity_stats = bn_df.describe()
print(similarity_stats)


fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()

for region_idx in range(9):
    axes[region_idx].hist(bn_df[f'region_{region_idx}'], bins=20, alpha=0.7, color='blue')
    axes[region_idx].set_title(f'Region {region_idx} Similarity Scores')
    axes[region_idx].set_xlabel('Cosine Similarity')
    axes[region_idx].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

for region in bn_df.columns[:-1]:  
    bn_df[region] = pd.qcut(bn_df[region], q=3, labels=['low', 'medium', 'high'])

# Display the first few rows of the discretized DataFrame
print("Discretized BN DataFrame:\n", bn_df.head())

hc = HillClimbSearch(bn_df)
model = hc.estimate(scoring_method=K2Score(bn_df))
bn_model = BayesianNetwork(model.edges())
bn_model.fit(bn_df, estimator=MaximumLikelihoodEstimator)



# Save the Bayesian Network model
with open('bayesian_network_model.pkl', 'wb') as file:
    pickle.dump(bn_model, file)
# load the bn model   
with open('bayesian_network_model.pkl', 'rb') as file:
    bn_model = pickle.load(file)
# Print the learned network structure
print("Learned Network Structure:", bn_model.edges())


#Plotting the BN

G = nx.DiGraph()
G.add_nodes_from(bn_model.nodes())
G.add_edges_from(bn_model.edges())

# Draw the network
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=12, font_weight='bold', arrowsize=20)
plt.title("Bayesian Network Structure")
plt.show()
#end of plot

for node in bn_model.nodes():
    parents = bn_model.get_parents(node)
    print(f"Parents of node '{node}': {parents}")

for cpd in bn_model.get_cpds():
    print(f"CPD of {cpd.variable}:")
    print(cpd)
    print("\n")



marginals = bn_model.predict_probability(pd.DataFrame([{}]))  
print("Marginal probabilities for each node:")
print(marginals)


def generate_counterfactuals(bn, region_nodes):
    counterfactual_results = []
    original_probabilities = bn.predict_probability(pd.DataFrame([{}]))  
    original_label_prob = original_probabilities['label_1'][0] 
    for region in region_nodes:
        original_probabilities_region = bn.predict_probability(pd.DataFrame([{}]))
        original_region_state = original_probabilities_region.filter(like=region).idxmax(axis=1).values[0]
        possible_states = ['low', 'medium', 'high']
        altered_states = [state for state in possible_states if state != original_region_state.split('_')[-1]]
        for altered_state in altered_states:
            new_evidence = {region: altered_state}
            new_probabilities = bn.predict_probability(pd.DataFrame([new_evidence]))
            new_label_prob = new_probabilities['label_1'][0]
            counterfactual_results.append({
                'Counterfactual': f"{region}_to_{altered_state}",
                'Changed Region': region,
                'Change (From -->To)': f"{original_region_state.split('_')[-1]} --> {altered_state}",
                'Original Prob (label_1)': original_label_prob,
                'New Prob (label_1)': new_label_prob,
                'Change in Prob (%)': (new_label_prob - original_label_prob) * 100
            })
    counterfactual_summary = pd.DataFrame(counterfactual_results)
    return counterfactual_summary


region_nodes = ['region_0', 'region_1', 'region_2', 'region_3', 'region_4', 'region_5', 'region_6', 'region_7', 'region_8']
counterfactual_summary = generate_counterfactuals(bn_model, region_nodes)

print(counterfactual_summary)

counterfactual_summary.to_csv('counterfactual_summary.csv', index=False)
print("Results saved to 'counterfactual_summary.csv'")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

counterfactual_summary =  pd.read_csv('counterfactual_summary.csv')
print(counterfactual_summary)





    

 
    





