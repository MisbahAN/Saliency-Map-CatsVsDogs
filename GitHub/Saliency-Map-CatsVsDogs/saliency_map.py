import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

testing_dog = "images/test/dog"

# Load the trained model (assumes model has been trained and saved from binary_classification_model.py)
model = tf.keras.models.load_model('output/cats_vs_dogs_model.h5')
print("Model loaded successfully!")

#loading the image and converting it to numpy array
img = tf.keras.preprocessing.image.load_img('images/test/dog/'+ random.sample(os.listdir(testing_dog), 1)[0],target_size=(300,300))
x = img_to_array(img)  # Numpy array with shape (300, 300, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 300, 300, 3)

# Normalize input
x = x / 255.0

# Get prediction
prediction = model.predict(x)
pred_class = "Cat" if prediction[0][0] > 0.5 else "Dog"
confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
print(f"Prediction: {pred_class} (confidence: {confidence:.3f})")

# Simple gradient-based saliency map
def compute_saliency_map(model, image):
    """Compute a simple gradient-based saliency map"""
    image_tensor = tf.convert_to_tensor(image)
    
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model(image_tensor)
        score = predictions[0][0]
    
    # Compute gradients
    gradients = tape.gradient(score, image_tensor)
    
    # Take the maximum across color channels for visualization
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
    
    # Normalize to [0, 1] range
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency))
    
    return saliency.numpy()

# Generate saliency map
print("Generating saliency map...")
saliency_map = compute_saliency_map(model, x)

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
original_img = img_to_array(img) / 255.0
axes[0].imshow(original_img)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Saliency map
im1 = axes[1].imshow(saliency_map[0], cmap='hot', alpha=0.8)
axes[1].set_title('Saliency Map')
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# Overlay
axes[2].imshow(original_img)
axes[2].imshow(saliency_map[0], cmap='hot', alpha=0.4)
axes[2].set_title(f'Overlay\\nPrediction: {pred_class} ({confidence:.3f})')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('output/saliency_visualization.png', dpi=300, bbox_inches='tight')
print("Saliency map saved as 'output/saliency_visualization.png'")