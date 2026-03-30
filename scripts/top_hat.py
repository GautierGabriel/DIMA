import cv2
import numpy as np
import matplotlib.pyplot as plt

img_id_test = '003' 
img_brute = visualizer.load_data(img_id_test, 'Image')
img_norm = cv2.normalize(img_brute, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 2. Définition des tailles à tester (en pixels)
# Les tailles doivent être des nombres impairs
tailles_kernel = [10, 15, 25, 35, 51]

# Création de la figure
fig, axes = plt.subplots(1, 6, figsize=(25, 5))
axes[0].imshow(img_norm, cmap='gray')
axes[0].set_title('Image Brute')
axes[0].axis('off')

# 3. Boucle d'application du Black Top-Hat
for idx, k_size in enumerate(tailles_kernel):
    # Création de l'élément structurant elliptique
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
    # Application de la morphologie (Black-Hat pour les creux sombres)
    creux = cv2.morphologyEx(img_norm, cv2.MORPH_BLACKHAT, kernel)
    
    # Normalisation pour l'affichage
    creux_norm = cv2.normalize(creux, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Affichage
    ax = axes[idx + 1]
    ax.imshow(creux_norm, cmap='gray')
    ax.set_title(f'Kernel : {k_size}x{k_size}')
    ax.axis('off')

plt.tight_layout()
plt.show()