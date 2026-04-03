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

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Ce code suppose que l'objet 'visualizer' est déjà défini dans votre environnement
# img_id_test = '003' 
# img_brute = visualizer.load_data(img_id_test, 'Image')

# tailles_kernel = [10, 15, 25, 35, 51]

# fig, axes = plt.subplots(1, 6, figsize=(25, 5))
# axes[0].imshow(img_brute, cmap='gray')
# axes[0].set_title('Image Brute')
# axes[0].axis('off')

# for idx, k_size in enumerate(tailles_kernel):

#     # 1. Remplacement de MORPH_RECT par MORPH_ELLIPSE pour respecter la forme des noyaux
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
#     # 2. Utilisation de l'opération native Black Top-Hat d'OpenCV
#     # Cela remplace mathématiquement et plus efficacement : cv2.subtract(cv2.morphologyEx(..., MORPH_CLOSE), img)
#     blackhat = cv2.morphologyEx(img_brute, cv2.MORPH_BLACKHAT, kernel)

#     # 3. Normalisation (étirement de l'histogramme) pour révéler les faibles variations d'intensité
#     blackhat_norm = cv2.normalize(blackhat, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

#     # Affichage
#     ax = axes[idx + 1]
#     ax.imshow(blackhat_norm, cmap='gray')
#     ax.set_title(f'Blackhat : {k_size}x{k_size}')
#     ax.axis('off')

# plt.tight_layout()
# plt.show()