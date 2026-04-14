# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# img_id_test = '003' 
# img_brute = visualizer.load_data(img_id_test, 'Image')
# img_norm = cv2.normalize(img_brute, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# # 2. Définition des tailles à tester (en pixels)
# # Les tailles doivent être des tnombres impairs
# tailles_kernel = [10, 15, 25, 35, 51]

# # Création de la figure
# fig, axes = plt.subplots(1, 6, figsize=(25, 5))
# axes[0].imshow(img_norm, cmap='gray')
# axes[0].set_title('Image Brute')
# axes[0].axis('off')

# # 3. Boucle d'application du Black Top-Hat
# for idx, k_size in enumerate(tailles_kernel):
#     # Création de l'élément structurant elliptique
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
#     # Application de la morphologie (Black-Hat pour les creux sombres)
#     creux = cv2.morphologyEx(img_norm, cv2.MORPH_CLOSE, kernel)
    
#     black_hat = cv2.subtract(creux, img_norm)

#     creux_norm = cv2.normalize(black_hat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
#     # Affichage
#     ax = axes[idx + 1]
#     ax.imshow(creux_norm, cmap='gray')
#     ax.set_title(f'Kernel : {k_size}x{k_size}')
#     ax.axis('off')

# plt.tight_layout()
# plt.show()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# img_id_test = '003' 
# img_brute = visualizer.load_data(img_id_test, 'Image')

# tailles_kernel = [10, 15, 25, 35, 51]

# fig, axes = plt.subplots(1, 6, figsize=(25, 5))
# axes[0].imshow(img_brute, cmap='gray')
# axes[0].set_title('Image Brute')
# axes[0].axis('off')

# for idx, k_size in enumerate(tailles_kernel):

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
#     blackhat = cv2.morphologyEx(img_brute, cv2.MORPH_BLACKHAT, kernel)

#     blackhat_norm = cv2.normalize(blackhat, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

#     ax = axes[idx + 1]
#     ax.imshow(blackhat_norm, cmap='gray')
#     ax.set_title(f'Blackhat : {k_size}x{k_size}')
#     ax.axis('off')

# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import morphology, exposure, util

# img_brute = visualizer.load_data(img_id_test, 'Image')

# img_float = util.img_as_float(img_brute)

# rayons = [5, 7, 12, 17, 25]

# fig, axes = plt.subplots(1, 6, figsize=(25, 5))
# axes[0].imshow(img_float, cmap='gray')
# axes[0].set_title('Image Brute')
# axes[0].axis('off')

# for idx, r in enumerate(rayons):
#     footprint = morphology.disk(r)
    
#     creux = morphology.black_tophat(img_float, footprint)
    
#     creux_norm = exposure.rescale_intensity(creux, out_range=(0, 1))
    
#     ax = axes[idx + 1]
#     ax.imshow(creux_norm, cmap='gray')
#     ax.set_title(f'Rayon : {r} (Disk)')
#     ax.axis('off')

# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, exposure, util

img_id_test = '001' 
img_brute = visualizer.load_data(img_id_test, 'Image')

img_float = util.img_as_float(img_brute)

area = [500, 600]

fig, axes = plt.subplots(1, 3, figsize=(25, 5))
axes[0].imshow(img_float, cmap='gray')
axes[0].set_title('Image Brute')
axes[0].axis('off')

for idx, r in enumerate(area):
    
    creux_norm = morphology.area_closing(img_float, area_threshold=r)

    creux_norm = creux_norm - img_float

    ax = axes[idx + 1]
    ax.imshow(creux_norm, cmap='gray')
    ax.set_title(f'Area : {r}')
    ax.axis('off')

plt.tight_layout()
plt.show()