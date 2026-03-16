import matplotlib.pyplot as plt
import cv2
from pathlib import Path

class CellVisualizer:
    def __init__(self, img_dir='./data/1-Images/01-Data/', ann_dir='./data/2-Annotations/02-Annotations/'):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)

    def _get_path(self, folder, pattern):
        return next(folder.glob(pattern))

    def load_data(self, id_unique, mode='Image'):
        """Charge les données pour un ID donné"""
        if mode == 'Cell':
            pattern = f"{id_unique}*Cell.bmp"
            p = self._get_path(self.ann_dir, pattern)
        
        elif mode == 'Nuc':
            pattern = f"{id_unique}*Nuc.bmp"
            p = self._get_path(self.ann_dir, pattern)

        else :
            pattern = f"{id_unique}*"
            p = self._get_path(self.img_dir, pattern)

        
        p = cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p else None
        
        return p
    
    def plot(self, i, mode='Image'):
    
        img = self.load_data(i, mode=mode)

        if mode == 'Image':
            plt.imshow(img, cmap='gray')
            plt.title(f"Image : {i}")
            plt.show()
        
        else:
            plt.imshow(img, cmap='nipy_spectral')
            plt.title(f"Cellules : {i}")
            plt.show()
