import matplotlib.pyplot as plt
import cv2
from pathlib import Path

import numpy as np
from cellpose import metrics

import os, shutil, owncloud
from datetime import datetime

class CellVisualizer:
    def __init__(self, 
                 img_dir='./data/1-Images/01-Data/', 
                 ann_dir='./data/2-Annotations/02-Annotations/',
                 proc_dir='./data/processed/'): # Ajout du dossier processed
        
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.proc_dir = Path(proc_dir)

    def _get_path(self, folder, pattern):
        return next(folder.rglob(pattern))
    

    def load_data(self, id_unique, mode='Image'):
        if mode == 'Cell':
            pattern = f"{id_unique}*Cell.bmp"
            p = self._get_path(self.ann_dir, pattern)
            
        elif mode == 'Nuc':
            pattern = f"{id_unique}*Nuc.bmp"
            p = self._get_path(self.ann_dir, pattern)
            
        elif mode == 'Cell_Pred':
            pattern = f"{id_unique}*Cell_pred.bmp"
            p = self._get_path(self.proc_dir, pattern)
            
        elif mode == 'Nuc_Pred':
            pattern = f"{id_unique}*Nuc_pred.bmp"
            p = self._get_path(self.proc_dir, pattern)

        else: # Image
            pattern = f"{id_unique}*"
            p = self._get_path(self.img_dir, pattern)
        
        return cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p else None
    
    def plot(self, i=None, data=None, mode='Image', show=True, ax=None, title=None):
        """
        Affiche ou retourne un graphique les images et les masques
        """
        
        if data is not None:
            img = data
        elif i is not None:
            img = self.load_data(i, mode=mode)

        if mode.lower() == 'image':
            cmap = 'gray'
            default_title = f"Image : {i}" if i is not None else "Image"
        else:
            cmap = 'nipy_spectral'
            default_title = f"Cellules : {i}" if i is not None else "Cellules"

        final_title = title if title is not None else default_title

        # 4. Gérer l'objet Axes
        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(img, cmap=cmap)
        ax.set_title(final_title)
        
        if show:
            plt.show()

        return ax
    
    def plot_overlay(self, i, mask_pred, ax=None, show=True):
            img = self.load_data(i, mode='Image')
            mask_true = self.load_data(i, mode='Cell')
            if img.ndim == 3:
                overlay = img.copy()
            else:
                overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            def to_cv2_bin(m):
                if m is None: return None
                m_bin = (m > 0).astype(np.uint8)
                if m_bin.ndim == 3:
                    m_bin = m_bin[:,:,0]
                return m_bin

            bin_true = to_cv2_bin(mask_true)
            bin_pred = to_cv2_bin(mask_pred)

            if bin_true is not None:
                contours_true, _ = cv2.findContours(bin_true, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours_true, -1, (0, 255, 0), 2) # Vert = Expert

            if bin_pred is not None:
                contours_pred, _ = cv2.findContours(bin_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours_pred, -1, (255, 0, 0), 2) # Rouge = Modèle
        
            if ax is None:
                fig, ax = plt.subplots(figsize=(8,8))

            ax.imshow(overlay)
            ax.set_title(f"Overlay {i} : Expert (Vert) vs Pred (Rouge)")
            ax.axis('off')
            
            if show: plt.show()
            return ax

class Owncloud:
    def __init__(self, client):
        self.oc = client 
    def upload_data(self):
        self.oc.get_directory_as_zip('/travail/Mines/DIMA/Segmentation/data', 'data.zip')
        os.system('unzip -o -q data.zip -d . && rm data.zip')

    def upload_scripts(self):
        self.oc.get_directory_as_zip('/travail/Mines/DIMA/Segmentation/scripts', 'scripts.zip')
        os.system('unzip -o -q scripts.zip -d . && rm scripts.zip')

    def download_results(self):
        tag = datetime.now().strftime("%Y-%m-%d_%Hh%M")
        name = f"resultats_{tag}"
        shutil.make_archive(name, 'zip', './data/processed/')
        path = '/travail/Mines/DIMA/Segmentation/data/processed/'
        try: self.oc.mkdir(path)
        except: pass
        self.oc.put_file(f"{path}{name}.zip", f"{name}.zip")
        os.remove(f"{name}.zip")
        print(f"Envoyé: {name}.zip")


class Stats:
    def summary(self, mask):
        ids, counts = np.unique(mask[mask > 0], return_counts=True)
        n = len(ids)
        return {"n": n, "area_total": int(np.sum(counts)), "area_mean": float(np.mean(counts)) if n>0 else 0}

    def summary_perf(self, mask_true, mask_pred, iou_threshold=0.5):
        
        if mask_true.ndim == 3: mask_true = mask_true[:,:,0]
        if mask_pred.ndim == 3: mask_pred = mask_pred[:,:,0]
        
        mask_true = mask_true.astype(np.uint32)
        mask_pred = mask_pred.astype(np.uint32)

        for m in [mask_true, mask_pred]: 
            if m is not None:
                vals = np.unique(m)
                # On crée une map : l'ancienne valeur devient son rang dans la liste
                for i, v in enumerate(vals):
                    m[m == v] = i

        ap, tp, fp, fn = metrics.average_precision(mask_true, mask_pred, threshold=iou_threshold)
        
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        return {"f1": float(f1), "ap": float(ap), "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    def report(self, name, mask_true, mask_pred):
        res = self.summary_perf(mask_true, mask_pred)
        print(f"[{name}] F1: {res['f1']:.3f} | TP: {res['tp']} | FN: {res['fn']}")
        return res