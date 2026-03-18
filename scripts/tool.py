import matplotlib.pyplot as plt
import cv2
from pathlib import Path

import numpy as np
from cellpose import metrics, utils

import os, shutil, owncloud
from datetime import datetime

class CellVisualizer:
    def __init__(self, 
                 img_dir='./data/1-Images/01-Data/', 
                 ann_dir='./data/2-Annotations/02-Annotations/',
                 nuc_dir = './data/2-Annotations/02-Annotations_Clean/',
                 proc_dir='./data/processed/'): # Ajout du dossier processed
        
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.nuc_dir = Path(nuc_dir)
        self.proc_dir = Path(proc_dir)

    def _get_path(self, folder, pattern):
        return next(folder.rglob(pattern))
    

    def load_data(self, id_unique, mode='Image'):
        if mode == 'Cell':
            pattern = f"{id_unique}*Cell.bmp"
            p = self._get_path(self.ann_dir, pattern)
            
        elif mode == 'Nuc':
            pattern = f"{id_unique}*Nuc.bmp"
            p = self._get_path(self.nuc_dir, pattern)
            
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

        # 1. Normalisation du fond pour qu'il soit bien visible
        img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if img_8u.ndim == 2:
            overlay = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2RGB)
        else:
            overlay = img_8u.copy()

        # 2. Traceur de cellules Infaillible
        def draw_individual_cells(mask, color, thickness=1):
            if mask is None: return
            
            # SÉCURITÉ 1 : Si c'est du RGB, on fusionne tout pour ne perdre aucune cellule
            if mask.ndim == 3: 
                mask = np.max(mask, axis=2)
                
            for cell_id in np.unique(mask):
                if cell_id == 0: continue
                
                # SÉCURITÉ 2 : On force OpenCV avec un vrai blanc (255)
                single_cell = np.uint8((mask == cell_id) * 255)
                
                contours, _ = cv2.findContours(single_cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, thickness)

        # 3. Tracés
        draw_individual_cells(mask_true, color=(0, 255, 0), thickness=2) # Vert = Expert
        draw_individual_cells(mask_pred, color=(255, 0, 0), thickness=2) # Rouge = Modèle

        # 4. Affichage
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,8))

        ax.imshow(overlay)
        ax.set_title(f"Overlay {i} : Expert (Vert) vs Pred (Rouge)")
        ax.axis('off')
        
        if show: plt.show()
        return ax
    
    def plot_loss(self, t_loss, v_loss, title="Courbe de Loss", ax=None, show=True):
        """Trace les courbes d'apprentissage (Train vs Val)"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 3))
            
        ax.plot(t_loss, label='Train Loss', color='blue', marker='o', markersize=4)
        ax.plot(v_loss, label='Test Loss (Val)', color='orange', marker='s', markersize=4)
        
        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
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

    def download_path(self, local_path, cloud_dest):
            """Zippe un dossier local et l'envoie sur le Cloud"""
            # 1. On définit le nom du zip à partir du dossier
            name = os.path.basename(local_path.rstrip('/'))
            
            # 2. On zippe
            shutil.make_archive(name, 'zip', local_path)
            
            # 3. On s'assure que le dossier de destination existe sur le Cloud
            try: self.oc.mkdir(os.path.dirname(cloud_dest))
            except: pass
            
            # 4. Envoi (ta commande fétiche) et nettoyage
            self.oc.put_file(f"{cloud_dest}.zip", f"{name}.zip")
            os.remove(f"{name}.zip")
            
            print(f"✅ Dossier {name} envoyé vers {cloud_dest}.zip")


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