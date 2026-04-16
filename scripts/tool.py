import matplotlib.pyplot as plt
import cv2
from pathlib import Path

import numpy as np
from cellpose import metrics, utils

import os, shutil, owncloud
from datetime import datetime
from scipy import ndimage
from skimage import morphology, exposure, util
import pandas as pd

class CellVisualizer:
    def __init__(self, 
                 img_dir='./data/1-Images/01-Data/', 
                 ann_dir='./data/2-Annotations/02-Annotations/',
                 nuc_dir = './data/02-Annotations_Clean/',
                 nuc_pred_dir='./data/prediction/cellpose_nuclei_200_train/masks/',
                 cell_pred_dir='./data/prediction/cellpose4_50/'):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.nuc_dir = Path(nuc_dir)
        self.nuc_pred_dir = Path(nuc_pred_dir)
        self.cell_pred_dir = Path(cell_pred_dir)

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
            p = self._get_path(self.cell_pred_dir, pattern)
            
        elif mode == 'Nuc_Pred':
            pattern = f"{id_unique}*Nuc_pred.bmp"
            p = self._get_path(self.nuc_pred_dir, pattern)

        else: # Image
            pattern = f"{id_unique}*"
            p = self._get_path(self.img_dir, pattern)
        
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None: return None

        # SI C'EST UN MASQUE ET QU'IL EST EN COULEUR (RGB)
        if mode in ['Cell', 'Nuc', 'Cell_Pred', 'Nuc_Pred'] and img.ndim == 3:
            # On transforme le RGB en un ID unique de 24 bits (R + G*256 + B*256^2)
            # C'est la méthode la plus sûre pour ne perdre aucune cellule BMP
            label_img = img[:,:,0].astype(np.int32) + \
                        img[:,:,1].astype(np.int32) * 256 + \
                        img[:,:,2].astype(np.int32) * 256**2
            
            # On re-mappe ces IDs géants vers des labels simples (0, 1, 2, 3...)
            unique_ids = np.unique(label_img)
            new_mask = np.zeros(label_img.shape, dtype=np.uint16)
            for i, val in enumerate(unique_ids):
                if val == 0: continue # Fond
                new_mask[label_img == val] = i
            return new_mask
            
        return img

    def load_channel(self, ids, mode='raw'):
        """
        Prépare les données multi-canaux (2, H, W) pour Cellpose.
        Modes disponibles : 
        - 'raw' : Duplication de l'image biphoton.
        - 'clahe' : Amélioration de contraste locale sur le second canal.
        - 'Nuc_Pred' : Utilisation des masques de noyaux prédits transformés en EDT.
        """
        X = []
        Y = [self.load_data(i, 'Cell') for i in ids]
        
        for i in ids:
            img = self.load_data(i, 'Image')
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            if mode == 'raw':
                ch2 = img_norm
                
            elif mode == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                ch2 = clahe.apply(img_norm)

            elif mode == 'top_hat':
                creux_norm = morphology.area_closing(img_norm, area_threshold=500)

                ch2 = creux_norm - img_norm
                
            elif mode == 'Nuc_Pred':
                nuc_mask = self.load_data(i, mode='Nuc_Pred')
                binary_nuc = (nuc_mask > 0).astype(np.uint8)
                
                if binary_nuc.max() > 0:
                    edt = ndimage.distance_transform_edt(binary_nuc)
                    ch2 = cv2.normalize(edt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else:
                    ch2 = np.zeros_like(img_norm)
        
            X.append(np.stack([img_norm, ch2], axis=0))
            
        return X, Y

    def load_set(self, ids, mode='Cell'):
        X = []
        Y = []
        
        for i in ids:
            # Chargement et normalisation de l'image
            img = self.load_data(i, 'Image')
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            X.append(img_norm)
            
            # Chargement du masque correspondant
            mask = self.load_data(i, mode)
            Y.append(mask)
            
        return X, Y
  

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
    
    # def plot_overlay(self, i, mask_pred, ax=None, show=True):
    #     img = self.load_data(i, mode='Image')
    #     mask_true = self.load_data(i, mode='Cell')

    #     # 1. Normalisation du fond pour qu'il soit bien visible
    #     img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #     if img_8u.ndim == 2:
    #         overlay = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2RGB)
    #     else:
    #         overlay = img_8u.copy()

    #     # 2. Traceur de cellules Infaillible
    #     def draw_individual_cells(mask, color, thickness=1):
    #         if mask is None: return
            
    #         # SÉCURITÉ 1 : Si c'est du RGB, on fusionne tout pour ne perdre aucune cellule
    #         if mask.ndim == 3: 
    #             mask = np.max(mask, axis=2)
                
    #         for cell_id in np.unique(mask):
    #             if cell_id == 0: continue
                
    #             # SÉCURITÉ 2 : On force OpenCV avec un vrai blanc (255)
    #             single_cell = np.uint8((mask == cell_id) * 255)
                
    #             contours, _ = cv2.findContours(single_cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #             cv2.drawContours(overlay, contours, -1, color, thickness)

    #     # 3. Tracés
    #     draw_individual_cells(mask_true, color=(0, 255, 0), thickness=2) # Vert = Expert
    #     draw_individual_cells(mask_pred, color=(255, 0, 0), thickness=2) # Rouge = Modèle

    #     # 4. Affichage
    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=(8,8))

    #     ax.imshow(overlay)
    #     ax.set_title(f"Overlay {i} : Expert (Vert) vs Pred (Rouge)")
    #     ax.axis('off')
        
    #     if show: plt.show()
    #     return ax

    def plot_overlay(self, i, mask_pred, ax=None, show=True, mode='Cell'):
        img = self.load_data(i, mode='Image')
        mask_true = self.load_data(i, mode=mode)

        img_rgb = cv2.cvtColor(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLOR_GRAY2RGB)

        def add_contours(overlay, mask, color):
            for l in np.unique(mask):
                if l == 0: continue
                contours, _ = cv2.findContours((mask==l).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 1)

        overlay = img_rgb.copy()
        add_contours(overlay, mask_true, (0, 255, 0))
        add_contours(overlay, mask_pred, (255, 0, 0)) 

        if ax is None: fig, ax = plt.subplots()
        ax.imshow(overlay)
        ax.set_title(f"Overlay {i} : Expert (Vert) vs Pred (Rouge)")
        ax.axis('off')
        if show: plt.show()
    
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

    def upload_models(self):
        self.oc.get_directory_as_zip('/travail/Mines/DIMA/Segmentation/results/models', 'models.zip')
        os.system('unzip -o -q models.zip -d . && rm models.zip')

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
            name = os.path.basename(local_path.rstrip('/'))
            
            shutil.make_archive(name, 'zip', local_path)
            
            try: self.oc.mkdir(os.path.dirname(cloud_dest))
            except: pass
            
            self.oc.put_file(f"{cloud_dest}.zip", f"{name}.zip")
            os.remove(f"{name}.zip")
            
            print(f"Dossier {name} envoyé vers {cloud_dest}.zip")


class Stats:
    def summary(self, mask):
        ids, counts = np.unique(mask[mask > 0], return_counts=True)
        n = len(ids)
        return {"n": n, "area_total": int(np.sum(counts)), "area_mean": float(np.mean(counts)) if n>0 else 0}

    def summary_perf(self, mask_true, mask_pred, iou_threshold=0.5):
        mask_true = np.array(mask_true).astype(np.uint32)
        mask_pred = np.array(mask_pred).astype(np.uint32)

        def relabel(m):
            unique_labels = np.unique(m)
            new_m = np.zeros_like(m)
            for i, v in enumerate(unique_labels):
                if v == 0: continue
                new_m[m == v] = i
            return new_m

        mask_true = relabel(mask_true)
        mask_pred = relabel(mask_pred)

        ap, tp, fp, fn = metrics.average_precision(mask_true, mask_pred, threshold=iou_threshold)
        
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        return {"f1": float(f1), "ap": float(ap), "tp": int(tp), "fp": int(fp), "fn": int(fn)}
    
    def compute_points_metrics(image_name, mask_pred, df_annotations):
        """
        Calcule les métriques pour une image donnée.
        
        image_name : str, nom du fichier image tel qu'inscrit dans le CSV
        mask_pred : np.array, masque de segmentation (labels uniques par noyau)
        df_annotations : pd.DataFrame, le contenu complet du fichier CSV ImageJ
        """
        # 1. Extraction des points pour cette image spécifique
        points = df_annotations[df_annotations['Image'] == image_name][['X', 'Y']].values
        
        if len(points) == 0:
            num_preds = len(np.unique(mask_pred)) - 1
            return {"tp": 0, "fp": num_preds, "fn": 0, "f1": 0.0}

        # 2. Identification des TP et FN
        tp = 0
        fn = 0
        labels_detected = set()
        
        for x, y in points:
            # Conversion coordonnées ImageJ (float) vers indices NumPy (int)
            # Note : X = colonne, Y = ligne
            row, col = int(round(y)), int(round(x))
            
            # Vérification des limites de l'image
            if 0 <= row < mask_pred.shape[0] and 0 <= col < mask_pred.shape[1]:
                label = mask_pred[row, col]
                if label > 0:
                    if label not in labels_detected:
                        tp += 1
                        labels_detected.add(label)
                else:
                    fn += 1
            else:
                fn += 1 # Point hors cadre considéré comme non détecté

        # 3. Identification des FP
        # Total des objets prédits moins ceux qui ont été validés par un point
        total_pred_labels = len(np.unique(mask_pred)) - 1
        fp = max(0, total_pred_labels - len(labels_detected))

        # 4. Calcul du score F1
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        return {
            "image": image_name,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f1": f1
        }
    
    def plot_f1_vs_iou(self, mask_true, mask_pred, thresholds=np.arange(0.5, 1.05, 0.05), ax=None, show=True):
        """
        Génère la courbe du F1-Score et des Faux Négatifs (FN) en fonction de l'exigence géométrique (IoU).
        Compatible avec une image unique (2D) ou une liste d'images (3D).
        Attend un tuple de deux axes : ax=(ax_f1, ax_fn).
        """
        f1_scores = []
        fn_counts = []
        
        # Vérification du type d'entrée (Batch vs Image unique)
        is_batch = isinstance(mask_true, list) or (isinstance(mask_true, np.ndarray) and mask_true.ndim == 3)
        
        # Itération sur la plage de tolérance
        for thresh in thresholds:
            if is_batch:
                total_tp, total_fp, total_fn = 0, 0, 0
                
                # Calcul des métriques image par image
                for mt, mp in zip(mask_true, mask_pred):
                    perf = self.summary_perf(mt, mp, iou_threshold=thresh)
                    total_tp += perf['tp']
                    total_fp += perf['fp']
                    total_fn += perf['fn']
                    
                # Calcul du F1-Score global pour le seuil courant
                if (total_tp + total_fp + total_fn) > 0:
                    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                    f1_global = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                else:
                    f1_global = 0.0
                    
                f1_scores.append(f1_global)
                fn_counts.append(total_fn)
                
            else:
                # Comportement original pour une image unique
                perf = self.summary_perf(mask_true, mask_pred, iou_threshold=thresh)
                f1_scores.append(perf["f1"])
                fn_counts.append(perf["fn"])
                
        # Création ou récupération des axes de tracé
        if ax is None:
            fig, (ax_f1, ax_fn) = plt.subplots(1, 2, figsize=(16, 5))
            ax_f1.set_title("F1-Score vs IoU")
            ax_fn.set_title("Faux Négatifs vs IoU")
        else:
            # Dépaquetage du tuple d'axes
            ax_f1, ax_fn = ax
            
        # Tracé des courbes (conservation de la couleur entre les deux graphiques)
        line, = ax_f1.plot(thresholds, f1_scores, marker='o', linewidth=2)
        ax_fn.plot(thresholds, fn_counts, marker='s', linestyle='--', linewidth=2, color=line.get_color())
        
        # Formatage du graphique F1
        ax_f1.set_xlabel('Seuil de tolérance (IoU)')
        ax_f1.set_ylabel('F1-Score')
        ax_f1.set_xlim(0.45, 1.05)
        ax_f1.set_ylim(0.0, 1.05)
        ax_f1.grid(True, linestyle='--', alpha=0.7)
        
        # Formatage du graphique FN
        ax_fn.set_xlabel('Seuil de tolérance (IoU)')
        ax_fn.set_ylabel('Total Faux Négatifs (FN)')
        ax_fn.set_xlim(0.45, 1.05)
        ax_fn.grid(True, linestyle='--', alpha=0.7)
        
        if show:
            ax_f1.legend()
            plt.tight_layout()
            plt.show()
        

    def report(self, name, mask_true, mask_pred, show = True):
        res = self.summary_perf(mask_true, mask_pred)
        if show:
            print(f"[{name}] F1: {res['f1']:.3f} | TP: {res['tp']} | FN: {res['fn']}")
        return res

    def global_report(self, results_list, model_name="Modèle"):
        """
        Agrège les résultats, affiche les statistiques globales et trace les graphiques d'analyse.
        """
        if not results_list:
            print("Aucun résultat à analyser.")
            return None

        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np

        df = pd.DataFrame(results_list)
        
        df['n_true_cells'] = df['tp'] + df['fn']

        total_tp = df['tp'].sum()
        total_fp = df['fp'].sum()
        total_fn = df['fn'].sum()

        global_f1 = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0
        mean_f1 = df['f1'].mean()

        print(f"\n{'='*50}")
        print(f"RESULTATS GLOBAUX : {model_name}")
        print(f"{'='*50}")
        print(f"Images evaluees          : {len(df)}")
        print(f"Total Vrais Positifs (TP): {total_tp}")
        print(f"Total Faux Positifs (FP) : {total_fp}")
        print(f"Total Faux Negatifs (FN) : {total_fn}")
        print("-" * 50)
        print(f"F1-Score Global (Micro)  : {global_f1:.3f}")
        print(f"F1-Score Moyen (Macro)   : {mean_f1:.3f}")
        print(f"{'='*50}\n")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.hist(df['f1'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(mean_f1, color='red', linestyle='dashed', linewidth=2, label=f'Moyenne ({mean_f1:.2f})')
        ax1.set_title("Distribution des scores F1")
        ax1.set_xlabel("F1 Score par image")
        ax1.set_ylabel("Nombre d'images")
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        ax2.scatter(df['n_true_cells'], df['f1'], color='coral', alpha=0.8, edgecolors='k')
        
        if len(df) > 1:
            z = np.polyfit(df['n_true_cells'], df['f1'], 1)
            p = np.poly1d(z)
            ax2.plot(df['n_true_cells'], p(df['n_true_cells']), "r--", alpha=0.8, label="Tendance")

        ax2.set_title("Performance vs Densite Cellulaire")
        ax2.set_xlabel("Nombre de cellules expertes (par image)")
        ax2.set_ylabel("F1 Score")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(f"Analyse des performances : {model_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return df