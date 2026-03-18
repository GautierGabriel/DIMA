from skimage import measure
import numpy as np
import cv2
from pathlib import Path
import tool

visualizer = tool.CellVisualizer()

all_ids = [f"{i:03d}" for i in range(1, 103)]

dossier_nuc_propre = visualizer.ann_dir.parent / '02-Annotations_Clean'
dossier_nuc_propre.mkdir(parents=True, exist_ok=True)

print("=== nettoyage des masques de noyaux ===\n")

for id_unique in all_ids:
    try:
        p_raw = visualizer._get_path(visualizer.ann_dir, f"{id_unique}*Nuc.bmp")
    except StopIteration:
        print(f" {id_unique} : Masque introuvable")
        continue
        
    mask_raw = cv2.imread(str(p_raw), cv2.IMREAD_UNCHANGED)
    if mask_raw is None: continue
    
    if mask_raw.ndim == 3:
        mask_raw = np.max(mask_raw, axis=2)
        
    nb_init = len(np.unique(mask_raw)) - 1 
        
    mask_bin = mask_raw > 0
    mask_fixed = measure.label(mask_bin, background=0)
    
    nb_final = np.max(mask_fixed)
    
    if nb_final > 255:
        mask_final = mask_fixed.astype(np.uint16)
    else:
        mask_final = mask_fixed.astype(np.uint8)
        
    cv2.imwrite(str(dossier_nuc_propre / p_raw.name), mask_final)
    
    print(f"Image {id_unique} | Labels originaux : {nb_init:3d}  -->  Noyaux réels : {nb_final:3d}")

print(f"done. Tous les masques propres sont dans : {dossier_nuc_propre}")