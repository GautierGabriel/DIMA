Segmentation — README


Ce dépôt contient des scripts pour entraîner et évaluer des modèles de segmentation sur la base de données locale.

dossier scripts :

- **Benchmark** : lance les modèles choisis sur l'ensemble de la base et sauve les mask et visualisations.
- **Scripts d'entraînement** : cellpose3_nuc, cellpose3, cellpose4 (notebooks/scripts d'entraînement pour Cellpose v3/v4).
- **Outils** : `scripts/tool.py` regroupe le visualiseur, les calculs de statistiques et le calcul du loss.
- **Pré-traitements** : `Nuclei_preprocessing.py` préparent les annotaions de noyaux pour l entrainement (nettoyage des masques)

Fichiers :
-dataset_splits.json : description des id des images a utliser pour la base de train val et test
-Nuclei.csv : contient les annotaions ( positions x,y) de noyaux pour les images de la granuleuse

Remarque : plusieurs scripts contiennent en tête des blocs de code commentés (OwnCloud/Nextcloud) servant à téléverser les données depuis Nextcloud.
ces blocs sont inutiles si les données sont déjà présentes sur le serveur, mais peuvent être utiles pour téléverser les données depuis Nextcloud si besoin.

```python
# USEFUL TO UPLOAD DATA TO OWNCLLOUD, BUT NOT NEEDED FOR THE TOOL IF DATA IS ALREADY ON THE SERVER
# import owncloud, getpass, cellpose
# 
# # Config
# url, user = 'url', 'user'
# oc_session = owncloud.Client(url)
# oc_session.login(user, getpass.getpass(f"PW {user}: "))
# oc_session.get_file('/travail/Mines/DIMA/Segmentation/scripts/tool.py', 'tool.py')
# 
# import tool
# oc = tool.Owncloud(oc_session)
# 
# print("Tool chargé et prêt.")
```



Pour exécuter les scripts : placez les données sur le serveur et mettez à jour les chemins dans `scripts/tool.py` (paramètres de `CellVisualizer` et autres chemins d'entrée/sortie).

Extrait des chemins par défaut utilisés par `CellVisualizer` :

```python
class CellVisualizer:
    def __init__(self, 
                img_dir='./data/1-Images/01-Data/', # for the input raw images
                ann_dir='./data/2-Annotations/02-Annotations/', # for the cell Annotations to train the model 
                nuc_dir = './data/02-Annotations_Clean/', # rerpocess the nuc annotaions using Nuclei_preprocessing.py and save the clean masks in this folder
                nuc_pred_dir='./data/prediction/cellpose_nuclei_200_train/masks/', # for the predicted masks of nuclei (to compute the metrics and visualize the results)
                cell_pred_dir='./data/prediction/cellpose4_50/',    # for the predicted masks of cells (to compute the metrics and visualize the results)
                z_stack_dir='./data/05-Stacks 3D/'):    # for the z-stacks
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.nuc_dir = Path(nuc_dir)
        self.nuc_pred_dir = Path(nuc_pred_dir)
        self.cell_pred_dir = Path(cell_pred_dir)
        self.z_stack_dir = Path(z_stack_dir)

    def _get_path(self, folder, pattern):
        try:
            return next(folder.rglob(pattern))
        except StopIteration:
            return None
```

