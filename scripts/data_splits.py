import owncloud, getpass

# Config
url, user = 'https://cloud.minesparis.psl.eu', 'gabriel.gautier'
oc_session = owncloud.Client(url)
oc_session.login(user, getpass.getpass(f"PW {user}: "))
oc_session.get_file('/travail/Mines/DIMA/Segmentation/scripts/tool.py', 'tool.py')

import tool
tool = tool.Owncloud(oc_session)

import random
import json

all_ids = [f"{i:03d}" for i in range(1, 103)] 

random.seed(42)
random.shuffle(all_ids)

n = len(all_ids)
splits = {
    "train": all_ids[:int(0.7 * n)],
    "val":   all_ids[int(0.7 * n):int(0.8 * n)],
    "test":  all_ids[int(0.8 * n):]
}
chemin_sauvegarde = './data/dataset_splits.json'

with open(chemin_sauvegarde, 'w') as f:
    json.dump(splits, f, indent=4)

print(f"Splits sauvegardés dans {chemin_sauvegarde}")
print(f"Train({len(splits['train'])}) | Val({len(splits['val'])}) | Test({len(splits['test'])})")
tool.oc.put_file(
    '/travail/Mines/DIMA/Segmentation/data/dataset_splits.json',
    './data/dataset_splits.json'                                   
)