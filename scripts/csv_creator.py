import os
import pandas as pd
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(script_dir, '..', 'data')

train_folder = os.path.join(BASE_DIR, 'train')
test_folder = os.path.join(BASE_DIR, 'test')

labels_csv_path = os.path.join(BASE_DIR, 'train_labels.csv')
labels_csv = pd.read_csv(labels_csv_path)
labels_dict = labels_csv.to_dict()
new_dict = {}
for i in labels_dict['id']:
    new_dict[labels_dict['id'][i]] = labels_dict['label'][i]

train_files = sorted(os.listdir(train_folder))
test_files = sorted(os.listdir(test_folder))
train_paths = []
train_labels = []
for i in train_files:
    if i != '.DS_Store':
        train_paths.append(os.path.join(train_folder, i))
        train_labels.append(new_dict[i[:-4]])

test_paths = []
for i in test_files:
    if i != '.DS_Store':
        test_paths.append(os.path.join(test_folder, i))


train_val_csv = pd.DataFrame()
train_val_csv['images'] = train_paths
train_val_csv['labels'] = train_labels

test_csv = pd.DataFrame()
test_csv['images'] = test_paths

train_csv, val_csv = train_test_split(train_val_csv, test_size=0.05, shuffle=True, random_state=42)

output_dir = os.path.join(script_dir, '..')
train_csv.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
val_csv.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
test_csv.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
