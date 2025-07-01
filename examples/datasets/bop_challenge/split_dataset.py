import os
import argparse
import shutil
import random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('merged_dir', help="Directory where SINGLESHOTPOSE label files will be saved")
parser.add_argument('train_ratio', help="train ratio")
parser.add_argument('val_ratio', help="val ratio")
parser.add_argument('test_ratio', help="test ratio")
args = parser.parse_args()

# --------- 資料夾設定 ---------
# merged_dir = '/home/user/BlenderProc/examples/datasets/bop_challenge/output/bop_data/itodd/merged'
merged_dir = args.merged_dir
img_dir    = os.path.join(merged_dir, 'rgb')
label_dir  = os.path.join(merged_dir, 'labels')
extra_dirs = ['depth', 'mask', 'mask_visib']

# --------- Split 比例 ---------
# train_ratio = 0.8
# val_ratio   = 0.2
# test_ratio  = 0.0
train_ratio = args.train_ratio
val_ratio = args.val_ratio
test_ratio = args.test_ratio

# --------- Split 輸出資料夾 ---------
split_dirs = {
    'train': os.path.join(merged_dir, 'train'),
    'val':   os.path.join(merged_dir, 'val'),
    'test':  os.path.join(merged_dir, 'test'),
}
for split in split_dirs:
    for subdir in ['rgb', 'labels'] + extra_dirs:
        os.makedirs(os.path.join(split_dirs[split], subdir), exist_ok=True)

# --------- 取得所有影像檔名 ---------
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])

# --------- 隨機分割 ---------
random.seed(42)
random.shuffle(img_files)
n = len(img_files)
n_train = int(n * train_ratio)
n_val   = int(n * val_ratio)
n_test  = n - n_train - n_val

train_set = img_files[:n_train]
val_set   = img_files[n_train:n_train+n_val]
test_set  = img_files[n_train+n_val:]

splits = {
    'train': train_set,
    'val':   val_set,
    'test':  test_set,
}
print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

# --------- Split + 複製檔案 ---------
for split, files in splits.items():
    for img_file in tqdm(files, desc=f'Copying {split}'):
        base = os.path.splitext(img_file)[0]

        # 影像
        shutil.copy(
            os.path.join(img_dir, img_file),
            os.path.join(split_dirs[split], 'rgb', img_file)
        )
        # label
        label_file = base + '.txt'
        src_label_path = os.path.join(label_dir, label_file)
        dst_label_path = os.path.join(split_dirs[split], 'labels', label_file)
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)
        else:
            print(f"Warning: label not found for {img_file}")

        # 其它 extra_dirs
        for extra in extra_dirs:
            # depth: 檔名為 000001.png
            if extra == 'depth':
                extra_file = base + '.png'
                src_extra_path = os.path.join(merged_dir, extra, extra_file)
                dst_extra_path = os.path.join(split_dirs[split], extra, extra_file)
                if os.path.exists(src_extra_path):
                    shutil.copy(src_extra_path, dst_extra_path)
                else:
                    print(f"Warning: {extra} not found for {img_file}")
            # mask/mask_visib: 可能多個物件，每個影像可能對應多個檔案（000001_000000.png, 000001_000001.png, ...）
            else:
                mask_files = [f for f in os.listdir(os.path.join(merged_dir, extra)) if f.startswith(base + '_') and f.endswith('.png')]
                if len(mask_files) == 0:
                    print(f"Warning: {extra} not found for {img_file}")
                for mf in mask_files:
                    src_mask_path = os.path.join(merged_dir, extra, mf)
                    dst_mask_path = os.path.join(split_dirs[split], extra, mf)
                    shutil.copy(src_mask_path, dst_mask_path)
