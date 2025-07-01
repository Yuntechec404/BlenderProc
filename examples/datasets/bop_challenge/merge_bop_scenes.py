import os
import argparse
import shutil
import json
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('input_root', help="Input directory containing generated BOP scenes")
parser.add_argument('output_root', help="Output directory where merged scenes will be saved")
args = parser.parse_args()

input_root = args.input_root
output_root = args.output_root

# input_root = '/home/user/BlenderProc/examples/datasets/bop_challenge/output/bop_data/itodd/train_pbr'
# output_root = '/home/user/BlenderProc/examples/datasets/bop_challenge/output/bop_data/itodd/merged'

os.makedirs(output_root, exist_ok=True)
for sub in ['depth', 'mask', 'mask_visib', 'rgb']:
    os.makedirs(os.path.join(output_root, sub), exist_ok=True)

# 最後要合併的 json
scene_camera = {}
scene_gt = {}
scene_gt_info = {}
scene_gt_coco = {"images": [], "annotations": []}

global_frame_idx = 0
global_anno_id = 0  # for COCO annotations

scenes = sorted([d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))])

for scene in tqdm(scenes, desc="Scenes"):
    scene_dir = os.path.join(input_root, scene)
    # 讀 scene json
    with open(os.path.join(scene_dir, "scene_camera.json")) as f:
        cam_data = json.load(f)
    with open(os.path.join(scene_dir, "scene_gt.json")) as f:
        gt_data = json.load(f)
    with open(os.path.join(scene_dir, "scene_gt_info.json")) as f:
        gtinfo_data = json.load(f)
    with open(os.path.join(scene_dir, "scene_gt_coco.json")) as f:
        coco_data = json.load(f)

    # 找所有 frame index
    rgb_files = sorted(glob(os.path.join(scene_dir, "rgb", "*.jpg")))
    frame_indices = [int(os.path.splitext(os.path.basename(f))[0]) for f in rgb_files]

    for i in frame_indices:
        # Copy depth
        src = os.path.join(scene_dir, "depth", f"{i:06d}.png")
        dst = os.path.join(output_root, "depth", f"{global_frame_idx:06d}.png")
        shutil.copy(src, dst)

        # Copy rgb
        src = os.path.join(scene_dir, "rgb", f"{i:06d}.jpg")
        dst = os.path.join(output_root, "rgb", f"{global_frame_idx:06d}.jpg")
        shutil.copy(src, dst)

        # Copy mask(s) and mask_visib(s) (注意：有多物件)
        # 例如 000000_000000.png, 000000_000001.png
        mask_files = sorted(glob(os.path.join(scene_dir, "mask", f"{i:06d}_*.png")))
        mask_visib_files = sorted(glob(os.path.join(scene_dir, "mask_visib", f"{i:06d}_*.png")))
        for mf in mask_files:
            obj_idx = mf.split("_")[-1].replace('.png', '')
            dst = os.path.join(output_root, "mask", f"{global_frame_idx:06d}_{int(obj_idx):06d}.png")
            shutil.copy(mf, dst)
        for mf in mask_visib_files:
            obj_idx = mf.split("_")[-1].replace('.png', '')
            dst = os.path.join(output_root, "mask_visib", f"{global_frame_idx:06d}_{int(obj_idx):06d}.png")
            shutil.copy(mf, dst)

        # 合併 scene_camera, scene_gt, scene_gt_info
        scene_camera[str(global_frame_idx)] = cam_data[str(i)]
        scene_gt[str(global_frame_idx)] = gt_data[str(i)]
        scene_gt_info[str(global_frame_idx)] = gtinfo_data[str(i)]

        # 合併 COCO（images + annotations）
        # 這裡假設 images/annotations 1:1 對應 frame idx
        # 需修正 id、image_id
        for img in coco_data.get("images", []):
            if img["file_name"].startswith(f"{i:06d}"):
                new_img = img.copy()
                new_img["id"] = global_frame_idx
                new_img["file_name"] = f"{global_frame_idx:06d}.jpg"
                scene_gt_coco["images"].append(new_img)
        for ann in coco_data.get("annotations", []):
            if ann["image_id"] == i:
                new_ann = ann.copy()
                new_ann["image_id"] = global_frame_idx
                new_ann["id"] = global_anno_id
                scene_gt_coco["annotations"].append(new_ann)
                global_anno_id += 1

        global_frame_idx += 1

# dump 合併結果
with open(os.path.join(output_root, "scene_camera.json"), "w") as f:
    json.dump(scene_camera, f)
with open(os.path.join(output_root, "scene_gt.json"), "w") as f:
    json.dump(scene_gt, f)
with open(os.path.join(output_root, "scene_gt_info.json"), "w") as f:
    json.dump(scene_gt_info, f)
with open(os.path.join(output_root, "scene_gt_coco.json"), "w") as f:
    json.dump(scene_gt_coco, f)
