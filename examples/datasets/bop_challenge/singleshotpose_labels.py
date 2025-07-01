import json
import argparse
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--camera_parser', help="Path to the camera intrinsic/extrinsic parameter file (e.g., camera.json)")
parser.add_argument('--merged_dir', help="Directory where SINGLESHOTPOSE label files will be saved")
parser.add_argument('--model_path', help="Path to the 3D object model (PLY file)")
args = parser.parse_args()

# ====== 讀取相機參數 ======
camera_parser = args.camera_parser
with open(camera_parser) as f:
    camera = json.load(f)
img_w = camera['width']
img_h = camera['height']
fx = camera['fx']
fy = camera['fy']
cx = camera['cx']
cy = camera['cy']
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

# 其它你的載入
# merged_dir = '/home/user/BlenderProc/examples/datasets/bop_challenge/output/bop_data/itodd/merged'
# model_path = '/home/user/BlenderProc/output/itodd/models/obj_000001.ply'
merged_dir = args.merged_dir
model_path = args.model_path
label_dir = os.path.join(merged_dir, 'labels')
os.makedirs(label_dir, exist_ok=True)

def compute_model_corners(model_path):
    mesh = trimesh.load(model_path, force='mesh')
    points3d = mesh.vertices
    min_xyz = points3d.min(axis=0)
    max_xyz = points3d.max(axis=0)
    centroid = (min_xyz + max_xyz) / 2
    corners = np.array([
        [min_xyz[0], min_xyz[1], min_xyz[2]],
        [min_xyz[0], min_xyz[1], max_xyz[2]],
        [min_xyz[0], max_xyz[1], min_xyz[2]],
        [min_xyz[0], max_xyz[1], max_xyz[2]],
        [max_xyz[0], min_xyz[1], min_xyz[2]],
        [max_xyz[0], min_xyz[1], max_xyz[2]],
        [max_xyz[0], max_xyz[1], min_xyz[2]],
        [max_xyz[0], max_xyz[1], max_xyz[2]],
    ])
    model_corners = np.vstack([centroid, corners])  # (9,3)
    return model_corners

model_corners = compute_model_corners(model_path)

with open(os.path.join(merged_dir, "scene_gt.json")) as f:
    gt_all = json.load(f)

rgb_dir = os.path.join(merged_dir, "rgb")
all_imgs = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg') or f.endswith('.png')])

def project_points(points_3d, R, t, K):
    pts_cam = (R @ points_3d.T).T + t.reshape(1,3)
    pts_2d = (K @ pts_cam.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    return pts_2d

for img_file in tqdm(all_imgs, desc='Processing images'):
    img_idx = int(os.path.splitext(img_file)[0])
    label_file = os.path.join(label_dir, f"{img_idx:06d}.txt")

    obj_inst = gt_all[str(img_idx)][0]   # 假設只一個物件
    obj_id = obj_inst["obj_id"]

    R = np.array(obj_inst["cam_R_m2c"]).reshape(3,3)
    t = np.array(obj_inst["cam_t_m2c"]).reshape(3,1)
    # t 單位若 mm，K也mm，通常不需特別轉

    pts_2d = project_points(model_corners, R, t, K)
    pts_2d[:,0] /= img_w
    pts_2d[:,1] /= img_h

    x_range = np.max(pts_2d[:,0]) - np.min(pts_2d[:,0])
    y_range = np.max(pts_2d[:,1]) - np.min(pts_2d[:,1])

    label_row = [obj_id] + pts_2d.flatten().tolist() + [x_range, y_range]
    with open(label_file, "w") as f:
        f.write(" ".join([f"{v:.6f}" for v in label_row]) + "\n")
