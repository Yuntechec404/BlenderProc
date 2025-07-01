import os
import sys
import json
import trimesh

def main(models_dir):
    info = {}
    files = sorted(f for f in os.listdir(models_dir) if f.endswith('.ply') or f.endswith('.obj'))
    for fname in files:
        model_id = int(fname.split('_')[1].split('.')[0])
        mesh = trimesh.load(os.path.join(models_dir, fname), force='mesh')
        bounds = mesh.bounds  # (min, max)
        min_x, min_y, min_z = bounds[0]
        max_x, max_y, max_z = bounds[1]
        # 轉 mm（BOP 格式要求 mm）
        min_x, min_y, min_z = [float(x)*1000 for x in (min_x, min_y, min_z)]
        max_x, max_y, max_z = [float(x)*1000 for x in (max_x, max_y, max_z)]
        diameter = float(mesh.bounding_sphere.primitive.radius) * 2 * 1000
        info[str(model_id)] = {
            "diameter": diameter,
            "min_x": min_x,
            "min_y": min_y,
            "min_z": min_z,
            "max_x": max_x,
            "max_y": max_y,
            "max_z": max_z
        }
        print(f"Processed {fname} (id {model_id})")
    # 輸出
    with open(os.path.join(models_dir, '..', 'models_info.json'), 'w') as f:
        json.dump(info, f, indent=4)
    print("models_info.json saved!")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python gen_models_info.py <models資料夾路徑>")
        sys.exit(1)
    main(sys.argv[1])
