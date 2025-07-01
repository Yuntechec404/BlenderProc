import blenderproc as bproc
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', help="Path to the BOP datasets parent directory")
parser.add_argument('cc_textures_path', help="Path to downloaded cc textures")
parser.add_argument('output_dir', help="Path to where the final files will be saved")
parser.add_argument('--num_scenes', type=int, default=10, help="Number of scenes (each with multiple images) to generate")
args = parser.parse_args()

bproc.init()

# **只載入蘋果模型**：使用 BOP 加載函數指定 obj_ids 只載入ID=15的物體:contentReference[oaicite:0]{index=0}
target_bop_objs = bproc.loader.load_bop_objs(
    bop_dataset_path=os.path.join(args.bop_parent_path, 'lm'),
    obj_ids=[15], 
    object_model_unit="m",

    mm2m=False  # 確保模型單位與原始單位一致（此處模型已是公尺單位）
)
if len(target_bop_objs) == 0:
    raise RuntimeError("蘋果模型(obj_id=15)未載入，請確認路徑及ID是否正確。")

# 設定相機內參（RealSense D435）
bproc.camera.set_intrinsics_from_K_matrix(
    np.array([[606.23, 0, 325.59],
              [0, 605.98, 243.27],
              [0,    0,    1]]), 
    image_width=640, image_height=480
)  # :contentReference[oaicite:1]{index=1}

# 設定輸出解析度
bproc.camera.set_resolution(640, 480)

# 設定物體外觀並隱藏（初始先隱藏，稍後逐個場景放置時再顯示）
for obj in target_bop_objs:
    obj.set_shading_mode('auto')  # 啟用自動陰影模式
    obj.hide(True)

# 建立簡單房間環境（地板和四周牆壁）
room_planes = [
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),  # 地板 (4m x 4m)
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.5708, 0, 0]),  # 牆壁
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0,  2, 2], rotation=[ 1.5708, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[ 2, 0, 2], rotation=[0, -1.5708, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0,  1.5708, 0])
]

# 建立光源：一個頂部發光平面和一個點光源
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_mat = bproc.material.create('light_material')
light_point = bproc.types.Light()
light_point.set_energy(200)

# 載入隨機材質貼圖
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

# 定義蘋果擺放的隨機位姿函數（小範圍隨機擺放，包含隨機朝向）
def sample_apple_center(obj: bproc.types.MeshObject):
    # 在近中心區域隨機放置蘋果
    loc_min = np.array([-0.03, -0.03, 0.02])
    loc_max = np.array([ 0.03,  0.03, 0.05])
    obj.set_location(np.random.uniform(loc_min, loc_max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())  # 隨機旋轉朝向

# 定義「初始」位姿函數，用於將物體放置在地板上（保持物體直立，只繞垂直軸隨機旋轉）
def sample_initial_pose(obj: bproc.types.MeshObject):
    obj.set_location(bproc.sampler.upper_region(
        objects_to_sample_on=[room_planes[0]],  # 在地板上采樣位置
        min_height=1, max_height=4, face_sample_range=[0.4, 0.6]
    ))
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, 2*np.pi]))  # 隨機水平旋轉（保持直立）

# **生成場景資料**
for scene_idx in range(args.num_scenes):
    # 選擇當前場景的目標物件（此處僅蘋果一個）
    if len(target_bop_objs) < 15:
        sampled_target_objs = target_bop_objs  # 若物件數少於15，直接使用全部物件:contentReference[oaicite:2]{index=2}
    else:
        sampled_target_objs = list(np.random.choice(target_bop_objs, size=15, replace=False))
    # 隨機化材質屬性並顯示物件
    for obj in sampled_target_objs:
        # 若物件自帶材質，這裡可調整粗糙度和高光反射以增加隨機性
        mat = obj.get_materials()[0]
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.0, 1.0))
# 安全設定 Specular IOR，如果存在該項目
        try:
            mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0.0, 1.0))
        except KeyError:
            pass  # 有些材質沒有這個輸入槽，可以忽略
        obj.hide(False)
    # 設定隨機光源參數
    light_plane.replace_materials(light_mat)
    light_mat.make_emissive(
        emission_strength=np.random.uniform(3, 6),
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0])
    )
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    light_point.set_location(bproc.sampler.shell(
        center=[0, 0, 0], radius_min=1, radius_max=1.5,
        elevation_min=5, elevation_max=89
    ))
    # 隨機選擇並應用一個環境材質（地板/牆壁貼圖）
    random_tex = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_tex)
    # 隨機樣本物件位姿
    for obj in sampled_target_objs:
        try:
            if obj.get_cp("category_id") == 15:  # 蘋果的 category_id 為15
                sample_apple_center(obj)
            else:
                # (若有其他物件，一律使用一般擺放函數)
                sample_apple_center(obj)
        except Exception as e:
            sample_apple_center(obj)
    # 將物件放置在地板上（確保不穿透地面且直立）
    bproc.object.sample_poses_on_surface(
        objects_to_sample=sampled_target_objs, 
        surface=room_planes[0],
        sample_pose_func=sample_initial_pose,
        min_distance=0.01, max_distance=0.2
    )
    # 建立BVH樹以供攝影機避障檢查
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_objs)
    # 隨機生成多個攝影機視角
    cam_poses = 0
    while cam_poses < 25:  # 每個場景生成25張圖像
        location = bproc.sampler.shell(
            center=[0, 0, 0],
            radius_min=0.35, radius_max=1.5,
            elevation_min=5, elevation_max=89
        )
        # 計算攝影機興趣點（視線焦點）為場景中一個物件的位置
        if len(sampled_target_objs) == 1:
            poi = sampled_target_objs[0].get_location()
        else:
            # 從當前場景物件中隨機挑選最多10個，計算其平均位置作為POI:contentReference[oaicite:3]{index=3}
            subset = np.random.choice(sampled_target_objs, size=min(10, len(sampled_target_objs)), replace=False)
            poi = bproc.object.compute_poi(subset)
        # 根據攝影機位置和興趣點計算朝向旋轉矩陣
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
        cam_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        # 檢查視角是否沒有被遮擋且物體在視野中
        if bproc.camera.perform_obstacle_in_view_check(cam_matrix, {"min": 0.3}, bop_bvh_tree):
            bproc.camera.add_camera_pose(cam_matrix)
            cam_poses += 1
    # 渲染當前場景所有視角的影像（僅RGB）
    data = bproc.renderer.render()  # 獲取渲染結果的顏色圖
    # 將蘋果物件標註寫出為BOP格式（scene_gt.json、scene_camera.json、rgb等）
    bproc.writer.write_bop(
        os.path.join(args.output_dir, 'bop_data'),
        target_objects=sampled_target_objs,
        dataset='lm',
        # 不輸出深度圖，因此省略 depths 和 depth_scale 參數
        colors=data["colors"],
        color_file_format="JPEG",
        ignore_dist_thres=10,
        append_to_existing_output=True  # 連續追加輸出
    )
    # 將物體重新隱藏，準備產生下一個場景
    for obj in sampled_target_objs:
        obj.hide(True)

print("資料生成完畢，請確認輸出資料夾中的 bop_data/ 格式。")
