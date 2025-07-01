import blenderproc as bproc
import argparse
import os
import numpy as np

# 參數設定
parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', help="Path to the BOP datasets parent directory")
parser.add_argument('cc_textures_path', help="Path to downloaded cc textures")
parser.add_argument('output_dir', help="Path to where the final files will be saved")
parser.add_argument('--num_scenes', type=int, default=20, help="How many scenes to generate (default: 20)")
args = parser.parse_args()

bproc.init()

# 設定相機內參（RealSense D435 640x480）
bproc.camera.set_intrinsics_from_K_matrix(
    np.array([[606.23, 0, 325.59],
              [0, 605.98, 243.27],
              [0,    0,    1]]),
    image_width=640, image_height=480
)

# 載入 itodd 物件, apple (obj_000001.ply) 應為 category_id=1
itodd_objs = bproc.loader.load_bop_objs(os.path.join(args.bop_parent_path, 'itodd'), mm2m=True)
apple_objs = [obj for obj in itodd_objs if obj.get_cp("category_id", None) == 1]
if len(apple_objs) == 0:
    raise RuntimeError("找不到 apple (category_id=1) in itodd dataset!")
apple = apple_objs[0]

# distractors = 其他 itodd 非 apple
distractor_itodd = [obj for obj in itodd_objs if obj != apple]
distractors = distractor_itodd

# 設 shading，初始隱藏
for obj in [apple] + distractors:
    obj.set_shading_mode('auto')
    obj.hide(True)

# 房間環境 & 光源
room_planes = [
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])
]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)

light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_point = bproc.types.Light()
light_point.set_energy(30)

cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

# 物件隨機初始位置（在上空，會掉下來）
def sample_initial_pose(obj):
    xy_min = -0.25
    xy_max = 0.25
    z_min = 0.15
    z_max = 0.35
    loc = np.array([
        np.random.uniform(xy_min, xy_max),
        np.random.uniform(xy_min, xy_max),
        np.random.uniform(z_min, z_max)
    ])
    obj.set_location(loc)
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

for scene_idx in range(args.num_scenes):

    # show
    apple.hide(False)
    num_distractors = min(30, len(distractors))
    sampled_distractors = list(np.random.choice(distractors, size=num_distractors, replace=False))
    for obj in sampled_distractors:
        obj.hide(False)

    # ==== Apple 永遠純紅色 =====
    apple_mat = apple.get_materials()[0]
    apple_mat.set_principled_shader_value("Base Color", [1.0, 0.0, 0.0, 1.0])  # RGBA 紅
    apple_mat.set_principled_shader_value("Roughness", 0.4)

    # 材質（只給牆面地面用 cc textures，蘋果保原生材質）
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)

    # 光源亂數
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(2, 5),
        emission_color=np.random.uniform([0.7, 0.7, 0.7, 1.0], [1.0, 1.0, 1.0, 1.0])
    )
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.6, 0.6, 0.6], [1, 1, 1]))
    light_point.set_location(bproc.sampler.shell(center=[0,0,0], radius_min=1, radius_max=1.5, elevation_min=5, elevation_max=89))

    # 初始隨機擺放 (apple + distractor)
    sample_initial_pose(apple)
    for obj in sampled_distractors:
        sample_initial_pose(obj)

    # 先設定剛體屬性
    apple.enable_rigidbody(True, mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99, collision_margin=0.0005)
    for obj in sampled_distractors:
        mat = obj.get_materials()[0]
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.0, 0.8))
        try:
            mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0.0, 1.0))
        except Exception:
            pass
        obj.enable_rigidbody(True, mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99, collision_margin=0.0005)

    # 物理模擬同步掉落
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=3,
        max_simulation_time=30,
        check_object_interval=1,
        substeps_per_frame=100,
        solver_iters=25
    )

    # 建 BVH
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects([apple] + sampled_distractors)
    cam_poses = 0
    while cam_poses < 25:
        location = bproc.sampler.shell(
            center=[0, 0, 0],
            radius_min=0.35,
            radius_max=1.0,
            elevation_min=5,
            elevation_max=89
        )
        poi = apple.get_location()
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            bproc.camera.add_camera_pose(cam2world_matrix)
            cam_poses += 1

    # 渲染
    data = bproc.renderer.render()

    # 只寫 apple 的標註
    bproc.writer.write_bop(
        os.path.join(args.output_dir, 'bop_data'),
        target_objects=[apple],
        dataset='itodd',
        depth_scale=0.1,
        depths=data["depth"],
        colors=data["colors"],
        color_file_format="JPEG",
        ignore_dist_thres=10,
        append_to_existing_output=True
    )

    apple.hide(True)
    for obj in sampled_distractors:
        obj.disable_rigidbody()
        obj.hide(True)

print("資料集產生完成。")
