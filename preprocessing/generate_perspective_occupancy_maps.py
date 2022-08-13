import os
import cv2
import numpy as np
from multiprocessing.pool import Pool


# Habitat dataset params
full_res_shape = (1024, 1024)
hfov = 90
f = full_res_shape[0]/ (2 * np.tan(np.deg2rad(hfov/2)))
cx, cy = full_res_shape[0] // 2, full_res_shape[1] // 2

K = np.array([    # Camera intrinsics
    [f, 0, cx],
    [0, f, cy],
    [0, 0, 1]])

SEM_IDX_OFFSET = 1  # We offset to set 0 as unknown and shift the ADE20k map from 0-130 to 1-131
FLOOR_INDICES = np.array([3, 28]) + SEM_IDX_OFFSET


# Setup params for bev
cw = ch = 0.05  # cell-size in m
px = 1.6  # origin offset along x-axis
pz = 3.3 # origin offset along z-axis

r = int(2 * px/ch)  # size for bev in pixels

T = np.array([     # Transformation to go from camera_coords (m) to bev_coords (px)
    [1/cw, 0, 0, px/cw],
    [0, 1, 0, 0],
    [0, 0, -1/ch, pz/ch]
])


# Setup output map params
img_w, img_h = 128, 128
Ko = np.array([   # To project flattened point cloud back to the camera's image plane
    [img_w//2, 0, img_w//2],
    [0, img_w//2, img_h//2],
    [0, 0, 1]])


# Perspective occupancy map generator fn
def generate_pom(fn_arg):
    try:
        data_dir, out_dir, scene, camera, filename = fn_arg.split()

        # Read data from disk

        depth = cv2.imread(os.path.join(data_dir, scene, '0', camera, 'DEPTH', f'{filename}.png'), -1)
        depth = depth / 6553.5

        semantics = cv2.imread(os.path.join(data_dir, scene, '0', camera, 'semantics', f'{filename}.png'), -1)
        semantics += SEM_IDX_OFFSET

        # Generate point cloud
        y_coords, x_coords = np.indices(depth.shape)
        sensor_coords = np.stack([x_coords, y_coords, np.ones_like(x_coords)])
        image_coords = (np.linalg.inv(K) @ sensor_coords.reshape((3, -1))).reshape((3, *full_res_shape))
        camera_coords = depth * image_coords

        # Filter points above height threshold
        obstacle_height_thresh = 0.1  # in m, measured from camera and not ground
        filter_idx = camera_coords[1] > -obstacle_height_thresh # Since y-axis points downwards in camera-coordinate frame
        filter_idx = filter_idx & (camera_coords[2] != 0) # Since depth is zero at some points
        valid_pts = camera_coords[:, filter_idx]
        valid_semantics = semantics[filter_idx]

        flattened_pts = np.copy(valid_pts)
        flattened_pts[1] = 1  # Project to floor


        # Discretize point coordinates to take highest per bin
        discrete_coords = T @ np.concatenate([valid_pts, np.ones((1, valid_pts.shape[1]))])
        filter_idx = (discrete_coords[0] > 0) & (discrete_coords[0] < r) & (discrete_coords[2] > 0) & (discrete_coords[2] < r)
        discrete_coords = discrete_coords[:, filter_idx]

        x_indices = discrete_coords[0].astype(np.uint32)
        z_indices = discrete_coords[2].astype(np.uint32)

        indices =  z_indices * r + x_indices

        sorting_keys = (z_indices * r * 10) + (x_indices * 10) - discrete_coords[1]
        sorting_order = np.argsort(sorting_keys)

        sorted_indices = indices[sorting_order]
        unique_max_filter = sorted_indices[1:] != sorted_indices[:-1]
        if len(sorted_indices) > 0:
            unique_max_filter = np.concatenate([unique_max_filter, [True]])

        filtered_indices = sorted_indices[unique_max_filter]

        bev_x, bev_z = filtered_indices % r, filtered_indices // r
        bev = np.zeros((r, r), dtype=np.uint16)
        filtered_semanticsColor = valid_semantics[filter_idx][sorting_order][unique_max_filter]
        bev[bev_z, bev_x] = filtered_semanticsColor


        # Reproject to image plane
        img_coords = flattened_pts / flattened_pts[2]
        sensor_coords = (Ko @ img_coords).astype(np.uint32)

        filter_idx = (sensor_coords[0] > 0) & (sensor_coords[0] < img_w) & \
            (sensor_coords[1] < img_h) & (sensor_coords[1] < img_h)
        x_coords, y_coords, _ = sensor_coords[:, filter_idx]
        valid_semantics = valid_semantics[filter_idx]

        sensor_indices = y_coords * full_res_shape[0] + x_coords

        sorting_keys = (y_coords * full_res_shape[0] * 10) + (x_coords * 10) - valid_pts[1, filter_idx]
        sorting_order = np.argsort(sorting_keys)

        sorted_sensor_indices = sensor_indices[sorting_order]
        unique_max_filter = sorted_sensor_indices[:-1] != sorted_sensor_indices[1:] 
        if len(sorted_sensor_indices) > 0:
            unique_max_filter = np.concatenate([unique_max_filter, [True]])

        filtered_indices = sorted_sensor_indices[unique_max_filter]
        x_coords, y_coords = filtered_indices % full_res_shape[1], filtered_indices // full_res_shape[1]

        sorted_semantics = valid_semantics[sorting_order]
        filtered_semantics = sorted_semantics[unique_max_filter]

        pers_semocc_map = np.zeros((img_h, img_w), dtype=np.uint16)
        pers_semocc_map[y_coords, x_coords] = filtered_semantics


        # Save both outputs to disk
        bev_outpath = os.path.join(out_dir, scene, camera, 'proj_bev', f'{filename}.png')
        os.makedirs(os.path.dirname(bev_outpath), exist_ok=True)
        cv2.imwrite(bev_outpath, bev)

        pom_outpath = os.path.join(out_dir, scene, camera, 'pom', f'{filename}.png')
        os.makedirs(os.path.dirname(pom_outpath), exist_ok=True)
        cv2.imwrite(pom_outpath, pers_semocc_map)
    except Exception as e:
        print(e)
        print(fn_arg)



if __name__ == "__main__":
    data_dir = '/scratch/shantanu/gibson4/new'
    camera = 'front'

    out_dir = '/scratch/shantanu/gibson4/generated'

    os.makedirs(out_dir, exist_ok=True)

    fn_args = ['{} {} {} {} {}'.format(
        data_dir, out_dir, scene, camera, os.path.splitext(x)[0])
         for scene in os.listdir(data_dir)
         for x in os.listdir(os.path.join(data_dir, scene, '0', camera, 'RGB'))]

    with Pool(processes=8) as p:
        p.map(generate_pom, fn_args)