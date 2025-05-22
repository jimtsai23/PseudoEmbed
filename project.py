
import numpy as np
from glob import glob
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def cam_extrinsic(r, azim, elev, outward=True):
    azim = azim / 180 * np.pi
    elev = elev / 180 * np.pi
    r1 = r*np.cos(elev)
    cam_center = np.array([r1*np.cos(azim), r1*np.sin(azim), r*np.sin(elev)])
    lookat = normalize(0 - cam_center)
    if outward:
        lookat *= -1
    r2 = r / np.cos(elev)
    up = normalize(cam_center - np.array([r2*np.cos(azim), r2*np.sin(azim), 0]))
    if elev<0:
        up *= -1
    if np.linalg.norm(up)==0: up = np.array([0,0,1])
    right = normalize(np.cross(lookat, up))
    c2w = np.eye(4)
    c2w[:3,:3] = np.concatenate([right[:,None],-1*up[:,None],lookat[:,None]],axis=1)
    c2w[:3,3] = cam_center
    return c2w

def preprocess_xyz(xyz):
    '''make xyz 0 to 1, then center at origin (-0.5,0.5)'''
    xyz = xyz - np.amin(xyz, axis=0)
    xyz = xyz / xyz.max()
    center = np.amax(xyz, axis=0) / 2
    xyz = xyz - center
    return xyz

def convert_array_to_pil(depth_map):
    # Input: depth_map -> HxW numpy array with depth values 
    # Output: colormapped_im -> HxW numpy array with colorcoded depth values
    eps = 1e-6
    mask = depth_map!=0
    disp_map = 1/(depth_map+eps)
    vmax = np.percentile(disp_map[mask], 95)
    vmin = np.percentile(disp_map[mask], 5)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    mask = np.repeat(np.expand_dims(mask,-1), 3, -1)
    colormapped_im = (mapper.to_rgba(disp_map)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im[~mask] = 255
    return colormapped_im

def main(out_path,dataset,split,exp):
    out_path = f'{out_path}/{dataset}/{exp}/{split}'
    Path(out_path).mkdir(parents=True,exist_ok=True)
    if 'scannet' in out_path:
        data_path = f'datasets/ScanNet/scenes/{split}_data'
        # azim_set = np.arange(8) * 45
        # elev_set = [45]
        # radius = 0.4
        # outward = False
        azim_set = np.arange(8) * 45
        elev_set = [0]
        radius = 0.4
        outward = True
    elif 's3dis' in out_path:
        data_path = f'datasets/S3DIS/scenes/{split}_data'
        azim_set = np.arange(6) * 60
        elev_set = np.arange(-2,3) * 15
        radius = 0.001
        outward = True

    # set synthetic camera
    H, W = 240, 320
    aspect_ratio = W / H
    fov = 90 / 180 * np.pi
    fx = 0.5 * W / np.tan(0.5*fov)
    fy = fx / aspect_ratio
    print(fx, fy)
    intrinsic = np.eye(3)
    intrinsic[:2,:3] = np.array([
        [fx, 0, W/2],
        [0, fy, H/2],
    ])
    np.save(f'{out_path}/intrinsic.npy',intrinsic)
    # extrinsic
    pose_lst = []
    num_pose = 0
    pose_dir = out_path + '/' + 'pose'
    Path(pose_dir).mkdir(parents=True,exist_ok=True)
    for elev in elev_set:
        for azim in azim_set:
            c2w = cam_extrinsic(r=radius,azim=azim,elev=elev, outward=outward)
            pose_lst.append(c2w)
            np.save(f'{pose_dir}/{num_pose}.npy',c2w)
            num_pose += 1
    w2c_lst = []
    for pose in pose_lst:
        w2c_lst.append(np.linalg.inv(pose))

    # read scenes
    file_lst = sorted(glob(f'{data_path}/*'))
    for scene_name in file_lst:
        scene = np.load(scene_name)
        scene_id = scene_name.split('/')[-1][:-4]
        print('scene name: ', scene_id)

        out_dir = out_path + '/' + scene_id
        Path(out_dir).mkdir(parents=True,exist_ok=True)
        
        print('point cloud size: ', scene.shape)
        label = scene[:,6:7]
        rgb = scene[:,3:6]
        xyz_raw = scene[:,0:3]
        xyz = preprocess_xyz(xyz_raw)
        
        for j in tqdm(range(num_pose)):
            w2c = w2c_lst[j]
            xyz_w_h = np.concatenate([xyz, np.ones([xyz.shape[0],1])],axis=1)
            xyz_c_h = (xyz_w_h @ w2c.T)[:,:3]
            uvd = (xyz_c_h @ intrinsic.T)
            depth = uvd[:,2:3]
            uv = uvd[:,:2] / depth
            uv = np.round(uv).astype(int)
            u, v = uv[:,0], uv[:,1]

            proj_depth = np.full([H,W], np.inf)
            # mapping = np.zeros_like(proj_depth) - 1
            proj_rgb = np.zeros([H,W,3])
            for i in np.arange(xyz.shape[0]):
                ui, vi, d = u[i], v[i], depth[i]
                if ui<0 or ui>=W:
                    continue
                if vi<0 or vi>=H:
                    continue
                if d<0:
                    continue
                if proj_depth[vi,ui] > d:
                    proj_depth[vi,ui] = d
                    proj_rgb[vi,ui] = rgb[i]
                    # mapping[vi,ui] = i
            proj_rgb = Image.fromarray(proj_rgb.astype(np.uint8))
            proj_rgb.save(f'{out_dir}/{j:03d}.png')
            # np.save(f'{out_dir}/{j:03d}.npy',mapping)

if __name__ == '__main__':
    dataset = 's3dis'
    split = 'test'
    exp = 'base0'
    out_path = 'z_projection'

    main(out_path,dataset,split,exp)