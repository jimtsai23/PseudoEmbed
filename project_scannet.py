
import numpy as np
from glob import glob
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import torch
from pytorch3d.io import IO
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    AmbientLights,
    HardPhongShader,
    BlendParams,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor)
from pytorch3d.structures import Pointclouds

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

def preprocess_xyz_01(xyz):
    '''make xyz 0 to 1'''
    xyz = xyz - np.amin(xyz, axis=0)
    xyz = xyz / xyz.max()
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

def generate_map(xyz_raw, label, novel_mask, out_dir):
    xyz = preprocess_xyz_01(xyz_raw) # (0,1)
    xyz_novel = xyz[novel_mask]
    x_len, y_len, z_len = xyz.max(0) - xyz.min(0)
    map_reso = 0.01
    map_xy = np.zeros([int(1/map_reso),int(1/map_reso),3])
    xy_wall = np.round(xyz[label==1,0:2]/map_reso).astype(int)
    xy_novel = np.round(xyz_novel[:,0:2]/map_reso).astype(int)
    for coord in xy_wall:
        try: map_xy[coord[0],coord[1],0] = 1
        except: pass
    for coord in xy_novel:
        try: map_xy[coord[0],coord[1],1] = 1
        except: pass
    map_xy*=255
    map_xy = Image.fromarray(map_xy.astype(np.uint8))
    map_xy.save(f'{out_dir}/map.png')

def pcd2img(xyz, c2w, intrinsic, rgb, novel_mask, out_dir, H, W, idx):
    w2c = np.linalg.inv(c2w)
    xyz_w_h = np.concatenate([xyz, np.ones([xyz.shape[0],1])],axis=1)
    xyz_c_h = (xyz_w_h @ w2c.T)[:,:3]
    uvd = (xyz_c_h @ intrinsic.T)
    depth = uvd[:,2:3]
    uv = uvd[:,:2] / depth
    uv = np.round(uv).astype(int)
    u, v = uv[:,0], uv[:,1]
    proj_depth = np.full([H,W], np.inf)
    # mapping = np.zeros_like(proj_depth) - 1
    proj_novel_mask = np.zeros([H,W])
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
            if proj_novel_mask[vi,ui]:
                continue
            proj_depth[vi,ui] = d
            proj_rgb[vi,ui] = rgb[i]
            if novel_mask[i]:
                proj_novel_mask[vi,ui] = 1
            # mapping[vi,ui] = i
    proj_rgb = Image.fromarray(proj_rgb.astype(np.uint8)) #.resize(size=(320,240))
    proj_rgb.save(f'{out_dir}/{idx:03d}.png')
    # np.save(f'{out_dir}/{j:03d}.npy',mapping)


def main(out_path,split,exp):
    generate_map = False
    pcd2img = False

    novel_cls = [9,10,11,13,16,18]
    out_path = f'{out_path}/{exp}/{split}'
    Path(out_path).mkdir(parents=True,exist_ok=True)

    # set synthetic camera
    H, W = 240, 320
    # H, W = 120, 160
    aspect_ratio = W / H
    fov = 60 / 180 * np.pi
    fx = 0.5 * W / np.tan(0.5*fov)
    fy = fx / aspect_ratio
    print(fx, fy)
    intrinsic = np.eye(3)
    intrinsic[:2,:3] = np.array([
        [fx, 0, W/2],
        [0, fy, H/2],
    ])
    np.save(f'{out_path}/intrinsic.npy',intrinsic)

    num_ref = 8
    ref_deg = 360/num_ref
    xy_azim = np.arange(num_ref) * ref_deg * (np.pi/180)
    x_dir = np.cos(xy_azim)
    y_dir = np.sin(xy_azim)
    xy_ref = np.stack([x_dir,y_dir],axis=1)

    # read scenes
    data_path = f'datasets/ScanNet/scenes/{split}_data'
    file_lst = sorted(glob(f'{data_path}/*'))

    for i, scene_name in enumerate(tqdm(file_lst)):
        scene_id = scene_name.split('/')[-1][:-4]

        scene = np.load(scene_name)
        # print(scene_id, ': ', scene.shape)
        label = scene[:,6]
        ## novel class filter
        novel_mask = label==novel_cls[0]
        for n_cls in novel_cls:
            novel_mask += (label==n_cls)
        if not novel_mask.any(): # no novel classes exist
            print(scene_id)
            continue
        rgb = scene[:,3:6]
        xyz_raw = scene[:,0:3]
        out_dir = out_path + '/' + scene_id
        Path(out_dir).mkdir(parents=True,exist_ok=True)
        if generate_map:
            generate_map(xyz_raw, label, novel_mask, out_dir)
            continue
        
        xyz = preprocess_xyz(xyz_raw) #(-0.5,0.5)
        xyz_novel = xyz[novel_mask]
        nxy = xyz_novel[:,:2]
        nxy_norm = np.linalg.norm(nxy,axis=1)[:,None]
        nxy_dir = nxy / nxy_norm


        active_dir = np.zeros(num_ref)
        thres = np.cos(ref_deg/2 * (np.pi/180))
        xy_region = np.zeros(nxy.shape[0]) - 1
        for i in range(num_ref):
            region_mask = (nxy_dir @ xy_ref[i][None].T)>thres
            xy_region[region_mask.squeeze()] = i
            active_dir[i] = region_mask.sum()

        upaxis = np.array([0,0,1])
        cam_center = np.array([0,0,0.1])
        idx = -1
        for j in range(num_ref):
            idx += 1
            if active_dir[j]<100:
                continue
            # camera pose
            tgt_center = xyz_novel[xy_region==j].mean(0)
            lookat = tgt_center - cam_center
            lookat = normalize(lookat)
            right = normalize(np.cross(lookat, upaxis))
            up = normalize(np.cross(right, lookat))
            c2w = np.eye(4)
            c2w[:3,:3] = np.concatenate([right[:,None],-1*up[:,None],lookat[:,None]],axis=1)
            c2w[:3,3] = cam_center
            np.save(f'{out_dir}/{idx:03d}.npy',c2w)
          
            # projection
            if pcd2img:
                pcd2img(xyz, c2w, intrinsic, rgb, novel_mask, out_dir, H, W, idx)

            # render_pcd
            render_pcd(xyz, c2w, intrinsic, rgb, out_dir, H, W, idx)



def render_pcd(xyz, c2w, intrinsic, rgb, out_dir, H, W, idx):
    xyz = torch.from_numpy(xyz).type(torch.float)
    rgb = torch.from_numpy(rgb).type(torch.float)
    device = "cuda"
    intrinsic_matrix = torch.zeros([4, 4])
    intrinsic_matrix[3, 3] = 1
    point_cloud = Pointclouds(points=[xyz], features=[rgb]).cuda()
    intrinsic = torch.from_numpy(intrinsic).type(torch.float)
    c2w = torch.from_numpy(c2w).type(torch.float)
    w2c = torch.inverse(c2w)
    fx, fy, cx, cy = (
        intrinsic[0, 0],
        intrinsic[1, 1],
        intrinsic[0, 2],
        intrinsic[1, 2],
    )
    width, height = W, H
    rotation_matrix = w2c[:3, :3].permute(1, 0).unsqueeze(0)
    translation_vector = w2c[:3, 3].reshape(-1, 1).permute(1, 0)
    focal_length = -torch.tensor([[fx, fy]])
    principal_point = torch.tensor([[cx, cy]])
    camera = PerspectiveCameras(focal_length=focal_length,
                                principal_point=principal_point,
                                R=rotation_matrix,
                                T=translation_vector,
                                image_size=torch.tensor([[height, width]]),
                                in_ndc=False,
                                device=device)
        
    raster_settings = PointsRasterizationSettings(
            image_size=(height, width), 
            radius = 0.03, #0.01, #0.007,
            points_per_pixel = 10
            )

    rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings) 
    renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color = 255)
        )
    rendered_image = renderer(point_cloud)
    rendered_image = rendered_image[0].cpu().numpy()
    color = rendered_image[..., :3]
    color_image = Image.fromarray((color).astype(np.uint8))
    color_image.save(f'{out_dir}/{idx:03d}.png')


if __name__ == '__main__':
    split = 'train'
    exp = 'pt_pcd_camz01'
    out_path = 'z_projection_scannet'

    main(out_path,split,exp)