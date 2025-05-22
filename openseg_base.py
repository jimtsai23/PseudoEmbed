import os
import torch
import imageio
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from os.path import join, exists
from fusion_util import extract_openseg_img_feature, PointCloudToImageMapper
from pathlib import Path

def preprocess_xyz(xyz):
    '''make xyz 0 to 1, then center at origin (-0.5,0.5)'''
    xyz = xyz - np.amin(xyz, axis=0)
    xyz = xyz / xyz.max()
    center = np.amax(xyz, axis=0) / 2
    xyz = xyz - center
    return xyz


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on ScanNet.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--openseg_model', type=str, default='', help='Where is the exported OpenSeg model')
    parser.add_argument('--exp', type=str, default='')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args

def save_fused_feature(feat_bank, point_ids, n_points, out_dir, scene_id):
    '''Save features.'''

    mask = torch.zeros(n_points, dtype=torch.bool)
    mask[point_ids] = True

    n = 0
    torch.save({"feat": feat_bank[mask].half().cpu(),
                "mask_full": mask
    },  os.path.join(out_dir, scene_id +'_%d.pt'%(n)))

def process_one_scene(data_path, out_dir, args):
    '''Process one scene.'''

    # short hand
    scene_id = data_path.split('/')[-1][:-4]
    feat_dim = args.feat_dim
    point2img_mapper = args.point2img_mapper
    openseg_model = args.openseg_model
    text_emb = args.text_emb
    keep_features_in_memory = args.keep_features_in_memory

    # load 3D data (point cloud)
    pcd = np.load(data_path)
    label = pcd[:,6:7]
    rgb = pcd[:,3:6]
    xyz_raw = pcd[:,0:3]
    locs_in = preprocess_xyz(xyz_raw)
    n_points = locs_in.shape[0]

    # short hand for processing 2D features
    scene = join(args.data_root_2d, scene_id,'*.png')
    img_dirs = sorted(glob(scene), key=lambda x: int(os.path.basename(x)[:-4]))
    num_img = len(img_dirs)
    if num_img==0:
        print(scene_id, num_img)
        return
    device = torch.device('cpu')

    # extract image features and keep them in the memory
    # default: False (extract image on the fly)
    if keep_features_in_memory and openseg_model is not None:
        img_features = []
        for img_dir in tqdm(img_dirs):
            img_features.append(extract_openseg_img_feature(img_dir, openseg_model, text_emb, img_size=[240, 320]))

    n_points_cur = n_points
    counter = torch.zeros((n_points_cur, 1), device=device)
    sum_features = torch.zeros((n_points_cur, feat_dim), device=device)

    ################ Feature Fusion ###################
    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    for idx, img_dir in enumerate(img_dirs):
        img_id = int(img_dir.split('/')[-1].split('.')[0])
        # load pose
        if args.dataset=='s3dis':
            pose = np.load(f'{args.data_root_2d}/pose/{img_id}.npy')
        elif args.dataset=='scannet':
            pose = np.load(img_dir.replace('.png','.npy'))

        # calculate the 3d-2d mapping
        mapping = np.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue

        mapping = torch.from_numpy(mapping).to(device)
        mask = mapping[:, 3]
        vis_id[:, idx] = mask
        if keep_features_in_memory:
            pass
        else:
            feat_2d = extract_openseg_img_feature(img_dir, openseg_model, text_emb, img_size=[240, 320]).to(device)

        feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)

        counter[mask!=0]+= 1
        sum_features[mask!=0] += feat_2d_3d[mask!=0]

    counter[counter==0] = 1e-5
    feat_bank = sum_features/counter
    point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

    save_fused_feature(feat_bank, point_ids, n_points, out_dir, scene_id)

def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # #!### Dataset specific parameters #####
    img_dim = (320, 240)
    visibility_threshold = 0.25 # threshold for the visibility check

    args.cut_num_pixel_boundary = 10 # do not use the features on the image boundary
    args.keep_features_in_memory = False # keep image features in the memory, very expensive
    args.feat_dim = 768 # CLIP feature dimension

    if args.dataset=='s3dis':
        data_root = f'datasets/S3DIS/scenes/{args.split}_data'
        data_root_2d = 'z_projection' + '/' + args.dataset + '/' + args.exp + '/' + args.split
        out_dir = args.output_dir + '/' + args.dataset + '/' + args.exp + '/' + args.split
    elif args.dataset=='scannet':
        data_root = f'datasets/ScanNet/scenes/{args.split}_data'
        data_root_2d = 'z_projection_' + args.dataset + '/' + args.exp + '/' + args.split
        out_dir = args.output_dir + '_' + args.dataset + '/' + args.exp + '/' + args.split

    args.data_root_2d = data_root_2d
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    process_id_range = None #args.process_id_range

    args.n_split_points = 2000000
    args.num_rand_file_per_scene = 1

    # load the openseg model
    saved_model_path = args.openseg_model
    args.text_emb = None
    if args.openseg_model != '':
        print(saved_model_path)
        args.openseg_model = tf2.saved_model.load(saved_model_path,
                    tags=[tf.saved_model.tag_constants.SERVING],)
        args.text_emb = tf.zeros([1, 1, args.feat_dim])
    else:
        args.openseg_model = None

    intrinsic = np.load(f'{data_root_2d}/intrinsic.npy')
    print('intrinsic,', intrinsic)

    # calculate image pixel-3D points correspondances
    args.point2img_mapper = PointCloudToImageMapper(
            image_dim=img_dim, intrinsics=intrinsic,
            visibility_threshold=visibility_threshold,
            cut_bound=args.cut_num_pixel_boundary)

    data_paths = sorted(glob(join(data_root, '*.npy')))
    total_num = len(data_paths)

    id_range = None
    if process_id_range is not None:
        id_range = [int(process_id_range[0].split(',')[0]), int(process_id_range[0].split(',')[1])]

    for i in trange(total_num):
        if id_range is not None and \
           (i<id_range[0] or i>id_range[1]):
            print('skip ', i, data_paths[i])
            continue
        
        process_one_scene(data_paths[i], out_dir, args)



if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)
    main(args)

