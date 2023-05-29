import argparse
import os
import sys
from collections import defaultdict

import numpy as np
# import smplx
import open3d as o3d
import torch
from smplx.lbs import batch_rodrigues
from tqdm import tqdm

from saga.utils.cfg_parser import Config
from saga.utils.utils import makelogger, makepath, rotmat2aa
from saga.WholeGraspPose.models.fittingop import FittingOP
from saga.WholeGraspPose.models.objectmodel import ObjectModel
from saga.WholeGraspPose.trainer import Trainer

def get_obstacle_info(obstacles_dict):
       """
       From obstacle_list with obstacle name, position and orientation, get vertices and normals from Mesh.

       :param obstacles_dict        (dict) with keys containing obstacle names and values list of [verts, faces]

       :return obstacles_info       (list) containing dicts with keys ['o_verts', 'o_faces'] - each a torch.Tensor.
       """
       obstacles_info = []
       for _, [verts, faces] in obstacles_dict.items():
            obj_verts = torch.from_numpy(verts.astype('float32'))  # (N, 3)
            obj_faces = torch.LongTensor(faces.astype('float32'))  # (F, 3)
            obstacles_info.append({'o_verts': obj_verts, 'o_faces': obj_faces})
       return obstacles_info

def load_obj_with_receptacle(obj_name, receptacle_name, ornt_name, idx):
    
    obj_meshes_dir = './dataset/contact_meshes'
    receptacles_path = './dataset/replicagrasp/receptacles.npz'
    dset_path = './dataset/replicagrasp/dset_info.npz'

    # === Load replica dataset from file.
    recept_dict = dict(np.load(receptacles_path, allow_pickle=1))
    dset_info_dict = dict(np.load(dset_path, allow_pickle=1))
    transl_grab, orient_grab, recept_idx = dset_info_dict[f'{obj_name}_{receptacle_name}_{ornt_name}_{idx}']
    test_pose = [torch.Tensor(transl_grab), rotmat2aa(torch.Tensor(orient_grab).reshape(1,1,1,9)).reshape(1,3)] # the pose of the object in the scence
    recept_v, recept_f = recept_dict[receptacle_name][recept_idx][0], recept_dict[receptacle_name][recept_idx][1]
    obstacles_dict = {receptacle_name: [recept_v, recept_f]}

    # # flex'way tp get object mesh
    # object_mesh = Mesh(filename=os.path.join(obj_meshes_dir, obj_name + '.ply'), vscale=1.)
    # object_mesh.reset_normals()
    # object_mesh = [object_mesh.v, object_mesh.f]

    transf_transl, global_orient = test_pose[0], test_pose[1]
    transl = torch.zeros_like(transf_transl) # for object model which is centered at object

    # get the object info
    obj_mesh_base = o3d.io.read_triangle_mesh(os.path.join(obj_meshes_dir, obj_name + '.ply'))
    obj_mesh_base.compute_vertex_normals()
    v_temp = torch.FloatTensor(np.asarray(obj_mesh_base.vertices)).to(grabpose.device).view(1, -1, 3)
    normal_temp = torch.FloatTensor(np.asarray(obj_mesh_base.vertex_normals)).to(grabpose.device).view(1, -1, 3)
    obj_model = ObjectModel(v_temp, normal_temp, 1)
    global_orient_rotmat = batch_rodrigues(global_orient.view(-1, 3)).to(grabpose.device)  

    object_output = obj_model(global_orient_rotmat, transl.to(grabpose.device), v_temp.to(grabpose.device), normal_temp.to(grabpose.device), rotmat=True)
    object_verts = object_output[0].detach().cpu().numpy()
    object_normal = object_output[1].detach().cpu().numpy()

    index = np.linspace(0, object_verts.shape[1], num=2048, endpoint=False,retstep=True,dtype=int)[0]
    
    verts_object = object_verts[:, index]
    normal_object = object_normal[:, index]
    global_orient_rotmat_6d = global_orient_rotmat.view(-1, 1, 9)[:, :, :6].detach().cpu().numpy()
    feat_object = np.concatenate([normal_object, global_orient_rotmat_6d.repeat(2048, axis=1)], axis=-1)
    
    verts_object = torch.from_numpy(verts_object).to(grabpose.device)
    feat_object = torch.from_numpy(feat_object).to(grabpose.device)
    transf_transl = transf_transl.to(grabpose.device)

    object_info = {'verts_object':verts_object, 'normal_object': normal_object, 'global_orient':global_orient, 
                   'global_orient_rotmat':global_orient_rotmat, 'feat_object':feat_object, 'transf_transl':transf_transl}

    # get the obstacle info (vertices, normals) required for penetration loss computation.
    obstacles_info = get_obstacle_info(obstacles_dict)

    # Define if optimization is over transl and/or global orient.
    extras = {'obstacles_info': obstacles_info, 'recept_idx': recept_idx, 'receptacle_name': receptacle_name}
    return object_info, extras


#### inference, conditioned ONLY on the object

def inference(grabpose, obj_data, n_rand_samples, save_dir):
    """ prepare test object data: [verts_object, feat_object(normal + rotmat), transf_transl] 
    :params obj_data: dict, info about the object
    :n_rand_samples: number of random samples generated per object
    """
    ### object centered
    # for obj in grabpose.cfg.object_class:
    obj_data['feat_object'] = obj_data['feat_object'].permute(0,2,1)
    obj_data['verts_object'] = obj_data['verts_object'].permute(0,2,1)

    n_samples_total = obj_data['feat_object'].shape[0]

    markers_gen = []
    object_contact_gen = []
    markers_contact_gen = []
    for i in range(n_samples_total):
        sample_results = grabpose.full_grasp_net.sample(obj_data['verts_object'][None, i].repeat(n_rand_samples,1,1), obj_data['feat_object'][None, i].repeat(n_rand_samples,1,1), obj_data['transf_transl'][None, i].repeat(n_rand_samples,1))
        markers_gen.append((sample_results[0]+obj_data['transf_transl'][None, i]))
        markers_contact_gen.append(sample_results[1])
        object_contact_gen.append(sample_results[2])

    markers_gen = torch.cat(markers_gen, dim=0)   # [B, N, 3]
    object_contact_gen = torch.cat(object_contact_gen, dim=0).squeeze()   # [B, 2048]
    markers_contact_gen = torch.cat(markers_contact_gen, dim=0)   # [B, N]

    output = {}
    output['markers_gen'] = markers_gen.detach().cpu().numpy()
    output['markers_contact_gen'] = markers_contact_gen.detach().cpu().numpy()
    output['object_contact_gen'] = object_contact_gen.detach().cpu().numpy()
    output['normal_object'] = obj_data['normal_object']#.repeat(n_rand_samples, axis=0)
    output['transf_transl'] = obj_data['transf_transl'].detach().cpu().numpy()#.repeat(n_rand_samples, axis=0)
    output['global_orient_object'] = obj_data['global_orient'].detach().cpu().numpy()#.repeat(n_rand_samples, axis=0)
    output['global_orient_object_rotmat'] = obj_data['global_orient_rotmat'].detach().cpu().numpy()#.repeat(n_rand_samples, axis=0)
    output['verts_object'] = (obj_data['verts_object']+obj_data['transf_transl'].view(-1,3,1).repeat(1,1,2048)).permute(0, 2, 1).detach().cpu().numpy()#.repeat(n_rand_samples, axis=0)

    save_path = os.path.join(save_dir, 'markers_results.npy')
    np.save(save_path, output)
    print('Saving to {}'.format(save_path))

    return output

def fitting_data_save(save_data,
              markers,
              markers_fit,
              smplxparams,
              gender,
              object_contact, body_contact,
              object_name, verts_object, global_orient_object, transf_transl_object, 
              extras):
    # markers & markers_fit
    save_data['markers'].append(markers)
    save_data['markers_fit'].append(markers_fit)
    # print('markers:', markers.shape)

    # body params
    for key in save_data['body'].keys():
        # print(key, smplxparams[key].shape)
        save_data['body'][key].append(smplxparams[key].detach().cpu().numpy())
    # object name & object params
    save_data['object_name'].append(object_name)
    save_data['gender'].append(gender)
    save_data['object']['transl'].append(transf_transl_object)
    save_data['object']['global_orient'].append(global_orient_object)
    save_data['object']['verts_object'].append(verts_object)

    # contact
    save_data['contact']['body'].append(body_contact)
    save_data['contact']['object'].append(object_contact)

    # obstacles
    save_data['obstacle']['recept_idx'].append(extras['recept_idx'])
    save_data['obstacle']['receptacle_name'].append(extras['receptacle_name'])

#### fitting: optimization taking the obstacles into account

def pose_opt(grabpose, samples_results, n_random_samples, obj, gender, save_dir, logger, device, extras):
    """
    :params extras: dict, info about the obstacle
    """
    # prepare objects
    n_samples = len(samples_results['verts_object'])
    verts_object = torch.tensor(samples_results['verts_object'])[:n_samples].to(device)  # (n, 2048, 3)
    normals_object = torch.tensor(samples_results['normal_object'])[:n_samples].to(device)  # (n, 2048, 3)
    global_orients_object = torch.tensor(samples_results['global_orient_object'])[:n_samples].to(device)  # (n, 2048, 3)
    transf_transl_object = torch.tensor(samples_results['transf_transl'])[:n_samples].to(device)  # (n, 2048, 3)

    # prepare body markers
    markers_gen = torch.tensor(samples_results['markers_gen']).to(device)  # (n*k, 143, 3)
    object_contacts_gen = torch.tensor(samples_results['object_contact_gen']).view(markers_gen.shape[0], -1, 1).to(device)  #  (n, 2048, 1)
    markers_contacts_gen = torch.tensor(samples_results['markers_contact_gen']).view(markers_gen.shape[0], -1, 1).to(device)   #  (n, 143, 1)

    print('Fitting {} {} samples for {} at the repreptacle {}...'.format(n_samples, cfg.gender, obj.upper(), extras['receptacle_name']))

    fittingconfig={ 'init_lr_h': 0.008,
                            'num_iter': [300,400,500],
                            'batch_size': 1,
                            'num_markers': 143,
                            'device': device,
                            'cfg': cfg,
                            'verbose': False,
                            'hand_ncomps': 24,
                            'only_rec': False,     # True / False 
                            'contact_loss': 'contact',  # contact / prior / False
                            'logger': logger,
                            'data_type': 'markers_143',                        
                            }
    fittingop = FittingOP(fittingconfig)

    save_data_gen = {}
    for data in [save_data_gen]:
        data['markers'] = []
        data['markers_fit'] = []
        data['body'] = {}
        for key in ['betas', 'transl', 'global_orient', 'body_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose']:
            data['body'][key] = []
        data['object'] = {}
        for key in ['transl', 'global_orient', 'verts_object']:
            data['object'][key] = []
        data['contact'] = {}
        for key in ['body', 'object']:
            data['contact'][key] = []
        data['gender'] = []
        data['object_name'] = []
        
        data['obstacle'] = {}
        for key in ['recept_idx', 'receptacle_name']:
            data['obstacle'][key] = []


    for i in tqdm(range(n_samples)):
        # prepare object 
        vert_object = verts_object[None, i, :, :]
        normal_object = normals_object[None, i, :, :]

        marker_gen = markers_gen[i*n_random_samples:(i+1)*n_random_samples, :, :]
        object_contact_gen = object_contacts_gen[i*n_random_samples:(i+1)*n_random_samples, :, :]
        markers_contact_gen = markers_contacts_gen[i*n_random_samples:(i+1)*n_random_samples, :, :]

        for k in range(n_random_samples):
            print('Fitting for {}-th GEN...'.format(k+1))
            markers_fit_gen, smplxparams_gen, loss_gen = fittingop.fitting(marker_gen[None, k, :], object_contact_gen[None, k, :], markers_contact_gen[None, k], vert_object, normal_object, gender, extras)
            fitting_data_save(save_data_gen,
                    marker_gen[k, :].detach().cpu().numpy().reshape(1, -1 ,3),
                    markers_fit_gen[-1].squeeze().reshape(1, -1 ,3),
                    smplxparams_gen[-1],
                    gender,
                    object_contact_gen[k].detach().cpu().numpy().reshape(1, -1), markers_contact_gen[k].detach().cpu().numpy().reshape(1, -1),
                    obj, vert_object.detach().cpu().numpy(), global_orients_object[i].detach().cpu().numpy(), transf_transl_object[i].detach().cpu().numpy(),
                    extras)


    for data in [save_data_gen]:
        # for data in [save_data_gt, save_data_rec, save_data_gen]:
            data['markers'] = np.vstack(data['markers'])  
            data['markers_fit'] = np.vstack(data['markers_fit'])
            for key in ['betas', 'transl', 'global_orient', 'body_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose']:
                data['body'][key] = np.vstack(data['body'][key])
            for key in ['transl', 'global_orient', 'verts_object']:
                data['object'][key] = np.vstack(data['object'][key])
            for key in ['body', 'object']:
                data['contact'][key] = np.vstack(data['contact'][key])
            for key in ['recept_idx', 'receptacle_name']:
                data['obstacle'][key] = np.vstack(data['obstacle'][key])

    np.savez(os.path.join(save_dir, 'fitting_results.npz'), **save_data_gen)
    
    
def findz()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='grabpose-Testing')

    parser.add_argument('--data_path', default = './dataset/GraspPose', type=str,
                        help='The path to the folder that contains grabpose data')

    parser.add_argument('--gender', default=None, type=str,
                        help='The gender of dataset')

    parser.add_argument('--exp_name', default = None, type=str,
                        help='experiment name')

    parser.add_argument('--pose_ckpt_path', default = None, type=str,
                        help='checkpoint path')

    parser.add_argument('--n_rand_samples_per_object', default = 1, type=int,
                        help='The number of whole-body poses random samples generated per object')

    parser.add_argument('--obj_name', default = 'camera', type=str, 
                        help='Name of the object.')

    parser.add_argument('--receptacle_name', default = 'receptacle_aabb_Tbl2_Top2_frl_apartment_table_02', type=str, 
                        help='Name of the receptacle.')

    parser.add_argument('--ornt_name', default = 'up', type=str, 
                        help='Orientation -- all or up.')

    parser.add_argument('--index', default=0, type=int, 
                        help='0/1/2')

    

    args = parser.parse_args()

    cwd = os.getcwd()

    best_net = os.path.join(cwd, args.pose_ckpt_path)

    vpe_path  = '/configs/verts_per_edge.npy'
    c_weights_path = cwd + '/WholeGraspPose/configs/rhand_weight.npy'
    work_dir = cwd + '/results/{}/GraspPose'.format(args.exp_name)

    print(work_dir)
    config = {
        'dataset_dir': args.data_path,
        'work_dir':work_dir,
        'vpe_path': vpe_path,
        'c_weights_path': c_weights_path,
        'exp_name': args.exp_name,
        'gender': args.gender,
        'best_net': best_net
    }

    cfg_path = 'WholeGraspPose/configs/WholeGraspPose.yaml'
    cfg = Config(default_cfg_path=cfg_path, **config)

    save_dir = os.path.join(work_dir, args.obj_name, args.receptacle_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)        

    logger = makelogger(makepath(os.path.join(save_dir, f'{args.obj_name}_{args.receptacle_name}.log'), isfile=True)).info
    
    grabpose = Trainer(cfg=cfg, inference=True, logger=logger)

    # load the object and the obstacles
    object_info, extras = load_obj_with_receptacle(args.obj_name, args.receptacle_name, args.ornt_name, args.index)

    
    samples_results = inference(grabpose, object_info, args.n_rand_samples_per_object, save_dir)
    fitting_results = pose_opt(grabpose, samples_results, args.n_rand_samples_per_object, args.obj_name, 
                               cfg.gender, save_dir, logger, grabpose.device, extras)
    
    

    
    
    

