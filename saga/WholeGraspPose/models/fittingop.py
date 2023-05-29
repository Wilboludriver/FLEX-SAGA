import os
import sys

sys.path.append(os.getcwd())
import copy
import json
import pickle
import time

import numpy as np
import open3d as o3d
import smplx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from human_body_prior.tools.model_loader import load_vposer
from torch.autograd import Variable
from utils.train_helper import EarlyStopping, point2point_signed
from utils.utils import RotConverter

# for flex
import kaolin
from utils import pytorch_geometric
import scipy.sparse as sp
from collections import Counter



class FittingOP:
    def __init__(self, fittingconfig):

        for key, val in fittingconfig.items():
            setattr(self, key, val)

        body_model_path = './body_utils/body_models'
        self.bm_male = smplx.create(body_model_path, model_type='smplx',
                                    gender='male', ext='npz',
                                    num_pca_comps=self.hand_ncomps,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=self.batch_size
                                    )
        self.bm_female = smplx.create(body_model_path, model_type='smplx',
                                    gender='female', ext='npz',
                                    num_pca_comps=self.hand_ncomps,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=self.batch_size
                                    )
        self.bm_male.to(self.device)
        self.bm_female.to(self.device)
        self.bm_male.eval()
        self.bm_female.eval()
        self.bm = None
        self.vposer, _ = load_vposer(body_model_path+'/vposer_v1_0', vp_model='snapshot')
        self.vposer.to(self.device)
        self.vposer.eval()
        self.fittingconfig = fittingconfig

        ## setup optim variables
        self.betas = Variable(torch.zeros(self.batch_size,10).to(self.device), requires_grad=True)
        self.transl_rec = Variable(torch.zeros(self.batch_size,3).to(self.device), requires_grad=True)
        self.glo_rot_rec = Variable(torch.FloatTensor([[-1,0,0,0,0,1]]).repeat(self.batch_size,1).to(self.device), requires_grad=True)
        self.vpose_rec = Variable(torch.zeros(self.batch_size,32).to(self.device), requires_grad=True)
        self.hand_pose = Variable(torch.zeros(self.batch_size,2*self.hand_ncomps).to(self.device), requires_grad=True)
        self.eye_pose = Variable(torch.zeros(self.batch_size,6).to(self.device), requires_grad=True)
        self.optimizer_s1 = optim.Adam([self.transl_rec, self.glo_rot_rec], lr=self.init_lr_h*2.0)
        self.optimizer_s2 = optim.Adam([self.betas, self.transl_rec, self.glo_rot_rec, self.vpose_rec], lr=self.init_lr_h*1.5)
        self.optimizer_s3 = optim.Adam([self.vpose_rec, self.hand_pose, self.eye_pose],
                                        lr=self.init_lr_h)
        
        self.optimizers = [self.optimizer_s1,
         self.optimizer_s2,
         self.optimizer_s3]

        self.v_weights = torch.from_numpy(np.load(self.cfg.c_weights_path)).to(torch.float32).to(self.device)
        self.v_weights2 = torch.pow(self.v_weights, 1.0 / 2.5)

        with open('./body_utils/smplx_markerset.json') as f:
            markerset = json.load(f)['markersets']

            self.markers_143 = []
            for marker in markerset:
                if marker['type'] not in ['palm_5']:
                    self.markers_143 += list(marker['indices'].values())

        mano_fname = './body_utils/smplx_mano_flame_correspondences/MANO_SMPLX_vertex_ids.pkl'
        with open(mano_fname, 'rb') as f:
            idxs_data = pickle.load(f)
            self.rhand_verts = idxs_data['right_hand']
            self.lhand_verts = idxs_data['left_hand']

        body_segments_dir = './body_utils/body_segments'
        with open(os.path.join(body_segments_dir, 'L_Leg.json'), 'r') as f:
            data = json.load(f)
            left_foot_verts_id = np.asarray(list(set(data["verts_ind"])))
        left_heel_verts_id = np.load(
            './body_utils/left_heel_verts_id.npy')
        left_toe_verts_id = np.load(
            './body_utils/left_toe_verts_id.npy')
        self.left_heel_verts_id = left_foot_verts_id[left_heel_verts_id]
        self.left_toe_verts_id = left_foot_verts_id[left_toe_verts_id]

        with open(os.path.join(body_segments_dir, 'R_Leg.json'), 'r') as f:
            data = json.load(f)
            right_foot_verts_id = np.asarray(list(set(data["verts_ind"])))
        right_heel_verts_id = np.load(
            './body_utils/right_heel_verts_id.npy')
        right_toe_verts_id = np.load(
            './body_utils/right_toe_verts_id.npy')
        self.right_heel_verts_id = right_foot_verts_id[right_heel_verts_id]
        self.right_toe_verts_id = right_foot_verts_id[right_toe_verts_id]
        
        self.foot_markers_all = np.concatenate([self.right_heel_verts_id, self.right_toe_verts_id, self.left_heel_verts_id, self.left_toe_verts_id], axis=0)

        # from flex
        # Misc.
        # self.sbj_verts_region_map = np.load(self.cfg.sbj_verts_region_map_pth, allow_pickle=True)  # (10475,)
        self.adj_matrix_original = np.load(self.cfg.adj_matrix_orig)
        if self.cfg.subsample_sbj:
            self.sbj_verts_id = np.load(self.cfg.sbj_verts_simplified)                             # (625,)
            self.sbj_faces_simplified = np.load(self.cfg.sbj_faces_simplified)
            self.adj_matrix_simplified = np.load(self.cfg.adj_matrix_simplified)

    def init_betas(self, betas):
        self.betas.data = torch.nn.Parameter(torch.FloatTensor(betas).to(self.device).repeat(self.batch_size, 1))

    def reset(self):
        self.betas.data = torch.nn.Parameter(torch.zeros(self.batch_size,10).to(self.device))
        self.transl_rec.data = torch.nn.Parameter(torch.zeros(self.batch_size,3).to(self.device))
        self.glo_rot_rec.data = torch.nn.Parameter(torch.FloatTensor([[-1,0,0,0,0,1]]).to(self.device).repeat(self.batch_size,1))
        self.vpose_rec.data = torch.nn.Parameter(torch.zeros(self.batch_size,32).to(self.device))
        self.hand_pose.data = torch.nn.Parameter(torch.zeros(self.batch_size,2*self.hand_ncomps).to(self.device))
        self.eye_pose.data = torch.nn.Parameter(torch.zeros(self.batch_size,6).to(self.device))
    


    def calc_loss_contact_map(self, body_markers, verts_object, normal_object, contacts_object, contacts_markers, gender, betas, stage, alpha, extras={}):
        """
        :param extras: dict, keys ['ov', 'obj_normals'] (stuff that is not optimized over and that can be copied over from GT and is necessary for loss computation)
        """
        body_param = {}
        body_param['transl'] = self.transl_rec
        body_param['global_orient'] = RotConverter.rotmat2aa(RotConverter.cont2rotmat(self.glo_rot_rec))
        body_param['betas'] = self.betas
        body_param['body_pose'] = self.vposer.decode(self.vpose_rec,
                                           output_type='aa').view(self.batch_size, -1)
        body_param['left_hand_pose'] = self.hand_pose[:,:self.hand_ncomps]
        body_param['right_hand_pose'] = self.hand_pose[:,self.hand_ncomps:]
        body_param['leye_pose'] = self.eye_pose[:,:3]
        body_param['reye_pose'] = self.eye_pose[:,3:]

        output = self.bm(return_verts=True, **body_param)
        verts_full = output.vertices
        joints = output.joints
        body_markers_rec = verts_full[:, self.markers_143, :]
        
        foot_markers_rec = verts_full[:,self.foot_markers_all,:]
        rhand_verts_rec = verts_full[:, self.rhand_verts, :]
        ############################
        # compute normal
        mesh = o3d.geometry.TriangleMesh()
        verts_full_new = verts_full.detach().cpu().numpy()[0]
        mesh.vertices = o3d.utility.Vector3dVector(verts_full_new)
        mesh.triangles = o3d.utility.Vector3iVector(self.faces) # the mesh for the reconstructed human
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)
        rh_normals = torch.tensor(normals[self.rhand_verts, :]).to(torch.float32).to(self.device).view(-1, 778, 3)
        ##################################################
        ##################################################
        # markers reconstruction loss
        loss_rec = torch.mean(torch.abs(body_markers_rec-body_markers.detach()))
        loss_body_rec = torch.mean(torch.abs(body_markers_rec[:, :49, :]-body_markers.detach()[:, :49, :]))

        #################################################
        # foot loss
        loss_foot = 0.1 * torch.mean(torch.abs(foot_markers_rec[:,:,-1]))

        #################################################
        # regularization
        loss_vpose_reg = 0.0001*torch.mean(self.vpose_rec**2)
        loss_hand_pose_reg = 0.0005*torch.mean(self.hand_pose**2)
        loss_eye_pose_reg = 0.0001*torch.mean(self.eye_pose**2)

        #################################################
        # contact map loss
        o2h_marker, h2o_signed_marker, o2h_idx_marker, _ = point2point_signed(body_markers_rec, verts_object.float(), normal_object.float())
        o2h_signed, h2o_signed, o2h_idx, _ = point2point_signed(rhand_verts_rec, verts_object.float(), rh_normals, normal_object.float())

        # map_weight = torch.gather(contacts_markers.view(1, -1), 1, o2h_idx_marker.long())
        # contacts_object_pred = (1 - 2 * (torch.sigmoid(o2h_marker*150)-0.5)) * map_weight
        # loss_contact_map = torch.mean((contacts_object_pred-contacts_object)**2)   # not used

        loss_marker_contact = torch.mean(torch.abs(h2o_signed_marker)*contacts_markers.view(1, -1))
        loss_object_contact = torch.mean(torch.abs(o2h_signed)*contacts_object.view(1, -1))
        ##########################################################
        v_contact = torch.zeros([1, h2o_signed.size(1)]).to(self.device)
        v_collision = torch.zeros([1, h2o_signed.size(1)]).to(self.device)
        v_dist = (h2o_signed < 0.02) * (h2o_signed > 0) * (self.v_weights2[None] > 0.7)
        v_dist_neg = h2o_signed < 0
        v_dist_marker_neg = h2o_signed_marker < 0

        v_contact[v_dist] = 1 * self.v_weights[None][v_dist] #  weight for close vertices
        v_collision[v_dist_neg] = 10 # large weight for penetration

        w = torch.zeros([1, o2h_signed.size(1)]).to(self.device)
        w_dist = (o2h_signed < 0.01) * (o2h_signed > 0)
        w_dist_neg = o2h_signed < 0
        w[w_dist] = 0 # small weight for far away vertices
        w[w_dist_neg] = 20 # large weight for penetration

        f = torch.nn.ReLU()

        loss_prior_contact = 1 * torch.mean(torch.einsum('ij,ij->ij', torch.abs(h2o_signed), v_contact))   # replace with key markers
        h_collision = 1 * torch.mean(torch.einsum('ij,ij->ij', torch.abs(h2o_signed), v_collision))  # keep it
        loss_dist_o = 1 * torch.mean(torch.einsum('ij,ij->ij', torch.abs(o2h_signed), w)) # 

        ############################################################################################################
        # from flex: calculate the human mesh & obstacle mesh penetration loss
        # alpha_obstacle_in=self.cfg.alpha_obstacle_in
        # alpha_obstacle_out=self.cfg.alpha_obstacle_out
        # loss_obstacle_in = loss_obstacle_out = 0
        # if alpha_obstacle_in + alpha_obstacle_out > 0:
        #     loss_obstacle_in, loss_obstacle_out = self.get_obstacle_penet_loss(mesh, extras)
        #     loss_obstacle_in = alpha_obstacle_in * torch.mean(loss_obstacle_in) # avgerage over bs
        #     loss_obstacle_out = alpha_obstacle_out * torch.mean(loss_obstacle_out) # avgerage over bs
        # # print(f"final losss shape: loss_obstacle_in =  {loss_obstacle_in}, loss_obstacle_out = {loss_obstacle_out}")
        
        ############################################################################################################
        loss = (1 * loss_rec
           + (self.only_rec==False) * (self.contact_loss=='contact') * (stage>1) * 15 * (loss_marker_contact+loss_object_contact)  # 2 / 0 / 0
           + (self.only_rec==False) * (self.contact_loss=='prior') * (stage>1) * 15 * loss_prior_contact   # 2 / 0 / 0
           + (self.only_rec==False) * (stage>1)* 10 * (h_collision+loss_dist_o)
           + (stage>1)*loss_hand_pose_reg
           + (stage>1)*loss_eye_pose_reg
           + (stage>0)*loss_vpose_reg
           + 0.1*loss_foot
        #    + loss_obstacle_in # from flex
        #    + loss_obstacle_out # from flex
           )

        loss_dict = {}
        loss_dict['total'] = loss.detach().cpu().numpy()
        loss_dict['rec'] = loss_rec.detach().cpu().numpy()
        loss_dict['body_rec'] = loss_body_rec.detach().cpu().numpy()
        # loss_dict['contact map diff'] = loss_contact_map.detach().cpu().numpy()
        loss_dict['marker contact'] = loss_marker_contact.detach().cpu().numpy()
        loss_dict['object contact'] = loss_object_contact.detach().cpu().numpy()
        loss_dict['prior contact'] = loss_prior_contact.detach().cpu().numpy()
        loss_dict['hand collision'] = h_collision.detach().cpu().numpy()
        loss_dict['object collision'] = loss_dist_o.detach().cpu().numpy()
        loss_dict['foot'] = loss_foot.detach().cpu().numpy()
        loss_dict['reg'] = (loss_vpose_reg+loss_hand_pose_reg+loss_eye_pose_reg).detach().cpu().numpy()
        loss_dict['obstacle in'] = loss_obstacle_in.detach().cpu().numpy()
        loss_dict['obstacle out'] = loss_obstacle_out.detach().cpu().numpy()

        vertices_info = {}
        vertices_info['hand colli'] = torch.where(v_dist_neg==True)[0].size()[0]
        vertices_info['obj colli'] = torch.where(w_dist_neg==True)[0].size()[0]
        vertices_info['contact'] = torch.where((h2o_signed < 0.001) * (h2o_signed > -0.001)==True)[0].size()[0]
        vertices_info['hand markers colli'] = torch.where(v_dist_marker_neg==True)[0].size()[0]

        return loss, loss_dict, body_markers_rec, body_param, vertices_info, rhand_verts_rec, rh_normals, h2o_signed, o2h_signed


    def calc_loss(self, body_markers, verts_object, normal_object, gender, betas, stage, alpha):
        body_param = {}
        body_param['transl'] = self.transl_rec
        body_param['global_orient'] = RotConverter.rotmat2aa(RotConverter.cont2rotmat(self.glo_rot_rec))
        body_param['betas'] = self.betas
        # body_param['betas'] = torch.tensor(betas, dtype=torch.float32, requires_grad=False).repeat(self.batch_size, 1).to(self.device)
        body_param['body_pose'] = self.vposer.decode(self.vpose_rec,
                                           output_type='aa').view(self.batch_size, -1)
        body_param['left_hand_pose'] = self.hand_pose[:,:self.hand_ncomps] #+ self.delta_hand_pose[:,:self.hand_ncomps]
        body_param['right_hand_pose'] = self.hand_pose[:,self.hand_ncomps:] #+ self.delta_hand_pose[:,self.hand_ncomps:]
        body_param['leye_pose'] = self.eye_pose[:,:3]
        body_param['reye_pose'] = self.eye_pose[:,3:]

        output = self.bm(return_verts=True, **body_param)
        verts_full = output.vertices
        body_markers_rec = verts_full[:,self.marker,:]
        foot_markers_rec = verts_full[:,self.foot_marker,:]
        rhand_verts_rec = verts_full[:, self.rhand_verts, :]

        ############################
        # compute normal
        mesh = o3d.geometry.TriangleMesh()
        verts_full_new = verts_full.detach().cpu().numpy()[0]
        mesh.vertices = o3d.utility.Vector3dVector(verts_full_new)
        mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)
        rh_normals = torch.tensor(normals[self.rhand_verts, :]).to(torch.float32).to(self.device).view(-1, 778, 3)
        ##################################################
        loss_rec = torch.mean(torch.abs(body_markers_rec-body_markers.detach()))
        loss_body_rec = torch.mean(torch.abs(body_markers_rec[:, :49, :]-body_markers.detach()[:, :49, :]))

        ##################################################
        # hand contact loss
        o2h_signed, h2o_signed, _ = point2point_signed(rhand_verts_rec, verts_object.float(), rh_normals, normal_object.float())

        v_contact = torch.zeros([1, h2o_signed.size(1)]).to(self.device)
        v_collision = torch.zeros([1, h2o_signed.size(1)]).to(self.device)
        v_dist = (h2o_signed < 0.02) * (h2o_signed > 0) * (self.v_weights2[None] > 0.7)
        v_dist_neg = h2o_signed < 0

        v_contact[v_dist] = 5 * self.v_weights[None][v_dist] #  weight for close vertices
        v_collision[v_dist_neg] = 10 # more weight for penetration

        w = torch.zeros([1, o2h_signed.size(1)]).to(self.device)
        w_dist = (o2h_signed < 0.01) * (o2h_signed > 0)
        w_dist_neg = o2h_signed < 0
        w[w_dist] = 0 # less weight for far away vertices
        w[w_dist_neg] = 20 # more weight for penetration

        f = torch.nn.ReLU()

        h_contact = 1 * torch.mean(torch.einsum('ij,ij->ij', torch.abs(h2o_signed), v_contact))   # replace with key markers
        h_collision = 1 * torch.mean(torch.einsum('ij,ij->ij', torch.abs(h2o_signed), v_collision))  # keep it
        loss_dist_o = 1 * torch.mean(torch.einsum('ij,ij->ij', torch.abs(o2h_signed), w)) # 
        
        loss_pen = 1 * torch.sum(f(-h2o_signed)**2)


        #################################################
        # foot loss
        loss_foot = 0.1 * torch.mean(torch.abs(foot_markers_rec[:,:,-1]-0.02))

        #################################################
        # reg loss
        loss_vpose_reg = 0.0001*torch.mean(self.vpose_rec**2)
        loss_hand_pose_reg = 0.0005*torch.mean(self.hand_pose**2)
        loss_eye_pose_reg = 0.0001*torch.mean(self.eye_pose**2)

        ##################################################
        loss = (5 * loss_rec
           + (self.only_rec==False) * (stage>1)* 2 * alpha * (h_contact + h_collision + loss_dist_o)   # 2 / 0 / 0
           + (stage>1)*loss_hand_pose_reg
           + (stage>1)*loss_eye_pose_reg
           + (stage>0)*loss_vpose_reg
           + (stage<2)*loss_foot
           )

        loss_dict = {}
        loss_dict['total'] = loss.detach().cpu().numpy()
        loss_dict['rec'] = loss_rec.detach().cpu().numpy()
        loss_dict['body_rec'] = loss_body_rec.detach().cpu().numpy()
        loss_dict['hand contact'] = h_contact.detach().cpu().numpy()
        loss_dict['hand collision'] = h_collision.detach().cpu().numpy()
        loss_dict['object collision'] = loss_dist_o.detach().cpu().numpy()
        loss_dict['foot'] = (stage<2)*loss_foot.detach().cpu().numpy()
        loss_dict['reg'] = (loss_vpose_reg+loss_hand_pose_reg+loss_eye_pose_reg).detach().cpu().numpy()

        vertices_info = {}
        vertices_info['hand colli'] = torch.where(v_dist_neg==True)[0].size()[0]
        vertices_info['obj colli'] = torch.where(w_dist_neg==True)[0].size()[0]
        vertices_info['contact'] = torch.where((h2o_signed < 0.001) * (h2o_signed > -0.001)==True)[0].size()[0]

        return loss, loss_dict, body_markers_rec, body_param, vertices_info


    def fitting(self, body_markers, object_contact, markers_contact, verts_object, normal_object, gender, extras, betas=None):
        if gender == 'male':
            self.bm = self.bm_male
        elif gender == 'female':
            self.bm = self.bm_female
        self.faces = self.bm.faces
        # print(self.bm.pose_mean.size())
        early_stopping = EarlyStopping(patience=300)
        smplxparams_list = []
        markers_fit_list = []

        best_eval_grasp = 10000
        early_stop = False
        tmp_info = None

        save_loss = {}
        save_loss['total'] = []
        save_loss['rec'] = []
        save_loss['body_rec'] = []
        save_loss['hand contact'] = []
        save_loss['hand collision'] = []
        save_loss['object collision'] = []
        save_loss['foot'] = []
        save_loss['reg'] = []
        save_loss['hand markers colli'] = []
        save_loss['hand colli'] = []
        save_loss['obj colli'] = []
        save_loss['contact'] = []
        save_loss['contact map diff'] = []
        save_loss['marker contact'] = []
        save_loss['object contact'] = []
        save_loss['prior contact'] = []

        # flex loss
        save_loss['obstacle in'] = []
        save_loss['obstacle out'] = []

        start = time.time()

        for ss, optimizer in enumerate(self.optimizers):
            for ii in range(self.num_iter[ss]):
                alpha = min(ii/self.num_iter[ss]*2, 1)
                optimizer.zero_grad()
                loss, loss_dict, markers_fit, body_param, vertices_info, rhand_verts_rec, rh_normals, h2o_signed, o2h_signed = self.calc_loss_contact_map(body_markers, verts_object, normal_object, object_contact, markers_contact, gender, betas, ss, alpha, extras) 
                loss.backward(retain_graph=False)
                optimizer.step()
                losses_str = ' '.join(['{}: {:.4f} | '.format(x, loss_dict[x]) for x in loss_dict.keys()])
                # print("losses_str", losses_str)
                verts_str = ' '.join(['{}: {} | '.format(x, int(vertices_info[x])) for x in vertices_info.keys()])
                if self.verbose and not (ii+1) % 50:
                    self.logger('[INFO][fitting][stage{:d}] iter={:d}, loss:{:s}, verts_info:{:s}'.format(ss,
                                            ii, losses_str, verts_str))

                    # #### (optional) debug here
                    # # import open3d as o3d 
                    # import sys
                    # object_pcd = o3d.geometry.PointCloud()
                    # rhand_pcd = o3d.geometry.PointCloud()
                    # object_pcd.points = o3d.utility.Vector3dVector(verts_object.squeeze().detach().cpu().numpy())
                    # object_pcd.normals = o3d.utility.Vector3dVector(normal_object.squeeze().detach().cpu().numpy())
                    # rhand_pcd.points = o3d.utility.Vector3dVector(rhand_verts_rec.squeeze().detach().cpu().numpy())
                    # rhand_pcd.normals = o3d.utility.Vector3dVector(rh_normals.squeeze().detach().cpu().numpy())

                    # # print(h2o_signed)
                    # h_in = torch.where(h2o_signed<0)[1].cpu().numpy()
                    # colors_rh = np.zeros((rhand_verts_rec.shape[1], 3))
                    # colors_rh[h_in, 0] = 1
                    # rhand_pcd.colors = o3d.utility.Vector3dVector(colors_rh)

                    # # print(h2o_signed)
                    # o_in = torch.where(o2h_signed<0)[1].cpu().numpy()
                    # print(o_in)
                    # colors_obj = np.zeros((2048, 3))
                    # colors_obj[:, 1] = 1
                    # colors_obj[o_in, 1] = 0
                    # object_pcd.colors = o3d.utility.Vector3dVector(colors_obj)

                    # # o3d.visualization.draw_geometries([rhand_pcd, object_pcd])
                    # # o3d.visualization.draw_geometries([rhand_pcd])
                    # o3d.visualization.draw_geometries([object_pcd])


                eval_grasp = loss
                # eval_grasp = vertices_info['hand colli'] + vertices_info['obj colli']#-8*vertices_info['contact']
                # contact_num = vertices_info['contact']

                for key in loss_dict.keys():
                    save_loss[key] = save_loss[key] + [loss_dict[key]]
                for key in vertices_info.keys():
                    save_loss[key] = save_loss[key] + [vertices_info[key]]
                
                if self.only_rec != True:
                    if ss==2 and ii>200 and eval_grasp < best_eval_grasp:# and contact_num>=5:
                        best_eval_grasp = eval_grasp
                        tmp_smplxparams = {}
                        tmp_smplxparams['transl'] = copy.deepcopy(self.transl_rec).detach()
                        tmp_smplxparams['global_orient'] = RotConverter.rotmat2aa(RotConverter.cont2rotmat(copy.deepcopy(self.glo_rot_rec).detach()))
                        # smplxparams['betas'] = torch.tensor(betas, dtype=torch.float32, requires_grad=False).repeat(self.batch_size, 1).to(self.device)
                        tmp_smplxparams['betas'] = copy.deepcopy(self.betas).detach()
                        tmp_smplxparams['body_pose'] = self.vposer.decode(copy.deepcopy(self.vpose_rec).detach(),
                                                           output_type='aa').view(self.batch_size, -1)
                        tmp_smplxparams['left_hand_pose'] = copy.deepcopy(self.hand_pose).detach()[:,:self.hand_ncomps]
                        tmp_smplxparams['right_hand_pose'] = copy.deepcopy(self.hand_pose).detach()[:,self.hand_ncomps:]
                        tmp_smplxparams['leye_pose'] = copy.deepcopy(self.eye_pose).detach()[:,:3]
                        tmp_smplxparams['reye_pose'] = copy.deepcopy(self.eye_pose).detach()[:,3:]
                        tmp_markers_fit = markers_fit
                        tmp_info = '[stage{:d}] iter={:d}, loss:{:s}, verts_info:{:s}'.format(ss,
                                                    ii, losses_str, verts_str)
                        if self.verbose:
                            self.logger('saving:{}'.format(tmp_info))
                    if ss==2 and ii>200:
                        if early_stopping(eval_grasp):
                            # print(early_stopping.counter)
                            if contact_num < 4:
                                early_stopping.counter = 0
                            else:
                                early_stop = True
                                self.logger('Early stop...')
                                self.logger('Save %s' % tmp_info)
                                break
                    if ss==2 and ii==self.num_iter[ss]-1:
                        early_stop = True
                        self.logger('Save %s' % tmp_info)

            # early_stop = False

            if not early_stop or tmp_info is None:
                # self.logger('No EARLY STOP!')
                smplxparams = {}
                smplxparams['transl'] = copy.deepcopy(self.transl_rec).detach()
                smplxparams['global_orient'] = RotConverter.rotmat2aa(RotConverter.cont2rotmat(copy.deepcopy(self.glo_rot_rec).detach()))
                smplxparams['betas'] = copy.deepcopy(self.betas).detach()
                smplxparams['body_pose'] = self.vposer.decode(copy.deepcopy(self.vpose_rec).detach(),
                                                   output_type='aa').view(self.batch_size, -1)
                smplxparams['left_hand_pose'] = copy.deepcopy(self.hand_pose).detach()[:,:self.hand_ncomps]
                smplxparams['right_hand_pose'] = copy.deepcopy(self.hand_pose).detach()[:,self.hand_ncomps:]
                smplxparams['leye_pose'] = copy.deepcopy(self.eye_pose).detach()[:,:3]
                smplxparams['reye_pose'] = copy.deepcopy(self.eye_pose).detach()[:,3:]
                # print('handpose:', self.hand_pose)

                # print(smplxparams['right_hand_pose'])
                # smplx_copy = copy.deepcopy(smplxparams)

                ### TO FIX THE bug
                smplxparams_list.append(smplxparams)

                markers_fit_list.append(markers_fit.detach().cpu().numpy()[0])
            else:
                smplxparams_list.append(tmp_smplxparams)

                markers_fit_list.append(tmp_markers_fit.detach().cpu().numpy()[0])

        # self.logger('beta after fitting: %s' % str(smplxparams['betas']))
        # time_0 = time.time() - start
        # self.logger('time per sample: %f' % time_0)
        self.reset()
        for key in save_loss.keys():
            save_loss[key] = np.asarray(save_loss[key])
            # print(save_loss[key].shape)


        return markers_fit_list, smplxparams_list, save_loss
    
    # ===========================================================================================================================#

    def intersection(self, sbj_verts, obj_verts, sbj_faces, obj_faces, full_body=True, adjacency_matrix=None):
        """
        Compute intersection penalty between body and object (or obstacle) given vertices and normals of both.

        :param sbj_verts                (torch.Tensor) on device - (bs, N_sbj, 3)
        :param obj_verts                (torch.Tensor) on device - (1, N_obj, 3)
        :param sbj_faces                (torch.Tensor) on device - (F_sbj, 3)
        :param obj_faces                (torch.Tensor) on device - (F_obj, 3)
        :param full_body                (bool) -- for full-body if True; else for rhand
        :param adjacency_matrix         (optional)

        :return penet_loss_batched_in   (torch.Tensor) - (bs,) - loss values for each batch element - penetration
        :return penet_loss_batched_out  (torch.Tensor) - (bs,) - loss values for each batch element - outside
        """
        device = sbj_verts.device
        bs = sbj_verts.shape[0]
        obj_verts = obj_verts.repeat(bs, 1, 1).to(device)                                                                         # (bs, N_obj, 3)
        num_obj_verts, num_sbj_verts = obj_verts.shape[1], sbj_verts.shape[1]
        penet_loss_batched_in, penet_loss_batched_out = torch.zeros(bs).to(device), torch.zeros(bs).to(device)
        thresh = self.cfg.intersection_thresh

        # (*) Object to subject.
        if self.cfg.obstacle_obj2sbj:
            # 1. Use Kaolin to calculate sign (True if inside, False if outside)
            sign = kaolin.ops.mesh.check_sign(sbj_verts, sbj_faces, obj_verts)                                               # (bs, N_obj)
            ones = torch.ones_like(sign.long())                                                                              # (bs, N_obj)
            # 2. Negative for penetration, Positive for outside (to keep consistent with previous format).
            sign = torch.where(sign, -ones, ones)                                                                            # (bs, N_obj)
            # 3. Calculate absolute distance of points from mesh, and multiply by sign.
            face_vertices = kaolin.ops.mesh.index_vertices_by_faces(sbj_verts, sbj_faces)                                    # (bs, F_sbj, 3, 3)
            dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(obj_verts.contiguous(), face_vertices)           # (bs, N_obj)
            obj2sbj = dist * sign                                                                                            # (bs, N_obj)
            # 4. Average across batch for negative and positive values.
            zeros_o2s, ones_o2s = torch.zeros_like(obj2sbj).to(device), torch.ones_like(obj2sbj).to(device)
            loss_o2s_in = torch.sum(abs(torch.where(obj2sbj<thresh, obj2sbj-thresh, zeros_o2s)), 1) / num_obj_verts          # (bs,) -- averaged across (bs, N_obj)
            loss_o2s_out = torch.sum(torch.log(torch.where(obj2sbj>thresh, obj2sbj+ones_o2s, ones_o2s)), 1) / num_obj_verts  # (bs,) -- averaged across (bs, N_obj)
            # 5. Add to final loss.
            penet_loss_batched_in += loss_o2s_in
            penet_loss_batched_out += loss_o2s_out

        # (*) Subject to object.
        if self.cfg.obstacle_sbj2obj:
            # 0. Simplify obstacle faces - many have determinant ~0, i.e., it is a degenerate triangle.
            # NOTE: No need to do for all elements in batch because faces are the same, so just repeat.
            face_vertices = kaolin.ops.mesh.index_vertices_by_faces(obj_verts, obj_faces)                                    # (bs, F_obj, 3, 3)
            indices_good_faces = (face_vertices[0].det().abs() > 0.001)                                                      # (F_obj)
            obj_faces = obj_faces[indices_good_faces]
            face_vertices = face_vertices[0][indices_good_faces][None].repeat(bs, 1, 1, 1)                                   # (bs, F_obj_good, 3, 3)
            # 1. Use Kaolin to calculate sign (True if inside, False if outside)
            sign = kaolin.ops.mesh.check_sign(obj_verts, obj_faces, sbj_verts)                                               # (bs, N_sbj)
            ones = torch.ones_like(sign.long())                                                                              # (bs, N_sbj)
            # 2. Negative for penetration, Positive for outside (to keep consistent with previous format).
            sign = torch.where(sign, -ones, ones)                                                                            # (bs, N_sbj)
            # 3. Calculate absolute distance of points from mesh, and multiply by sign.
            dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(sbj_verts.contiguous(), face_vertices)           # (bs, N_sbj)
            sbj2obj = dist * sign                                                                                            # (bs, N_sbj)
            # 4. Average across batch for negative and positive values.
            zeros_s2o, ones_s2o = torch.zeros_like(sbj2obj).to(device), torch.ones_like(sbj2obj).to(device)
            loss_s2o_out = torch.sum(torch.log(torch.where(sbj2obj>thresh, sbj2obj+ones_s2o, ones_s2o)), 1) / num_sbj_verts  # (bs,)  -- averaged across (bs, N_sbj)
            # 4.1. Special case for sbj2obj negative values - check whether to do connected components or not.
            loss_s2o_in = torch.sum(abs(torch.where(sbj2obj<thresh, sbj2obj-thresh, zeros_s2o)), 1) / num_sbj_verts          # (bs,)  -- averaged across (bs, N_sbj)
            if full_body and self.cfg.obstacle_sbj2obj_extra == 'connected_components' and loss_s2o_in.mean() > 0:
                # Connected components based loss.
                edges = np.stack(np.where(adjacency_matrix))
                num_nodes = adjacency_matrix.shape[0]
                v_to_edges = torch.zeros((num_nodes, edges.shape[1]))
                v_to_edges[edges[0], range(edges.shape[1])] = 1
                v_to_edges[edges[1], range(edges.shape[1])] = 1

                indices_inter = (sbj2obj < thresh)
                v_to_edges = v_to_edges[None].expand(bs, -1, -1).clone()
                v_to_edges[torch.where(indices_inter)] = 0
                edges_indices = v_to_edges.sum(1) == 2
                num_inter_v = indices_inter.sum(-1)

                for i in range(bs):
                    if loss_s2o_in[i] > 0:
                        edges_i = edges[:, edges_indices[i]]
                        adj = pytorch_geometric.to_scipy_sparse_matrix(edges_i, num_nodes=num_nodes)
                        n_components, labels = sp.csgraph.connected_components(adj)

                        n_components -= num_inter_v[i]  # Inside obstacles are not taken into account
                        if n_components > 1:
                            indices_out = torch.ones([num_sbj_verts])
                            indices_out[indices_inter[i]] = 0
                            labels_ = labels[indices_out.bool()]
                            # We penalize only the vertices that are out, but the penalization is wrt the original
                            # edge, not including the threshold.
                            most_common_label = Counter(labels_).most_common()[0][0]
                            penalized_joints = (labels != most_common_label) * indices_out.bool().numpy()
                            loss_s2o_in[i] += sbj2obj[i][penalized_joints].sum() / num_sbj_verts

            # 5. Add to final loss.
            penet_loss_batched_in += loss_s2o_in
            penet_loss_batched_out += loss_s2o_out

        # (*) Return final.
        return penet_loss_batched_in, penet_loss_batched_out


    def get_rh_obstacle_penet_loss(self, rv, rf, extras):
        """
        Compute penetration loss between right-hand grasp and all provided obstacle vertices.

        :param rv                           (torch.Tensor) - (bs, 778, 3)
        :param rf                           (torch.Tensor) - (bs, 1538, 3)
        :param extras                       (dict)         - keys ['ov', 'obj_normals', 'o_verts_wts'] (stuff that is not optimized over and that can be copied over from GT and is necessary for loss computation)

        :return obstacle_loss_batched_in   (torch.tensor) - (bs,)
        :return obstacle_loss_batched_out  (torch.tensor) - (bs,)
        """
        bs = self.batch_size
        obstacle_loss_batched_in, obstacle_loss_batched_out = torch.zeros(bs).to(self.device), torch.zeros(bs).to(self.device)
        for obstacle in extras['obstacles_info']:
            olb_in, olb_out = self.intersection(rv, obstacle['o_verts'][None].to(self.device), rf, obstacle['o_faces'].to(self.device), False)
            obstacle_loss_batched_in += olb_in
            obstacle_loss_batched_out += olb_out
        if len(extras['obstacles_info']):
            obstacle_loss_batched_in /= len(extras['obstacles_info'])
            obstacle_loss_batched_out /= len(extras['obstacles_info'])
        return obstacle_loss_batched_in, obstacle_loss_batched_out


    def get_obstacle_penet_loss(self, bm_output, extras):
        """
        Compute penetration loss between human and all provided obstacle vertices.

        :param bm_output                    (SMPLX body-model output)
        :param extras                       (dict)         - keys ['o_verts', 'o_faces'] (stuff that is not optimized over and that can be copied over from GT and is necessary for loss computation)

        :return obstacle_loss_batched_in   (torch.tensor) - (bs,)
        :return obstacle_loss_batched_out  (torch.tensor) - (bs,)
        """
        # Preliminaries.
        bs = self.batch_size

        # Load subject vertices and faces.
        # bv = bm_output.vertices 
        bv = np.asarray(bm_output.vertices).reshape(bs,-1,3)  # (bs, 10475, 3)
        bv = torch.FloatTensor(bv)
                                                      
        bf = torch.LongTensor(self.faces.astype('float32')).to(self.device)                     # (20908, 3)
        if self.cfg.subsample_sbj:
            bf = torch.LongTensor(self.sbj_faces_simplified).to(self.device)                                                        # (1269, 3)
            bv = bv[:, self.sbj_verts_id, :]                                                    # (bs, 625, 3)
            adjacency_matrix = self.adj_matrix_simplified
        else:
            adjacency_matrix = self.adj_matrix_original

        # Compute loss for each obstacle.
        obstacle_loss_batched_in, obstacle_loss_batched_out = torch.zeros(bs).to(self.device), torch.zeros(bs).to(self.device)
        for obstacle in extras['obstacles_info']:
            olb_in, olb_out = self.intersection(bv.to(self.device), obstacle['o_verts'][None].to(self.device), bf, obstacle['o_faces'].to(self.device), True, adjacency_matrix)
            obstacle_loss_batched_in += olb_in
            obstacle_loss_batched_out += olb_out
        if len(extras['obstacles_info']):
            obstacle_loss_batched_in /= len(extras['obstacles_info'])
            obstacle_loss_batched_out /= len(extras['obstacles_info'])
        # print(f"returned obstacle_loss_batched_in: {obstacle_loss_batched_in}, obstacle_loss_batched_out: {obstacle_loss_batched_out}")
        return obstacle_loss_batched_in, obstacle_loss_batched_out





