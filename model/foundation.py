
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


from model.dgcnn import DGCNN
from model.attention import SelfAttention


manual_seed=321
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
random.seed(manual_seed)


class SelfDistil(nn.Module):
    def __init__(self, args=None, main_dim=256):
        super().__init__()

        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)
        self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)

        self.feat_dim = args.edgeconv_widths[0][-1] + args.output_dim + args.base_widths[-1]

        self.args = args
        self.fusion = nn.Sequential(nn.Conv1d(in_channels=self.feat_dim, out_channels=main_dim, kernel_size=1),
                                    nn.BatchNorm1d(main_dim),
                                    nn.LeakyReLU(0.2))
                                    # nn.Conv1d(in_channels=main_dim, out_channels=main_dim, kernel_size=1))
        
    def forward(self, x):
        edge_convs, feat_level2 = self.encoder(x)
        feat_level3 = self.base_learner(feat_level2)
        att_feat = self.att_learner(feat_level2)
        feat_level1 = edge_convs[0] # (b, d, n)
        pnt_feat = torch.cat((feat_level1, att_feat, feat_level3), dim=1) # (b, d, n)
        pnt_feat = self.fusion(pnt_feat)

        return pnt_feat


class CAPL(nn.Module):
    def __init__(self, classes=13, criterion=nn.CrossEntropyLoss(), args=None, main_dim=256):
        super(CAPL, self).__init__()

        self.criterion = criterion
        self.classes = classes  # 13
        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)
        self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)

        self.feat_dim = args.edgeconv_widths[0][-1] + args.output_dim + args.base_widths[-1]

        self.args = args
        self.fusion = nn.Sequential(nn.Conv1d(in_channels=self.feat_dim, out_channels=main_dim, kernel_size=1),
                                    nn.BatchNorm1d(main_dim),
                                    nn.LeakyReLU(0.2))
        
        self.loss_fn = args.loss
        # 
        
    def forward(self, x, fm_feat=None, label=None, is_eval=False, fm_feat_test=None, base_num=None, is_label=False):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return:
        segment_feature: features with shape (k+d, m1+m2...). m1 is the number of segment for query 1.
        point_feature: (b, d, n)
        """
        loss_ctt = 0
        edge_convs, feat_level2 = self.encoder(x)
        feat_level3 = self.base_learner(feat_level2)
        att_feat = self.att_learner(feat_level2)
        feat_level1 = edge_convs[0] # (b, d, n)
        semantic_feat = torch.cat((feat_level1, att_feat, feat_level3), dim=1) # (b, d, n)
        semantic_feat = self.fusion(semantic_feat)
        if self.args.use_dct:
            semantic_feat = semantic_feat / semantic_feat.norm(dim=1, keepdim=True)
            attn = semantic_feat.permute(0,2,1) @ fm_feat_test.T # (b,n, N_feat)
            res = F.softmax(attn,dim=-1) @ fm_feat_test
            semantic_feat += res.permute(0,2,1)

        if is_eval:
            return semantic_feat, edge_convs

        similarity = F.cosine_similarity(semantic_feat[:,None],fm_feat[None,...,None],dim=2) # sem:(B,1,d,N), feat:(1,N_feat,d,1), sim:(B,N_feat,N)
        if is_label:
            return similarity.max(1)[1]
        if self.args.use_calibration!=0:
            similarity[:,:7,:] -= self.args.use_calibration
        if self.loss_fn == 'ce':
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fn(similarity, label)
        elif self.loss_fn == 'cos':
            x_feat = semantic_feat.permute(0,2,1).reshape(-1,512)
            label = label.view(-1)
            loss = 0
            for idx in label.unique():
                if idx>0:
                    loss += 1 - F.cosine_similarity(x_feat[label==idx],fm_feat[idx][None]).mean()
        elif self.loss_fn == 'ctt': # contrastive
            x_feat = semantic_feat.permute(0,2,1).reshape(-1,512)
            label = label.view(-1)
            sim_l = 1 - F.cosine_similarity(x_feat[:,None],fm_feat_test[None],dim=-1) # (N_p, N_feat_test), (0 most similar,2)
            sim_l_b = sim_l[:,:base_num]
            sim_l_n = sim_l[:,base_num:]
            slb_min = sim_l_b.min(1)[0]
            sln_min = sim_l_n.min(1)[0]
            diff = sln_min - slb_min
            diff_l = diff[(label==0)&(diff>0)]
            loss_ctt = diff_l.mean() if diff_l.shape[0]>0 else 0
            loss = 0
            for idx in label.unique():
                if idx>0:
                    loss += 1 - F.cosine_similarity(x_feat[label==idx],fm_feat[idx][None]).mean()

        else:
            raise TypeError('loss is not defined!')
            
        # return similarity, loss
        return similarity.max(1)[1], loss, loss_ctt
        

class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, in_channels, params):
        super(BaseLearner, self).__init__()

        self.num_convs = len(params)
        self.convs = nn.ModuleList()

        for i in range(self.num_convs):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = params[i-1]
            self.convs.append(nn.Sequential(
                              nn.Conv1d(in_dim, params[i], 1),
                              nn.BatchNorm1d(params[i])))

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.convs[i](x)
            if i != self.num_convs-1:
                x = F.relu(x)
        return x
    


######

class PseudoCAPL(nn.Module):
    def __init__(self, classes=13, criterion=nn.CrossEntropyLoss(), args=None, base_num=7, main_dim=256, fm_feat=None):
        super(PseudoCAPL, self).__init__()
        assert classes > 1
        self.criterion = criterion
        self.classes = classes  # 13
        # define model
        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)
        self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)

        self.feat_dim = args.edgeconv_widths[0][-1] + args.output_dim + args.base_widths[-1]
        
        self.args = args
        self.fusion = nn.Sequential(nn.Conv1d(in_channels=self.feat_dim, out_channels=main_dim, kernel_size=1),
                                    nn.BatchNorm1d(main_dim),
                                    nn.LeakyReLU(0.2))
        
        self.base_num = base_num
        self.novel_num = classes - base_num
        self.main_proto = nn.Parameter(torch.randn((classes, main_dim)))  # (13, 192) this is for the testing classes
        self.bg_proto = nn.Parameter(torch.randn((1, main_dim))) # during base training, set all the novel classes as bg. no use in the test.
        # if fm_feat is not None:
        #     with torch.no_grad():
        #         self.main_proto[:base_num] = self.main_proto[:base_num]*0.01 + fm_feat.requires_grad_()

    def Get_Fg_Feat(self, x, y):
        '''
        Args:
            x: (1, d, 2048)
            y: (1, 2048) binary mask. support point cloud
        Returns: fg feature (n, d)
        '''
        y = y[0] # (2048)
        # get feature of x
        point_feat = self.getFeatures(x)  # (1, c, 2048)
        point_feat = point_feat[0] # (d, 2048)
        fg_feat = point_feat[:,y==1] # (d, n)

        return fg_feat.transpose(1,0) # (n,k)


    def forward(self, x, y=None, px=None, py=None, gened_proto=None, eval_model=False):

        base_num = self.base_num
        point_feat = self.getFeatures(x) # (k+d, m1+m2..) . (b, k_d, n)

        if eval_model:
            #### evaluation
            if len(gened_proto.size()[:]) == 3:
                gened_proto = gened_proto[0] # p_orig

            refine_proto = self.post_refine_proto_v2(proto=self.main_proto, x=point_feat, point_feat=point_feat) # (b, classes, c)
            refine_proto[:, :base_num] = refine_proto[:, :base_num] + gened_proto[:base_num].unsqueeze(0) # refine proto is not l2 norm, but gened_proto is l2 norm. mismatch?
            refine_proto[:, base_num:] = refine_proto[:, base_num:] * 0 + gened_proto[base_num:].unsqueeze(0)
            x_pre = self.get_pred(point_feat, refine_proto) # (b, cls, n)
            return x_pre

        else:
            ##### training
            # fake novel + fake base
            px_feat = self.getFeatures(px)
            unique_y = list(py.unique())  # classes exist in the x # (0,16)
            if 0 in unique_y:
                unique_y.remove(0)
            fake_novel = random.sample(unique_y, self.novel_num)
            cls_py = torch.zeros_like(py)
            new_proto = F.normalize(self.main_proto.clone(),p=2,dim=1) # (n_base, d)
            px_feat_norm = F.normalize(px_feat,p=2,dim=1)
            for i, fn in enumerate(fake_novel):  # if it is fake novel, then its classifier is the prototype of the support set. Otherwise, use the main_proto.
                tmp_mask = (py == fn).float().unsqueeze(1) # (b, 1, n)
                tmp_feat = (px_feat_norm * tmp_mask).sum(0).sum(-1) / (tmp_mask.sum(0).sum(-1) + 1e-12)  # (d,). proto
                new_proto[self.base_num+i] = tmp_feat[None]
                cls_py[py==fn] = self.base_num+i+1
            ori_proto = new_proto
            # ori_proto = self.use_pseudo_proto(x=px_feat, y=py, main_proto=self.main_proto.clone())  # ori_new_proto is eqn.8
            ###
            point_feat = torch.cat([point_feat, px_feat], dim=0)
            y = torch.cat([y,cls_py],dim=0)
            ###
            x_pre_1 = self.get_pred(x=point_feat, proto=ori_proto, use_bg_proto=True)  # logits. pred via eqn.8. (b, bg+cls, n). use the whole batch as query... and the first half batch is support...
            loss_ce_1 = self.criterion(x_pre_1, y)

            # query pred: should be same as testing
            refine_proto = self.post_refine_proto_v2(proto=self.main_proto.clone(), x=point_feat, point_feat=point_feat, use_bg_proto=True)  # eqn.6 only update base proto. (b, classes, c)
            post_refine_proto = refine_proto.clone()
            post_refine_proto[:, :base_num] = post_refine_proto[:, :base_num] + ori_proto[:base_num].unsqueeze(0)
            post_refine_proto[:, base_num:] = post_refine_proto[:, base_num:] * 0 + ori_proto[base_num:].unsqueeze(0) # (b, cls, d)
            x_pre_2 = self.get_pred(x=point_feat, proto=post_refine_proto, use_bg_proto=True)  # (b, cls, n)
            loss_ce_2 = self.criterion(x_pre_2, y)
            ce_loss = 0.5 * loss_ce_2 + 0.5 * loss_ce_1
            return x_pre_2.max(1)[1], ce_loss, y

    def post_refine_proto_v2(self, proto, x, point_feat, use_bg_proto=False):
        ''' refine the base proto via query prediction. eqn. 6. use segment_feat(x) to predict label. Then aggregate feature using point_feat.
        Args: n: number of point. c: feature dim.

            proto: (13, 192)
            x: point feature of this batch (b, d, n)
            point_feat: (b, d, n)
            segment_label: (b,n)
        Returns: eqn.6 (b, classes, c)

        '''
        if use_bg_proto == False:
            b, c, n = point_feat.shape[:]
            pred = self.get_pred(x, proto).view(b, proto.shape[0], n)  # (b, 13, n)
            pred = F.softmax(pred, 2)
            pred_proto = pred @ point_feat.view(b, c, n).permute(0, 2, 1) # (b, classes, c)
            pred_proto_norm = F.normalize(pred_proto, 2, -1)  # (b, classes, c)
            proto_norm = F.normalize(proto, 2, -1).unsqueeze(0)  # (1, classes, c)
            pred_weight = (pred_proto_norm * proto_norm).sum(-1).unsqueeze(-1)  # (b, classes, 1)
            pred_weight = pred_weight * (pred_weight > 0).float()
            pred_proto = pred_weight * pred_proto + (1 - pred_weight) * proto.unsqueeze(0)  # b, cls, c
            
        else:
            # base training
            # raw_x = x.clone()
            # b, c, n = raw_x.shape[:]
            b, c, n = point_feat.shape[:]
            pred = self.get_pred(x, proto, use_bg_proto).view(b, proto.shape[0]+1, n)  # (b, bg+13, n)
            pred = F.softmax(pred, 2)  # (b, bg+13, n)
            pred_proto = pred @ point_feat.view(b, c, n).permute(0, 2, 1)
            pred_proto = pred_proto[:,1:,:]  # (b, classes, c). exclude 'bg proto'
            pred_proto_norm = F.normalize(pred_proto, 2, -1)  # (b, classes, c)
            proto_norm = F.normalize(proto, 2, -1).unsqueeze(0)  # (1, classes, c)
            pred_weight = (pred_proto_norm * proto_norm).sum(-1).unsqueeze(-1)  # (b, classes, 1)
            pred_weight = pred_weight * (pred_weight > 0).float()
            pred_proto = pred_weight * pred_proto + (1 - pred_weight) * proto.unsqueeze(0)  # b, cls, c

        return pred_proto

    def get_pred(self, x, proto, use_bg_proto=False):
        ''' cosine similairty between x and proto
        Args: n: number of point. c: feature dim. cls: number of classes.
            x: is the whole batch feature. (b, c, n)
            proto: prototype. (13, 192)
            use_bg_proto: in the base stage, we treat the novel classes as bg class. and use self.bg_proto to classify. but no use in the final test.
        Returns: prediction of x. (b, cls, n)

        '''
        b, c, n = x.size()[:]

        if len(proto.shape[:]) == 3:
            # x: [b, c, n]
            # proto: [b, cls, c]
            if use_bg_proto:
                proto = torch.cat([self.bg_proto.unsqueeze(0).repeat(proto.shape[0],1,1), proto], dim=1) # (b, bg+cls, c)

            cls_num = proto.size(1)
            x = F.normalize(x, p=2, dim=1)
            proto = F.normalize(proto, p=2, dim=-1)  # b, cls, c
            x = x.contiguous().view(b, c, n)  # b, c, n
            pred = proto @ x  # b, cls, n
        elif len(proto.shape[:]) == 2:
            if use_bg_proto:
                proto = torch.cat([self.bg_proto, proto], dim=0) # (bg+cls, c)
            cls_num = proto.size(0)
            x = F.normalize(x, p=2, dim=1)  # l2 norm
            proto = F.normalize(proto, p=2, dim=1)  # l2 norm
            x = x.contiguous().view(b, c, n)  # b, c, n
            proto = proto.unsqueeze(0)  # 1, cls, c
            pred = proto @ x  # b, cls, n
        pred = pred.contiguous().view(b, cls_num, n)  # (b, cls, n)
        return pred * 10 # scaling

    def getFeatures(self, x, segment_label=None):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return:
        segment_feature: features with shape (k+d, m1+m2...). m1 is the number of segment for query 1.
        point_feature: (b, d, n)
        """
        edge_convs, feat_level2 = self.encoder(x)
        feat_level3 = self.base_learner(feat_level2)
        att_feat = self.att_learner(feat_level2)
        feat_level1 = edge_convs[0] # (b, d, n)
        semantic_feat = torch.cat((feat_level1, att_feat, feat_level3), dim=1) # (b, d, n)
        semantic_feat = self.fusion(semantic_feat)
        return semantic_feat

######

class BgCAPL(nn.Module):
    def __init__(self, classes=13, criterion=nn.CrossEntropyLoss(), args=None, base_num=7, main_dim=256, N_cluster=0):
        super(BgCAPL, self).__init__()
        assert classes > 1
        self.criterion = criterion
        self.classes = classes  # 13
        # define model
        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)
        self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)

        self.feat_dim = args.edgeconv_widths[0][-1] + args.output_dim + args.base_widths[-1]
        
        self.args = args
        self.N_cluster = N_cluster
        # self.fusion = nn.Sequential(nn.Conv1d(in_channels=self.feat_dim, out_channels=main_dim, kernel_size=1),
        #                             nn.BatchNorm1d(main_dim),
        #                             nn.LeakyReLU(0.2))
        
        self.base_num = base_num
        self.novel_num = classes - base_num
        if args.attn_infer:
            self.main_proto = nn.Parameter(torch.randn((base_num+N_cluster, main_dim)))  # (13, 192) this is for the testing classes
        else:
            self.main_proto = nn.Parameter(torch.randn((classes, main_dim)))  # (13, 192) this is for the testing classes
        self.bg_proto = nn.Parameter(torch.randn((1, main_dim))) # during base training, set all the novel classes as bg. no use in the test.

    def Get_Fg_Feat(self, x, y):
        '''
        Args:
            x: (1, d, 2048)
            y: (1, 2048) binary mask. support point cloud
        Returns: fg feature (n, d)
        '''
        y = y[0] # (2048)
        # get feature of x
        point_feat = self.getFeatures(x)  # (1, c, 2048)
        point_feat = point_feat[0] # (d, 2048)
        fg_feat = point_feat[:,y==1] # (d, n)

        return fg_feat.transpose(1,0) # (n,k)


    def forward(self, x, y=None, gened_proto=None, eval_model=False, N_bg=None, args=None):

        base_num = self.base_num
        point_feat = self.getFeatures(x) # (k+d, m1+m2..) . (b, k_d, n)

        if eval_model:
            #### evaluation
            if args.use_grad:
                if args.attn_infer:
                    # breakpoint()
                    x_pre = self.get_pred(point_feat, gened_proto)  # (b, cls, n)
                    return x_pre
                else:
                    x_pre = self.get_pred(point_feat, self.main_proto)  # (b, cls, n)
                    return x_pre
            else:
                if len(gened_proto.size()[:]) == 3:
                    gened_proto = gened_proto[0] # p_orig
                refine_proto = self.post_refine_proto_v2(proto=self.main_proto, x=point_feat, point_feat=point_feat) # (b, classes, c)
                refine_proto[:, :base_num] = refine_proto[:, :base_num] + gened_proto[:base_num].unsqueeze(0) # refine proto is not l2 norm, but gened_proto is l2 norm. mismatch?
                refine_proto[:, base_num:] = refine_proto[:, base_num:] * 0 + gened_proto[base_num:].unsqueeze(0)
                x_pre = self.get_pred(point_feat, refine_proto) # (b, cls, n)
                return x_pre

        else:
            ##### training
            if args.use_grad:
                if args.attn_infer:
                    x_pre_2 = self.get_pred(x=point_feat, proto=self.main_proto, use_bg_proto=True)  # (b, cls, n)
                    loss_ce_2 = self.criterion(x_pre_2, y)
                    ce_loss = loss_ce_2
                    return x_pre_2.max(1)[1], ce_loss
                else:
                    x_pre_2 = self.get_pred(x=point_feat, proto=self.main_proto, use_bg_proto=True)  # (b, cls, n)
                    loss_ce_2 = self.criterion(x_pre_2, y)
                    ce_loss = loss_ce_2
                    return x_pre_2.max(1)[1], ce_loss
            else:
                new_proto = F.normalize(self.main_proto.clone(),p=2,dim=1) # (n_base, d)
                point_feat_norm = F.normalize(point_feat.clone(),p=2,dim=1) # (n_base, d)
                if args.use_bg_gt:
                    for i in range(base_num,base_num+self.novel_num): # 7,8,9,10,11,12
                        if i < base_num+N_bg:
                            tmp_mask = (y == i+1).float().unsqueeze(1) # (b, 1, n)
                            tmp_feat = (point_feat_norm * tmp_mask).sum(0).sum(-1) / (tmp_mask.sum(0).sum(-1) + 1e-12)  # (d,). proto
                            new_proto[i] = tmp_feat[None]
                        else:
                            new_proto[i] = new_proto[i]*0
                else:
                    for i in range(base_num,base_num+self.novel_num): # 7,8,9,10,11,12
                        tmp_mask = (y == i+1).float().unsqueeze(1) # (b, 1, n)
                        tmp_feat = (point_feat_norm * tmp_mask).sum(0).sum(-1) / (tmp_mask.sum(0).sum(-1) + 1e-12)  # (d,). proto
                        new_proto[i] = tmp_feat[None]
                ori_proto = new_proto
                x_pre_1 = self.get_pred(x=point_feat, proto=ori_proto, use_bg_proto=True)  # logits. pred via eqn.8. (b, bg+cls, n). use the whole batch as query... and the first half batch is support...
                loss_ce_1 = self.criterion(x_pre_1, y)

                # query pred: should be same as testing
                refine_proto = self.post_refine_proto_v2(proto=self.main_proto.clone(), x=point_feat, point_feat=point_feat, use_bg_proto=True)  # eqn.6 only update base proto. (b, classes, c)
                post_refine_proto = refine_proto.clone()
                post_refine_proto[:, :base_num] = post_refine_proto[:, :base_num] + ori_proto[:base_num].unsqueeze(0)
                post_refine_proto[:, base_num:] = post_refine_proto[:, base_num:] * 0 + ori_proto[base_num:].unsqueeze(0) # (b, cls, d)
                x_pre_2 = self.get_pred(x=point_feat, proto=post_refine_proto, use_bg_proto=True)  # (b, cls, n)
                loss_ce_2 = self.criterion(x_pre_2, y)
                ce_loss = 0.5 * loss_ce_2 + 0.5 * loss_ce_1
                return x_pre_2.max(1)[1], ce_loss
    
    def get_pred(self, x, proto, use_bg_proto=False):
        ''' cosine similairty between x and proto
        Args: n: number of point. c: feature dim. cls: number of classes.
            x: is the whole batch feature. (b, c, n)
            proto: prototype. (13, 192)
            use_bg_proto: in the base stage, we treat the novel classes as bg class. and use self.bg_proto to classify. but no use in the final test.
        Returns: prediction of x. (b, cls, n)

        '''
        b, c, n = x.size()[:]

        if len(proto.shape[:]) == 3:
            # x: [b, c, n]
            # proto: [b, cls, c]
            if use_bg_proto:
                proto = torch.cat([self.bg_proto.unsqueeze(0).repeat(proto.shape[0],1,1), proto], dim=1) # (b, bg+cls, c)

            cls_num = proto.size(1)
            x = F.normalize(x, p=2, dim=1)
            proto = F.normalize(proto, p=2, dim=-1)  # b, cls, c
            x = x.contiguous().view(b, c, n)  # b, c, n
            pred = proto @ x  # b, cls, n
        elif len(proto.shape[:]) == 2:
            if use_bg_proto:
                proto = torch.cat([self.bg_proto, proto], dim=0) # (bg+cls, c)
            cls_num = proto.size(0)
            x = F.normalize(x, p=2, dim=1)  # l2 norm
            proto = F.normalize(proto, p=2, dim=1)  # l2 norm
            x = x.contiguous().view(b, c, n)  # b, c, n
            proto = proto.unsqueeze(0)  # 1, cls, c
            pred = proto @ x  # b, cls, n
        pred = pred.contiguous().view(b, cls_num, n)  # (b, cls, n)
        return pred * 10 # scaling

    def use_pseudo_proto(self, x, y, main_proto, fake_novel=None, post_processing=False):
        ''' only used during training!
        Args: during training
            x: x is the feature of the support set. (b, d, n)
            y: label of x. (b, n). {0,1,2,...} 0 is 'bg' in base stage. but main_proto doesn't count 'bg'!
            main_proto: (n, d)
            fake_novel: fake_novel class ids.
        Returns: l2 normed proto.

        '''
        b, c, n = x.size()[:]  # x is feature
        # get fake novel idx

        tmp_y = y.unsqueeze(1)  # (b, 1, n) # the label set of the support set
        unique_y = list(tmp_y.unique())  # classes exist in the x # (0,16)

        # get fake_novel and fake_context classes. exclude 'bg' in fake_novel and fake_context.
        if fake_novel == None:
            if 0 in unique_y:
                unique_y.remove(0)
            fake_novel = random.sample(unique_y, self.novel_num)  # fake novel classes in the support set.

        new_proto = main_proto  # (n_base, d)
        new_proto = F.normalize(new_proto,p=2,dim=1)
        x = F.normalize(x,p=2,dim=1)
        # for fake_novel classes, we use the feature average as the classifier.
        for i, fn in enumerate(fake_novel):  # if it is fake novel, then its classifier is the prototype of the support set. Otherwise, use the main_proto.
            tmp_mask = (tmp_y == fn).float() # (b, 1, n)
            tmp_feat = (x * tmp_mask).sum(0).sum(-1) / (tmp_mask.sum(0).sum(-1) + 1e-12)  # (d,). proto
            new_proto[self.base_num+i] = tmp_feat[None]
        return new_proto

    def post_refine_proto_v2(self, proto, x, point_feat, use_bg_proto=False):
        ''' refine the base proto via query prediction. eqn. 6. use segment_feat(x) to predict label. Then aggregate feature using point_feat.
        Args: n: number of point. c: feature dim.

            proto: (13, 192)
            x: point feature of this batch (b, d, n)
            point_feat: (b, d, n)
            segment_label: (b,n)
        Returns: eqn.6 (b, classes, c)

        '''
        if use_bg_proto == False:
            b, c, n = point_feat.shape[:]
            pred = self.get_pred(x, proto).view(b, proto.shape[0], n)  # (b, 13, n)
            pred = F.softmax(pred, 2)
            pred_proto = pred @ point_feat.view(b, c, n).permute(0, 2, 1) # (b, classes, c)
            pred_proto_norm = F.normalize(pred_proto, 2, -1)  # (b, classes, c)
            proto_norm = F.normalize(proto, 2, -1).unsqueeze(0)  # (1, classes, c)
            pred_weight = (pred_proto_norm * proto_norm).sum(-1).unsqueeze(-1)  # (b, classes, 1)
            pred_weight = pred_weight * (pred_weight > 0).float()
            pred_proto = pred_weight * pred_proto + (1 - pred_weight) * proto.unsqueeze(0)  # b, cls, c
            
        else:
            # base training
            # raw_x = x.clone()
            # b, c, n = raw_x.shape[:]
            b, c, n = point_feat.shape[:]
            pred = self.get_pred(x, proto, use_bg_proto).view(b, proto.shape[0]+1, n)  # (b, bg+13, n)
            pred = F.softmax(pred, 2)  # (b, bg+13, n)
            pred_proto = pred @ point_feat.view(b, c, n).permute(0, 2, 1)
            pred_proto = pred_proto[:,1:,:]  # (b, classes, c). exclude 'bg proto'
            pred_proto_norm = F.normalize(pred_proto, 2, -1)  # (b, classes, c)
            proto_norm = F.normalize(proto, 2, -1).unsqueeze(0)  # (1, classes, c)
            pred_weight = (pred_proto_norm * proto_norm).sum(-1).unsqueeze(-1)  # (b, classes, 1)
            pred_weight = pred_weight * (pred_weight > 0).float()
            pred_proto = pred_weight * pred_proto + (1 - pred_weight) * proto.unsqueeze(0)  # b, cls, c

        return pred_proto

    def getFeatures(self, x, segment_label=None):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return:
        segment_feature: features with shape (k+d, m1+m2...). m1 is the number of segment for query 1.
        point_feature: (b, d, n)
        """
        edge_convs, feat_level2 = self.encoder(x)
        feat_level3 = self.base_learner(feat_level2)
        att_feat = self.att_learner(feat_level2)
        feat_level1 = edge_convs[0] # (b, d, n)
        semantic_feat = torch.cat((feat_level1, att_feat, feat_level3), dim=1) # (b, d, n)
        # semantic_feat = self.fusion(semantic_feat)
        return semantic_feat