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



class CAPL(nn.Module):
    def __init__(self, classes=13, criterion=nn.CrossEntropyLoss(), args=None, base_cls=None, novel_cls=None, use_grad=False):
        ''' use geometric primiytives as visual words to describe any point cloud. Only use three layer of edge conv'''
        ''' hard coding + No atten
        classes: number of total classes in the whole dataset
        gp: precomputed geometric primitives. (k, d). cpu tensor. no grad.
        '''
        super(CAPL, self).__init__()
        assert classes > 1
        self.criterion = criterion
        self.classes = classes  # 13

        # define model
        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        # # freeze edge convs:
        # for param in self.encoder.edge_convs.parameters():
        #     param.requires_grad = False

        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)
        self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        self.feat_dim = args.edgeconv_widths[0][-1] + args.output_dim + args.base_widths[-1]

        # define classifier
        main_dim = self.feat_dim
        self.main_proto = nn.Parameter(torch.randn((classes, main_dim)))  # (13, 192) this is for the testing classes
        self.bg_proto = nn.Parameter(torch.randn((1, main_dim))) # during base training, set all the novel classes as bg. no use in the test.

        self.args = args
        self.base_num = len(base_cls)
        self.base_cls = base_cls
        self.novel_cls = novel_cls
        self.pseudo_cls = np.arange(len(novel_cls)) + 1 + len(base_cls)
        print('novel_cls', novel_cls)
        print('pseudo_cls', self.pseudo_cls)
        self.use_grad = use_grad

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
    
    def pseudo_proto_feat(self, x):
        point_feat = self.getFeatures(x)  # (1, c, 2048)
        point_feat = point_feat[0] # (d, 2048)
        return point_feat.transpose(1,0) # (n,d)


    def forward(self, x, y=None, gened_proto=None, eval_model=False):
        '''
        Args: point-level prediction. No use of segment
            x: (b, d, 2048)
            y: (b, 2048)
            gened_proto:
            base_num:
            novel_num:
            epoch: define as epoch
            gen_proto:
            eval_model:
            visualize:
            target_cls: in the novel class stage, the novel class idx in the whole testing classes. also is the oder in the main_proto.
            base_class_coding: (n_base, k)
            novel_class_coding: (n_new, k)
            bg_class_coding: (k,)
        Returns:

        '''
        base_num = self.base_num
        point_feat = self.getFeatures(x) # (k+d, m1+m2..) . (b, k_d, n)

        if self.use_grad:
            if eval_model:
                refine_proto = self.main_proto.clone()
                refine_proto[base_num:] = gened_proto[base_num:]
                x_pre = self.get_pred(point_feat, refine_proto) # (b, cls, n)
                return x_pre
            else:
                ori_proto = self.main_proto.clone()
                x_pre_1 = self.get_pred(x=point_feat, proto=ori_proto, use_bg_proto=True)
                loss_ce_1 = self.criterion(x_pre_1, y)
                ce_loss = loss_ce_1
                return x_pre_1.max(1)[1], ce_loss
        else:
            if eval_model:
                #### evaluation
                # gened_proto = gened_proto.unsqueeze(0).repeat(8, 1, 1) # (b, 13, 192)
                # if len(gened_proto.size()[:]) == 3:
                #     gened_proto = gened_proto[0] # p_orig
                refine_proto = self.post_refine_proto_v2(proto=self.main_proto, x=point_feat, point_feat=point_feat) # (b, classes, c)
                refine_proto[:, :base_num] = refine_proto[:, :base_num] + gened_proto[:base_num].unsqueeze(0) # refine proto is not l2 norm, but gened_proto is l2 norm. mismatch?
                refine_proto[:, base_num:] = refine_proto[:, base_num:] * 0 + gened_proto[base_num:].unsqueeze(0)
                x_pre = self.get_pred(point_feat, refine_proto) # (b, cls, n)
                return x_pre

            else:
                ##### training
                # fake novel + fake base
                fake_num = x.size(0) // 2  # B/2
                ori_proto = self.pseudo_proto(x=point_feat[fake_num:], y=y[fake_num:], main_proto=self.main_proto.clone(), fake_novel=self.pseudo_cls)  # ori_new_proto is eqn.8
                # ori_proto, fake_novel = self.generate_fake_proto(x=point_feat[fake_num:], y=y[fake_num:], main_proto=self.main_proto.clone())  # ori_new_proto is eqn.8
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


    def pseudo_proto(self, x, y, main_proto, fake_novel=None):
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
        unique_y = list(tmp_y.unique())  # classes exist in the x

        # get fake_novel and fake_context classes. exclude 'bg' in fake_novel and fake_context.
        if False and fake_novel == None:
            if 0 in unique_y:
                unique_y.remove(0)
            novel_num = len(unique_y) // 2
            fake_novel = random.sample(unique_y, novel_num)  # fake novel classes in the support set.
            for fn in fake_novel:
                unique_y.remove(fn)
            fake_context = unique_y  # fake context classes in this support set

        new_proto = main_proto  # (n_base, d)
        # l2 norm. need l2 norm here!!
        new_proto = new_proto / (torch.norm(new_proto, 2, 1, True) + 1e-12)  # l2 norm
        # input feat l2 norm
        x = x / (torch.norm(x, 2, 1, True) + 1e-12)  # l2 norm (b, d, n)

        # for fake_novel classes, we use the feature average as the classifier.
        for fn in fake_novel:  # if it is fake novel, then its classifier is the prototype of the support set. Otherwise, use the main_proto.
            tmp_mask = (tmp_y == fn).float() # (b, 1, n)
            if tmp_mask.sum()==0:
                continue
            tmp_feat = (x * tmp_mask).sum(0).sum(-1) / (tmp_mask.sum(0).sum(-1) + 1e-12)  # (d,). proto

            fake_vec = torch.zeros(new_proto.size(0), 1).cuda()  # (n_base, 1)
            fake_vec[fn - 1] = 1
            new_proto = new_proto * (1 - fake_vec) + tmp_feat.unsqueeze(0) * fake_vec

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
            pred_weight = (pred_proto_norm * proto_norm).sum(-1).unsqueeze(-1)  # (b, classes, 1), inner product
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

    def getFeatures(self, x):
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
        return semantic_feat


    def generate_fake_proto(self, x, y, main_proto, fake_novel=None, post_processing=False):
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
        unique_y = list(tmp_y.unique())  # classes exist in the x

        # get fake_novel and fake_context classes. exclude 'bg' in fake_novel and fake_context.
        if fake_novel == None:
            if 0 in unique_y:
                unique_y.remove(0)
            novel_num = len(unique_y) // 2
            fake_novel = random.sample(unique_y, novel_num)  # fake novel classes in the support set.
            for fn in fake_novel:
                unique_y.remove(fn)
            fake_context = unique_y  # fake context classes in this support set

        new_proto = main_proto  # (n_base, d)
        # l2 norm. need l2 norm here!!
        new_proto = new_proto / (torch.norm(new_proto, 2, 1, True) + 1e-12)  # l2 norm
        # input feat l2 norm
        x = x / (torch.norm(x, 2, 1, True) + 1e-12)  # l2 norm (b, d, n)

        # for fake_novel classes, we use the feature average as the classifier.
        for fn in fake_novel:  # if it is fake novel, then its classifier is the prototype of the support set. Otherwise, use the main_proto.
            tmp_mask = (tmp_y == fn).float() # (b, 1, n)
            tmp_feat = (x * tmp_mask).sum(0).sum(-1) / (tmp_mask.sum(0).sum(-1) + 1e-12)  # (d,). proto

            fake_vec = torch.zeros(new_proto.size(0), 1).cuda()  # (n_base, 1)
            fake_vec[fn.long() - 1] = 1
            new_proto = new_proto * (1 - fake_vec) + tmp_feat.unsqueeze(0) * fake_vec


        return new_proto, fake_novel


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






