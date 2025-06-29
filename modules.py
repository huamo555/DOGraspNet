from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import pointnet2.pytorch_utils as pt_utils
from pointnet2.pointnet2_utils import CylinderQueryAndGroup


class GraspableNet(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv_graspable = nn.Conv1d(self.in_dim, 3, 1)

    def forward(self, seed_features, end_points):
        graspable_score = self.conv_graspable(seed_features)  # (B, 3, num_seed)
        end_points['objectness_score'] = graspable_score[:, :2]
        end_points['graspness_score'] = graspable_score[:, 2]
        return end_points


class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim, is_training=True):
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.is_training = is_training
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        res_features = F.relu(self.conv1(seed_features), inplace=True)
        features = self.conv2(res_features)
        view_score = features.transpose(1, 2).contiguous()  # (B, num_seed, num_view)
        end_points['view_score'] = view_score

        if self.is_training:
            # normalize view graspness score to 0~1
            view_score_ = view_score.clone().detach()
            view_score_max, _ = torch.max(view_score_, dim=2)
            view_score_min, _ = torch.min(view_score_, dim=2)
            view_score_max = view_score_max.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_min = view_score_min.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_ = (view_score_ - view_score_min) / (view_score_max - view_score_min + 1e-8)

            top_view_inds = []
            for i in range(B):
                top_view_inds_batch = torch.multinomial(view_score_[i], 1, replacement=False)
                top_view_inds.append(top_view_inds_batch)
            top_view_inds = torch.stack(top_view_inds, dim=0).squeeze(-1)  # B, num_seed
        else:
            _, top_view_inds = torch.max(view_score, dim=2)  # (B, num_seed)

            top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
            template_views = generate_grasp_views(self.num_view).to(features.device)  # (num_view, 3)
            template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous()
            vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
            vp_xyz_ = vp_xyz.view(-1, 3)
            batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
            vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
            end_points['grasp_top_view_xyz'] = vp_xyz
            end_points['grasp_top_view_rot'] = vp_rot

        end_points['grasp_top_view_inds'] = top_view_inds
        return end_points, res_features


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x): # 输入是B, N, C
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # Offset-Attention
        #x = x - xx

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CloudCrop(nn.Module):
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=0.04):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [3 + self.in_dim, 256, 256]  # use xyz, so plus 3

        self.grouper = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample,
                                             use_xyz=True, normalize_xyz=True)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

        self.conv1 = nn.Conv1d(896, 256, 1)

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        grouped_feature = self.grouper(seed_xyz_graspable, seed_xyz_graspable, vp_rot,
                                       seed_features_graspable)  # B，3 + feat_dim，M，nsample


        # 使用偏移特征！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        xyz_000 = torch.zeros([seed_features_graspable.size()[0],3,seed_features_graspable.size()[2]]).cuda()
        seed_features_graspable_re = torch.cat(
                    [xyz_000, seed_features_graspable], dim=1
                )  # (B, C + 3, npoint, nsample)
        seed_features_graspable_re = seed_features_graspable_re.unsqueeze(-1).repeat(1, 1, 1, self.nsample) ## B，C，M -> B，C，M，K
        offset_feature = grouped_feature - seed_features_graspable_re
        # 使用偏移特征！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！          



        new_features = self.mlps(offset_feature)  # (batch_size, mlps[-1], M, nsample)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (batch_size, mlps[-1], M, 1)
        new_features = new_features.squeeze(-1)  # (batch_size, mlps[-1], M)

        # 在这儿拼接
        new_features = torch.cat([new_features, seed_features_graspable],dim=1)

        new_features = self.conv1(new_features)

        
        return new_features


class SWADNet(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 256, 1)  # input feat dim need to be consistent with CloudCrop module
        self.conv_swad = nn.Conv1d(256, 2 * num_angle * num_depth, 1)

    def forward(self, vp_features, end_points):#B,C,N
        B, _, num_seed = vp_features.size()
        vp_features = F.relu(self.conv1(vp_features), inplace=True)
        vp_features = self.conv_swad(vp_features)
        vp_features = vp_features.view(B, 2, self.num_angle, self.num_depth, num_seed)
        vp_features = vp_features.permute(0, 1, 4, 2, 3)

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0]  # B * num_seed * num angle * num_depth B,N，12，4
        end_points['grasp_width_pred'] = vp_features[:, 1]  # B,N，12，4
        return end_points
    

def fuseFeature_rgbd_pointCloud(rgbd_features, point_features,point_clouds_idx_in_rgbd):
    """_summary_
        找到20000个点对应的像素特征并进行拼接融合

    Args:
        point_features (_type_): 点云特征，（B，C1，20000）
        rgbd_features (_type_): rgbd特征， （B，C2，H，W）
        point_clouds_idx_in_rgbd (_type_): 点云点在rgbd中的索引，（B，N）

    Returns:
        fusion_features: 点云与rgbd特征拼接后的特征，（B，C1+C2，N）
    """


    B, C2, H, W = rgbd_features.size()

    fusion_features_list = []
    for i in range(B):
        #根据索引找到相应的特征
        rgb_fe = rgbd_features[i].reshape(C2, H*W)#想一下这里怎么验证是否选到了对的点？？？？？
        rgb_fe = torch.index_select(rgb_fe, dim=1, index=point_clouds_idx_in_rgbd[i]) #(C2,20000)#采用多种方法进行校验，如转换维度后再验证看结果是否一致
        point_fe = point_features[i]#( C1,20000)
        #根据找到的特征进行拼接
        fusion_features_list.append(torch.cat((point_fe,rgb_fe),dim=0))#(C1+C2,20000)
    fusion_features = torch.stack(fusion_features_list, dim=0) #（B，C1+C2，N）
    return fusion_features

#
class ScaleAttention_Scale(nn.Module):
    def __init__(self, scaleNum=16, normType='sigmod', in_channels=256):#in_channels每个尺度的点的特征通道数
        super(ScaleAttention_Scale, self).__init__()
        self.scaleNum = scaleNum#薄片的数量（尺度数量16）
        self.normType = normType
        self.in_channels = in_channels
        self.relu0 = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(nn.Conv2d(scaleNum, out_channels=64, kernel_size=1, bias=False),
                                    nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(64, out_channels=128, kernel_size=1, bias=False),
                                    nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(128, out_channels=scaleNum, kernel_size=1, bias=False),
                                    nn.ReLU(inplace=True))

        self.pooling = nn.MaxPool2d(kernel_size=(1, self.in_channels), stride=1)
        #self.pooling = nn.AvgPool2d(kernel_size=(1, self.in_channels), stride=1)

        self.norm = {'sigmod': nn.Sigmoid(),
                     'softmax': nn.Softmax(dim=1)}[self.normType]#具体哪个合适需要消融
        self.layer6 = nn.Conv1d(in_channels*scaleNum, out_channels=in_channels, kernel_size=1)
        # 沿着通道维度复制。
        # 对应位置乘积。

    def forward(self, x):  # (B, in_channels, npoint, scaleNum)
        # 维度交换
        x = x.permute(0, 3, 2, 1)
        # 该值后续进行残差相加

        B, C, npoint, p = x.shape

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pooling(out)
        out = self.norm(out)  # 输出归一化的权重

        # 沿着列维度进行复制为和原有形状一致
        out = out.repeat(1, 1, 1, self.in_channels)
        # 对多尺度进行加权，并进行残差链接
        out = x + torch.mul(x, out)
        out = self.relu0(out)
        # 转化为原有形状，(B, in_channels, npoint, scaleNum)
        out = out.permute(0, 3, 2, 1)
        
        #三种处理方式：最大池化，加权相加，加权拼接
        
        #方式2：加权相加
        #out = torch.sum(input=out,dim=3)
        #return out
        #输出维度B，C（256），N


        #方式1： 拼接方式，
        # 将列维度逐个取出，并拼接到通道维度，进行分离并拼接
        out_list = []
        for scaleNum_idx in range(self.scaleNum):
            out_list.append(out[:, :, :, scaleNum_idx])
        #维度压缩到256！！！！
        out =  self.layer6(torch.cat(out_list, dim=1))

        return out

       

        