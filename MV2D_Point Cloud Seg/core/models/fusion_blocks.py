import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsparse import SparseTensor
import torchsparse.nn as spnn

class IA_Layer(nn.Module):
    def __init__(self, channels, return_att=False):
        """
        ic: [64, 128, 256, 512]
        pc: [96, 256, 512, 1024]
        """
        super(IA_Layer, self).__init__()

        self.return_att = return_att
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                   nn.BatchNorm1d(self.pc),
                                   nn.ReLU(True))
        # self.fc1 = nn.Linear(self.ic, rc)
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(self.ic),
            nn.ReLU(True),
            nn.Linear(self.ic, rc)
        )
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)

    def forward(self, img_feats, point_feats):
        """

        Args:
            img_feas: <Tensor, N, C> image_feature conv+bn
            point_feas: <Tensor, N, C'> point_feature conv+bn+relu

        Returns:

        """
        img_feats = img_feats.contiguous()
        point_feats = point_feats.contiguous()
        # 将图像特征和点云特征映射成相同维度
        ri = self.fc1(img_feats)
        rp = self.fc2(point_feats)
        # 直接逐元素相加作为融合手段，基于假设：如果相同位置图像特征和点云特征比较相似，那么图像特征将有利于提高网络的performance
        att = torch.sigmoid(self.fc3(torch.tanh(ri + rp)))  # BNx1
        att = att.unsqueeze(1)
        att = att.view(1, 1, -1)  # B1N
        img_feats_c = img_feats.unsqueeze(0).transpose(1, 2).contiguous()
        img_feas_new = self.conv1(img_feats_c)
        # 依据图像特征和点云特征的相关程度筛选图像特征
        out = img_feas_new * att

        if self.return_att:
            return out, att
        else:
            return out, None

class LI_Fusion_Layer(nn.Module):
    def __init__(self, channels, return_att=False):
        """
        ic: [64, 128, 256, 512]
        pc: [96, 256, 512, 1024]
        """
        super(LI_Fusion_Layer, self).__init__()
        self.return_att = return_att
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                   nn.BatchNorm1d(self.pc),
                                   nn.ReLU())
        # self.fc1 = nn.Linear(self.ic, rc)
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(self.ic),
            nn.ReLU(),
            nn.Linear(self.ic, rc)
        )
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, self.pc)

    def forward(self, img_feats, point_feats):
        """

        Args:
            img_feats: <Tensor, N, C> image_feature
            point_feats: <Tensor, N, C'> point_feature

        Returns:

        """
        img_feats = img_feats.contiguous()
        point_feats = point_feats.contiguous()
        # 将图像特征和点云特征映射成相同维度
        ri = self.fc1(img_feats)
        rp = self.fc2(point_feats)
        # 直接逐元素相加作为融合手段，基于假设：如果相同位置图像特征和点云特征比较相似，那么图像特征将有利于提高网络的performance
        att = torch.sigmoid(self.fc3(torch.tanh(ri + rp)))  # BNx1
        att = att.unsqueeze(1)
        att = att.view(1, self.pc, -1)  # [B, C', N]
        img_feats_c = img_feats.unsqueeze(0).transpose(1, 2)
        img_feas_new = self.conv1(img_feats_c)
        # 依据图像特征和点云特征的相关程度筛选图像特征
        out = img_feas_new * att
        if self.return_att:
            return out, att
        else:
            return out

class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes, return_att=False):
        """
        inplanes_I: [64, 128, 256, 512]
        inplanes_P: [96, 256, 512, 1024]
        outplanes: [96, 256, 512, 1024]
        """
        super(Atten_Fusion_Conv, self).__init__()

        self.return_att = return_att

        self.ai_layer = IA_Layer(channels=[inplanes_I, inplanes_P], return_att=return_att)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        """
        point_feature: 点云特征 [B, C, N] conv+bn+relu
        img_feature: 图像特征 [B, N, C]  conv+bn
        """

        img_features, att = self.ai_layer(img_features, point_features)  # [B, C, N]
        point_feats = point_features.unsqueeze(0).transpose(1, 2)
        # 将筛选的图像特征与点云特征直接拼接
        fusion_features = torch.cat([point_feats, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))
        fusion_features = fusion_features.squeeze(0).transpose(0, 1)

        if att is not None:
            return fusion_features, att
        else:
            return fusion_features


class ResidualBasedFusionBlock(nn.Module):
    def __init__(self, pcd_channels, img_channels):
        super(ResidualBasedFusionBlock, self).__init__()
        self.fuse_conv = nn.Sequential(
            spnn.Conv3d(pcd_channels+img_channels, pcd_channels, kernel_size=3, dilation=1, stride=1),
            spnn.BatchNorm(pcd_channels),
            spnn.LeakyReLU(True)
        )
        self.attention = nn.Sequential(
            spnn.Conv3d(pcd_channels, pcd_channels, kernel_size=3, dilation=1, stride=1),
            spnn.BatchNorm(pcd_channels),
            spnn.ReLU(inplace=True),
            spnn.Conv3d(pcd_channels, pcd_channels, kernel_size=3, dilation=1, stride=1),
            spnn.BatchNorm(pcd_channels)
        )

    def forward(self, pcd_feature, img_feature):
        if isinstance(img_feature, SparseTensor):
            img_feature = img_feature.F
        cat_feature = SparseTensor(feats=torch.cat((pcd_feature.F, img_feature), dim=1), coords=pcd_feature.C, stride=pcd_feature.s)
        cat_feature.cmaps = pcd_feature.cmaps
        cat_feature.kmaps = pcd_feature.kmaps
        fuse_out = self.fuse_conv(cat_feature)
        attention_map = self.attention(fuse_out)
        attention_map.F = torch.sigmoid(attention_map.F)
        out = SparseTensor(feats=fuse_out.F*attention_map.F + pcd_feature.F, coords=pcd_feature.C, stride=pcd_feature.s)
        out.cmaps = pcd_feature.cmaps
        out.kmaps = pcd_feature.kmaps
        return out

def Feature_Gather(feature_map, xy):
    """
    :param xy:(B,N,2), normalize to [-1,1], (width, height)
    :param feature_map:(B,C,H,W)
    :return:
    """

    # use grid_sample for this.
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)

    interpolate_feature = nn.functional.grid_sample(feature_map, xy, padding_mode='border', align_corners=True)  # (B,C,1,N)

    return interpolate_feature.squeeze(2)  # (B,C,N)


def Feature_Fetch(masks, pix_coord, imfeats):
    """

    Args:
        masks:
        pix_coord:
        imfeats: <Tensor, B, 6, C, H, W>

    Returns:

    """
    imfs = []
    for mask, coord, img in zip(masks, pix_coord, imfeats):
        mask = mask.cuda()
        imf = torch.zeros(size=(mask.size(1), img.size(1))).cuda()
        imf_list = Feature_Gather(img, coord.cuda()).permute(0, 2, 1)  # [6, N, C]
        # assert mask.size(0) == coord.size(0)
        for idx in range(mask.size(0)):
            imf[mask[idx]] = imf_list[idx, mask[idx], :]
        imfs.append(imf)
    return torch.cat(imfs, dim=0)


def Predict_Fetch(masks, pix_coord, imfeats):
    imfs = []
    for mask, coord, img in zip(masks, pix_coord, imfeats):
        mask = mask.cuda()
        imf = torch.zeros(size=(mask.size(1), img.size(1))).cuda()
        for m, co, im in zip(mask, coord, img):
            co[:, 0] = (co[:, 0] + 1.0) / 2 * (imfeats.size(-1) - 1.0)
            co[:, 1] = (co[:, 1] + 1.0) / 2 * (imfeats.size(-2) - 1.0)
            coord = torch.floor(coord).long()


def Feature_Project(masks, pix_coord, pix_feat_tensor, pix_feat):
    pix_feat_c = pix_feat
    cur = 0
    for mask, coord, pix in zip(masks, pix_coord, pix_feat_c):
        pf = pix_feat_tensor[cur:cur+mask.size(1), :]
        h, w = pix.size(2), pix.size(3)
        pc = coord[mask, :]
        pc[:, :, 0] = (pc[:, :, 0] + 1.0) / 2 * (h - 1.0)
        pc[:, :, 1] = (pc[:, :, 1] + 1.0) / 2 * (w - 1.0)
        pc = torch.floor(pc).long()
        for idx in range(mask.size(0)):
            pf_i = pf[mask[idx], :]
            pc_i = pc[mask[idx], :]
            uq, invs = torch.unique(pc_i, sort=False, return_inverse=True, dim=0)
            f_ten = torch.zeros(size=(uq.size(0), pf_i.size(1))).cuda()






