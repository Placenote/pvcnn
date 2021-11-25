import torch
import torch.nn as nn

from models.utils import create_pointnet_components, create_mlp_components

__all__ = ['PVCNN']


class PVCNN(nn.Module):
    blocks = [(32, 2, 64), (64, 1, 32), (128, 1, 16), (256, 1, 8)]

    def __init__(self, extra_feature_channels=2, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = extra_feature_channels + 3

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=False, normalize=True,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.point_features = nn.ModuleList(layers)

        layers, _ = create_mlp_components(in_channels=(channels_point + concat_channels_point),
                                          out_channels=[512, 0.2, 256, 0.2, 128],
                                          classifier=False, dim=2,
                                          width_multiplier=width_multiplier)
        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Linear(128, 1))

    def forward(self, inputs):
        # inputs : [B, in_channels + S, N]
        features = inputs[:, :self.in_channels, :]
        num_points = features.size(-1)

        coords = features[:, :3, :]
        out_features_list = []
        for i in range(len(self.point_features)):
            features, _ = self.point_features[i]((features, coords))
            out_features_list.append(features)
        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))

        feature_stack = torch.cat(out_features_list, dim=1)
        mlp_out = self.mlp(feature_stack)
        pool = mlp_out.mean(dim=2)
        result = self.classifier(pool)
        return result.flatten()
