import torch
from torch import Tensor, nn
import timm
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple, Union

OptIntSeq = Optional[Sequence[int]]


def conv_layer(ni, no, kernel_size, stride=1):
    return nn.Sequential(
        nn.Conv2d(ni, no, kernel_size, stride),
        nn.ReLU(),
        nn.BatchNorm2d(no),
        nn.MaxPool2d(2)
    )

class Vit_b(nn.Module):
    def __init__(self,
                 pretrained=True,
                 out_channels=11,
                 deconv_out_channels: OptIntSeq = (256, 256, 256 , 256, 128),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4, 4, 4),
                 ):
        super(Vit_b, self).__init__()
        self.vit_backbone = timm.create_model('pvt_v2_b2', pretrained=pretrained,
                                 pretrained_cfg_overlay=dict(file="./model/pvt_v2_b2/pytorch_model.bin"),
                                 num_classes = 0,
                                 features_only=True
                                            #   global_pool='avg',
                                            #   out_indices=[-1],
                                              )
        # hard-code to remove the head and use global_pool instead,

        self.vit_backbone.head = nn.Identity()
        # self.vit_backbone.global_pool = 'avg'
        self.out_channels = out_channels


        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            self.deconv_layers = self._make_deconv_layers(
                num_layers=5,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()
            in_channels = 512

        # import pdb; pdb.set_trace()
        self.logit_conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1)
        self.offset_conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels*2,
                               kernel_size=1)
        self.num_keypoints = out_channels

        # self.deconv_C4 = self._make_one_deconv_layers(
        #     layer_in_channels=512,layer_out_channels=256,layer_kernel_size = 4
        # )
        # self.deconv_C3 = self._make_one_deconv_layers(
        #     layer_in_channels=256,layer_out_channels=256,layer_kernel_size = 4
        # )
        # self.deconv_C2 = self._make_one_deconv_layers(
        #     layer_in_channels=256,layer_out_channels=256,layer_kernel_size = 4
        # )
        # self.deconv_C1 = self._make_one_deconv_layers(
        #     layer_in_channels=256,layer_out_channels=256,layer_kernel_size = 4
        # )
        # self.deconv_C0 = self._make_one_deconv_layers(
        #     layer_in_channels=256,layer_out_channels=128,layer_kernel_size = 4
        # )

    def _make_one_deconv_layers(self,
                                layer_in_channels: int,
                            layer_out_channels: int,
                            layer_kernel_size: int) -> nn.Module:
        """Create deconvolutional layers by given parameters."""
        layers = []
        if layer_kernel_size == 4:
            padding = 1
            output_padding = 0
        elif layer_kernel_size == 3:
            padding = 1
            output_padding = 1
        elif layer_kernel_size == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Unsupported kernel size {layer_kernel_size} for'
                             'deconvlutional layers in '
                             f'{self.__class__.__name__}')
        layers.append(
            nn.ConvTranspose2d(
                in_channels=layer_in_channels,       #!!! wether change deconv input
                out_channels=layer_out_channels,
                kernel_size=layer_kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False))
        layers.append(nn.BatchNorm2d(layer_out_channels))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)


    def _make_deconv_layers(self, num_layers: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""
        assert num_layers == len(layer_out_channels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(layer_out_channels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        planes = 512
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=planes,       #!!! wether change deconv input
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            # layers.append(nn.functional.interpolate(pred_mask_0, size=(img.shape[2],img.shape[3]), mode='bilinear', align_corners=True)

            planes = out_channels
        return nn.Sequential(*layers)


    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
        return y

    @torch.no_grad()
    def locations(self, features):
        h, w = features.size()[-2:]
        device = features.device
        shifts_x = torch.arange(0, w, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, h, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1) / w
        shift_y = shift_y.reshape(-1) / h
        locations = torch.stack((shift_x, shift_y), dim=1)
        locations = locations.reshape(h, w, 2).permute(2, 0, 1)
        return locations

    def forward(self, x):
        # C1, C2, C3, C4 = self.vit_backbone(x)
        _, _, _, C4 = self.vit_backbone(x)


        x = self.deconv_layers(C4)      #N x 128 x 480 x 768

        # x = self.deconv_C4(C4)
        # x = self.deconv_C3(x)
        # # x = self.deconv_C2(x) + C1
        # x = torch.cat([self.deconv_C2(x),C1], dim=1)
        # x = self.deconv_C1(x)
        # x = self.deconv_C0(x)

        # import pdb;pdb.set_trace()
        # x = torch.nn.functional.interpolate(x, size=(x.shape[2]*2, x.shape[3]*2), mode='bilinear', align_corners=True)

        bs, c, h, w = x.size()
        logit = self.logit_conv(x).sigmoid()

        # logit = self.logit_conv(x).reshape(bs, self.num_keypoints, h*w)
        # logit = F.softmax(logit, dim=-1).reshape(bs, self.num_keypoints, h,w)
        # import pdb;pdb.set_trace()
        offset = self.offset_conv(x).reshape(bs, self.num_keypoints, 2, h, w)
        location = self.locations(offset)[None, None]
        keypoint = location - offset

        ret = [logit, keypoint]



        # import pdb;pdb.set_trace()

        return ret

    def decode(self, batch_rets):
        logit, keypoint = batch_rets[:2]
        bs, k, h, w = logit.shape
        logits = logit.reshape(bs*k, h*w)
        logits = logits / logits.sum(dim=1, keepdim=True)
        keypoints = keypoint.reshape(bs*k, 2, h*w).permute(0, 2, 1)
        maxvals, maxinds = logits.max(dim=1)
        coords = keypoints[torch.arange(bs*k, dtype=torch.long).to(keypoints.device), maxinds]
        coords[..., 0] *= w
        coords[..., 1] *= h

        # # hmvals
        # heatmap = batch_rets[2]
        # bs, k, h, w = heatmap.shape
        # heatmap = heatmap.reshape(bs*k, 1, h, w).sigmoid()
        # coord_inds = torch.stack((
        #     coords[:, 0] / (w - 1) * 2 - 1,
        #     coords[:, 1] / (h - 1) * 2 - 1,
        # ), dim=-1)
        # coord_inds = coord_inds[:, None, None, :]
        # keypoint_scores = torch.nn.functional.grid_sample(
        #     heatmap, coord_inds,
        #     padding_mode='border').reshape(bs*k, -1)
        # maxvals = keypoint_scores
        # preds = coords.reshape(bs, k, 2).cpu().numpy()
        #
        # if self.codec.get('type', 'MSRAHeatmap') == 'UDPHeatmap':
        #     preds = preds / [w-1, h-1]
        #     preds = preds * self.codec['input_size']
        # else:
        #     stride = self.codec['input_size'][0] / self.codec['heatmap_size'][0]
        #     preds = preds * stride
        preds = coords.reshape(bs, k, 2).cpu().numpy()
        maxvals = maxvals.reshape(bs, k).cpu().numpy()

        stride = self.codec['input_size'][0] / self.codec['heatmap_size'][0]
        preds = preds * stride


        return preds, maxvals





