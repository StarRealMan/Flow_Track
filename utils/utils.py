import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])


def mask2point(mask):
    height, width = mask.shape
    point_u = torch.arange(0, width)
    point_v = torch.arange(0, height)
    point_u, point_v = torch.meshgrid(point_u, point_v, indexing="xy")
    point_u = point_u.to(mask.device)
    point_v = point_v.to(mask.device)
    
    point = torch.stack([point_u, point_v], dim=0)
    mask_point = point[:, mask]
    
    return point, mask_point

def mask_spread(mask, mask_point, flow):
    mask_flow = flow[:, mask]
    spread_point = mask_point + mask_flow

    out_mask = (spread_point[0] < 0) | (spread_point[0] > mask.shape[1]) | \
               (spread_point[1] < 0) | (spread_point[1] > mask.shape[0])
    
    spread_point = spread_point[:, ~out_mask]
    
    return spread_point

def point2mask(mask, spread_point):
    new_mask = torch.zeros_like(mask)
    spread_point = spread_point.long()
    new_mask[spread_point[1], spread_point[0]] = True
    
    return new_mask

def mask_AABB(spread_point):
    min_x = torch.min(spread_point[0])
    min_y = torch.min(spread_point[1])
    max_x = torch.max(spread_point[0])
    max_y = torch.max(spread_point[1])
    
    return torch.tensor([min_x, min_y, max_x, max_y])
