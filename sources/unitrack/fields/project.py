from typing import Dict, List, Union

import torch
from torch import Tensor

from ... import projection
from .base_field import Field

__all__ = ["DepthProjection"]


class DepthProjection(Field):
    """
    A :class:`.Field` that tracks the world-positon of an object using a
    projection of the object 2D mask (i.e. segmentation) and estimated mean
    depth.
    """

    max_depth: Tensor

    def __init__(
        self,
        id: str,
        key_mask: str,
        key_depth: str,
        max_depth: Union[Tensor, float],
        scale: float = 4.0,
    ):
        """
        Parameters
        ----------
        id
            Unique ID
        key_mask
            Data key for the mask tensor.
        key_depth
            Data key for the depth kernel that has the mean instance depth as
            its first element.
        max_depth
            Maximum depth for denoramlization of the mean depth, which should be
            fixed for a given dataset.
        scale
            Value to upscale screen by, e.g. when the network gives masks
            with size (h,w) then they will be projected as if they had size
            (h*scale, w*scale). This is useful because predictions are often
            made on a downscaled version of the image.
        """

        super().__init__(
            id, required_keys=[key_mask, key_depth], required_data=[]
        )

        self.key_mask = key_mask
        self.key_depth = key_depth
        self.scale = scale

        self.register_buffer(
            "max_depth",
            torch.as_tensor(max_depth, dtype=torch.float32),
            persistent=False,
        )
        self.max_depth.requires_grad_(False)

    def extract(
        self, kvmap: Dict[str, Tensor], data: Dict[str, Tensor]
    ) -> Tensor:
        masks = kvmap[self.key_mask]
        if len(masks) == 0:
            return torch.empty((0, 3), dtype=torch.float32, device=masks.device)

        mean_depths = self._read_depth(kvmap[self.key_depth])
        cams = self._make_cameras()
        points_xyd = self._read_points_xyd(masks, mean_depths)
        points_world = self._unproject(cams, points_xyd)

        return points_world

    def _read_depth(self, depth_kernels: Tensor):
        """
        Read the mean depths for each detection from the kernels
        """
        d = depth_kernels[:, 0]
        d = d.sigmoid() * self.max_depth

        return d

    def _read_points_xyd(self, masks: Tensor, mean_depths: Tensor) -> Tensor:
        """
        Create a list of points using the mass center and mean depth of each
        instance.
        """

        assert masks.ndim == 3, masks.ndim  # n, h, w
        assert mean_depths.ndim == 1, mean_depths.ndim  # n x d
        assert len(masks) == len(mean_depths), (len(masks), len(mean_depths))

        indices, yx = masks.argwhere().split([1, 2], dim=1)
        _, counts = torch.unique(indices, return_counts=True)
        indices_len: List[int] = counts.tolist()

        # Convert yx -> xy
        xy = torch.flip(yx, [1])

        # Compensate for downscaling of masks
        xy = xy * self.scale

        # Split indices
        indices_split = indices.float().split(indices_len)
        for i in indices_split:
            assert i.std() == 0

        xy_split = xy.float().split(indices_len)

        # Compute means (mass center)
        xy_means = [torch.mean(xy, dim=0) for xy in xy_split]

        assert len(xy_means) == len(mean_depths), (
            len(xy_means),
            len(mean_depths),
        )

        points = torch.cat(
            [torch.stack(xy_means), mean_depths.unsqueeze(1)],
            dim=1,
        )

        return points

    def _make_cameras(self) -> projection.render.Cameras:
        """
        Create a camera object.
        """
        # TODO: Support metadata from dataset for values

        fx = 2268.36
        fy = 2225.5405988775956
        u0 = 1048.64
        v0 = 519.277
        w = 2048
        h = 1024

        K = projection.render.build_calibration_matrix(
            focal_lengths=[(fx, fy)], principal_points=[(u0, v0)]
        ).to(device=self.max_depth.device)
        T = torch.zeros((1, 3), device=self.max_depth.device)
        R = torch.eye(3, device=self.max_depth.device)[None, :]

        return projection.render.Cameras(
            K=K,
            R=R,
            T=T,
            image_size=(h, w),
        )

    def _unproject(
        self, cams: projection.render.Cameras, points_xyd: Tensor
    ) -> Tensor:
        points_xyd_batch = points_xyd[None, :]
        points_world = cams.unproject_points(points_xyd_batch)
        return points_world[0]


# from pytorch3d.renderer import CamerasBase, PerspectiveCameras
# def _make_cameras(
#     device: torch.device,
# ) -> CamerasBase:
#     fx = 2268.36
#     fy = 2225.5405988775956
#     u0 = 1048.64
#     v0 = 519.277
#     w = 2048
#     h = 1024

#     translation = [0, 0, 0]

#     rotation = [
#         [1, 0, 0],
#         [0, 1, 0],
#         [0, 0, 1],
#     ]

#     cameras = PerspectiveCameras(
#         focal_length=((fx, fy),),
#         principal_point=((u0, v0),),
#         R=torch.as_tensor(
#             rotation, dtype=torch.float32, device=device
#         ).unsqueeze(0),
#         T=torch.as_tensor(
#             translation, dtype=torch.float32, device=device
#         ).unsqueeze(0),
#         image_size=((h, w),),
#         in_ndc=False,
#         device=device,
#     )
#     return cameras


# def make_points_xyd(
#     boxes: Tensor, depths: Tensor
# ) -> Tensor:
#     centers_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
#     centers_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2

#     points = torch.stack([centers_x, centers_y, depths], dim=1)

#     return points


# @torch.jit.unused()
# def _make_points_xyd(masks: Tensor, mean_depths: Tensor):
#     """
#     Create a list of points using the mass center and mean depth of each
#     instance.
#     """
#     indices, xy = masks.argwhere().split([1, 2], dim=1)
#     _, indices_len = indices.unique(return_counts=True)
#     indices_len = indices_len.tolist()

#     indices_split = indices.float().split(indices_len)
#     assert all(i.std() == 0 for i in indices_split)

#     xy_split = xy.float().split(indices_len)
#     xy_means = torch.stack(list(map(partial(torch.mean, dim=0), xy_split)))

#     assert len(xy_means) == len(mean_depths)

#     points = torch.cat([xy_means, mean_depths.unsqueeze(1)], dim=1)

#     return points


# @torch.jit.ignore()
# def _unproject(masks: Tensor, depths: Tensor) -> Tensor:
#     cams = _make_cameras(masks.device)

#     points_2d = _make_points_xyd(masks, depths)
#     points_2d.unsqueeze_(0)  # Add batch dimension

#     points_3d = cams.unproject_points(points_2d, world_coordinates=True)
#     points_3d.squeeze_(0)  # Remove batch dimension

#     return points_3d
