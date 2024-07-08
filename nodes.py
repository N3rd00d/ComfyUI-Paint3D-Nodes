import sys
import argparse
import os
import cv2
from tqdm import tqdm
import torch
import torchvision
import time
import numpy as np
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import List, Tuple
from einops import rearrange
import folder_paths
# import tensorflow_io as tfio

from .Paint3D.paint3d import utils
from .Paint3D.paint3d.models.textured_mesh import TexturedMeshModel
from .Paint3D.paint3d.dataset import init_dataloaders
from .Paint3D.paint3d.config.train_config_paint3d import GuideConfig, OptimConfig, LogConfig


def to_pil_image(hwc_tensor: torch.Tensor) -> Image.Image:
    return Image.fromarray((hwc_tensor.cpu().numpy() * 255).astype(np.uint8))


def chw_to_bhwc(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)  # [c, h, w]-->[b, c, h, w]
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)  #  [b, 1, h, w]-->[b, 3, h, w]
    tensor = tensor.permute(0, 2, 3, 1).detach()  # [b, c, h, w]-->[b, h, w, c]
    return tensor


def bhwc_to_chw(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # [b, h, w, c]-->[h, w, c]
    if tensor.shape[2] == 1:
        tensor = tensor.repeat(1, 1, 3)  # [h, w, 1]-->[h, w, 3]
    tensor = tensor.permute(2, 0, 1).detach()  # [h, w, c]-->[c, h, w]
    return tensor


def bchw_to_chw(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # [b, c, h, w]-->[c, h, w]
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)  #  [1, h, w]-->[3, h, w]
    return tensor.detach()


def chw_to_hwc(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)  #  [1, h, w]-->[3, h, w]
    tensor = tensor.permute(1, 2, 0).detach()  # [3, h, w]-->[h, w, 3]
    return tensor


def bhwc_to_hwc(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # [b, h, w, c]-->[h, w, c]
    return tensor.detach()


def to_tensor_image(image: Image.Image) -> torch.Tensor:
    to_tensor = torchvision.transforms.ToTensor()
    tensor_image = to_tensor(image)
    return chw_to_bhwc(tensor_image)


def get_output_directory(obj_path):
    dir = os.path.dirname(obj_path)
    new_dir = os.path.join(dir, "output")
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir



def inpaint_viewpoint(save_result_dir: str, mesh_model: TexturedMeshModel, dataloaders, inpaint_view_ids):
    view_angle_info = {i: data for i, data in enumerate(dataloaders)}
    inpaint_used_key = ["image", "depth", "uncolored_mask"]
    results = {}

    batch_img = []
    for view_id in inpaint_view_ids:
        data = view_angle_info[view_id]
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        outputs = mesh_model.render(theta=theta, phi=phi, radius=radius)
        view_img_info = [outputs[k] for k in inpaint_used_key]
        batch_img.append(view_img_info)

    for i, img in enumerate(zip(*batch_img)):
        img = torch.cat(img, dim=3)
        if img.size(1) == 1:
            img = img.repeat(1, 3, 1, 1)
        t = '_'.join(map(str, inpaint_view_ids))
        name = inpaint_used_key[i]
        if name == "uncolored_mask":
            img[img > 0] = 1
        save_path = os.path.join(save_result_dir, f"view_{t}_{name}.png")
        utils.save_tensor_image(img, save_path=save_path)
        results[name] = img

    return results["image"], results["depth"], results["uncolored_mask"]


@torch.no_grad()
def generate_video(dataloaders, mesh_model, save_result_dir):
    all_render_rgb_frames = []
    mesh_model.renderer.clear_seen_faces()
    for i, data in tqdm(enumerate(dataloaders), desc="Evalating textured mesh~"):
        phi = data['phi']
        phi = float(phi + 2 * np.pi if phi < 0 else phi)

        outputs = mesh_model.render(theta=data['theta'], phi=phi, radius=data['radius'], dims=(1024, 1024))
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        mask, uncolored_masks = outputs['mask'], outputs['uncolored_mask']
        color_with_shade_img = utils.color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals, light_coef=0.3)
        rgb_render = outputs['image'] * (1 - uncolored_masks) + color_with_shade_img * uncolored_masks
        all_render_rgb_frames.append(utils.tensor2numpy(rgb_render))

    safe_path = os.path.join(save_result_dir, "render_rgb.mp4")
    utils.save_video(np.stack(all_render_rgb_frames, axis=0), safe_path)

    return safe_path


@dataclass
class RenderConfig:
    """ Parameters for the Mesh Renderer """
    grid_size: int = 512
    radius: float = 1.5
    look_at_height = 0.25
    base_theta: float = 60
    # Suzanne
    fov_para: float = 0.8  # 0.61 or 0.8 for Orthographic ; np.pi / 3 for Pinhole
    remove_mesh_part_names: List[str] = field(default_factory=["MI_CH_Top"].copy)
    remove_unsupported_buffers: List[str] = field(default_factory=["filamat"].copy)
    n_views: int = 24  # 16
    # Additional views to use before rotating around shape
    views_before: List[Tuple[float, float]] = field(default_factory=list)
    # Additional views to use after rotating around shape
    views_after: List[Tuple[float, float]] = field(default_factory=[[180, 30], [180, 150]].copy)
    # Whether to alternate between the rotating views from the different sides
    alternate_views: bool = True
    calcu_uncolored_mode: str = "WarpGrid"  # FACE_ID, DIFF, WarpGrid
    projection_mode: str = "Orthographic"  # Pinhole, Orthographic
    texture_interpolation_mode: str = 'bilinear'
    texture_default_color: List[float] = field(default_factory=[0.8, 0.1, 0.8].copy)
    texturify_blend_alpha: float = 1.0
    render_angle_thres: float = 68
    # Suzanne
    views_init: List[float] = field(default_factory=[0, 23].copy)
    views_inpaint: List[Tuple[float, float]] = field(default_factory=[(5, 6), (24, 25)].copy)


@dataclass
class TrainConfig:
    """ The main configuration for the coach trainer """
    log: LogConfig = field(default_factory=LogConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    guide: GuideConfig = field(default_factory=GuideConfig)


class MultiviewDataset:
    def __init__(self, train_config: TrainConfig, device):
        super().__init__()

        self.cfg = train_config.render
        self.device = device
        self.type = type  # train, val, tests
        size = self.cfg.n_views
        self.phis = [(index / size) * 360 for index in range(size)]
        self.thetas = [self.cfg.base_theta for _ in range(size)]

        # Alternate lists
        alternate_lists = lambda l: [l[0]] + [i for j in zip(l[1:size // 2], l[-1:size // 2:-1]) for i in j] + [
            l[size // 2]]
        if self.cfg.alternate_views:
            self.phis = alternate_lists(self.phis)
            self.thetas = alternate_lists(self.thetas)

        for phi, theta in self.cfg.views_before:
            self.phis = [phi] + self.phis
            self.thetas = [theta] + self.thetas
        for phi, theta in self.cfg.views_after:
            self.phis = self.phis + [phi]
            self.thetas = self.thetas + [theta]

        self.size = len(self.phis)

    def collate(self, index):
        phi = self.phis[index[0]]
        theta = self.thetas[index[0]]
        radius = self.cfg.radius
        thetas = torch.FloatTensor([np.deg2rad(theta)]).to(self.device).item()
        phis = torch.FloatTensor([np.deg2rad(phi)]).to(self.device).item()

        return {'theta': thetas, 'phi': phis, 'radius': radius}

    def dataloader(self):
        loader = DataLoader(
            list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=False, num_workers=0)
        loader._data = self
        return loader


class GenerateTrainConfig:
    RETURN_TYPES = ("TRAINCONFIG",)
    FUNCTION = "generate"
    CATEGORY = "Paint3D"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file_path": ("STRING", {"default": '', "multiline": False}),
            }
        }

    def generate(self, mesh_file_path):
        config = TrainConfig()
        config.guide.shape_path = mesh_file_path
        return (config,)


class GenerateTextureMeshModel:
    RETURN_TYPES = ("MESHMODEL",)
    FUNCTION = "generate"
    CATEGORY = "Paint3D"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "train_config": ("TRAINCONFIG",),
            }
        }

    def generate(self, train_config: TrainConfig):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mesh_model = TexturedMeshModel(cfg=train_config, device=device).to(device)
        mesh_model.initial_texture_path = None
        mesh_model.refresh_texture()
        return (mesh_model,)


class GenerateDilateDepthImage:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Paint3D"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_model": ("MESHMODEL",),
                "cam1": ("INT", {"default": 0, "min": 0, "max": 26}),
                "cam2": ("INT", {"default": 23, "min": 0, "max": 26}),
            }
        }

    def dilate_depth_outline(self, img: Image.Image, iters=5, dilate_kernel=3):
        img_gray = img.convert('L')  # grayscale로 변환
        img = np.array(img_gray)
        for i in range(iters):
            _, mask = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
            mask = mask / 255
            img_dilate = cv2.dilate(img, np.ones((dilate_kernel, dilate_kernel), np.uint8))
            img = (mask * img + (1 - mask) * img_dilate).astype(np.uint8)

        return np.dstack([img.astype(np.uint8)] * 3).copy(order='C')  #  to rgb

    def generate(self, mesh_model: TexturedMeshModel, cam1=0, cam2=23):
        init_depth_map = []
        init_rgb_map = []

        train_config = mesh_model.cfg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = MultiviewDataset(train_config, device)
        dataloaders = dataset.dataloader()

        view_angle_info = {i: data for i, data in enumerate(dataloaders)}
        view_ids = [cam1, cam2]
        for view_id in view_ids:
            data = view_angle_info[view_id]
            theta, phi, radius = data['theta'], data['phi'], data['radius']
            outputs = mesh_model.render(theta=theta, phi=phi, radius=radius)
            depth_render = outputs['depth']
            init_depth_map.append(depth_render)
            rgb_render = outputs['image']
            init_rgb_map.append(rgb_render)

        init_depth_map = torch.cat(init_depth_map, dim=0).repeat(1, 3, 1, 1)
        init_depth_map = torchvision.utils.make_grid(init_depth_map, nrow=2, padding=0)  # CHW
        init_depth_map = chw_to_hwc(init_depth_map)
        depth_pil_img = to_pil_image(hwc_tensor=init_depth_map)

        depth_dilated = self.dilate_depth_outline(depth_pil_img, iters=5, dilate_kernel=3)
        tensor_depth_dilated = to_tensor_image(depth_dilated)

        # init_rgb_map = torch.cat(init_rgb_map, dim=0).repeat(1, 1, 1, 1)
        # init_rgb_map = torchvision.utils.make_grid(init_rgb_map, nrow=2, padding=0)
        # rgb_map = init_rgb_map.unsqueeze(0).permute(0, 2, 3, 1)  # [c, h, w]-->[n, h, w, c]

        return (tensor_depth_dilated,)


class ProjectToMeshModel:
    RETURN_TYPES = ("MESHMODEL", "IMAGE",)
    RETURN_NAMES = ("model_model", "albedo",)
    FUNCTION = "project"
    CATEGORY = "Paint3D"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mesh_model": ("MESHMODEL",),
                "cam1": ("INT", {"default": 0, "min": 0, "max": 26}),
                "cam2": ("INT", {"default": 23, "min": 0, "max": 26}),
            }
        }

    def forward_texturing(self, cfg, dataloaders, mesh_model, save_result_dir, device, view_imgs=[], view_ids=[],
                          verbose=False):
        assert len(view_imgs) == len(view_ids), "the number of view_imgs should equal to the number of view_ids"

        view_info = {}
        for view_id, view_img in zip(view_ids, view_imgs):
            view_info[view_id] = {"img": view_img}

        for view_id, data in tqdm(enumerate(dataloaders)):
            if view_id not in view_ids:
                continue
            theta, phi, radius = data['theta'], data['phi'], data['radius']
            view_info[view_id]["pos"] = (theta, phi, radius)

        for view_id in view_ids:
            view_img = view_info[view_id]["img"]
            theta, phi, radius = view_info[view_id]["pos"]
            target_sd = utils.pil2tensor(
                Image.fromarray(view_img).convert('RGB').resize(
                    (cfg.render.grid_size, cfg.render.grid_size), resample=Image.BILINEAR), device)
            mesh_model.forward_texturing(
                theta=theta,
                phi=phi,
                radius=radius,
                view_target=target_sd,
                save_result_dir=save_result_dir,
                view_id=view_id,
                verbose=verbose)

        return mesh_model.export_mesh(save_result_dir)

    def project(self, image: torch.Tensor, mesh_model: TexturedMeshModel, cam1=0, cam2=23):
        train_config: TrainComfig = mesh_model.cfg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = MultiviewDataset(train_config, device)
        dataloaders = dataset.dataloader()
        hwc_image = bhwc_to_hwc(image)
        pil_image = to_pil_image(hwc_tensor=hwc_image)
        view_imgs = utils.split_grid_image(img=np.array(pil_image), size=(1, 2))
        output_dir = get_output_directory(train_config.guide.shape_path)

        albedo = self.forward_texturing(
            cfg=train_config,
            dataloaders=dataloaders,
            mesh_model=mesh_model,
            save_result_dir=output_dir,
            device=device,
            view_imgs=view_imgs,
            view_ids=[cam1, cam2],
            verbose=False, )
        albedo = to_tensor_image(albedo)
        return mesh_model, albedo,


class GenerateInpaintImageAndMask:
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE",)
    RETURN_NAMES = ("image", "mask", "depth")
    FUNCTION = "generate"
    CATEGORY = "Paint3D"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_model": ("MESHMODEL",),
                "cam1": ("INT", {"default": 0, "min": 0, "max": 26}),
                "cam2": ("INT", {"default": 23, "min": 0, "max": 26}),
            }
        }

    def generate(self, mesh_model: TexturedMeshModel, cam1=0, cam2=23):
        train_config: TrainComfig = mesh_model.cfg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = MultiviewDataset(train_config, device)
        dataloaders = dataset.dataloader()
        output_dir = get_output_directory(train_config.guide.shape_path)
        image, depth, mask = inpaint_viewpoint(
            save_result_dir=output_dir, mesh_model=mesh_model, dataloaders=dataloaders, inpaint_view_ids=[cam1, cam2], )

        image = chw_to_bhwc(image)
        mask = chw_to_bhwc(mask)
        mask = mask[:, :, :, 0]
        depth = chw_to_bhwc(depth)

        return image, mask, depth


class GeneratePreviewVideo:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "Paint3D"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_model": ("MESHMODEL",),
            }
        }

    def generate(self, mesh_model: TexturedMeshModel):
        train_config: TrainComfig = mesh_model.cfg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = MultiviewDataset(train_config, device)
        dataloaders = dataset.dataloader()
        output_dir = get_output_directory(train_config.guide.shape_path)
        video_url = generate_video(
            dataloaders=dataloaders, mesh_model=mesh_model, save_result_dir=output_dir, )
        return (video_url, )


NODE_CLASS_MAPPINGS = {
    "3D_GenerateDepthImage": GenerateDilateDepthImage,
    "3D_TrainConfig": GenerateTrainConfig,
    "3D_LoadMeshModel": GenerateTextureMeshModel,
    "3D_Projection": ProjectToMeshModel,
    "3D_GenerateInpaintImageAndMask": GenerateInpaintImageAndMask,
    "3D_GeneratePreviewVideo": GeneratePreviewVideo,
}
