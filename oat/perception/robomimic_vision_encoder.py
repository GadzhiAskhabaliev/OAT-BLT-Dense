from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import robomimic.models.obs_nets as rmon

from oat.common.pytorch_util import replace_submodules
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.perception.crop_randomizer import CropRandomizer
from oat.model.common.normalizer import LinearNormalizer, _normalize


class RobomimicRgbEncoder(BaseObservationEncoder):
    """
    Assumes rgb input: B,H,W,C
    """
    def __init__(self,
        shape_meta: dict,
        crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
        use_group_norm: bool=True,
        eval_fixed_crop: bool = False,
        share_rgb_model: bool=False,
    ):
        super().__init__()

        rgb_ports = list()
        port_shape = dict()
        for key, attr in shape_meta['obs'].items():
            type = attr['type']
            shape = attr['shape']
            if type == 'rgb':
                rgb_ports.append(key)
                port_shape[key] = (shape[2], shape[0], shape[1])  # H,W,C -> C,H,W

        # init global state
        ObsUtils.initialize_obs_modality_mapping_from_dict({"rgb": rgb_ports})

        def crop_randomizer(shape, crop_shape):
            if crop_shape is None:
                return None
            return rmbn.CropRandomizer(
                input_shape=shape,
                crop_height=crop_shape[0],
                crop_width=crop_shape[1],
                num_crops=1,
                pos_enc=False,
            )
            
        def visual_net(shape, crop_shape):
            if crop_shape is not None:
                shape = (shape[0], crop_shape[0], crop_shape[1])
            net = rmbn.VisualCore(
                input_shape=shape,
                feature_dimension=64,
                backbone_class='ResNet18Conv',
                backbone_kwargs={
                    'input_channels': shape[0],
                    'input_coord_conv': False,
                },
                pool_class='SpatialSoftmax',
                pool_kwargs={
                    'num_kp': 32,
                    'temperature': 1.0,
                    'noise': 0.0,
                },
                flatten=True,
            )
            return net

        obs_encoder = rmon.ObservationEncoder()
        if share_rgb_model:
            this_shape = port_shape[rgb_ports[0]]
            net = visual_net(this_shape, crop_shape)
            obs_encoder.register_obs_key(
                name=rgb_ports[0],
                shape=this_shape,
                net=net,
                randomizer=crop_randomizer(this_shape, crop_shape),
            )
            for port in rgb_ports[1:]:
                assert port_shape[port] == this_shape
                obs_encoder.register_obs_key(
                    name=port,
                    shape=this_shape,
                    randomizer=crop_randomizer(this_shape, crop_shape),
                    share_net_from=rgb_ports[0],
                )
        else:
            for port in rgb_ports:
                shape = port_shape[port]
                net = visual_net(shape, crop_shape)
                obs_encoder.register_obs_key(
                    name=port,
                    shape=shape,
                    net=net,
                    randomizer=crop_randomizer(shape, crop_shape),
                )

        if use_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16,
                    num_channels=x.num_features,
                )
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        obs_encoder.make()
        self.encoder = obs_encoder
        self.rgb_keys = list(obs_encoder.obs_shapes.keys())
        self.normalizer = LinearNormalizer()

    def forward(self, obs_dict) -> Dict[str,torch.Tensor]:
        # normalize
        nobs = self._normalize_obs_dict(obs_dict)   # [B, To, H, W, C]
        
        sample = next(iter(nobs.values()))
        B, To, H, W, C = sample.shape
        
        for key in self.rgb_keys:
            # rgb H,W,C -> C,H,W
            nobs[key] = nobs[key].reshape(B*To, H, W, C).permute(0,3,1,2)
        image_feats = self.encoder(nobs)
        image_feats = image_feats.reshape(B, To, -1)
        return image_feats

    @torch.no_grad()
    def output_feature_dim(self) -> int:
        D = self.encoder.output_shape()[0]
        N = len(self.rgb_keys)
        dims = dict(zip(self.rgb_keys, [D // N] * N))
        return sum(dims.values())

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _normalize_obs_dict(self, obs_dict: Dict) -> Dict:
        nobs = dict()
        for port, value in obs_dict.items():
            if port in self.rgb_keys:
                params = self.normalizer.params_dict.get(port, None)
                if params is None:
                    nobs[port] = value
                    print(f"no normalizer params for port {port}, skipping normalization.")
                else:
                    nobs[port] = _normalize(value, params, forward=True)
        return nobs


class DenseRgbEncoder(BaseObservationEncoder):
    """
    Spatial RGB encoder: ResNet18 feature map -> patch tokens (no SpatialSoftmax / global pool).

    Input per camera: obs_dict[rgb_key] [B, To, H, W, C]
    Output: [B*To, L, d_model] with L = h*w (e.g. 8*8=64 for 128x128 input, stride 16).
    """

    def __init__(
        self,
        shape_meta: dict,
        d_model: int,
        crop_shape: Union[Tuple[int, int], None] = None,
        use_group_norm: bool = True,
        eval_fixed_crop: bool = False,
        share_rgb_model: bool = False,
        rgb_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.d_model = d_model

        if rgb_keys is None:
            rgb_keys = [
                key for key, attr in shape_meta["obs"].items()
                if attr.get("type") == "rgb"
            ]
        assert rgb_keys, "DenseRgbEncoder requires at least one rgb port."
        self.rgb_keys = list(rgb_keys)

        port_shape = dict()
        for key in self.rgb_keys:
            shape = shape_meta["obs"][key]["shape"]
            port_shape[key] = (shape[2], shape[0], shape[1])  # H,W,C -> C,H,W

        ObsUtils.initialize_obs_modality_mapping_from_dict({"rgb": self.rgb_keys})

        def crop_randomizer(shape, crop_shape):
            if crop_shape is None:
                return None
            return rmbn.CropRandomizer(
                input_shape=shape,
                crop_height=crop_shape[0],
                crop_width=crop_shape[1],
                num_crops=1,
                pos_enc=False,
            )

        def spatial_backbone(shape, crop_shape):
            if crop_shape is not None:
                shape = (shape[0], crop_shape[0], crop_shape[1])
            return rmbn.ResNet18Conv(
                input_channel=shape[0],
                input_coord_conv=False,
            )

        obs_encoder = rmon.ObservationEncoder()
        if share_rgb_model:
            this_shape = port_shape[self.rgb_keys[0]]
            net = spatial_backbone(this_shape, crop_shape)
            obs_encoder.register_obs_key(
                name=self.rgb_keys[0],
                shape=this_shape,
                net=net,
                randomizer=crop_randomizer(this_shape, crop_shape),
            )
            for port in self.rgb_keys[1:]:
                assert port_shape[port] == this_shape
                obs_encoder.register_obs_key(
                    name=port,
                    shape=this_shape,
                    randomizer=crop_randomizer(this_shape, crop_shape),
                    share_net_from=self.rgb_keys[0],
                )
        else:
            for port in self.rgb_keys:
                shape = port_shape[port]
                net = spatial_backbone(shape, crop_shape)
                obs_encoder.register_obs_key(
                    name=port,
                    shape=shape,
                    net=net,
                    randomizer=crop_randomizer(shape, crop_shape),
                )

        if use_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features,
                ),
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc,
                ),
            )

        obs_encoder.make()
        self.encoder = obs_encoder

        # Probe output channels from backbone (typically 256 for ResNet18 layer3)
        with torch.no_grad():
            c, h, w = port_shape[self.rgb_keys[0]]
            if crop_shape is not None:
                h, w = crop_shape
            dummy = torch.zeros(1, c, h, w)
            # Probe net directly (without randomizer) to avoid crop assertions
            # when dummy size equals crop size.
            feat = self.encoder.obs_nets[self.rgb_keys[0]](dummy)
            backbone_c = feat.shape[1]

        self.proj = nn.Conv2d(backbone_c, d_model, kernel_size=1)
        self.token_norm = nn.LayerNorm(d_model)
        self.normalizer = LinearNormalizer()

    def _encode_spatial_map(self, key: str, images: torch.Tensor) -> torch.Tensor:
        """images: [N, C, H, W] -> [N, C', h, w]"""
        net = self.encoder.obs_nets[key]
        randomizer = self.encoder.obs_randomizers[key]
        if randomizer is not None:
            images = randomizer.forward_in(images)
        return net(images)

    def encode_key(self, obs_dict: Dict, key: str) -> torch.Tensor:
        """
        Returns patch tokens for one rgb key: [B*To, L, d_model]
        """
        nobs = self._normalize_obs_dict(obs_dict)
        sample = nobs[key]
        B, To, H, W, C = sample.shape
        x = sample.reshape(B * To, H, W, C).permute(0, 3, 1, 2)
        feat_map = self._encode_spatial_map(key, x)
        feat_map = self.proj(feat_map)
        feat_map = feat_map.flatten(2).transpose(1, 2)  # [B*To, L, d_model]
        feat_map = self.token_norm(feat_map)
        return feat_map

    def forward(self, obs_dict: Dict) -> Dict[str, torch.Tensor]:
        """Per-key spatial tokens."""
        return {key: self.encode_key(obs_dict, key) for key in self.rgb_keys}

    def modalities(self) -> List[str]:
        return ["rgb"]

    @torch.no_grad()
    def output_feature_dim(self) -> int:
        return self.d_model

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _normalize_obs_dict(self, obs_dict: Dict) -> Dict:
        nobs = dict()
        for port, value in obs_dict.items():
            if port in self.rgb_keys:
                params = self.normalizer.params_dict.get(port, None)
                if params is None:
                    nobs[port] = value
                else:
                    nobs[port] = _normalize(value, params, forward=True)
        return nobs
