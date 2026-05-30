import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from oat.policy.base_policy import BasePolicy
from oat.tokenizer.oat.tokenizer import OATTok
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.perception.fused_obs_encoder import FusedObservationEncoder
from oat.model.autoregressive.transformer_cache import AutoregressiveModel

if TYPE_CHECKING:
    from oat.perception.robomimic_vision_encoder import DenseRgbEncoder


class OATPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        action_tokenizer: OATTok,
        n_action_steps: int,
        n_obs_steps: int,
        # policy model params
        embed_dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        # dense visual memory (cross-attn input)
        use_dense_visual_memory: bool = False,
        use_cross_attn: Optional[bool] = None,
        dense_feature_dim: Optional[int] = None,
        max_memory_len: int = 1024,
        num_state_tokens: int = 1,
        num_tasks: int = 10,
        rgb_camera_keys: Optional[List[str]] = None,
        dense_crop_shape: Optional[Tuple[int, int]] = (76, 76),
        share_dense_rgb_encoder: bool = True,
        # policy inference params
        temperature: float = 1.0,
        topk: int = 10,
    ):
        super().__init__()
        
        modalities = obs_encoder.modalities()
        obs_feature_dim = obs_encoder.output_feature_dim()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_key_shapes = dict()
        obs_ports = []
        for key, attr in shape_meta['obs'].items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr['type']
            if type in modalities:
                obs_ports.append(key)

        # freeze action tokenizer
        for param in action_tokenizer.parameters():
            param.requires_grad_(False)
        action_tokenizer.eval()

        # Backward/plan compatibility: allow use_cross_attn alias
        if use_cross_attn is not None:
            use_dense_visual_memory = use_cross_attn
        self.use_dense_visual_memory = use_dense_visual_memory
        d_model = dense_feature_dim if dense_feature_dim is not None else embed_dim
        self.d_model = d_model
        self.max_memory_len = max_memory_len
        self.num_state_tokens = num_state_tokens

        if rgb_camera_keys is None:
            rgb_camera_keys = [
                k for k, attr in shape_meta["obs"].items()
                if attr.get("type") == "rgb"
            ]
        self.rgb_camera_keys = list(rgb_camera_keys)
        assert len(self.rgb_camera_keys) >= 1, "dense/legacy policy needs rgb cameras"

        # create AR model
        codebook_size = action_tokenizer.quantizer.codebook_size
        latent_horizon = action_tokenizer.latent_horizon
        model = AutoregressiveModel(
            vocab_size=codebook_size + 1,  # +1 for <BOS>
            max_seq_len=latent_horizon + 1,
            max_cond_len=n_obs_steps,
            max_memory_len=max_memory_len if use_dense_visual_memory else n_obs_steps,
            cond_dim=embed_dim if use_dense_visual_memory else obs_feature_dim,
            n_layer=n_layers,
            n_head=n_heads,
            n_emb=embed_dim,
            p_drop_emb=dropout,
            p_drop_attn=dropout,
        )
        bos_id = codebook_size  # last token id for <BOS>

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.action_tokenizer = action_tokenizer
        self.model = model
        self.max_seq_len = latent_horizon
        self.bos_id = bos_id
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.temperature = temperature
        self.topk = topk

        self.dense_rgb_encoder: Optional["DenseRgbEncoder"] = None
        self.time_embed: Optional[nn.Embedding] = None
        self.camera_embed: Optional[nn.Embedding] = None
        self.task_uid_embed: Optional[nn.Embedding] = None
        self.state_to_memory: Optional[nn.Module] = None
        self._state_encoder = None
        self.memory_index_map: Optional[torch.Tensor] = None

        if use_dense_visual_memory:
            assert isinstance(obs_encoder, FusedObservationEncoder), (
                "use_dense_visual_memory expects FusedObservationEncoder "
                "(for state normalization / legacy fallback)."
            )
            self._state_encoder = obs_encoder.state_encoder
            state_dim = self._state_encoder.output_feature_dim() if self._state_encoder else 0
            has_task_uid = "task_uid" in shape_meta["obs"]
            from oat.perception.robomimic_vision_encoder import DenseRgbEncoder

            self.dense_rgb_encoder = DenseRgbEncoder(
                shape_meta=shape_meta,
                d_model=d_model,
                crop_shape=dense_crop_shape,
                share_rgb_model=share_dense_rgb_encoder,
                rgb_keys=self.rgb_camera_keys,
            )
            self.time_embed = nn.Embedding(n_obs_steps, d_model)
            self.camera_embed = nn.Embedding(len(self.rgb_camera_keys), d_model)
            if has_task_uid:
                self.task_uid_embed = nn.Embedding(num_tasks, d_model)
            in_state_dim = state_dim + (d_model if has_task_uid else 0)
            self.state_to_memory = nn.Sequential(
                nn.Linear(in_state_dim, d_model * num_state_tokens),
                nn.Mish(),
                nn.LayerNorm(d_model * num_state_tokens),
            )

        # report
        num_obs_params = sum(p.numel() for p in obs_encoder.parameters())
        num_trainable_obs_params = sum(p.numel() for p in obs_encoder.parameters() if p.requires_grad)
        obs_trainable_ratio = num_trainable_obs_params / max(num_obs_params, 1)
        num_tok_params = sum(p.numel() for p in action_tokenizer.parameters())
        num_trainable_tok_params = sum(p.numel() for p in action_tokenizer.parameters() if p.requires_grad)
        tok_trainable_ratio = num_trainable_tok_params / max(num_tok_params, 1)
        num_model_params = sum(p.numel() for p in model.parameters())
        num_trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_trainable_ratio = num_trainable_model_params / max(num_model_params, 1)
        mem_mode = "dense_visual_memory" if use_dense_visual_memory else "pooled_obs"
        print(
            f"{self.get_policy_name()} initialized ({mem_mode}) with\n"
            f"  obs enc: {num_obs_params/1e6:.1f}M ({obs_trainable_ratio:.5%} trainable)\n"
            f"  act tok: {num_tok_params/1e6:.1f}M ({tok_trainable_ratio:.5%} trainable)\n"
            f"  policy : {num_model_params/1e6:.1f}M ({model_trainable_ratio:.5%} trainable)\n"
        )

    def get_observation_encoder(self):
        return self.obs_encoder

    def get_observation_modalities(self):
        return self.modalities
    
    def get_observation_ports(self):
        return self.obs_ports
    
    def get_policy_name(self):
        base_name = 'oatpolicy_'
        if self.use_dense_visual_memory:
            base_name += 'dense|'
        for modality in self.modalities:
            if modality != 'state':
                base_name += modality + '|'
        return base_name[:-1]

    def create_dummy_observation(self,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        return super().create_dummy_observation(
            batch_size=batch_size,
            horizon=self.n_obs_steps,
            obs_key_shapes=self.obs_key_shapes,
            device=device
        )

    def set_normalizer(self, normalizer):
        self.obs_encoder.set_normalizer(normalizer)
        if self.dense_rgb_encoder is not None:
            self.dense_rgb_encoder.set_normalizer(normalizer)

    def get_optimizer(
        self, 
        policy_lr: float,
        obs_enc_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        """Create an AdamW optimizer with weight decay for 2D parameters only."""
        encoder_modules = [self.obs_encoder]
        if self.dense_rgb_encoder is not None:
            encoder_modules.append(self.dense_rgb_encoder)
        encoder_param_ids = set()
        for enc in encoder_modules:
            for p in enc.parameters():
                encoder_param_ids.add(id(p))

        encoder_decay_params = []
        encoder_nodecay_params = []
        policy_decay_params = []
        policy_nodecay_params = []
        for param in self.parameters():
            if not param.requires_grad:
                continue
            is_encoder = id(param) in encoder_param_ids
            if param.dim() >= 2:
                (encoder_decay_params if is_encoder else policy_decay_params).append(param)
            else:
                (encoder_nodecay_params if is_encoder else policy_nodecay_params).append(param)
        
        optim_groups = [
            {'params': policy_decay_params, 'lr': policy_lr, 'weight_decay': weight_decay},
            {'params': policy_nodecay_params, 'lr': policy_lr, 'weight_decay': 0.0},
            {'params': encoder_decay_params, 'lr': obs_enc_lr, 'weight_decay': weight_decay},
            {'params': encoder_nodecay_params, 'lr': obs_enc_lr, 'weight_decay': 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, betas=betas)
        return optimizer

    def _get_conditioning(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, bool]:
        if self.use_dense_visual_memory:
            return self.get_dense_memory(obs_dict), True
        return self.obs_encoder(obs_dict), False

    def get_dense_memory(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Build dense visual + state memory for cross-attention.

        Returns:
            memory: [B, N_total, d_model]
        """
        assert self.dense_rgb_encoder is not None
        sample_key = self.rgb_camera_keys[0]
        B = obs_dict[sample_key].shape[0]
        device = self.device
        d_model = self.d_model
        To = obs_dict[sample_key].shape[1]

        memory_parts = []
        index_rows = []

        for cam_idx, cam_key in enumerate(self.rgb_camera_keys):
            tokens = self.dense_rgb_encoder.encode_key(obs_dict, cam_key)  # [B*To, L, d]
            L = tokens.shape[1]
            tokens = tokens.view(B, To, L, d_model)
            time_idx = torch.arange(To, device=device).view(1, To, 1).expand(B, To, L)
            cam_idx_t = torch.full((B, To, L), cam_idx, device=device, dtype=torch.long)
            tokens = tokens + self.time_embed(time_idx) + self.camera_embed(cam_idx_t)
            tokens = tokens.reshape(B, To * L, d_model)
            memory_parts.append(tokens)

            hw = int(L ** 0.5)
            for t in range(To):
                for p in range(L):
                    h_pos = p // hw
                    w_pos = p % hw
                    index_rows.append((cam_idx, t, h_pos, w_pos, 0))  # type 0 = visual

        memory = torch.cat(memory_parts, dim=1)

        if self._state_encoder is not None:
            state_feat = self._state_encoder(obs_dict)  # [B, To, Ds]
            state_in = state_feat
            if self.task_uid_embed is not None and "task_uid" in obs_dict:
                uid = obs_dict["task_uid"].long()
                if uid.dim() == 3 and uid.shape[-1] == 1:
                    uid = uid.squeeze(-1)
                if uid.dim() == 1:
                    uid = uid.unsqueeze(1).expand(-1, To)
                elif uid.dim() == 2 and uid.shape[1] == 1 and To > 1:
                    uid = uid.expand(-1, To)
                # Keep task ids in embedding range for multi-suite/global ids.
                uid = torch.remainder(uid, self.task_uid_embed.num_embeddings)
                task_e = self.task_uid_embed(uid)
                state_in = torch.cat([state_feat, task_e], dim=-1)
            state_flat = self.state_to_memory(state_in)  # [B, To, K*d]
            state_tokens = state_flat.view(B, To * self.num_state_tokens, d_model)
            memory = torch.cat([memory, state_tokens], dim=1)
            for t in range(To):
                for k in range(self.num_state_tokens):
                    index_rows.append((-1, t, k, -1, 1))  # type 1 = state

        N = memory.shape[1]
        if N > self.max_memory_len:
            raise ValueError(
                f"memory length {N} exceeds max_memory_len={self.max_memory_len}"
            )

        self.memory_index_map = torch.tensor(index_rows, device=device, dtype=torch.long)
        return memory

    def predict_action(self, 
        obs_dict: Dict[str, torch.Tensor],
        use_k_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if use_k_tokens is None:
            use_k_tokens = self.max_seq_len
        else:
            use_k_tokens = min(use_k_tokens, self.max_seq_len)
        if temperature is None:
            temperature = self.temperature
        if topk is None:
            topk = self.topk

        cond, memory_is_embedded = self._get_conditioning(obs_dict)
        B = cond.shape[0]

        action_tokens = torch.full(
            (B, 1), self.bos_id, 
            dtype=torch.long, device=self.device
        )
        action_tokens = self.model.generate(
            action_tokens,
            cond=cond,
            memory_is_embedded=memory_is_embedded,
            max_new_tokens=use_k_tokens,
            temperature=temperature,
            top_k=topk,
        )[:, 1:]

        with torch.inference_mode():
            action_pred = self.action_tokenizer.detokenize(
                tokens=action_tokens,
            )

        action = action_pred[:,:self.n_action_steps]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result


    def forward(self, batch) -> torch.Tensor:
        with torch.inference_mode():
            action_tokens = self.action_tokenizer.tokenize(batch['action'])

        B = batch['action'].shape[0]
        device = batch['action'].device

        cond, memory_is_embedded = self._get_conditioning(batch['obs'])

        action_tokens = torch.cat([
            torch.full(
                (B, 1), self.bos_id, 
                dtype=torch.long, device=device
            ),
            action_tokens
        ], dim=1)

        logits = self.model(
            action_tokens[:, :-1],
            cond=cond,
            memory_is_embedded=memory_is_embedded,
        )

        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            action_tokens[:, 1:].reshape(-1)
        )
        return loss
