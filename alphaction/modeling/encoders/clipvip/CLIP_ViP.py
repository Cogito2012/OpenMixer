# coding=utf-8
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch CLIP model."""


from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
import copy


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "openai/clip-vit-base-patch32"

CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/clip-vit-base-patch32",
    # See all CLIP models at https://huggingface.co/models?filter=clip
]


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


@dataclass
class CLIPOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

class CLIPVisionViPEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig, additional_vision_config=None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.temporal_size = additional_vision_config.temporal_size
        self.if_use_temporal_embed = additional_vision_config.if_use_temporal_embed
        self.patch_size = config.patch_size

        self.add_cls_num = additional_vision_config.add_cls_num
        self.added_cls = nn.Parameter(torch.randn(self.add_cls_num, self.embed_dim))

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.grid_size = [self.image_size // self.patch_size, self.image_size // self.patch_size]
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))
        if self.if_use_temporal_embed:
            self.temporal_embedding = nn.Parameter(torch.zeros(1, self.temporal_size, self.embed_dim))

        # runtime position embeddings
        self.runtime_spe, self.runtime_tpe = None, None


    def forward(self, pixel_values: torch.FloatTensor, interpolator=None, return_ws=False) -> torch.Tensor:
        B, T, C, H, W = pixel_values.shape
        if self.if_use_temporal_embed:
            if T != self.temporal_embedding.shape[1]:
                time_embed = self.temporal_embedding.transpose(1, 2)
                time_embed = F.interpolate(time_embed, size=(T), mode='linear')
                time_embed = time_embed.transpose(1, 2)  
            else:
                time_embed = self.temporal_embedding
            # # record the runtime time_embed
            # self.runtime_tpe = time_embed
        
        patch_embeds = self.patch_embedding(pixel_values.reshape(-1, C, H, W))
        ws = patch_embeds.size()[-2:]  
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        C = patch_embeds.shape[-1]
        patch_embeds = patch_embeds.reshape(B, T, -1, C)
        
        if self.if_use_temporal_embed:
            patch_embeds = patch_embeds + time_embed.unsqueeze(2)
        
        cls_pos_embed = self.position_embedding(self.position_ids[:, 0:1])
        pos_embed = self.position_embedding(self.position_ids[:, 1:])
        # need to spatially interpolate the positional embedding
        if self.num_patches != ws[0] * ws[1]:
            assert interpolator is not None
            pos_embed = interpolator(pos_embed, self.grid_size, ws, 0)
        # record the runtime pos_embed
        self.runtime_spe = pos_embed.squeeze(0)

        patch_embeds = patch_embeds + pos_embed.unsqueeze(1)

        class_embeds = self.class_embedding.expand(B, 1, -1)
        class_embeds = class_embeds + cls_pos_embed

        added_cls = self.added_cls.expand(B, self.add_cls_num, -1)
        added_cls = added_cls + cls_pos_embed

        N, L = patch_embeds.shape[1], patch_embeds.shape[2]

        embeds = torch.cat([class_embeds, added_cls, patch_embeds.reshape(patch_embeds.shape[0], -1, patch_embeds.shape[-1])], dim=1)
        M = 1 + self.add_cls_num
        if return_ws:
            return (embeds, (M, N, L), ws)
        return (embeds, (M, N, L))  # [B, M+N*L, C]

class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.temporal_winsize = getattr(config, 'temporal_winsize', 1)

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_size,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        attn_forward_hook = None,
        attn_backward_hook = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        if inputs_size is not None:
            return self.forward2(hidden_states, inputs_size, 
                                 attn_forward_hook=attn_forward_hook,
                                 attn_backward_hook=attn_backward_hook), None

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # use hooks for the attention weights if necessary
        if attn_forward_hook is not None:
            attn_forward_hook(attn_probs)
        
        if attn_backward_hook is not None:
            attn_probs.register_hook(attn_backward_hook)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped
    
    def forward2(self, hidden_states, inputs_size, attn_forward_hook=None, attn_backward_hook=None):
        """
        hidden_states: [B, M+N*L, C]
        inputs_size: (M, N, L)
        """
        M, N, L = inputs_size
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        if self.temporal_winsize > 1:
            W = self.temporal_winsize
            # slicing over the N frames with sliding widow size W(=3)
            start_indices = torch.arange(0, N * L, L, device=hidden_states.device)
            range_indices = torch.arange(0, W*L, device=hidden_states.device).unsqueeze(0).repeat(N, 1) - int(W//2)*L
            all_indices = start_indices.unsqueeze(1) + range_indices  # (N, WL)
            all_indices = torch.clamp(all_indices, min=0, max=N*L-1)
            # compute the multi-frame sliced patch features
            k_frames = key_states[:, M:].index_select(1, all_indices.view(-1)).view(-1, W*L, self.head_dim)  # [B*num_heads*N, WL, head_dim]
            v_frames = value_states[:, M:].index_select(1, all_indices.view(-1)).view(-1, W*L, self.head_dim)  # [B*num_heads*N, WL, head_dim]
        else:
            # compute the single-frame sliced patch features
            k_frames = key_states[:, M:].reshape(-1, L, self.head_dim)
            v_frames = value_states[:, M:].reshape(-1, L, self.head_dim)

        # qkv: [B*num_heads, M+N*L, head_dim]
        # in-frame attention:
        q = query_states[:, M:].reshape(-1, L, self.head_dim)  #[B*num_heads*N, L, head_dim]
        k = key_states[:, :M].repeat(1, N, 1).reshape(-1, M, self.head_dim)  #[B*num_heads*N, M, head_dim]
        k = torch.cat([k, k_frames], dim=1)   #[B*num_heads*N, M+L, head_dim]
        v = value_states[:, :M].repeat(1, N, 1).reshape(-1, M, self.head_dim)  #[B*num_heads*N, M, head_dim]
        v = torch.cat([v, v_frames], dim=1)   #[B*num_heads*N, M+L, head_dim]
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs_intra = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs_intra, v)   # [B*num_heads*N, L, head_dim]
        attn_output = attn_output.view(bsz, self.num_heads, N, L, self.head_dim)
        attn_output = attn_output.permute(0, 2, 3, 1, 4)
        attn_output_frames = attn_output.reshape(bsz, N*L, embed_dim)  # [B, N*L, C]

        # cls divided attention:
        q = query_states[:, :M]    # [B*num_heads, M, head_dim]
        k = key_states  # [B*num_heads, M+N*L, head_dim]
        v = value_states  # [B*num_heads, M+N*L, head_dim]
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs_inter = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs_inter, v)   # [B*num_heads, M, head_dim]
        attn_output = attn_output.view(bsz, self.num_heads, M, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output_cls = attn_output.reshape(bsz, M, embed_dim)  # [B, M, C]

        attn_output = torch.cat([attn_output_cls, attn_output_frames], dim=1)

        attn_output = self.out_proj(attn_output)

        # use hooks for the attention weights if necessary
        if attn_forward_hook is not None:
            attn_forward_hook(attn_probs_intra, attn_probs_inter)
        
        if attn_backward_hook is not None:
            # register gradient hooks
            attn_probs_intra.register_hook(lambda grad: attn_backward_hook(grad, 'intra'))  # [B*num_heads*N, L=196, M+L] where L=196
            attn_probs_inter.register_hook(lambda grad: attn_backward_hook(grad, 'inter'))  # [B*num_heads, M=4, M+N*L] where M=4

        return attn_output


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        self.get_attn_prob = getattr(config, 'attn_prob', False)
        self.get_attn_grad = getattr(config, 'attn_grad', False)
        self.attn_forward_hook = None
        self.attn_backward_hook = None
        
        if self.get_attn_prob:
            self.attn_probs = {'intra': None, 'inter': None}
            self.attn_forward_hook = self.set_attn_probs
        
        if self.get_attn_grad:
            self.attn_grads = {'intra': None, 'inter': None}
            self.attn_backward_hook = self.set_attn_grad


    def set_attn_probs(self, attn_probs_intra, attn_probs_inter):
        self.attn_probs['intra'] = attn_probs_intra
        self.attn_probs['inter'] = attn_probs_inter

    def set_attn_grad(self, attn_grad, name):
        self.attn_grads[name] = attn_grad

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_size,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        if isinstance(hidden_states, tuple):
            residual = hidden_states

            hidden_states = (self.layer_norm1(hidden_states[0]), self.layer_norm1(hidden_states[1]))
            hidden_states, attn_weights = self.self_attn(
                hidden_states=hidden_states,
                inputs_size=inputs_size,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                attn_forward_hook=self.attn_forward_hook,
                attn_backward_hook=self.attn_backward_hook,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        else:
            residual = hidden_states

            hidden_states = self.layer_norm1(hidden_states)
            hidden_states, attn_weights = self.self_attn(
                hidden_states=hidden_states,
                inputs_size=inputs_size,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                attn_forward_hook=self.attn_forward_hook,
                attn_backward_hook=self.attn_backward_hook,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CLIPConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, CLIPTextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, CLIPVisionEmbeddings) or isinstance(module, CLIPVisionViPEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, CLIPAttention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, CLIPMLP):
            factor = self.config.initializer_factor
            in_proj_std = (
                (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            )
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, CLIPModel):
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, CLIPEncoder):
            module.gradient_checkpointing = value


CLIP_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
    Parameters:
        config ([`CLIPConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`CLIPTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`CLIPFeatureExtractor`]. See [`CLIPFeatureExtractor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

CLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`CLIPTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`CLIPFeatureExtractor`]. See [`CLIPFeatureExtractor.__call__`] for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].
    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        
        self.attn_last_only = getattr(config, 'attn_last_only', False)
        if self.attn_last_only:
            config_temp = copy.deepcopy(config)
            # previous 11 layers
            config_temp.attn_prob = False
            layers_prev = [CLIPEncoderLayer(config_temp) for _ in range(config.num_hidden_layers-1)]
            # the last layer
            config_temp.attn_prob = True
            layers_last = [CLIPEncoderLayer(config_temp)]
            self.layers = nn.ModuleList(layers_prev + layers_last)
        else:
            self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        inputs_size = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    inputs_size,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    inputs_size,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class CLIPTextTransformer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        bsz, seq_len = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        if_fp16 = hidden_states.dtype == torch.float16
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, fp16=if_fp16).to(hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), input_ids.argmax(dim=-1)]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len, fp16=False):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        if fp16:
            mask = mask.half()
        return mask


class CLIPTextModel(CLIPPreTrainedModel):
    config_class = CLIPTextConfig

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = CLIPTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        Examples:
        ```python
        >>> from transformers import CLIPTokenizer, CLIPTextModel
        >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig, additional_vision_config=None):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionViPEmbeddings(config, additional_vision_config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim)

        self.use_cross_attn = getattr(additional_vision_config, 'st_cross_attn', False)  # has attr and True
        if self.use_cross_attn:
            num_xattn = getattr(additional_vision_config, 'num_xattn', 1)
            from .custom_layers import CrossAttnModules, CrossAttnBlock
            self.cross_attn_layer = CrossAttnModules(*[CrossAttnBlock(embed_dim, embed_dim // 64, return_kv=True) 
                                                       for _ in range(num_xattn)])

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_ws: Optional[bool] = False,
        interpolator=None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embed_outs = self.embeddings(pixel_values, interpolator=interpolator, return_ws=return_ws)   # tuple
        hidden_states, inputs_size = embed_outs[0], embed_outs[1]
        if return_ws:
            ws = embed_outs[2]
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            inputs_size=inputs_size,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]

        # insert cross-attention layer
        if self.use_cross_attn:
            pooled_output = self.cross_attn_layer(pooled_output,   # (B, D)
                                                  last_hidden_state[:, 4:, :].permute(1, 0, 2).contiguous())  # (hw, B, D)

        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return_item = BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        if return_ws:
            return return_item, ws
        return return_item


class CLIPVisionModel(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_ws: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, CLIPVisionModel
        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            return_ws=return_ws,
        )


@add_start_docstrings(CLIP_START_DOCSTRING)
class CLIPModel(CLIPPreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                f"config.text_config is expected to be of type CLIPTextConfig but is of type {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                f"config.vision_config is expected to be of type CLIPVisionConfig but is of type {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        if hasattr(config, "vision_additional_config"):
            additional_vision_config = config.vision_additional_config
        else:
            additional_vision_config = None

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)
        self.vision_model = CLIPVisionTransformer(vision_config, additional_vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        if_norm: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].
        Examples:
        ```python
        >>> from transformers import CLIPTokenizer, CLIPModel
        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)
        
        if_norm = if_norm if if_norm is not None else False
        if if_norm:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        if_norm: Optional[bool] = None,
        interpolator=None,
        return_ws: Optional[bool] = False,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, CLIPModel
        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolator=interpolator,
            return_ws=return_ws,
        )
        if return_ws:
            vision_outputs, ws = vision_outputs[0], vision_outputs[1]

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        if_norm = if_norm if if_norm is not None else False
        if if_norm:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        if return_ws:
            return image_features, ws

        return image_features

    @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPOutput, config_class=CLIPConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolator=None,
    ) -> Union[Tuple, CLIPOutput]:
        r"""
        Returns:
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, CLIPModel
        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )
        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolator=interpolator,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )



def split_cls(x, size=(8, 49)):
    """
    x [B, 1+N*L, C]
    size [N, L]
    return [B, 1, C], [B, N, L, C]
    """
    B = x.shape[0]
    N, L = size
    return x[:, 0:1], x[:, 1:].reshape(B, N, L, -1)

def merge_cls(x1, x2):
    return torch.cat([x1, x2.reshape(x2.shape[0], -1, x2.shape[-1])], dim=1)