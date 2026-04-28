# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import ip_adapter.config as config
print("config.attention_map_save:",config.attention_map_save)
try:
    import xformers
    import xformers.ops
    xformers_available = True
except Exception as e:
    xformers_available = False

class RegionControler(object):
    def __init__(self) -> None:
        self.prompt_image_conditioning = []
region_control = RegionControler()

class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def forward(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
    
class IPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, scale_style=1.0, num_tokens=4,dtype=torch.float16):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.scale_style = scale_style
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False).to(dtype)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False).to(dtype)

        self.to_k_style = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False).to(dtype)
        self.to_v_style = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False).to(dtype)

    def forward(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens*2
            encoder_hidden_states, ip_hidden_states, style_hidden_states = encoder_hidden_states[:, :end_pos, :], encoder_hidden_states[:, end_pos:self.num_tokens+end_pos, :], encoder_hidden_states[:, self.num_tokens+end_pos:, :]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if xformers_available:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)
        
        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)
        
        if xformers_available:
            ip_hidden_states = self._memory_efficient_attention_xformers(query, ip_key, ip_value, None)
        else:
            ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
            ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        # region control
        if len(region_control.prompt_image_conditioning) == 1:
            region_mask = region_control.prompt_image_conditioning[0].get('region_mask', None)
            if region_mask is not None:
                h, w = region_mask.shape[:2]
                ratio = (h * w / query.shape[1]) ** 0.5
                mask = F.interpolate(region_mask[None, None], scale_factor=1/ratio, mode='nearest').reshape([1, -1, 1])
            else:
                mask = torch.ones_like(ip_hidden_states)
            ip_hidden_states = ip_hidden_states * mask    

        # for style-adapter
        style_key = self.to_k_style(style_hidden_states)
        style_value = self.to_v_style(style_hidden_states)
        
        style_key = attn.head_to_batch_dim(style_key)
        style_value = attn.head_to_batch_dim(style_value)
        
        if xformers_available:
            style_hidden_states = self._memory_efficient_attention_xformers(query, style_key, style_value, None)
        else:
            style_attention_probs = attn.get_attention_scores(query, style_key, None)
            style_hidden_states = torch.bmm(style_attention_probs, style_value)
        style_hidden_states = attn.batch_to_head_dim(style_hidden_states)

        # style region control
        if len(region_control.prompt_image_conditioning) == 1:
            region_mask = region_control.prompt_image_conditioning[0].get('region_mask', None)
            if region_mask is not None:
                h, w = region_mask.shape[:2]
                ratio = (h * w / query.shape[1]) ** 0.5
                mask = F.interpolate(region_mask[None, None], scale_factor=1/ratio, mode='nearest').reshape([1, -1, 1])
            else:
                mask = torch.ones_like(style_hidden_states)
            style_hidden_states = style_hidden_states * mask    

        hidden_states = hidden_states + self.scale * ip_hidden_states + self.scale_style * style_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        # TODO attention_mask
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        # hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        name='',
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.name=name

    def forward(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        self.hidden_states=hidden_states.to('cpu')
        
        
        
        if config.attention_map_save is not None:
            if config.bg is None:
            
                from PIL import Image
                import numpy as np
                # region_mask = region_control.prompt_image_conditioning[0].get('region_mask', None)
                bg_image = Image.open('bg_mask.png').convert('L')  # 转为灰度图
                bg_np = np.array(bg_image)  # 转为 NumPy 数组

                # 转为 PyTorch 张量并归一化
                config.bg = torch.tensor(bg_np / 255.0)  # 转为布尔张量
                
            mask_4d = config.bg.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 512, 512]
            import math

            mask_resized = F.interpolate(
                mask_4d, 
                size=int(math.sqrt(query.shape[2])),                # 目标尺寸 64x64
                mode='bilinear',        # 插值模式 (对浮点型 mask) 或 'nearest' (对二值 mask)
                align_corners=False     # 对齐方式
            ).reshape([1, -1, 1] )  
            newmask = mask_resized.to(dtype=hidden_states.dtype, device=hidden_states.device)
            
            hidden_states_save = config.attention_map_save[config.cur_step][self.name]['hidden_states'].to(device=hidden_states.device)
            

            hidden_states = hidden_states * newmask+hidden_states_save * (1-newmask)        

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class IPAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, scale_style=1.0, num_tokens=4,name='',dtype=torch.float16):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.scale_style = scale_style
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False).to(dtype)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False).to(dtype)

        self.to_k_style = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False).to(dtype)
        self.to_v_style = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False).to(dtype)
        
        self.name=name

    def forward(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens*2
            encoder_hidden_states, ip_hidden_states, style_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:self.num_tokens+end_pos, :],
                encoder_hidden_states[:, self.num_tokens+end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
                
#         key = attn.to_k(encoder_hidden_states)
#         value = attn.to_v(encoder_hidden_states)

#         inner_dim = key.shape[-1]
#         head_dim = inner_dim // attn.heads

#         query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         # the output of sdp = (batch, num_heads, seq_len, head_dim)
#         # TODO: add support for attn.scale when we move to Torch 2.1
#         hidden_states = F.scaled_dot_product_attention(
#             query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#         )

#         hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#         hidden_states = hidden_states.to(query.dtype)

       
        # for style-adapter
        style_key = self.to_k_style(style_hidden_states)
        style_value = self.to_v_style(style_hidden_states)
        
        inner_dim = style_key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        style_key = style_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        style_value = style_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # print("query, style_key, style_value:",query.shape, style_key.shape, style_value.shape)
        style_hidden_states = F.scaled_dot_product_attention(
            query, style_key, style_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        # print("style_hidden_states:",style_hidden_states.shape)
        with torch.no_grad():
            self.attn_map = query @ style_key.transpose(-2, -1).softmax(dim=-1)
            #print(self.attn_map.shape)

        style_hidden_states = style_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        style_hidden_states = style_hidden_states.to(query.dtype)
        
        

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        ip_hidden_states = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        with torch.no_grad():
            self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)
            #print(self.attn_map.shape)

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)
        
        hidden_states_ip = hidden_states + self.scale * ip_hidden_states
        self.hidden_states =hidden_states_ip.to('cpu')
        
        if config.attention_map_save is not None:
            style_hidden_states_save=config.attention_map_save[config.cur_step][self.name]['style_hidden_states']
            if config.bg is None:
            
                from PIL import Image
                import numpy as np
                # region_mask = region_control.prompt_image_conditioning[0].get('region_mask', None)
                bg_image = Image.open('bg_mask.png').convert('L')  # 转为灰度图
                bg_np = np.array(bg_image)  # 转为 NumPy 数组

                # 转为 PyTorch 张量并归一化
                config.bg = torch.tensor(bg_np / 255.0)  # 转为布尔张量
                
            mask_4d = config.bg.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 512, 512]
            import math

            mask_resized = F.interpolate(
                mask_4d, 
                size=int(math.sqrt(query.shape[2])),                # 目标尺寸 64x64
                mode='bilinear',        # 插值模式 (对浮点型 mask) 或 'nearest' (对二值 mask)
                align_corners=False     # 对齐方式
            ).reshape([1, -1, 1] )  
            newmask = mask_resized.to(dtype=style_hidden_states.dtype, device=style_hidden_states.device)
            

            style_hidden_states = style_hidden_states * newmask+style_hidden_states_save.to(device=style_hidden_states.device) * (1-newmask)
            hidden_states_ip= hidden_states_ip* newmask+ config.attention_map_save[config.cur_step][self.name]['hidden_states'].to(device=style_hidden_states.device) * (1-newmask)

            
            
            hidden_states = hidden_states_ip + self.scale_style * style_hidden_states 
        else:
            hidden_states = hidden_states_ip + self.scale_style * style_hidden_states 
        self.style_hidden_states=style_hidden_states.to('cpu')


        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
