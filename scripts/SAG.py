
from inspect import isfunction
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from modules.processing import StableDiffusionProcessing

import math

import modules.scripts as scripts
from modules import shared
import gradio as gr

from modules.script_callbacks import on_cfg_denoiser,CFGDenoiserParams, CFGDenoisedParams, on_cfg_denoised, AfterCFGCallbackParams, on_cfg_after_cfg

import os

from scripts import xyz_grid_support_sag

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")
def exists(val):
    return val is not None
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class LoggedSelfAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.attn_probs = None

    def forward(self, x, context=None, mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],	
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
        
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            with torch.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        self.attn_probs = sim

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        if additional_tokens is not None:
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)

def xattn_forward_log(self, x, context=None, mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
    if additional_tokens is not None:
        # get the number of masked tokens at the beginning of the output sequence
        n_tokens_to_mask = additional_tokens.shape[1]
        # add additional token
        x = torch.cat([additional_tokens, x], dim=1)

    if n_times_crossframe_attn_in_self:
        # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
        assert x.shape[0] % n_times_crossframe_attn_in_self == 0
        # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
        k = repeat(
            k[::n_times_crossframe_attn_in_self],	
            "b ... -> (b n) ...",
            n=n_times_crossframe_attn_in_self,
        )
        v = repeat(
            v[::n_times_crossframe_attn_in_self],
            "b ... -> (b n) ...",
            n=n_times_crossframe_attn_in_self,
        )
            
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    # force cast to fp32 to avoid overflowing
    if _ATTN_PRECISION == "fp32":
        with torch.autocast(enabled=False, device_type='cuda'):
            q, k = q.float(), k.float()
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    del q, k

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    self.attn_probs = sim
    global current_selfattn_map
    current_selfattn_map = sim
    
    sim = sim.to(dtype=v.dtype)
    out = einsum('b i j, b j d -> b i d', sim, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    out = self.to_out(out)

    if additional_tokens is not None:
        out = out[:, n_tokens_to_mask:]
    global current_outsize
    current_outsize = out.shape[-2:]
    return out

saved_original_selfattn_forward = None
current_selfattn_map = None
current_sag_guidance_scale = 1.0
sag_enabled = False
sag_mask_threshold = 1.0
sag_blur_sigma = 1.0
sag_method_bilinear = False
sag_attn_target = ""

current_max_sigma = 0.0
current_sag_block_index = -1
current_xin = None
current_outsize = (64,64)
current_batch_size = 1
current_degraded_pred= None
current_unet_kwargs = {}
current_uncond_pred = None
current_degraded_pred_compensation = None

def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img
class Script(scripts.Script):

    def __init__(self):
        pass

    def title(self):
        return "Self Attention Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def denoiser_callback(self, parms: CFGDenoiserParams):
        if not sag_enabled:
            return
        global current_xin, current_batch_size

        # logging current uncond size for cond/uncond output separation
        current_batch_size = parms.text_uncond.shape[0]
        # logging current input for eps calculation later
        current_xin = parms.x[-current_batch_size:]

        # logging necessary information for SAG pred
        current_uncond_emb = parms.text_uncond
        current_sigma = parms.sigma
        current_image_cond_in = parms.image_cond

        global current_max_sigma
        if params.sampling_step == 0:
            current_max_sigma = current_sigma[-current_batch_size:][0]
        
        global current_unet_kwargs
        current_unet_kwargs = {
            "sigma": current_sigma[-current_batch_size:],
            "image_cond": current_image_cond_in[-current_batch_size:],
            "text_uncond": current_uncond_emb,
        }

        global saved_original_selfattn_forward
        global current_sag_block_index
        # replace target self attention module in unet with ours
        if sag_attn_target == "dynamic":
            if current_sag_block_index == -1:
                org_attn_module = shared.sd_model.model.diffusion_model.middle_block._modules['1'].transformer_blocks._modules['0'].attn1
                saved_original_selfattn_forward = org_attn_module.forward
                org_attn_module.forward = xattn_forward_log.__get__(org_attn_module,org_attn_module.__class__)
                current_sag_block_index = 0
            elif current_unet_kwargs['sigma'] < current_max_sigma / 6.25: #4:
                if current_sag_block_index == 1:
                    attn_module = shared.sd_model.model.diffusion_model.output_blocks[5]._modules['1'].transformer_blocks._modules['0'].attn1
                    attn_module.forward = saved_original_selfattn_forward
                    org_attn_module = shared.sd_model.model.diffusion_model.output_blocks[8]._modules['1'].transformer_blocks._modules['0'].attn1
                    saved_original_selfattn_forward = org_attn_module.forward
                    org_attn_module.forward = xattn_forward_log.__get__(org_attn_module,org_attn_module.__class__)
                    current_sag_block_index = 2
            elif current_unet_kwargs['sigma'] < current_max_sigma / 2.5: #2:
                if current_sag_block_index == 0:
                    attn_module = shared.sd_model.model.diffusion_model.middle_block._modules['1'].transformer_blocks._modules['0'].attn1
                    attn_module.forward = saved_original_selfattn_forward
                    org_attn_module = shared.sd_model.model.diffusion_model.output_blocks[5]._modules['1'].transformer_blocks._modules['0'].attn1
                    saved_original_selfattn_forward = org_attn_module.forward
                    org_attn_module.forward = xattn_forward_log.__get__(org_attn_module,org_attn_module.__class__)
                    current_sag_block_index = 1

    def denoised_callback(self, params: CFGDenoisedParams):
        if not sag_enabled:
            return
        # output from DiscreteEpsDDPMDenoiser is already pred_x0
        uncond_output = params.x[-current_batch_size:]
        original_latents = uncond_output
        global current_uncond_pred
        current_uncond_pred = uncond_output

        # Produce attention mask
        # We're only interested in the last current_batch_size*head_count slices of logged self-attention map
        attn_map = current_selfattn_map[-current_batch_size*8:]
        bh, hw1, hw2 = attn_map.shape
        b, latent_channel, latent_h, latent_w = original_latents.shape
        h=8
        sdxl=False

        block_scale = 1
        if sag_attn_target == "dynamic":
            block_scale = 2 ** current_sag_block_index
        elif sag_attn_target == "block5":
            block_scale = 2
        elif sag_attn_target == "block8":
            block_scale = 4
        
        middle_layer_latent_size = [math.ceil(latent_h/h) * block_scale, math.ceil(latent_w/h) * block_scale]
        if (middle_layer_latent_size[0] * middle_layer_latent_size[1] < hw1):
            middle_layer_latent_size = [math.ceil(latent_h/(h/2)) * block_scale, math.ceil(latent_w/(h/2)) * block_scale]
            sdxl=True

        attn_map = attn_map.reshape(b, h, hw1, hw2)
        attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > sag_mask_threshold
        attn_mask = (
            attn_mask.reshape(b, middle_layer_latent_size[0], middle_layer_latent_size[1])
            .unsqueeze(1)
            .repeat(1, latent_channel, 1, 1)
            .type(attn_map.dtype)
        )
        filter = "nearest-exact" if not sag_method_bilinear else "bilinear"
        attn_mask = F.interpolate(attn_mask, (latent_h, latent_w), mode=filter)

        # Blur according to the self-attention mask
        if not sdxl:
            degraded_latents = gaussian_blur_2d(original_latents, kernel_size=25, sigma=sag_blur_sigma)
        else:
            degraded_latents = gaussian_blur_2d(original_latents, kernel_size=49, sigma=sag_blur_sigma)
        degraded_latents = degraded_latents * attn_mask + original_latents * (1 - attn_mask)

        renoised_degraded_latent = degraded_latents - (uncond_output - current_xin)

        if shared.sd_model.model.conditioning_key == "crossattn-adm":
            make_condition_dict = lambda c_crossattn, c_adm: {"c_crossattn": c_crossattn, "c_adm": c_adm}
        elif sdxl:
            make_condition_dict = lambda c_crossattn, c_concat: {**c_crossattn, "c_concat": [c_concat]}
        else:
            make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": c_crossattn, "c_concat": [c_concat]}

        if sdxl:
            degraded_pred = params.inner_model(renoised_degraded_latent, current_unet_kwargs['sigma'], cond=make_condition_dict(current_unet_kwargs['text_uncond'], [current_unet_kwargs['image_cond']]))
        else:
            degraded_pred = params.inner_model(renoised_degraded_latent, current_unet_kwargs['sigma'], cond=make_condition_dict([current_unet_kwargs['text_uncond']], [current_unet_kwargs['image_cond']]))

        # renoised_degraded_latent = degraded_latents
        # get predicted x0 in degraded direction
        global current_degraded_pred_compensation
        current_degraded_pred_compensation = uncond_output - degraded_latents
        
        global current_degraded_pred
        current_degraded_pred = degraded_pred

    def cfg_after_cfg_callback(self, params: AfterCFGCallbackParams):
        if not sag_enabled:
            return
        params.x = params.x + (current_uncond_pred - (current_degraded_pred + current_degraded_pred_compensation)) * float(current_sag_guidance_scale)
        params.output_altered = True

    def ui(self, is_img2img):
        with gr.Accordion('Self Attention Guidance', open=False):
            with gr.Row():
                enabled = gr.Checkbox(value=False, label="Enable Self Attention Guidance")
                method = gr.Checkbox(value=False, label="Use bilinear interpolation")
                attn = gr.Dropdown(label="Attention target", choices=["middle", "block5", "block8", "dynamic"], value="middle")	
            with gr.Group():
                scale = gr.Slider(label='Guidance Scale', minimum=-2.0, maximum=10.0, step=0.01, value=0.75)
                mask_threshold = gr.Slider(label='Mask Threshold', minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                blur_sigma = gr.Slider(label='Gaussian Blur Sigma', minimum=0.0, maximum=10.0, step=0.01, value=1.0)
                enabled.change(
                    fn=None,
                    inputs=[enabled],
                    show_progress = False)
        self.infotext_fields = (
            (enabled, lambda d: gr.Checkbox.update(value="SAG Guidance Enabled" in d)),
            (scale, "SAG Guidance Scale"),
            (mask_threshold, "SAG Mask Threshold"),
            (blur_sigma, "SAG Blur Sigma"),
            (method, lambda d: gr.Checkbox.update(value="SAG bilinear interpolation" in d)),
            (attn, "SAG Attention Target"))
        return [enabled, scale, mask_threshold, blur_sigma, method, attn]

    def process(self, p: StableDiffusionProcessing, *args, **kwargs):
        enabled, scale, mask_threshold, blur_sigma, method, attn = args
        global sag_enabled, sag_mask_threshold
        if enabled:
            sag_enabled = True
            sag_mask_threshold = mask_threshold
            sag_blur_sigma = blur_sigma
            sag_method_bilinear = method
            sag_attn_target = attn
            global current_sag_guidance_scale
            current_sag_guidance_scale = scale
            global saved_original_selfattn_forward
            # replace target self attention module in unet with ours
            if attn != "dynamic":
                if attn == "middle":
                    org_attn_module = shared.sd_model.model.diffusion_model.middle_block._modules['1'].transformer_blocks._modules['0'].attn1
                elif attn == "block5":
                    org_attn_module = shared.sd_model.model.diffusion_model.output_blocks[5]._modules['1'].transformer_blocks._modules['0'].attn1
                elif attn == "block8":
                    org_attn_module = shared.sd_model.model.diffusion_model.output_blocks[8]._modules['1'].transformer_blocks._modules['0'].attn1
                saved_original_selfattn_forward = org_attn_module.forward
                org_attn_module.forward = xattn_forward_log.__get__(org_attn_module,org_attn_module.__class__)

            p.extra_generation_params["SAG Guidance Enabled"] = enabled
            p.extra_generation_params["SAG Guidance Scale"] = scale
            p.extra_generation_params["SAG Mask Threshold"] = mask_threshold
            p.extra_generation_params["SAG Blur Sigma"] = blur_sigma
            p.extra_generation_params["SAG bilinear interpolation"] = method
            p.extra_generation_params["SAG Attention Target"] = attn
        else:
            sag_enabled = False

        if not hasattr(self, 'callbacks_added'):
            on_cfg_denoiser(self.denoiser_callback)
            on_cfg_denoised(self.denoised_callback)
            on_cfg_after_cfg(self.cfg_after_cfg_callback)
            self.callbacks_added = True
        return

    def postprocess_batch(self, p, processed, *args):
        enabled, scale, sag_mask_threshold, blur_sigma, method, attn = args
        if enabled:
            # restore original self attention module forward function
            if attn == "middle":
                attn_module = shared.sd_model.model.diffusion_model.middle_block._modules['1'].transformer_blocks._modules['0'].attn1
            elif attn == "block5":
                attn_module = shared.sd_model.model.diffusion_model.output_blocks[5]._modules['1'].transformer_blocks._modules['0'].attn1
            elif attn == "block8":
                attn_module = shared.sd_model.model.diffusion_model.output_blocks[8]._modules['1'].transformer_blocks._modules['0'].attn1

            global current_sag_block_index
            if attn == "dynamic":
                if current_sag_block_index == 0:
                    attn_module = shared.sd_model.model.diffusion_model.middle_block._modules['1'].transformer_blocks._modules['0'].attn1
                elif current_sag_block_index == 1:
                    attn_module = shared.sd_model.model.diffusion_model.output_blocks[5]._modules['1'].transformer_blocks._modules['0'].attn1
                elif current_sag_block_index == 2:
                    attn_module = shared.sd_model.model.diffusion_model.output_blocks[8]._modules['1'].transformer_blocks._modules['0'].attn1
                current_sag_block_index = -1

            attn_module.forward = saved_original_selfattn_forward
        return

xyz_grid_support_sag.initialize(Script)
