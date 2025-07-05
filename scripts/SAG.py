import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import gradio as gr

from inspect import isfunction
from torch import nn, einsum

from modules.processing import StableDiffusionProcessing
import modules.scripts as scripts
from modules import shared
from modules.script_callbacks import on_cfg_denoiser, CFGDenoiserParams, CFGDenoisedParams, on_cfg_denoised, AfterCFGCallbackParams, on_cfg_after_cfg

import os
from scripts import xyz_grid_support_sag

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def adaptive_gaussian_blur_2d(img, sigma, kernel_size=9):
    #if kernel_size is None:
    #    kernel_size = max(5, int(sigma * 4 + 1))
    #    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    #    kernel_size = min(81, kernel_size ** 2)

    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)
    
    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = kernel_size // 2
    img = F.pad(img, (padding, padding, padding, padding), mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img

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
            n_tokens_to_mask = additional_tokens.shape[1]
            x = torch.cat([additional_tokens, x], dim=1)

        if n_times_crossframe_attn_in_self:
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
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

        sim = sim.softmax(dim=-1)

        self.attn_probs = sim

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        if additional_tokens is not None:
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)

def xattn_forward_log(self, x, context=None, mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
    if additional_tokens is not None:
        n_tokens_to_mask = additional_tokens.shape[1]
        x = torch.cat([additional_tokens, x], dim=1)

    if n_times_crossframe_attn_in_self:
        assert x.shape[0] % n_times_crossframe_attn_in_self == 0
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

# Global variable declarations
current_degraded_pred = None # Initialize as None to indicate it hasn't been set yet
current_degraded_pred_compensation = None
current_attn = None
org_attn_module = None
saved_original_selfattn_forward = None

def get_attention_module_for_block(block, layer_name):
    fallback_order = ["0", "1"]  # Fallback layer names within the block

    for layer_name in fallback_order:
        try:
            # First, try to get the layer directly
            if hasattr(block, "_modules") and layer_name in block._modules:
                layer = block._modules[layer_name]
            else:
                # If not found directly, search deeper within the block
                for name, module in block.named_modules():
                    if name.endswith(layer_name) and hasattr(module, "_modules"):
                        layer = module
                        break
                else:
                    raise ValueError(f"Layer {layer_name} not found in block of type {type(block).__name__}")

            # Check if the layer itself is the attention module
            if hasattr(layer, "transformer_blocks"):
                return layer.transformer_blocks._modules['0'].attn1
            elif hasattr(layer, "attn1"):
                return layer.attn1

            # Handle nested Sequential modules
            if isinstance(layer, torch.nn.modules.container.Sequential):
                for sublayer_name, sublayer in layer._modules.items():
                    try:
                        return get_attention_module_for_block(sublayer, "0")  # Recursively check for the attention module
                    except ValueError:
                        pass  # If not found in this sublayer, continue to the next

            # Generic attention module search (based on module names)
            for name, module in layer.named_modules():
                if any(attn_keyword in name.lower() for attn_keyword in ["attn", "attention", "selfattn"]):
                    if hasattr(module, "transformer_blocks"):
                        return module.transformer_blocks._modules['0'].attn1
                    elif hasattr(module, "attn1"):
                        return module.attn1

        except (AttributeError, KeyError, ValueError) as e:
            logger.warning(f"Error accessing attention layer '{layer_name}': {e}. Trying next fallback.")
            continue # Try the next layer in the fallback order

    # If all fallbacks fail, raise an error
    raise ValueError(f"No valid attention layer found within block {type(block).__name__}.")     

class Script(scripts.Script):
    def __init__(self):
        self.custom_resolution = 512

    def title(self):
        return "Self Attention Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

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
                custom_resolution = gr.Slider(label='Base Reference Resolution', minimum=0, maximum=2048, step=8, value=512, info="Default base resolution for models: SD 1.5= 512, SD 2.1= 768, SDXL= 1024")
            enabled.change(fn=None, inputs=[enabled], show_progress=False)
         
        self.infotext_fields = (
            (enabled, lambda d: gr.Checkbox.update(value=d.get("SAG Guidance Enabled").lower() == "true")),
            (scale, "SAG Guidance Scale"),
            (mask_threshold, "SAG Mask Threshold"),
            (blur_sigma, "SAG Blur Sigma"),
            (method, lambda d: gr.Checkbox.update(value=d.get("SAG bilinear interpolation").lower() == "true")),
            (attn, "SAG Attention Target"),
            (custom_resolution, "SAG Custom Resolution"))
        return [enabled, scale, mask_threshold, blur_sigma, method, attn, custom_resolution]
    
    def reset_attention_target(self):
        global sag_attn_target
        sag_attn_target = self.original_attn_target
    
    def process(self, p: StableDiffusionProcessing, *args):
        enabled, scale, mask_threshold, blur_sigma, method, attn, custom_resolution = args
        global sag_enabled, sag_mask_threshold, sag_blur_sigma, sag_method_bilinear, sag_attn_target, current_sag_guidance_scale, current_attn, saved_original_selfattn_forward
         
        if enabled:
            if saved_original_selfattn_forward is not None:  # Check if we successfully restore the attention module
                attn_module = self.get_attention_module(current_attn)  # Get the attention module
                attn_module.forward = saved_original_selfattn_forward  # Restore the original forward method
                current_attn = saved_original_selfattn_forward = None
            
            sag_enabled = True
            sag_mask_threshold = mask_threshold
            sag_blur_sigma = blur_sigma
            sag_method_bilinear = method
            self.original_attn_target = attn  # Save the original attention target
            sag_attn_target = attn
            current_sag_guidance_scale = scale
            self.custom_resolution = custom_resolution
           
            if attn != "dynamic":
                current_attn = attn
                org_attn_module = self.get_attention_module(attn)
                global saved_original_selfattn_forward
                saved_original_selfattn_forward = org_attn_module.forward
                org_attn_module.forward = xattn_forward_log.__get__(org_attn_module, org_attn_module.__class__)

            p.extra_generation_params.update({
                "SAG Guidance Enabled": enabled,
                "SAG Guidance Scale": scale,
                "SAG Mask Threshold": mask_threshold,
                "SAG Blur Sigma": blur_sigma,
                "SAG bilinear interpolation": method,
                "SAG Attention Target": attn,
                "SAG Base Model": base_model,
                "SAG Custom Resolution": custom_resolution
            })
        else:
            sag_enabled = False

        if not hasattr(self, 'callbacks_added'):
            on_cfg_denoiser(self.denoiser_callback)
            on_cfg_denoised(self.denoised_callback)
            on_cfg_after_cfg(self.cfg_after_cfg_callback)
            self.callbacks_added = True

        # Reset attention target for each image
        self.reset_attention_target()
        return

    def denoiser_callback(self, params: CFGDenoiserParams):
        if not sag_enabled:
            return

        global current_xin, current_batch_size, current_max_sigma, current_sag_block_index, current_unet_kwargs, sag_attn_target, current_sigma, current_attn, org_attn_module

        current_batch_size = params.text_uncond.shape[0]
        current_xin = params.x[-current_batch_size:]
        current_uncond_emb = params.text_uncond
        current_sigma = params.sigma
        current_image_cond_in = params.image_cond

        if params.sampling_step == 0:
            current_max_sigma = current_sigma[-current_batch_size:][0]
            current_sag_block_index = -1
            org_attn_module = None

        current_unet_kwargs = {
            "sigma": current_sigma[-current_batch_size:],
            "image_cond": current_image_cond_in[-current_batch_size:],
            "text_uncond": current_uncond_emb,
        }

        #divided by 6.25 (0.15) and 2.5 (0.4) are decided by testing, there might be better scale number
        global saved_original_selfattn_forward
        if sag_attn_target == "dynamic":
            if current_sag_block_index == -1:
                current_attn = "middle"
                org_attn_module = get_attention_module_for_block(shared.sd_model.model.diffusion_model.middle_block, '1')
                saved_original_selfattn_forward = org_attn_module.forward
                org_attn_module.forward = xattn_forward_log.__get__(org_attn_module, org_attn_module.__class__)
                current_sag_block_index = 0
            elif torch.any(current_unet_kwargs['sigma'] < current_max_sigma * 0.15):
                if current_sag_block_index == 1:
                    current_attn = "block8"
                    attn_module = get_attention_module_for_block(shared.sd_model.model.diffusion_model.output_blocks[5], '0')
                    attn_module.forward = saved_original_selfattn_forward
                    # Fallback logic for block8
                    try:
                        if shared.sd_model.is_sd1:
                            org_attn_module = shared.sd_model.model.diffusion_model.output_blocks[8]._modules['1'].transformer_blocks._modules['0'].attn1
                        # Handle potential variations in SDXL architecture
                        if shared.sd_model.is_sdxl:
                            if hasattr(org_attn_module, 'resnets'):  
                                org_attn_module = org_attn_module.resnets[1].spatial_transformer.transformer_blocks._modules['0'].attn1
                    except AttributeError:
                        logger.warning("Attention layer not found in block8. Switching attention target to 'middle' block.")
                        sag_attn_target = "middle"  # Change to middle block
                        org_attn_module = get_attention_module_for_block(shared.sd_model.model.diffusion_model.middle_block, '1')

                    saved_original_selfattn_forward = org_attn_module.forward
                    org_attn_module.forward = xattn_forward_log.__get__(org_attn_module, org_attn_module.__class__)
                    current_sag_block_index = 2

            # Handle the absence of '1' for the output_blocks[5] module 
            elif torch.any(current_unet_kwargs['sigma'] < current_max_sigma * 0.4):
                if current_sag_block_index == 0:
                    current_attn = "block5"
                    attn_module = get_attention_module_for_block(shared.sd_model.model.diffusion_model.middle_block, '1')
                    attn_module.forward = saved_original_selfattn_forward
                    # Fallback logic for block5
                    try:
                        org_attn_module = shared.sd_model.model.diffusion_model.output_blocks[5]._modules['1'].transformer_blocks._modules['0'].attn1
                    except AttributeError:
                        logger.warning("Attention layer not found in block5. Switching attention target to 'middle' block.")
                        sag_attn_target = "middle"  # Change to middle block
                        org_attn_module = get_attention_module_for_block(shared.sd_model.model.diffusion_model.middle_block, '1')

                    saved_original_selfattn_forward = org_attn_module.forward
                    org_attn_module.forward = xattn_forward_log.__get__(org_attn_module, org_attn_module.__class__)
                    current_sag_block_index = 1

              
    def denoised_callback(self, params: CFGDenoisedParams):
        if not sag_enabled:
            return

        uncond_output = params.x[-current_batch_size:]
        original_latents = uncond_output
        global current_uncond_pred
        current_uncond_pred = uncond_output

        attn_map = current_selfattn_map[-current_batch_size*8:]
        bh, hw1, hw2 = attn_map.shape
        b, latent_channel, latent_h, latent_w = original_latents.shape
        h = 8

        # Detect model type (SD 1.x/2.x or SDXL)
        is_sdxl = shared.sd_model.is_sdxl if hasattr(shared.sd_model, 'is_sdxl') else False
        is_cross_attention_control = shared.sd_model.model.conditioning_key == "crossattn-adm" # This is for SDXL

        # Dynamic block scale calculation
        block_scale = {
            "dynamic": 2 ** current_sag_block_index,
            "block5": 2,
            "block8": 4,
            "middle": 1
        }.get(sag_attn_target, 1) # Default to 1 if sag_attn_target is invalid
         
        # Dynamic middle layer size calculation
        middle_layer_latent_size = [
            math.ceil(latent_h / h) * block_scale,
            math.ceil(latent_w / h) * block_scale
        ]
        if middle_layer_latent_size[0] * middle_layer_latent_size[1] < hw1:
            middle_layer_latent_size = [
                math.ceil(latent_h / (h/2)) * block_scale,
                math.ceil(latent_w / (h/2)) * block_scale
            ]

        # Get reference resolution
        reference_resolution = self.custom_resolution

        # Calculate scale factor and adaptive mask threshold
        scale_factor = 1 if reference_resolution == 0 else math.sqrt((latent_h * latent_w) / (reference_resolution / 8) ** 2)
        adaptive_threshold = sag_mask_threshold * scale_factor
       
        # Calculate attention mask and ensure correct dimensions
        attn_map = attn_map.reshape(b, h, hw1, hw2)        
        attn_mask = (attn_map.mean(1).sum(1) > adaptive_threshold)
        attn_mask = (
            attn_mask.reshape(b, middle_layer_latent_size[0], middle_layer_latent_size[1])
            .unsqueeze(1)
            .repeat(1, latent_channel, 1, 1)
            .type(attn_map.dtype)
        )
        attn_mask = F.interpolate(attn_mask, (latent_h, latent_w), mode="nearest-exact" if not sag_method_bilinear else "bilinear", antialias=True)

        # Adaptive blur sigma and Gaussian blur
        adaptive_sigma = sag_blur_sigma * scale_factor
        attn_mask = adaptive_gaussian_blur_2d(attn_mask, sigma=adaptive_sigma)
        degraded_latents = adaptive_gaussian_blur_2d(original_latents, sigma=adaptive_sigma) * attn_mask + original_latents * (1 - attn_mask)

        renoised_degraded_latent = degraded_latents - (uncond_output - current_xin)
       
        # Handle the cond parameter for inner_model differently for SDXL
        if is_sdxl:
            if is_cross_attention_control:
                # Use crossattn_latent and text_embedding if it is using cross attention control
                degraded_pred = params.inner_model(renoised_degraded_latent, current_unet_kwargs['sigma'], crossattn_latent=current_unet_kwargs['image_cond'], text_embedding=current_unet_kwargs['text_uncond'])
            else:
                # If not using cross attention control, then use the normal cond dict format. 
                cond = {**current_unet_kwargs['text_uncond'], "c_concat": [current_unet_kwargs['image_cond']]}
                degraded_pred = params.inner_model(renoised_degraded_latent, current_unet_kwargs['sigma'], cond=cond)
        else:
            # For SD1.5 and SD2.1
            cond = {"c_crossattn": [current_unet_kwargs['text_uncond']], "c_concat": [current_unet_kwargs['image_cond']]}
            degraded_pred = params.inner_model(renoised_degraded_latent, current_unet_kwargs['sigma'], cond=cond)

        global current_degraded_pred_compensation, current_degraded_pred
        current_degraded_pred_compensation = uncond_output - degraded_latents
        current_degraded_pred = degraded_pred

        logger.info(f"Attention map shape: {attn_map.shape}")
        logger.info(f"Original latents shape: {original_latents.shape}")
        logger.info(f"Middle layer latent size: {middle_layer_latent_size}")


    def cfg_after_cfg_callback(self, params: AfterCFGCallbackParams):
        if not sag_enabled:
            return
           
        if current_degraded_pred is not None: # Check if current_degraded_pred is defined
            logger.info(f"params.x shape: {params.x.shape}")
            logger.info(f"current_uncond_pred shape: {current_uncond_pred.shape}")
            logger.info(f"current_degraded_pred shape: {current_degraded_pred.shape}")
            logger.info(f"current_degraded_pred_compensation shape: {current_degraded_pred_compensation.shape}")

        # Ensure tensors have matching sizes
        if params.x.size() != current_uncond_pred.size() or params.x.size() != current_degraded_pred.size() or params.x.size() != current_degraded_pred_compensation.size():
            # Resize tensors to match params.x
            filter_method = "nearest-exact" if not sag_method_bilinear else "bilinear"
            current_uncond_pred_resized = F.interpolate(current_uncond_pred, size=params.x.shape[2:], mode=filter_method)
            current_degraded_pred_resized = F.interpolate(current_degraded_pred, size=params.x.shape[2:], mode=filter_method)
            current_degraded_pred_compensation_resized = F.interpolate(current_degraded_pred_compensation, size=params.x.shape[2:], mode=filter_method)
           
            params.x = params.x + (current_uncond_pred_resized - (current_degraded_pred_resized + current_degraded_pred_compensation_resized)) * float(current_sag_guidance_scale)
        else:
            params.x = params.x + (current_uncond_pred - (current_degraded_pred + current_degraded_pred_compensation)) * float(current_sag_guidance_scale)
         
        params.output_altered = True

    def postprocess_batch(self, p, processed, *args):
        enabled, scale, sag_mask_threshold, blur_sigma, method, attn, custom_resolution = args
        if enabled: # Check if SAG was enabled
            attn_module = self.get_attention_module(attn)  # Get the attention module
            if attn_module is not None:  # Check if we successfully got the attention module
                global current_attn, saved_original_selfattn_forward
                attn_module.forward = saved_original_selfattn_forward  # Restore the original forward method
                current_attn = saved_original_selfattn_forward = None
        return

    def get_attention_module(self, attn):
        try:
            if attn == "middle":
                return shared.sd_model.model.diffusion_model.middle_block._modules['1'].transformer_blocks._modules['0'].attn1
            elif attn == "block5":
                return shared.sd_model.model.diffusion_model.output_blocks[5]._modules['1'].transformer_blocks._modules['0'].attn1
            elif attn == "block8":
                if shared.sd_model.is_sdxl:
                    if hasattr(shared.sd_model.model.diffusion_model.output_blocks[8], 'resnets'):  
                        return shared.sd_model.model.diffusion_model.output_blocks[8].resnets[1].spatial_transformer.transformer_blocks._modules['0'].attn1
                    else:
                        # If spatial_transformer not present, use standard SDXL attention block location
                        return shared.sd_model.model.diffusion_model.output_blocks[8].transformer_blocks._modules['0'].attn1
                else:
                    # Non-SDXL logic
                    return get_attention_module_for_block(shared.sd_model.model.diffusion_model.output_blocks[8], '0')
            elif attn == "dynamic":
                if current_sag_block_index == 0:
                    return shared.sd_model.model.diffusion_model.middle_block._modules['1'].transformer_blocks._modules['0'].attn1
                elif current_sag_block_index == 1:
                    return shared.sd_model.model.diffusion_model.output_blocks[5]._modules['1'].transformer_blocks._modules['0'].attn1
                elif current_sag_block_index == 2:
                    if shared.sd_model.is_sdxl:
                        if hasattr(shared.sd_model.model.diffusion_model.output_blocks[8], 'resnets'):  
                            return shared.sd_model.model.diffusion_model.output_blocks[8].resnets[1].spatial_transformer.transformer_blocks._modules['0'].attn1
                        else:
                            return shared.sd_model.model.diffusion_model.output_blocks[8].transformer_blocks._modules['0'].attn1
                    else:
                        return get_attention_module_for_block(shared.sd_model.model.diffusion_model.output_blocks[8], '0')
        except (KeyError, AttributeError):
            # If the specific block can't be accessed, gracefully fall back to the middle block
            logger.warning(f"Attention target {attn} not found. Falling back to 'middle'.")
            return shared.sd_model.model.diffusion_model.middle_block._modules['1'].transformer_blocks._modules['0'].attn1

# Initialize the script (if needed)
xyz_grid_support_sag.initialize(Script)
