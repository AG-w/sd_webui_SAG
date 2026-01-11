# Implementation of Self Attention Guidance in webui
https://arxiv.org/abs/2210.00939

Demos:
![xyz_grid-0014-232592377.png](resources%2Fimg%2Fxyz_grid-0014-232592377.png)
![xyz_grid-0001-232592377.png](resources%2Fimg%2Fxyz_grid-0001-232592377.png)

For SDXL model, try to tune down everything if you are getting blurry/overfried/distorted results (or lower base resolution), SDXL is more sensitive than SD1.5 

Bilinear interpolation: Attention mask is interpolated using bilinear method, resulting sharper image    
Attention target: Choose the block Attention mask would apply to, `dynamic` means depending on noise sigma value (kinda broken for SDXL, only use dynamic on SD1.5)    
Base resolution: Change attention resolution scaling, set it to 0 to ignore this setting    
Smooth Vectors: inspired by [Smooth Energy Guidance](https://github.com/logtd/ComfyUI-SEGAttention), we smoothed and applied median blur on QKV vectors first to make latents cleaner, set it to 0 to disable it    
Perturbed Vectors: inspired by [Perturbed-Attention Guidance](https://github.com/cvlab-kaist/Perturbed-Attention-Guidance), keep it at low scale (around 0.05~0.2)
