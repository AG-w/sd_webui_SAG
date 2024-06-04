# Implementation of Self Attention Guidance in webui
https://arxiv.org/abs/2210.00939

Demos:
![xyz_grid-0014-232592377.png](resources%2Fimg%2Fxyz_grid-0014-232592377.png)
![xyz_grid-0001-232592377.png](resources%2Fimg%2Fxyz_grid-0001-232592377.png)

For SDXL model, adjust Blur Sigma around 0.34 to prevent blurry image    
Bilinear interpolation: Attention mask is interpolated using bilinear method, resulting sharper image    
Attention target: Choose the block Attention mask would apply to, `dynamic` means depending on noise sigma value
