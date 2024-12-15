# Implementation of Self Attention Guidance in webui
https://arxiv.org/abs/2210.00939

Demos:
![xyz_grid-0014-232592377.png](resources%2Fimg%2Fxyz_grid-0014-232592377.png)
![xyz_grid-0001-232592377.png](resources%2Fimg%2Fxyz_grid-0001-232592377.png)

For SDXL model, try to tune down everything if you are getting blurry results, SDXL is more sensitive than SD1.5    
Bilinear interpolation: Attention mask is interpolated using bilinear method, resulting sharper image    
Attention target: Choose the block Attention mask would apply to, `dynamic` means depending on noise sigma value    
Base resolution: Change attention scaling according to resolution
