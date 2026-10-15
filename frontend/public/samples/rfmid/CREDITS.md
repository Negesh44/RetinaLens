# RFMiD demo images

These are images from the official **test split** of the Retinal Fundus Multi-Disease Image Dataset (RFMiD).
The RetinaLens disease model never saw them during training or threshold tuning.

Selection rule, fixed before looking at model output: for each condition, the first test image
(by dataset order) whose only RetinaLens-mapped label is that condition; for "normal", the first
test image with Disease_Risk = 0.

Source: Pachade S. et al., "Retinal Fundus Multi-Disease Image Dataset (RFMiD): A Dataset for
Multi-Disease Detection Research", Data 2021, 6(2):14. https://doi.org/10.3390/data6020014
License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Modifications: cropped to the fundus field, padded to square, resized to 512 px, saved as JPEG.

| File | RFMiD test image (0-based order) | Expert label |
|---|---|---|
| rfmid_normal.jpg | 244 | normal |
| rfmid_diabetic_retinopathy.jpg | 0 | DR |
| rfmid_amd.jpg | 11 | ARMD |
| rfmid_glaucoma.jpg | 17 | ODC (optic disc cupping) |
| rfmid_pathologic_myopia.jpg | 78 | MYA |
| rfmid_vascular_occlusion.jpg | 90 | BRVO / CRVO / CRAO / BRAO |
| rfmid_csc.jpg | 46 | CSR |
| rfmid_papilledema.jpg | 60 | ODE (optic disc edema) |
| rfmid_chorioretinitis.jpg | 22 | CRS / RS |
| rfmid_macular_hole_pucker.jpg | 7 | MHL / ERM |
