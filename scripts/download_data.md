# RetinaLens: Clinical Retinal Fundus Dataset Acquisition Guide

This guide details the procedure for downloading, organizing, and licensing each of the external diabetic retinopathy benchmark datasets supported by **RetinaLens**.

> **IMPORTANT REGULATORY & PRIVACY NOTE:**
> - Never commit raw datasets, individual patient scans, or model weights exceeding 50 MB to git version control.
> - Ensure all local patient scans comply with HIPAA / GDPR and regional de-identification mandates.
> - RetinaLens will automatically detect and gracefully skip any external dataset not present on disk, defaulting to in-memory/synthetic validation cohorts.

---

## Standardized Clinical Grading Scale (ICDR)

All external datasets are mapped to the 5-point **International Clinical Diabetic Retinopathy (ICDR)** severity scale:

| Grade | Clinical Label | Key Biomarkers / Pathologic Findings | Screening Triage Action | Referable DR? |
| :---: | :--- | :--- | :--- | :---: |
| **0** | **No Apparent DR** | No microaneurysms, hemorrhages, or exudates. | Routine annual rescreening. | **No** |
| **1** | **Mild NPDR** | Microaneurysms only. | 6–12 month follow-up rescreening. | **No** |
| **2** | **Moderate NPDR** | Microaneurysms, blot hemorrhages, hard exudates. | Refer to ophthalmologist within 3–6 months. | **YES** |
| **3** | **Severe NPDR** | >20 hemorrhages in 4 quadrants, venous beading, IRMA. | Prompt referral within 2–4 weeks. | **YES** |
| **4** | **Proliferative DR** | Neovascularization of disc/retina, vitreous hemorrhage. | Urgent specialist referral within 24–48 hours. | **YES** |

*Note: In screening workflows, **Grade >= 2** defines "Referable Diabetic Retinopathy" warranting clinical triage.*

---

## 1. APTOS 2019 Blindness Detection (Primary Training Cohort)

- **Source:** [Kaggle APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
- **Origin:** Aravind Eye Hospital, India.
- **Modality:** Color retinal fundus photography taken under various field conditions.
- **Size:** ~3,662 labeled images.
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

### Download Steps:
```bash
# Using official Kaggle CLI:
kaggle competitions download -c aptos2019-blindness-detection
unzip aptos2019-blindness-detection.zip -d data/aptos/
```

### Expected Directory Layout:
```
data/aptos/
├── train.csv
└── train_images/
    ├── 000c1434d8d7.png
    └── ...
```

---

## 2. EyePACS (Kaggle Diabetic Retinopathy Detection)

- **Source:** [Kaggle EyePACS Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- **Origin:** EyePACS screening network, California, USA.
- **Size:** ~35,126 labeled images taken with varied fundus cameras (Centervue, Topcon, Canon).
- **License:** Open for academic and scientific research purposes.

### Download Steps:
```bash
kaggle competitions download -c diabetic-retinopathy-detection
unzip trainLabels.csv.zip -d data/eyepacs/
# Images can be unpacked into:
mkdir -p data/eyepacs/train
unzip train.zip -d data/eyepacs/train/
```

### Expected Directory Layout:
```
data/eyepacs/
├── trainLabels.csv
└── train/
    ├── 10_left.jpeg
    ├── 10_right.jpeg
    └── ...
```

---

## 3. IDRiD (Indian Diabetic Retinopathy Image Dataset)

- **Source:** [IEEE Dataport IDRiD](https://idrid.grand-challenge.org/)
- **Origin:** Eye Clinic in Nanded, India.
- **Size:** 516 fundus images acquired using Kowa VX-10alpha digital fundus camera with 50° FOV.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

### Expected Directory Layout:
```
data/idrid/
├── IDRiD_disease_grading.csv
└── images/
    ├── IDRiD_001.jpg
    └── ...
```

---

## 4. Messidor-2

- **Source:** [Messidor-2 Grand Challenge / University of Iowa](https://www.adcis.net/en/third-party/messidor2/)
- **Origin:** Four French ophthalmology departments.
- **Size:** 1,748 images (874 patient examinations).
- **Adjudicated DR Labels:** Available via Google Research Messidor-2 adjudication.
- **License:** Academic and scientific research agreement.

### Expected Directory Layout:
```
data/messidor2/
├── messidor_data.csv
└── images/
    ├── 20051019_38557_0100_PP.tif
    └── ...
```

---

## Automated Verification

Once any dataset is placed in the designated `data/` subfolder, verify ingestion via:
```bash
python -c "from ml.data import load_dataset_metadata; print(load_dataset_metadata('aptos'))"
```
If a dataset is not downloaded, RetinaLens prints an informative skipping message and continues uninterrupted.
