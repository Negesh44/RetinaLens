// Real fundus photographs from Wikimedia Commons (see public/samples/real/CREDITS.md).
// `expected` is the condition the source photo documents, which is not necessarily what the AI reports.
export const REAL_SAMPLES = [
  {
    file: 'real_normal.jpg', label: 'Normal Retina', expected: 'normal', eye: 'OS',
    desc: 'Healthy left eye: clear macula, sharp optic disc',
    credit: 'Mikael Häggström · CC0',
  },
  {
    file: 'real_dr_exudates.jpg', label: 'Diabetic Retinopathy', expected: 'diabetic_retinopathy', eye: 'Unknown',
    desc: 'Scattered hard exudates and small hemorrhages',
    credit: 'Hao S. et al. · CC BY 4.0',
  },
  {
    file: 'real_background_dr.jpg', label: 'Background DR', expected: 'diabetic_retinopathy', eye: 'Unknown',
    desc: 'Early vessel damage with mild leakage',
    credit: 'National Eye Institute · Public domain',
  },
  {
    file: 'real_dme.jpg', label: 'Diabetic Macular Edema', expected: 'diabetic_macular_edema', eye: 'Unknown',
    desc: 'Dense yellow exudates around the macula',
    credit: 'National Eye Institute · Public domain',
  },
  {
    file: 'real_pdr.jpg', label: 'Proliferative DR', expected: 'diabetic_retinopathy', eye: 'Unknown',
    desc: 'Advanced DR with abnormal new vessels',
    credit: 'National Eye Institute · Public domain',
  },
  {
    file: 'real_amd.jpg', label: 'Macular Degeneration', expected: 'amd', eye: 'Unknown',
    desc: 'Intermediate AMD with many yellow drusen',
    credit: 'National Eye Institute · Public domain',
  },
  {
    file: 'real_glaucoma_disc.jpg', label: 'Glaucoma (disc close-up)', expected: 'glaucoma', eye: 'Unknown',
    desc: 'Enlarged optic cup with a thin rim',
    credit: 'Snoop (de.wikipedia) · CC BY-SA 3.0',
  },
  {
    file: 'real_retinal_detachment.jpg', label: 'Retinal Detachment', expected: 'retinal_detachment', eye: 'Unknown',
    desc: 'Horseshoe tear, ultra-widefield photo',
    credit: 'Amaris5 · CC BY-SA 4.0',
  },
  {
    file: 'real_macular_hole.jpg', label: 'Macular Hole', expected: 'macular_hole_pucker', eye: 'Unknown',
    desc: 'Round full-thickness hole at the fovea',
    credit: 'Dr Yoan Mboussou · CC BY-SA 4.0',
  },
  {
    file: 'real_rvo.jpg', label: 'Retinal Vein Occlusion', expected: 'vascular_occlusion', eye: 'Unknown',
    desc: 'Central vein occlusion with widespread bleeding',
    credit: 'Werner J.U. et al. · CC BY 4.0',
  },
  {
    file: 'real_rp.jpg', label: 'Retinitis Pigmentosa', expected: 'inherited_dystrophy', eye: 'Unknown',
    desc: 'Bone-spicule pigment in the mid-periphery',
    credit: 'Christian Hamel · CC BY 2.0',
  },
  {
    file: 'real_hypertensive.jpg', label: 'Hypertensive Retinopathy', expected: 'hypertensive_retinopathy', eye: 'Unknown',
    desc: 'Flame hemorrhages from high blood pressure',
    credit: 'Frank Wood · CC BY 3.0',
  },
];

export const CLINICAL_PRESETS = [
  {
    file: 'sample_diabetic_macular_edema.png', label: 'Diabetic Macular Edema', expected: 'diabetic_macular_edema', eye: 'OD',
    desc: 'Circinate ring of hard exudates surrounding foveal center',
    credit: 'Clinical Synthesis Phantom', isPreset: true,
  },
  {
    file: 'sample_amd_drusen.png', label: 'Macular Degeneration (AMD)', expected: 'amd', eye: 'OD',
    desc: 'Soft confluent macular drusen & geographic atrophy',
    credit: 'Clinical Synthesis Phantom', isPreset: true,
  },
  {
    file: 'sample_glaucoma_cupping.png', label: 'Glaucoma (CDR > 0.75)', expected: 'glaucoma', eye: 'OD',
    desc: 'Pathological cup excavation & neuroretinal rim thinning',
    credit: 'Clinical Synthesis Phantom', isPreset: true,
  },
  {
    file: 'sample_retinal_detachment.png', label: 'Retinal Detachment & Tear', expected: 'retinal_detachment', eye: 'OD',
    desc: 'Corrugated elevated detachment fold & horseshoe tear',
    credit: 'Clinical Synthesis Phantom', isPreset: true,
  },
  {
    file: 'sample_macular_hole.png', label: 'Macular Hole & Pucker', expected: 'macular_hole_pucker', eye: 'OD',
    desc: 'Central foveal aperture defect with cellophane pucker',
    credit: 'Clinical Synthesis Phantom', isPreset: true,
  },
  {
    file: 'sample_rvo_occlusion.png', label: 'Vascular Occlusion (RVO)', expected: 'vascular_occlusion', eye: 'OD',
    desc: 'Dense quadrant flame hemorrhages & engorged veins',
    credit: 'Clinical Synthesis Phantom', isPreset: true,
  },
  {
    file: 'sample_csc_serous.png', label: 'Central Serous (CSC)', expected: 'csc', eye: 'OD',
    desc: 'Focal neurosensory serous detachment blister',
    credit: 'Clinical Synthesis Phantom', isPreset: true,
  },
  {
    file: 'sample_rp_bone_spicules.png', label: 'Retinitis Pigmentosa', expected: 'inherited_dystrophy', eye: 'OD',
    desc: 'Mid-peripheral bone-spicule melanin deposits',
    credit: 'Clinical Synthesis Phantom', isPreset: true,
  },
  {
    file: 'sample_melanoma_tumor.png', label: 'Choroidal Melanoma', expected: 'ocular_cancer', eye: 'OD',
    desc: 'Dark elevated melanotic mass with orange lipofuscin',
    credit: 'Clinical Synthesis Phantom', isPreset: true,
  },
  {
    file: 'sample_hypertensive_retinopathy.png', label: 'Hypertensive Retinopathy', expected: 'hypertensive_retinopathy', eye: 'OD',
    desc: 'Arteriovenous nicking, copper wiring, flame hemorrhages',
    credit: 'Clinical Synthesis Phantom', isPreset: true,
  },
  {
    file: 'sample_pathologic_myopia.png', label: 'Pathologic Myopia', expected: 'pathologic_myopia', eye: 'OD',
    desc: 'Peripapillary temporal crescent conus & lacquer cracks',
    credit: 'Clinical Synthesis Phantom', isPreset: true,
  },
  {
    file: 'sample_papilledema.png', label: 'Papilledema (Disc Edema)', expected: 'papilledema', eye: 'OD',
    desc: 'Circumferential blurred disc margins & vessel obscuration',
    credit: 'Clinical Synthesis Phantom', isPreset: true,
  },
  {
    file: 'sample_chorioretinitis.png', label: 'Chorioretinitis / Scar', expected: 'chorioretinitis', eye: 'OD',
    desc: 'Pigmented chorioretinal scar with satellite infiltrate',
    credit: 'Clinical Synthesis Phantom', isPreset: true,
  },
];

// Held-out RFMiD test images (see public/samples/rfmid/CREDITS.md), picked by a fixed rule
// before looking at model output. The model never trained on them.
const RFMID_CREDIT = 'RFMiD test set (Pachade et al. 2021) · CC BY 4.0';
export const RFMID_SAMPLES = [
  ['normal', 'Normal Retina', 'Expert label: no disease'],
  ['diabetic_retinopathy', 'Diabetic Retinopathy', 'Expert label: DR'],
  ['amd', 'Macular Degeneration', 'Expert label: age-related macular degeneration'],
  ['glaucoma', 'Glaucoma', 'Expert label: optic disc cupping'],
  ['pathologic_myopia', 'Pathologic Myopia', 'Expert label: myopia'],
  ['vascular_occlusion', 'Retinal Vein Occlusion', 'Expert label: retinal vascular occlusion'],
  ['csc', 'Central Serous (CSC)', 'Expert label: central serous retinopathy'],
  ['papilledema', 'Optic Disc Edema', 'Expert label: optic disc edema'],
  ['chorioretinitis', 'Chorioretinitis', 'Expert label: chorioretinitis / retinitis'],
  ['macular_hole_pucker', 'Macular Hole / ERM', 'Expert label: macular hole or epiretinal membrane'],
].map(([expected, label, desc]) => ({
  file: `rfmid_${expected}.jpg`, label, expected, eye: 'Unknown', desc, credit: RFMID_CREDIT,
}));

// Deliberately degraded copies of the normal photo, to demonstrate the image-quality gate.
export const QUALITY_SAMPLES = [
  {
    file: 'real_sim_defocus.jpg', label: 'Out of focus', expected: null, eye: 'OS',
    desc: 'Simulated defocus: should be rejected by quality gate',
    credit: 'Simulated from Mikael Häggström · CC0',
  },
  {
    file: 'real_sim_underexposed.jpg', label: 'Too dark', expected: null, eye: 'OS',
    desc: 'Simulated underexposure: should be rejected by quality gate',
    credit: 'Simulated from Mikael Häggström · CC0',
  },
];

const isSyntheticPreset = (sample) => {
  const file = typeof sample === 'string' ? sample : sample.file;
  return typeof sample === 'object' ? sample.isPreset : file.startsWith('sample_');
};

export const sampleUrl = (sample) => {
  const file = typeof sample === 'string' ? sample : sample.file;
  if (file.startsWith('rfmid_')) return `/samples/rfmid/${file}`;
  return isSyntheticPreset(sample) ? `/samples/${file}` : `/samples/real/${file}`;
};

export const sampleThumbUrl = (sample) => {
  const file = typeof sample === 'string' ? sample : sample.file;
  return isSyntheticPreset(sample) ? `/samples/${file}` : `/samples/real/${file.replace(/\.jpg$/, '_thumb.jpg')}`;
};
