"""Clinical Retinal Pathology Catalog and Biomarker Knowledge Base.

Defines all ophthalmic disease categories, visual hallmark findings, ICD/SNOMED terminology,
clinical triage urgency, and multi-disease diagnostic criteria:
1. Diabetic Retinopathy & Diabetic Macular Edema (DR/DME)
2. Age-Related Macular Degeneration (AMD)
3. Glaucoma (Optic Neuropathy)
4. Retinal Detachment & Retinal Tears (RD/RT)
5. Macular Hole & Epiretinal Membrane / Pucker (MH/ERM)
6. Retinal Vein & Artery Occlusions (RVO/CRAO)
7. Central Serous Chorioretinopathy (CSC)
8. Inherited Retinal Dystrophies (Retinitis Pigmentosa & Best Disease)
9. Ocular Cancers (Choroidal Melanoma & Retinoblastoma)
10. Hypertensive Retinopathy (HR)
11. Pathologic Myopia (PM)
12. Normal Healthy Retina (Baseline)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class DiseaseDefinition:
    id: str
    name: str
    short_name: str
    acronym: str
    category: str
    summary: str
    detailed_description: str
    hallmark_findings: List[str]
    biomarker_classes: List[str]
    triage_urgency: str  # "Routine", "Prompt (2-4 wks)", "Urgent (24-48h)", "Emergency (Same Day)"
    clinical_action: str
    color: str
    bg_color: str
    border_color: str
    badge_color: str


DISEASE_CATALOG: Dict[str, DiseaseDefinition] = {
    "diabetic_retinopathy": DiseaseDefinition(
        id="diabetic_retinopathy",
        name="Diabetic Retinopathy & Diabetic Macular Edema",
        short_name="Diabetic Retinopathy",
        acronym="DR / DME",
        category="Retinal Vascular Disease",
        summary="Shows leaking, swollen, or damaged blood vessels caused by high blood sugar.",
        detailed_description=(
            "Microvascular complication of chronic diabetes mellitus characterized by capillary basement "
            "membrane thickening, pericyte loss, microaneurysm formation, vascular leakage, and progressive ischemia."
        ),
        hallmark_findings=[
            "Microaneurysms and intraretinal dot-and-blot hemorrhages",
            "Hard lipid exudates in outer plexiform layer threatening the fovea",
            "Cotton-wool spots representing precapillary arteriolar occlusions",
            "Venous beading, IRMA, and neovascularization of the disc/retina (NVD/NVE)",
        ],
        biomarker_classes=["microaneurysm", "hemorrhage", "hard_exudate", "cotton_wool_spot"],
        triage_urgency="Prompt (2-4 wks)",
        clinical_action="Dilated fundus examination, optical coherence tomography (OCT), and glycemic optimization.",
        color="#0284c7",
        bg_color="bg-sky-950/40",
        border_color="border-sky-500/40",
        badge_color="bg-sky-600",
    ),
    "amd": DiseaseDefinition(
        id="amd",
        name="Age-Related Macular Degeneration",
        short_name="Macular Degeneration",
        acronym="AMD",
        category="Macular Disease",
        summary="Reveals yellow deposits called drusen, fluid leaks, or tissue thinning in the macula.",
        detailed_description=(
            "Progressive neurodegenerative disorder of the choriocapillaris, Bruch's membrane, and retinal pigment "
            "epithelium (RPE) in patients over 50 years, leading to central scotoma and loss of fine vision."
        ),
        hallmark_findings=[
            "Soft confluent and cuticular hard drusen in the central macula",
            "Geographic atrophy with sharply demarcated RPE hypopigmentation and thinning",
            "Subretinal fluid, intraretinal cystoid spaces, or subretinal hemorrhage from CNVM",
            "Macular pigmentary mottling and focal hyperpigmentation",
        ],
        biomarker_classes=["drusen", "geographic_atrophy", "subretinal_fluid", "hard_exudate"],
        triage_urgency="Prompt (2-4 wks)",
        clinical_action="Macular OCT, Amsler grid self-monitoring, AREDS2 supplementation, and anti-VEGF referral if wet AMD.",
        color="#f59e0b",
        bg_color="bg-amber-950/40",
        border_color="border-amber-500/40",
        badge_color="bg-amber-600",
    ),
    "glaucoma": DiseaseDefinition(
        id="glaucoma",
        name="Glaucoma & Optic Neuropathy",
        short_name="Glaucoma",
        acronym="POAG / Glaucoma",
        category="Optic Nerve",
        summary="Displays damage, thinning, or pressure changes in the optic nerve head.",
        detailed_description=(
            "Progressive optic neuropathy characterized by loss of retinal ganglion cells and axonal atrophy, "
            "producing structural optic nerve head cupping and corresponding peripheral visual field defects."
        ),
        hallmark_findings=[
            "Pathological optic disc cupping (vertical Cup-to-Disc Ratio CDR > 0.65)",
            "Neuroretinal rim thinning and notch defects violating the ISNT rule",
            "Peripapillary chorioretinal atrophy (beta-zone PPA)",
            "Drance flame hemorrhages crossing the optic disc margin",
        ],
        biomarker_classes=["optic_cup_enlargement", "neuroretinal_rim_thinning", "peripapillary_atrophy"],
        triage_urgency="Prompt (2-4 wks)",
        clinical_action="Intraocular pressure (IOP) tonometry, pachymetry, OCT retinal nerve fiber layer (RNFL), and visual fields.",
        color="#06b6d4",
        bg_color="bg-cyan-950/40",
        border_color="border-cyan-500/40",
        badge_color="bg-cyan-600",
    ),
    "retinal_detachment": DiseaseDefinition(
        id="retinal_detachment",
        name="Retinal Detachment & Retinal Tears",
        short_name="Retinal Detachment",
        acronym="RD / Tear",
        category="Vitreoretinal Emergency",
        summary="Identifies areas where the retina has pulled away, torn, or developed a hole.",
        detailed_description=(
            "Separation of the neurosensory retina from the underlying retinal pigment epithelium, typically "
            "initiated by vitreous traction causing a full-thickness horseshoe tear or round operculated retinal break."
        ),
        hallmark_findings=[
            "Corrugated, translucent, elevated folds of detached neurosensory retina",
            "Full-thickness horseshoe flap tear or round break with subretinal fluid demarcation",
            "Loss of choroidal architectural pattern under detached retina",
            "Pigment granules in the vitreous cavity (Shafer's sign / tobacco dust)",
        ],
        biomarker_classes=["retinal_detachment_fold", "retinal_tear"],
        triage_urgency="Emergency (Same Day)",
        clinical_action="Emergency vitreoretinal consultation within 12-24 hours for laser retinopexy, cryotherapy, or vitrectomy.",
        color="#ec4899",
        bg_color="bg-pink-950/40",
        border_color="border-pink-500/40",
        badge_color="bg-pink-600",
    ),
    "macular_hole_pucker": DiseaseDefinition(
        id="macular_hole_pucker",
        name="Macular Hole & Epiretinal Membrane (Pucker)",
        short_name="Macular Hole / Pucker",
        acronym="MH / ERM",
        category="Vitreomacular Interface",
        summary="Pinpoints small openings or scar tissue on the central part of the retina.",
        detailed_description=(
            "Disorders of the vitreomacular interface where tangential vitreofoveal traction causes a full-thickness "
            "foveal defect (macular hole) or avascular fibrocellular proliferation produces a glistening wrinkling membrane (pucker)."
        ),
        hallmark_findings=[
            "Sharply circumscribed circular red aperture at the anatomical fovea",
            "Surrounding cuff of subretinal fluid and yellow drusenoid nodular deposits",
            "Glistening cellophane maculopathy reflex with distorted, tortuous retinal vessels",
            "Retinal striae radiating outward from a contracted fibrocellular epicentre",
        ],
        biomarker_classes=["macular_hole", "epiretinal_membrane"],
        triage_urgency="Prompt (2-4 wks)",
        clinical_action="High-resolution macular OCT to stage hole/pucker; surgical evaluation for pars plana vitrectomy with membrane peel.",
        color="#d946ef",
        bg_color="bg-fuchsia-950/40",
        border_color="border-fuchsia-500/40",
        badge_color="bg-fuchsia-600",
    ),
    "vascular_occlusion": DiseaseDefinition(
        id="vascular_occlusion",
        name="Retinal Vein & Artery Occlusions",
        short_name="Vascular Occlusion",
        acronym="RVO / CRAO",
        category="Acute Vascular",
        summary="Detects blocked blood vessels, hemorrhages, or areas lacking blood flow (ischemia).",
        detailed_description=(
            "Thrombotic or embolic occlusion of the retinal venous or arterial vasculature. Central/Branch Retinal "
            "Vein Occlusion (CRVO/BRVO) causes severe venous stasis and hemorrhage; Central Retinal Artery Occlusion (CRAO) causes acute infarction."
        ),
        hallmark_findings=[
            "Widespread flame and blot intraretinal hemorrhages along a blocked vascular arcade ('blood and thunder')",
            "Severe dilation and tortuosity of affected retinal veins",
            "Profound opaque retinal whitening / ischemic edema with classic foveal 'cherry-red spot' in CRAO",
            "Extensive cotton-wool spots indicating dense capillary non-perfusion",
        ],
        biomarker_classes=["vascular_occlusion_ischemia", "hemorrhage", "cotton_wool_spot"],
        triage_urgency="Urgent (24-48h)",
        clinical_action="CRAO is an acute ocular stroke requiring immediate emergency intervention; RVO requires macular OCT and anti-VEGF/steroid evaluation.",
        color="#6366f1",
        bg_color="bg-indigo-950/40",
        border_color="border-indigo-500/40",
        badge_color="bg-indigo-600",
    ),
    "csc": DiseaseDefinition(
        id="csc",
        name="Central Serous Chorioretinopathy",
        short_name="Central Serous (CSC)",
        acronym="CSC",
        category="Pachychoroid Spectrum",
        summary="Shows fluid accumulation or small blisters beneath the retina.",
        detailed_description=(
            "Idiopathic condition characterized by serous detachment of the neurosensory retina over the posterior "
            "pole secondary to focal retinal pigment epithelium leak and choroidal hyperpermeability."
        ),
        hallmark_findings=[
            "Well-circumscribed, round/oval dome-shaped neurosensory detachment blister",
            "Loss of normal foveal light reflex with a halo ring of subretinal fluid",
            "Focal yellow-white subretinal fibrin precipitate deposits",
            "Subtle mottled RPE depigmentation at the site of primary focal leakage",
        ],
        biomarker_classes=["serous_detachment_blister"],
        triage_urgency="Prompt (2-4 wks)",
        clinical_action="Observation in acute cases (frequently self-resolving in 2-3 months); discontinue exogenous corticosteroids; OCT & FFA imaging.",
        color="#0ea5e9",
        bg_color="bg-sky-950/40",
        border_color="border-sky-400/40",
        badge_color="bg-sky-500",
    ),
    "inherited_dystrophy": DiseaseDefinition(
        id="inherited_dystrophy",
        name="Inherited Retinal Dystrophies",
        short_name="Retinal Dystrophy",
        acronym="RP / Best",
        category="Genetic Dystrophy",
        summary="Spots genetic and degenerative changes such as retinitis pigmentosa or Best disease.",
        detailed_description=(
            "Heterogeneous group of genetically determined progressive disorders affecting photoreceptors and the "
            "RPE. Classical Retinitis Pigmentosa involves rod-cone degeneration; Best vitelliform disease causes macular lipofuscin accumulation."
        ),
        hallmark_findings=[
            "Mid-peripheral 'bone-spicule' melanin pigment deposits along retinal vessels",
            "Marked retinal arteriolar attenuation (thread-like narrowing)",
            "Waxy optic disc pallor with circular halo reflex",
            "Round, elevated, bright yellow 'egg-yolk' vitelliform subretinal cyst in Best disease",
        ],
        biomarker_classes=["bone_spicule_pigment"],
        triage_urgency="Prompt (2-4 wks)",
        clinical_action="Electroretinography (ERG), genetic testing/counseling, visual field perimetry, and low-vision aids.",
        color="#a855f7",
        bg_color="bg-purple-950/40",
        border_color="border-purple-500/40",
        badge_color="bg-purple-600",
    ),
    "ocular_cancer": DiseaseDefinition(
        id="ocular_cancer",
        name="Ocular & Intraocular Tumors",
        short_name="Ocular Oncology",
        acronym="Melanoma / RB",
        category="Ocular Oncology",
        summary="Identifies pigmented lesions or tumors like choroidal melanoma or retinoblastoma.",
        detailed_description=(
            "Primary intraocular malignancies. Choroidal melanoma is the most common primary intraocular tumor "
            "in adults; Retinoblastoma is the most common primary malignant intraocular neoplasm of childhood."
        ),
        hallmark_findings=[
            "Elevated, dome-shaped pigmented brown, slate-gray, or amelanotic choroidal mass",
            "Clumps of bright orange lipofuscin pigment overlying the tumor surface",
            "Associated exudative neurosensory retinal detachment over tumor margins",
            "Chalky white/pinkish vascularized mass with calcification and vitreous seeds in pediatric fundus",
        ],
        biomarker_classes=["ocular_tumor_mass"],
        triage_urgency="Emergency (Same Day)",
        clinical_action="Immediate ocular oncology referral for B-scan ultrasound, UBM, fine-needle biopsy, plaque brachytherapy, or enucleation.",
        color="#b91c1c",
        bg_color="bg-rose-950/40",
        border_color="border-rose-500/40",
        badge_color="bg-rose-700",
    ),
    "hypertensive_retinopathy": DiseaseDefinition(
        id="hypertensive_retinopathy",
        name="Hypertensive Retinopathy",
        short_name="Hypertensive Retinopathy",
        acronym="HR",
        category="Systemic Microvascular",
        summary="Demonstrates vascular constriction, arteriovenous nicking, and flame hemorrhages from high BP.",
        detailed_description=(
            "Retinal microvascular response to systemic arterial hypertension with vasoconstriction, sclerotic "
            "thickening, vessel wall opacification, and blood-retina barrier disruption."
        ),
        hallmark_findings=[
            "Arteriovenous (AV) crossing changes and venular deflection (Gunn/Salus signs)",
            "Diffuse or focal arteriolar narrowing with copper or silver wiring reflex",
            "Flame-shaped superficial nerve fiber layer hemorrhages",
            "Macular star lipid exudate configuration and disc edema (malignant hypertension)",
        ],
        biomarker_classes=["arteriovenous_nicking", "hemorrhage", "cotton_wool_spot"],
        triage_urgency="Prompt (2-4 wks)",
        clinical_action="Systemic blood pressure control, cardiovascular risk stratification, and renal function panel.",
        color="#ea580c",
        bg_color="bg-orange-950/40",
        border_color="border-orange-500/40",
        badge_color="bg-orange-600",
    ),
    "pathologic_myopia": DiseaseDefinition(
        id="pathologic_myopia",
        name="Pathologic & High Degenerative Myopia",
        short_name="Pathologic Myopia",
        acronym="PM",
        category="Degenerative Axial Myopia",
        summary="Reveals posterior staphyloma, lacquer cracks, myopic chorioretinal atrophy, or tilted disc with conus.",
        detailed_description=(
            "Excessive axial elongation of the globe causing mechanical stretching, scleral thinning, "
            "lacquer cracks in Bruch's membrane, patchy chorioretinal atrophy, and tilted disc with temporal crescent."
        ),
        hallmark_findings=[
            "Peripapillary temporal crescent / conus around tilted optic disc",
            "Lacquer cracks (fine irregular linear breaks in Bruch's membrane)",
            "Patchy yellowish-white chorioretinal atrophy exposing underlying sclera and choroidal vessels",
            "Fuchs spot (pigmented elevated scar following subretinal neovascularization)",
        ],
        biomarker_classes=["myopic_chorioretinal_atrophy", "peripapillary_atrophy"],
        triage_urgency="Prompt (2-4 wks)",
        clinical_action="Optical coherence tomography (OCT) for myopic traction maculopathy, axial length biometry, and anti-VEGF referral if myopic CNV develops.",
        color="#0d9488",
        bg_color="bg-teal-950/40",
        border_color="border-teal-500/40",
        badge_color="bg-teal-600",
    ),
    "diabetic_macular_edema": DiseaseDefinition(
        id="diabetic_macular_edema",
        name="Diabetic Macular Edema (Center-Involving / CSME)",
        short_name="Macular Edema (DME)",
        acronym="DME / CSME",
        category="Macular Microvascular",
        summary="Identifies center-involving foveal thickening, retinal cystoid spaces, and circinate hard exudate rings.",
        detailed_description=(
            "Fluid accumulation in the macula secondary to breakdown of the blood-retinal barrier in diabetic "
            "retinopathy, representing the leading cause of moderate visual impairment in diabetic patients."
        ),
        hallmark_findings=[
            "Circinate rings of hard lipid exudates encircling leaking microaneurysm clusters",
            "Retinal thickening within 500 microns of the foveal center (Clinically Significant Macular Edema)",
            "Cystoid macular edema with loss of the foveal depression reflex",
            "Focal or diffuse fluorescein dye leakage on angiography",
        ],
        biomarker_classes=["circinate_exudate_ring", "hard_exudate", "microaneurysm"],
        triage_urgency="Prompt (2-4 wks)",
        clinical_action="Macular spectral-domain OCT scan to measure central subfield thickness (CST) and intravitreal anti-VEGF or focal grid laser therapy.",
        color="#f59e0b",
        bg_color="bg-amber-950/40",
        border_color="border-amber-400/40",
        badge_color="bg-amber-600",
    ),
    "papilledema": DiseaseDefinition(
        id="papilledema",
        name="Papilledema & Optic Disc Edema",
        short_name="Papilledema",
        acronym="Disc Edema",
        category="Neuro-Ophthalmic",
        summary="Displays blurred disc margins, hyperaemia, vascular engorgement, and peripapillary hemorrhages from intracranial pressure.",
        detailed_description=(
            "Bilateral swelling of the optic nerve head secondary to elevated intracranial pressure (ICP), "
            "requiring urgent neuroimaging to rule out space-occupying intracranial mass, venous sinus thrombosis, or idiopathic intracranial hypertension."
        ),
        hallmark_findings=[
            "Circumferential blurring and elevation of optic disc margins (Frisén grade 1-5)",
            "Obscuration of major retinal vessels as they cross the optic disc boundary",
            "Peripapillary flame-shaped hemorrhages and cotton-wool spots on the disc border",
            "Paton's lines (concentric retinal folds circumferential to the swollen disc)",
        ],
        biomarker_classes=["disc_edema_margin_blur", "hemorrhage"],
        triage_urgency="Emergency (Same Day)",
        clinical_action="Urgent emergency neurology referral, urgent brain MRI/MRV, blood pressure measurement, and lumbar puncture with opening pressure.",
        color="#dc2626",
        bg_color="bg-red-950/40",
        border_color="border-red-500/40",
        badge_color="bg-red-600",
    ),
    "chorioretinitis": DiseaseDefinition(
        id="chorioretinitis",
        name="Infectious & Inflammatory Chorioretinitis",
        short_name="Chorioretinitis",
        acronym="Uveitis / Scar",
        category="Chorioretinal Inflammation",
        summary="Presents focal necrotizing retinal whitening, pigmented chorioretinal scarring ('headlight in fog'), and vitreous hazing.",
        detailed_description=(
            "Inflammation of the choroid and retina resulting from infectious pathogens (Toxoplasma gondii, "
            "Cytomegalovirus, HSV/VZV, Syphilis, Tuberculosis) or autoimmune uveitic syndromes, leading to permanent chorioretinal scarring."
        ),
        hallmark_findings=[
            "Sharply circumscribed hyperpigmented chorioretinal atrophic scar with dark borders",
            "Active satellite fluffy whitish inflammatory infiltrate adjacent to an old scar",
            "Vitreous cells and inflammatory haze ('headlight in the fog' sign)",
            "Retinal vasculitis with perivascular cuffing / sheathing",
        ],
        biomarker_classes=["chorioretinal_scar", "cotton_wool_spot"],
        triage_urgency="Urgent (24-48h)",
        clinical_action="Uveitis specialist referral, serologic workup (Toxoplasma IgG/IgM, QuantiFERON, RPR/VDRL), and targeted antimicrobial/corticosteroid therapy.",
        color="#8b5cf6",
        bg_color="bg-purple-950/40",
        border_color="border-purple-400/40",
        badge_color="bg-purple-600",
    ),
    "normal": DiseaseDefinition(
        id="normal",
        name="Physiological Fundus (No Apparent Abnormality)",
        short_name="Normal Retina",
        acronym="Normal",
        category="Physiological Baseline",
        summary="Healthy retina displaying sharp optic disc borders, standard foveal reflex, and intact vessels.",
        detailed_description=(
            "Normal funduscopic examination without evidence of microaneurysms, hemorrhages, exudation, "
            "drusen, cupping, detachment, or pigmentary alteration."
        ),
        hallmark_findings=[
            "Crisp optic nerve head margin with physiological cup-to-disc ratio (CDR < 0.45)",
            "Uniform background retinal pigmentation with intact macular foveal reflex",
            "Normal arteriolar-to-venular caliber ratio (~2:3) without crossing defects",
            "Uncompromised peripheral retina free of tears, holes, or detachment folds",
        ],
        biomarker_classes=[],
        triage_urgency="Routine",
        clinical_action="Routine annual rescreening as part of proactive wellness monitoring.",
        color="#10b981",
        bg_color="bg-emerald-950/40",
        border_color="border-emerald-500/40",
        badge_color="bg-emerald-600",
    ),
}


# Complete visual definitions for all segmented retinal pathology biomarkers
PATHOLOGY_BIOMARKER_METADATA = {
    # 1. Diabetic Retinopathy & Vascular
    "microaneurysm": {
        "name": "Microaneurysm",
        "disease_id": "diabetic_retinopathy",
        "hex": "#3b82f6",  # Blue
        "bgr": (246, 130, 59),
        "rgba": "rgba(59, 130, 246, 0.60)",
        "stroke": "#2563eb",
        "description": "Focal punctate dilation of retinal capillaries; earliest hallmark sign of diabetic retinopathy.",
        "advice": "Routine dilated retinal examination in 6–12 months and strict glycemic/BP management.",
    },
    "hemorrhage": {
        "name": "Retinal Hemorrhage",
        "disease_id": "diabetic_retinopathy",
        "hex": "#ef4444",  # Red
        "bgr": (68, 68, 239),
        "rgba": "rgba(239, 68, 68, 0.45)",
        "stroke": "#dc2626",
        "description": "Intraretinal pooling of extravasated blood from ruptured or decompensated microvessels.",
        "advice": "Comprehensive ophthalmic evaluation within 3–6 months to assess macular integrity and ischemia.",
    },
    "hard_exudate": {
        "name": "Hard Exudate",
        "disease_id": "diabetic_retinopathy",
        "hex": "#eab308",  # Amber Yellow
        "bgr": (8, 179, 234),
        "rgba": "rgba(234, 179, 8, 0.50)",
        "stroke": "#ca8a04",
        "description": "Waxy lipoprotein deposits secondary to chronic breakdown of the blood-retinal barrier.",
        "advice": "Macular OCT imaging is advised to evaluate for diabetic macular edema threatening central vision.",
    },
    "cotton_wool_spot": {
        "name": "Cotton-Wool Spot",
        "disease_id": "diabetic_retinopathy",
        "hex": "#a855f7",  # Purple
        "bgr": (247, 85, 168),
        "rgba": "rgba(168, 85, 247, 0.50)",
        "stroke": "#9333ea",
        "description": "Superficial nerve fiber layer infarct characterized by focal axoplasmic flow stasis.",
        "advice": "Ophthalmic review in 2–4 weeks; indicates active precapillary arteriolar ischemia.",
    },

    # 2. AMD Biomarkers
    "drusen": {
        "name": "Drusen Deposits",
        "disease_id": "amd",
        "hex": "#facc15",  # Gold Yellow
        "bgr": (21, 204, 250),
        "rgba": "rgba(250, 204, 21, 0.55)",
        "stroke": "#eab308",
        "description": "Extracellular yellow accumulations of lipid, protein, and cellular debris between RPE and Bruch's membrane.",
        "advice": "Macular OCT evaluation, daily Amsler grid monitoring, and AREDS2 antioxidant formula supplementation.",
    },
    "geographic_atrophy": {
        "name": "Geographic Atrophy",
        "disease_id": "amd",
        "hex": "#fb923c",  # Orange Coral
        "bgr": (60, 146, 251),
        "rgba": "rgba(251, 146, 60, 0.50)",
        "stroke": "#f97316",
        "description": "Circumscribed areas of retinal pigment epithelium thinning and choriocapillaris loss in the macula.",
        "advice": "Low vision rehabilitation, complement-inhibitor evaluation, and visual field surveillance.",
    },

    # 3. Glaucoma Biomarkers
    "optic_cup_enlargement": {
        "name": "Optic Cup Excavation (CDR > 0.65)",
        "disease_id": "glaucoma",
        "hex": "#06b6d4",  # Cyan
        "bgr": (212, 182, 6),
        "rgba": "rgba(6, 182, 212, 0.50)",
        "stroke": "#0891b2",
        "description": "Pathological excavation and enlargement of the central optic disc cup with high cup-to-disc ratio.",
        "advice": "Tonometry for intraocular pressure (IOP), visual field perimetry, and RNFL OCT imaging.",
    },
    "neuroretinal_rim_thinning": {
        "name": "Neuroretinal Rim Thinning",
        "disease_id": "glaucoma",
        "hex": "#14b8a6",  # Teal
        "bgr": (166, 184, 20),
        "rgba": "rgba(20, 184, 166, 0.50)",
        "stroke": "#0d9488",
        "description": "Localized thinning or notching of the neuroretinal rim tissue violating the physiological ISNT rule.",
        "advice": "Glaucoma specialist evaluation for topical pressure-lowering drops or laser trabeculoplasty.",
    },

    # 4. Retinal Detachment & Tears
    "retinal_detachment_fold": {
        "name": "Retinal Detachment Fold",
        "disease_id": "retinal_detachment",
        "hex": "#ec4899",  # Pink
        "bgr": (153, 72, 236),
        "rgba": "rgba(236, 72, 153, 0.55)",
        "stroke": "#db2777",
        "description": "Pale, elevated corrugated fold of neurosensory retina separated from the underlying pigment epithelium.",
        "advice": "EMERGENCY: Immediate vitreoretinal surgery consultation to prevent permanent macular detachment.",
    },
    "retinal_tear": {
        "name": "Retinal Tear / Break",
        "disease_id": "retinal_detachment",
        "hex": "#f43f5e",  # Crimson Rose
        "bgr": (94, 63, 244),
        "rgba": "rgba(244, 63, 94, 0.65)",
        "stroke": "#e11d48",
        "description": "Full-thickness flap or operculated break in the retina through which vitreous fluid can enter the subretinal space.",
        "advice": "EMERGENCY: Prompt laser retinopexy or cryotherapy to seal the tear before complete retinal detachment.",
    },

    # 5. Macular Hole & Pucker
    "macular_hole": {
        "name": "Macular Hole Defect",
        "disease_id": "macular_hole_pucker",
        "hex": "#d946ef",  # Fuchsia
        "bgr": (239, 70, 217),
        "rgba": "rgba(217, 70, 239, 0.60)",
        "stroke": "#c026d3",
        "description": "Full-thickness round anatomical aperture at the central fovea causing profound reduction in central acuity.",
        "advice": "Macular OCT imaging and surgical consultation for vitrectomy with ILM peel and gas tamponade.",
    },
    "epiretinal_membrane": {
        "name": "Epiretinal Membrane (Pucker)",
        "disease_id": "macular_hole_pucker",
        "hex": "#8b5cf6",  # Indigo Violet
        "bgr": (246, 92, 139),
        "rgba": "rgba(139, 92, 246, 0.50)",
        "stroke": "#7c3aed",
        "description": "Avascular fibrocellular contractile sheet over the inner retinal surface causing cellophane sheen and vessel distortion.",
        "advice": "Monitor visual distortion with Amsler grid; consider membrane peel surgery if metamorphopsia is disabling.",
    },

    # 6. Vascular Occlusion & Ischemia
    "vascular_occlusion_ischemia": {
        "name": "Retinal Vascular Occlusion & Ischemia",
        "disease_id": "vascular_occlusion",
        "hex": "#6366f1",  # Indigo
        "bgr": (241, 102, 99),
        "rgba": "rgba(99, 102, 241, 0.55)",
        "stroke": "#4f46e5",
        "description": "Dense venous dilation, extensive sectorial flame hemorrhages, or acute diffuse ischemic retinal whitening.",
        "advice": "Urgent cardiovascular/stroke workup, carotid Doppler, and retinal specialist evaluation for anti-VEGF therapy.",
    },

    # 7. Central Serous Chorioretinopathy
    "serous_detachment_blister": {
        "name": "Serous Detachment Blister",
        "disease_id": "csc",
        "hex": "#0ea5e9",  # Sky Blue
        "bgr": (233, 165, 14),
        "rgba": "rgba(14, 165, 233, 0.55)",
        "stroke": "#0284c7",
        "description": "Well-demarcated circular dome-shaped accumulation of serous fluid under the neurosensory retina in the posterior pole.",
        "advice": "Assess stress factors and discontinue systemic/topical steroids; monitor resolution with macular OCT.",
    },

    # 8. Inherited Retinal Dystrophies
    "bone_spicule_pigment": {
        "name": "Bone-Spicule Pigmentation",
        "disease_id": "inherited_dystrophy",
        "hex": "#78716c",  # Warm Charcoal
        "bgr": (108, 113, 120),
        "rgba": "rgba(120, 113, 108, 0.60)",
        "stroke": "#57534e",
        "description": "Perivascular bone-spicule melanin deposits in the mid-periphery indicative of photoreceptor degeneration.",
        "advice": "Full-field ERG, Goldmann visual field testing, genetic counseling, and evaluation for gene replacement therapy.",
    },

    # 9. Ocular Oncology
    "ocular_tumor_mass": {
        "name": "Ocular Tumor / Choroidal Mass",
        "disease_id": "ocular_cancer",
        "hex": "#b91c1c",  # Deep Crimson
        "bgr": (28, 28, 185),
        "rgba": "rgba(185, 28, 28, 0.60)",
        "stroke": "#991b1b",
        "description": "Elevated pigmented or amelanotic intraocular mass lesion with orange lipofuscin pigment clumps or calcification.",
        "advice": "EMERGENCY: Immediate referral to ocular oncology center for B-scan ultrasound, MRI, and specialized management.",
    },

    # 10. Hypertensive Retinopathy
    "arteriovenous_nicking": {
        "name": "Arteriovenous Nicking",
        "disease_id": "hypertensive_retinopathy",
        "hex": "#ea580c",  # Orange
        "bgr": (12, 88, 234),
        "rgba": "rgba(234, 88, 12, 0.55)",
        "stroke": "#c2410c",
        "description": "Sclerotic deflection and tapering of retinal venules at crossing points beneath thickened arterioles.",
        "advice": "Prompt primary care referral for aggressive systemic blood pressure monitoring and medication adjustment.",
    },

    # 11. Pathologic Myopia Biomarkers
    "myopic_chorioretinal_atrophy": {
        "name": "Myopic Chorioretinal Atrophy",
        "disease_id": "pathologic_myopia",
        "hex": "#0d9488",  # Teal
        "bgr": (136, 148, 13),
        "rgba": "rgba(13, 148, 136, 0.55)",
        "stroke": "#0f766e",
        "description": "Patchy scleral and choroidal exposure due to mechanical posterior pole stretching and choriocapillaris loss.",
        "advice": "Axial length surveillance and OCT macular scanning to rule out myopic foveoschisis or macular hole.",
    },

    # 12. Diabetic Macular Edema Biomarkers
    "circinate_exudate_ring": {
        "name": "Circinate Exudate Ring",
        "disease_id": "diabetic_macular_edema",
        "hex": "#f59e0b",  # Amber Gold
        "bgr": (11, 158, 245),
        "rgba": "rgba(245, 158, 11, 0.60)",
        "stroke": "#d97706",
        "description": "Circular lipid-rich hard exudate ring encircling leaking microaneurysm clusters threatening central fovea.",
        "advice": "Specialist referral within 2–4 weeks for optical coherence tomography (OCT) and anti-VEGF / focal laser evaluation.",
    },

    # 13. Papilledema Biomarkers
    "disc_edema_margin_blur": {
        "name": "Optic Disc Margin Elevation & Blur",
        "disease_id": "papilledema",
        "hex": "#dc2626",  # Red
        "bgr": (38, 38, 220),
        "rgba": "rgba(220, 38, 38, 0.55)",
        "stroke": "#b91c1c",
        "description": "Circumferential blurring and elevated opacification of the neuroretinal rim obscuring crossing retinal vessels.",
        "advice": "EMERGENCY: Immediate neurological workup and brain neuroimaging (MRI/MRV) to rule out elevated intracranial pressure.",
    },

    # 14. Chorioretinitis Biomarkers
    "chorioretinal_scar": {
        "name": "Chorioretinal Inflammatory Scar",
        "disease_id": "chorioretinitis",
        "hex": "#8b5cf6",  # Purple
        "bgr": (246, 92, 139),
        "rgba": "rgba(139, 92, 246, 0.55)",
        "stroke": "#7c3aed",
        "description": "Circumscribed atrophic lesion with hyperpigmented borders indicating resolved or satellite active necrotizing retinitis.",
        "advice": "Uveitis consultation, serum infectious titer screening, and monitoring for inflammatory vitreous haze recurrence.",
    },
}

