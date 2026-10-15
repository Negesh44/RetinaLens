"""Generate clinical preset sample fundus images for each retinal disease condition.

Produces realistic phantom fundus images for:
- Diabetic Retinopathy (DR & DME)
- Age-Related Macular Degeneration (AMD - Drusen & Geographic Atrophy)
- Glaucoma (Optic Nerve Head Cupping & CDR > 0.75)
- Retinal Detachment & Tears (Corrugated Folds & Horseshoe Tear)
- Macular Hole & Pucker (Central Foveal Defect & Halo)
- Retinal Vein Occlusion (RVO - Sectorial Hemorrhages & Engorged Venules)
- Central Serous Chorioretinopathy (CSC - Neurosensory Serous Fluid Blister)
- Retinitis Pigmentosa (Inherited Dystrophy - Mid-Peripheral Bone Spicules)
- Choroidal Melanoma (Ocular Oncology - Dark Pigmented Lesion with Lipofuscin)
- Hypertensive Retinopathy (A/V Nicking & Flame Hemorrhages)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

SAMPLES_DIR = PROJECT_ROOT / "frontend" / "public" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def create_base_fundus(size: int = 400, seed: int = 42) -> tuple:
    """Creates a base healthy fundus disc with disc center, macula center, and arcade vessels."""
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.44)

    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    fundus_mask = dist_from_center <= radius

    # Natural fundus color gradient (orange-red)
    red_val = np.clip(185 - (dist_from_center / radius * 50) + rng.normal(0, 3, (size, size)), 85, 215)
    green_val = np.clip(78 - (dist_from_center / radius * 35) + rng.normal(0, 3, (size, size)), 25, 115)
    blue_val = np.clip(26 - (dist_from_center / radius * 15) + rng.normal(0, 2, (size, size)), 5, 50)

    img[fundus_mask, 0] = red_val[fundus_mask].astype(np.uint8)
    img[fundus_mask, 1] = green_val[fundus_mask].astype(np.uint8)
    img[fundus_mask, 2] = blue_val[fundus_mask].astype(np.uint8)

    # Optic disc location (nasal side: left-ish)
    disc_center = (int(center[0] - radius * 0.42), int(center[1] + radius * 0.05))
    disc_radius = int(radius * 0.16)

    # Base optic disc
    cv2.circle(img, disc_center, disc_radius, (255, 220, 130), -1)

    # Physiological cup (small central depression: CDR ~ 0.35)
    cup_radius = int(disc_radius * 0.35)
    cv2.circle(img, disc_center, cup_radius, (255, 245, 180), -1)

    # Macula center (temporal side: right-center)
    macula_center = (int(center[0] + radius * 0.22), int(center[1] + radius * 0.04))
    macula_radius = int(radius * 0.20)
    # Subtle macular luteal pigment shading (darker amber-brown)
    cv2.circle(img, macula_center, macula_radius, (160, 55, 20), -1)
    # Foveal avascular zone (FAZ)
    cv2.circle(img, macula_center, int(macula_radius * 0.3), (135, 40, 15), -1)

    # Arcades
    for angle_deg in [25, 65, 115, 155, 205, 245, 295, 335]:
        rad = np.radians(angle_deg)
        pts = [disc_center]
        cur_x, cur_y = disc_center
        length = int(radius * 0.85)
        step = 8
        for _ in range(0, length, step):
            cur_x += int(step * np.cos(rad) + rng.randint(-2, 3))
            cur_y += int(step * np.sin(rad) + rng.randint(-2, 3))
            if np.sqrt((cur_x - center[0]) ** 2 + (cur_y - center[1]) ** 2) < radius - 5:
                pts.append((cur_x, cur_y))
        if len(pts) > 1:
            thickness = max(1, int(size * 0.007))
            for p1, p2 in zip(pts[:-1], pts[1:]):
                cv2.line(img, p1, p2, (100, 20, 15), thickness)

    return img, center, radius, disc_center, disc_radius, macula_center, rng


def generate_amd_drusen():
    """Generates fundus with soft drusen clusters and geographic atrophy in the macula."""
    img, center, radius, disc_center, disc_radius, macula_center, rng = create_base_fundus(400, seed=201)

    # 1. Soft Drusen: yellow-white dome-shaped confluent deposits in macula
    for _ in range(35):
        offset = rng.normal(0, radius * 0.12, size=2)
        dx = int(macula_center[0] + offset[0])
        dy = int(macula_center[1] + offset[1])
        dr = rng.randint(3, 8)
        # Soft yellow drusen color
        cv2.circle(img, (dx, dy), dr, (245, 230, 140), -1)

    # 2. Hard Drusen: tiny discrete yellow spots
    for _ in range(25):
        offset = rng.normal(0, radius * 0.18, size=2)
        dx = int(macula_center[0] + offset[0])
        dy = int(macula_center[1] + offset[1])
        cv2.circle(img, (dx, dy), rng.randint(1, 3), (255, 245, 170), -1)

    # 3. Geographic Atrophy patch: pale scalloped hypopigmentation
    ga_center = (macula_center[0] - 15, macula_center[1] + 10)
    cv2.ellipse(img, ga_center, (22, 16), 30, 0, 360, (220, 150, 90), -1)
    cv2.ellipse(img, ga_center, (18, 12), 30, 0, 360, (235, 175, 110), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return img


def generate_glaucoma_cupping():
    """Generates fundus with large optic cup excavation (CDR = 0.78) and rim thinning."""
    img, center, radius, disc_center, disc_radius, macula_center, rng = create_base_fundus(400, seed=202)

    # Enlarge optic cup significantly: CDR = 0.78 (cup radius = 0.78 * disc_radius)
    glaucoma_cup_radius = int(disc_radius * 0.78)
    # Pale excavated optic cup
    cv2.circle(img, disc_center, glaucoma_cup_radius, (255, 250, 205), -1)
    # Inferior neuroretinal rim thinning notch
    notch_center = (disc_center[0], disc_center[1] + int(disc_radius * 0.7))
    cv2.circle(img, notch_center, int(disc_radius * 0.28), (255, 252, 215), -1)

    # Nasal displacement of central retinal vessels (bayoneting)
    for dy in [-15, -8, 0, 8, 16]:
        p1 = (disc_center[0] - 10, disc_center[1] + dy)
        p2 = (disc_center[0] - disc_radius - 12, disc_center[1] + dy + rng.randint(-3, 3))
        cv2.line(img, p1, p2, (95, 15, 10), 2)

    # Subtle superior RNFL wedge defect (darker arcuate band)
    cv2.ellipse(img, disc_center, (int(radius * 0.45), int(radius * 0.3)), -40, 30, 85, (140, 45, 15), 6)

    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return img


def generate_retinal_detachment():
    """Generates fundus with elevated corrugated detachment folds and horseshoe tear."""
    img, center, radius, disc_center, disc_radius, macula_center, rng = create_base_fundus(400, seed=203)

    # Detached retina: pale, opaque, grayish-green undulating ballooning bulla
    detachment_poly = np.array([
        [center[0] - 20, center[1] - radius + 15],
        [center[0] + int(radius * 0.6), center[1] - radius + 25],
        [center[0] + radius - 20, center[1] - int(radius * 0.2)],
        [center[0] + int(radius * 0.5), center[1] + 10],
        [center[0] + 10, center[1] - int(radius * 0.15)],
        [center[0] - int(radius * 0.2), center[1] - int(radius * 0.5)],
    ], dtype=np.int32)

    cv2.fillPoly(img, [detachment_poly], (185, 175, 155))

    # Undulating corrugated folds (lighter white streaks)
    cv2.ellipse(img, (center[0] + 70, center[1] - 80), (55, 12), -25, 0, 360, (225, 218, 205), 3)
    cv2.ellipse(img, (center[0] + 110, center[1] - 50), (45, 10), -20, 0, 360, (230, 222, 210), 3)

    # Horseshoe retinal tear: flap with intense dark red underlying choroid exposed
    tear_center = (center[0] + 85, center[1] - 110)
    cv2.ellipse(img, tear_center, (14, 8), -35, 160, 380, (90, 10, 8), -1)  # exposed choroid
    # Flap apex (pale flap edge)
    cv2.ellipse(img, (tear_center[0] - 2, tear_center[1] - 4), (11, 4), -35, 0, 180, (240, 235, 225), 2)

    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return img


def generate_macular_hole():
    """Generates fundus with round central full-thickness foveal macular hole with halo."""
    img, center, radius, disc_center, disc_radius, macula_center, rng = create_base_fundus(400, seed=204)

    # Macular Hole: round, sharp red aperture right at foveal center
    # 1. Cuff/halo of subretinal fluid (pale yellow-gray circle)
    cv2.circle(img, macula_center, 18, (195, 165, 120), -1)

    # 2. Yellow deposits at the base
    for _ in range(6):
        ox = macula_center[0] + rng.randint(-5, 6)
        oy = macula_center[1] + rng.randint(-5, 6)
        cv2.circle(img, (ox, oy), 1, (255, 245, 170), -1)

    # 3. Deep red central aperture exposing RPE/choroid
    cv2.circle(img, macula_center, 8, (95, 10, 8), -1)

    # Subtle epiretinal membrane pucker wrinkles radiating out
    for angle in [15, 75, 145, 215, 285]:
        rad = np.radians(angle)
        p1 = (int(macula_center[0] + 18 * np.cos(rad)), int(macula_center[1] + 18 * np.sin(rad)))
        p2 = (int(macula_center[0] + 45 * np.cos(rad)), int(macula_center[1] + 45 * np.sin(rad)))
        cv2.line(img, p1, p2, (200, 140, 100), 1)

    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return img


def generate_rvo():
    """Generates Retinal Vein Occlusion with blood-and-thunder hemorrhages along superior arcade."""
    img, center, radius, disc_center, disc_radius, macula_center, rng = create_base_fundus(400, seed=205)

    # Engorged, dilated, tortuous superior temporal vein
    pts = [
        disc_center,
        (disc_center[0] + 30, disc_center[1] - 40),
        (disc_center[0] + 70, disc_center[1] - 65),
        (disc_center[0] + 115, disc_center[1] - 75),
        (disc_center[0] + 155, disc_center[1] - 70),
    ]
    for p1, p2 in zip(pts[:-1], pts[1:]):
        cv2.line(img, p1, p2, (80, 5, 5), 4)

    # Sectorial "blood-and-thunder" flame and blot hemorrhages across superior quadrant
    for _ in range(65):
        hx = int(disc_center[0] + rng.uniform(25, 180))
        hy = int(disc_center[1] - rng.uniform(20, 110))
        # Flame hemorrhage (brush stroke)
        axes = (rng.randint(8, 20), rng.randint(3, 7))
        angle = rng.randint(-35, 10)
        cv2.ellipse(img, (hx, hy), axes, angle, 0, 360, (70, 5, 5), -1)

    # Cotton wool spots (ischemic whitening)
    for _ in range(5):
        cx = int(disc_center[0] + rng.uniform(40, 140))
        cy = int(disc_center[1] - rng.uniform(30, 95))
        cv2.circle(img, (cx, cy), rng.randint(5, 10), (230, 220, 210), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return img


def generate_csc():
    """Generates Central Serous Chorioretinopathy with circular subretinal fluid bleb in macula."""
    img, center, radius, disc_center, disc_radius, macula_center, rng = create_base_fundus(400, seed=206)

    # CSC: well-demarcated circular dome-shaped blister of transparent subretinal fluid
    bleb_center = (macula_center[0] - 6, macula_center[1] + 4)
    bleb_radius = int(radius * 0.18)

    # Subretinal fluid dome with highlighted crescent light reflex
    cv2.circle(img, bleb_center, bleb_radius, (190, 115, 75), -1)
    cv2.circle(img, bleb_center, bleb_radius - 4, (180, 105, 65), -1)
    # Bright halo reflex border
    cv2.circle(img, bleb_center, bleb_radius, (230, 195, 145), 2)
    # Central punctate yellow precipitates at RPE level
    for _ in range(12):
        ox = bleb_center[0] + rng.randint(-12, 13)
        oy = bleb_center[1] + rng.randint(-12, 13)
        cv2.circle(img, (ox, oy), 1, (245, 235, 160), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return img


def generate_retinitis_pigmentosa():
    """Generates Retinitis Pigmentosa with mid-peripheral bone-spicule hyperpigmentation."""
    img, center, radius, disc_center, disc_radius, macula_center, rng = create_base_fundus(400, seed=207)

    # Normal cup on disc (not cupping)
    cv2.circle(img, disc_center, int(disc_radius * 0.32), (255, 235, 160), -1)

    # Bone spicules: black/dark-brown jagged branching deposits in mid-periphery
    for _ in range(160):
        # Ring in mid-periphery (0.50 to 0.88 radius)
        r_mid = rng.uniform(radius * 0.50, radius * 0.88)
        theta = rng.uniform(0, 2 * np.pi)
        bx = int(center[0] + r_mid * np.cos(theta))
        by = int(center[1] + r_mid * np.sin(theta))

        # Branching 3-point bone spicule star
        cv2.circle(img, (bx, by), 3, (15, 10, 10), -1)
        for arm_deg in rng.choice([0, 60, 120, 180, 240], size=2, replace=False):
            rad = np.radians(arm_deg + rng.randint(-15, 15))
            ex = int(bx + rng.randint(4, 9) * np.cos(rad))
            ey = int(by + rng.randint(4, 9) * np.sin(rad))
            cv2.line(img, (bx, by), (ex, ey), (15, 10, 10), 2)

    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return img



def generate_melanoma_tumor():
    """Generates Choroidal Melanoma with dark elevated pigmented mass and orange lipofuscin."""
    img, center, radius, disc_center, disc_radius, macula_center, rng = create_base_fundus(400, seed=208)

    # Elevated choroidal mass in supero-temporal periphery
    tumor_center = (center[0] + int(radius * 0.45), center[1] - int(radius * 0.35))
    tumor_axes = (int(radius * 0.22), int(radius * 0.17))

    # Dark brown-slate gray pigmented tumor dome
    cv2.ellipse(img, tumor_center, tumor_axes, -20, 0, 360, (30, 20, 15), -1)
    cv2.ellipse(img, tumor_center, (tumor_axes[0] - 6, tumor_axes[1] - 4), -20, 0, 360, (20, 15, 12), -1)

    # Clumps of orange lipofuscin pigment overlying tumor
    for _ in range(12):
        ox = tumor_center[0] + rng.randint(-18, 19)
        oy = tumor_center[1] + rng.randint(-14, 15)
        cv2.circle(img, (ox, oy), rng.randint(2, 4), (230, 110, 20), -1)

    # Associated localized exudative subretinal fluid cuff around base
    cv2.ellipse(img, tumor_center, (tumor_axes[0] + 8, tumor_axes[1] + 6), -20, 0, 360, (180, 140, 100), 2)

    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return img


def generate_hypertensive_retinopathy():
    """Generates Hypertensive Retinopathy with copper wiring and flame hemorrhages."""
    img, center, radius, disc_center, disc_radius, macula_center, rng = create_base_fundus(400, seed=209)

    # Flame-shaped hemorrhages along nerve fiber layer
    for _ in range(18):
        theta = rng.uniform(0, 2 * np.pi)
        r_off = rng.uniform(radius * 0.25, radius * 0.65)
        hx = int(center[0] + r_off * np.cos(theta))
        hy = int(center[1] + r_off * np.sin(theta))
        angle = int(np.degrees(theta))
        cv2.ellipse(img, (hx, hy), (rng.randint(7, 14), rng.randint(2, 4)), angle, 0, 360, (80, 8, 8), -1)

    # Macular star hard exudates (radiating yellow rays centered on fovea)
    for ray_deg in range(0, 360, 30):
        rad = np.radians(ray_deg)
        for d in [16, 24, 32]:
            fx = int(macula_center[0] + d * np.cos(rad) + rng.randint(-2, 3))
            fy = int(macula_center[1] + d * np.sin(rad) + rng.randint(-2, 3))
            cv2.circle(img, (fx, fy), 2, (255, 245, 150), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return img


def generate_pathologic_myopia():
    """Generates Pathologic Myopia with peripapillary conus, lacquer cracks, and atrophy."""
    img, center, radius, disc_center, disc_radius, macula_center, rng = create_base_fundus(400, seed=210)

    # 1. Peripapillary crescent (temporal conus): bright white/yellowish crescent on temporal side of disc
    crescent_pts = []
    for deg in range(-80, 80, 10):
        rad = np.radians(deg)
        cx = int(disc_center[0] + (disc_radius + 12) * np.cos(rad))
        cy = int(disc_center[1] + (disc_radius + 12) * np.sin(rad))
        crescent_pts.append((cx, cy))
    for deg in range(80, -80, -10):
        rad = np.radians(deg)
        cx = int(disc_center[0] + disc_radius * np.cos(rad))
        cy = int(disc_center[1] + disc_radius * np.sin(rad))
        crescent_pts.append((cx, cy))
    if len(crescent_pts) > 4:
        cv2.fillPoly(img, [np.array(crescent_pts, dtype=np.int32)], (240, 235, 210))

    # 2. Patchy chorioretinal atrophy (scleral visibility)
    for _ in range(4):
        ax = int(macula_center[0] + rng.randint(-30, 30))
        ay = int(macula_center[1] + rng.randint(-30, 30))
        cv2.ellipse(img, (ax, ay), (rng.randint(18, 30), rng.randint(10, 18)), rng.randint(0, 180), 0, 360, (230, 225, 195), -1)

    # 3. Lacquer cracks (fine linear breaks in Bruch's membrane)
    for _ in range(3):
        lx1 = int(macula_center[0] + rng.randint(-25, 25))
        ly1 = int(macula_center[1] + rng.randint(-25, 25))
        lx2 = int(lx1 + rng.randint(15, 35))
        ly2 = int(ly1 + rng.randint(-15, 15))
        cv2.line(img, (lx1, ly1), (lx2, ly2), (210, 190, 140), 1)

    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return img


def generate_diabetic_macular_edema():
    """Generates Diabetic Macular Edema with classic circinate hard exudate ring."""
    img, center, radius, disc_center, disc_radius, macula_center, rng = create_base_fundus(400, seed=211)

    # Central leaking microaneurysm cluster in fovea
    for _ in range(8):
        mx = int(macula_center[0] + rng.randint(-10, 11))
        my = int(macula_center[1] + rng.randint(-10, 11))
        cv2.circle(img, (mx, my), rng.randint(2, 4), (130, 10, 10), -1)

    # Circinate ring of bright waxy lipid hard exudates encircling the center
    ring_radius = int(radius * 0.14)
    for deg in range(0, 360, 18):
        rad = np.radians(deg)
        r_dist = ring_radius + rng.randint(-6, 7)
        ex_x = int(macula_center[0] + r_dist * np.cos(rad))
        ex_y = int(macula_center[1] + r_dist * np.sin(rad))
        cv2.circle(img, (ex_x, ex_y), rng.randint(3, 7), (255, 240, 120), -1)

    # Few small blot hemorrhages
    for _ in range(6):
        hx = int(macula_center[0] + rng.randint(-40, 41))
        hy = int(macula_center[1] + rng.randint(-40, 41))
        cv2.ellipse(img, (hx, hy), (rng.randint(4, 7), rng.randint(3, 5)), rng.randint(0, 180), 0, 360, (110, 12, 12), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return img


def generate_papilledema():
    """Generates Papilledema with circumferential disc margin blurring and elevated hyperaemia."""
    img, center, radius, disc_center, disc_radius, macula_center, rng = create_base_fundus(400, seed=212)

    # Swollen, hyperaemic enlarged optic disc with blurred edges
    swollen_radius = int(disc_radius * 1.45)
    # Layered gradient blurring to simulate elevation & Frisén grade 3/4
    for r_i in range(swollen_radius, disc_radius, -3):
        alpha = (r_i - disc_radius) / (swollen_radius - disc_radius)
        c_val = int(235 * (1 - alpha) + 185 * alpha)
        cv2.circle(img, disc_center, r_i, (c_val, int(c_val * 0.8), int(c_val * 0.5)), -1)

    # Obscuration of vessel segments crossing the margin
    for angle_deg in [30, 75, 120, 160, 210, 250, 300, 340]:
        rad = np.radians(angle_deg)
        vx = int(disc_center[0] + disc_radius * np.cos(rad))
        vy = int(disc_center[1] + disc_radius * np.sin(rad))
        # Vessel obscuration patch
        cv2.circle(img, (vx, vy), 7, (230, 185, 125), -1)

    # Flame-shaped splinter hemorrhages on disc margin
    for _ in range(10):
        theta = rng.uniform(0, 2 * np.pi)
        r_dist = rng.uniform(disc_radius * 0.85, swollen_radius * 1.1)
        hx = int(disc_center[0] + r_dist * np.cos(theta))
        hy = int(disc_center[1] + r_dist * np.sin(theta))
        cv2.ellipse(img, (hx, hy), (rng.randint(6, 12), rng.randint(2, 4)), int(np.degrees(theta)), 0, 360, (75, 8, 8), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return img


def generate_chorioretinitis():
    """Generates Chorioretinitis with sharply demarcated pigmented atrophic chorioretinal scar."""
    img, center, radius, disc_center, disc_radius, macula_center, rng = create_base_fundus(400, seed=213)

    # Main toxoplasmosis-style chorioretinal scar in upper temporal quadrant
    scar_center = (int(center[0] + radius * 0.35), int(center[1] - radius * 0.30))
    scar_axes = (int(radius * 0.16), int(radius * 0.12))

    # Inner pale atrophic bare sclera
    cv2.ellipse(img, scar_center, scar_axes, 25, 0, 360, (235, 230, 205), -1)

    # Dense perilesional hyperpigmented melanin border
    cv2.ellipse(img, scar_center, (scar_axes[0] + 3, scar_axes[1] + 3), 25, 0, 360, (25, 15, 12), 4)

    # Focal clumps of black pigment inside scar
    for _ in range(12):
        px = int(scar_center[0] + rng.randint(-scar_axes[0] + 4, scar_axes[0] - 4))
        py = int(scar_center[1] + rng.randint(-scar_axes[1] + 4, scar_axes[1] - 4))
        cv2.circle(img, (px, py), rng.randint(2, 4), (20, 15, 10), -1)

    # Fluffy satellite active inflammatory focus adjacent to old scar
    sat_center = (int(scar_center[0] - scar_axes[0] * 0.8), int(scar_center[1] + scar_axes[1] * 0.8))
    cv2.circle(img, sat_center, int(radius * 0.08), (220, 215, 200), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return img


def run_all():
    print(f"Generating multi-disease clinical presets in {SAMPLES_DIR}...")

    presets = [
        ("sample_amd_drusen.png", generate_amd_drusen()),
        ("sample_glaucoma_cupping.png", generate_glaucoma_cupping()),
        ("sample_retinal_detachment.png", generate_retinal_detachment()),
        ("sample_macular_hole.png", generate_macular_hole()),
        ("sample_rvo_occlusion.png", generate_rvo()),
        ("sample_csc_serous.png", generate_csc()),
        ("sample_rp_bone_spicules.png", generate_retinitis_pigmentosa()),
        ("sample_melanoma_tumor.png", generate_melanoma_tumor()),
        ("sample_hypertensive_retinopathy.png", generate_hypertensive_retinopathy()),
        ("sample_pathologic_myopia.png", generate_pathologic_myopia()),
        ("sample_diabetic_macular_edema.png", generate_diabetic_macular_edema()),
        ("sample_papilledema.png", generate_papilledema()),
        ("sample_chorioretinitis.png", generate_chorioretinitis()),
    ]

    for filename, rgb_img in presets:
        out_path = SAMPLES_DIR / filename
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)
        print(f"  [+] Wrote preset {filename} ({rgb_img.shape[1]}x{rgb_img.shape[0]})")

    print("[SUCCESS] All 13 multi-disease clinical presets generated!")


if __name__ == "__main__":
    run_all()

