"""Build the report fonts: each Noto Sans Indic font merged with the Latin glyphs of Noto Sans.

ReportLab shapes a word with a single font, so a word that mixes scripts (e.g. "(काला" or a
Tamil word followed by ":") renders wrong if Latin and Indic text come from different fonts.
One merged font per script avoids switching fonts inside a word.

Noto fonts are licensed under the SIL Open Font License 1.1; the merged fonts are renamed
("RetinaLens Report <Script>") as modified versions.

Usage:
    python scripts/build_report_fonts.py
"""

from pathlib import Path

from fontTools import subset
from fontTools.merge import Merger, Options
from fontTools.ttLib import TTFont

FONT_DIR = Path(__file__).resolve().parents[1] / "backend" / "fonts"
SCRIPTS = ["Devanagari", "Bengali", "Tamil", "Telugu", "Kannada", "Malayalam"]
# Basic Latin, Latin-1, general punctuation, bullet, rupee sign, dotted circle.
LATIN_UNICODES = list(range(0x20, 0x7F)) + list(range(0xA0, 0x100)) + list(range(0x2010, 0x2028)) + [0x2022, 0x20B9, 0x25CC]
HINTING_TABLES = ["fpgm", "prep", "cvt ", "gasp", "hdmx", "LTSH", "VDMX"]


def strip_hinting(path_in: Path, path_out: Path, unicodes=None):
    opts = subset.Options()
    opts.hinting = False
    opts.drop_tables += HINTING_TABLES
    opts.layout_features = ["*"]
    opts.name_IDs = ["*"]
    opts.notdef_outline = True
    font = TTFont(path_in)
    sub = subset.Subsetter(opts)
    sub.populate(unicodes=unicodes or font.getBestCmap().keys())
    sub.subset(font)
    font.save(path_out)


def set_name(font: TTFont, family: str, weight: str):
    full = f"{family} {weight}"
    ps = f"{family.replace(' ', '')}-{weight}"
    for rec in font["name"].names:
        if rec.nameID in (1, 16):
            rec.string = family
        elif rec.nameID in (2, 17):
            rec.string = weight
        elif rec.nameID == 4:
            rec.string = full
        elif rec.nameID == 6:
            rec.string = ps
        elif rec.nameID == 3:
            rec.string = f"{ps};modified-from-Noto"


def main():
    tmp = FONT_DIR / "_build"
    tmp.mkdir(exist_ok=True)
    for weight in ("Regular", "Bold"):
        latin = tmp / f"latin-{weight}.ttf"
        strip_hinting(FONT_DIR / f"NotoSans-{weight}.ttf", latin, LATIN_UNICODES)
        for script in SCRIPTS:
            native = tmp / f"{script}-{weight}.ttf"
            strip_hinting(FONT_DIR / f"NotoSans{script}-{weight}.ttf", native)
            # The script font goes first so its own glyphs win any cmap overlap.
            merged = Merger(options=Options(drop_tables=HINTING_TABLES)).merge([str(native), str(latin)])
            set_name(merged, f"RetinaLens Report {script}", weight)
            out = FONT_DIR / f"RetinaLensReport{script}-{weight}.ttf"
            merged.save(out)
            print("built", out.name)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


if __name__ == "__main__":
    main()
