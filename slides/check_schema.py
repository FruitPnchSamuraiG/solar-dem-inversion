"""Validate a deck's slides and charts against the OOXML schemas Google Slides enforces.

    uv run --with lxml python slides/check_schema.py slides/DEM_deck_v2.pptx

The pptx skill's validate.py passed decks that Google Slides then refused to
open; checking slide and chart parts directly against pml.xsd / dml-chart.xsd
catches what it missed (fill-before-border in table cells, float coordinates,
negative chart axis ids).
"""
import re
import sys
import zipfile

from lxml import etree

SCHEMAS = ("/home/hriday/.claude/skills/synced/d2605b18-73c4-4dd0-b319-52562620f3ae_"
           "9e97cdb3-4d3f-44a4-80c6-60cb9f566c98/pptx/scripts/office/schemas/ISO-IEC29500-4_2016/")

pml = etree.XMLSchema(etree.parse(SCHEMAS + "pml.xsd"))
chart = etree.XMLSchema(etree.parse(SCHEMAS + "dml-chart.xsd"))
z = zipfile.ZipFile(sys.argv[1])
bad = 0
for name in z.namelist():
    if re.match(r"ppt/slides/slide\d+\.xml$", name):
        schema = pml
    elif re.match(r"ppt/charts/chart\d+\.xml$", name):
        schema = chart
    else:
        continue
    if not schema.validate(etree.fromstring(z.read(name))):
        bad += 1
        print(name, schema.error_log.filter_from_errors()[:3])
print("schema errors in", bad, "parts")
sys.exit(1 if bad else 0)
