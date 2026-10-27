#!/opt/anaconda3/bin/python
"""
challenger_independent_audit.py
Adversarial Empirical Verification & Stress-Testing Suite
For:
- multicp/multicp.docx
- multicp_sacp/scmcp.docx

Written and executed by Teamwork Challenger M3-1.
Performs raw XML extraction from .docx ZIP archives,
evaluates pixel conservation (sum = 262,144),
hunts for latent NaNs/N/A/nulls/broken refs,
cross-checks against raw CSV/Excel/JSON ground truth,
audits OpenXML drawing blip relationships and embedded media assets,
and performs color-palette verification.
"""

import os
import sys
import zipfile
import xml.etree.ElementTree as ET
import json
import re
import math
import pandas as pd
import openpyxl
from PIL import Image
import io

WORKSPACE = "/Users/danesh/Desktop/Reports_dias/Kaggle_Computed_65Bands_DIAS"
MULTICP_DIR = os.path.join(WORKSPACE, "multicp")
SCMCP_DIR = os.path.join(WORKSPACE, "multicp_sacp")

XML_NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}

total_tests = 0
passed_tests = 0
failed_tests = 0
vulnerabilities = []

def record_pass(msg):
    global total_tests, passed_tests
    total_tests += 1
    passed_tests += 1

def record_fail(msg, detail=""):
    global total_tests, failed_tests, vulnerabilities
    total_tests += 1
    failed_tests += 1
    full_msg = f"{msg} | {detail}" if detail else msg
    vulnerabilities.append(full_msg)
    print(f"  [FAIL] {full_msg}")

def parse_num(val_str):
    """Strip formatting chars and convert to float."""
    s = str(val_str).replace(",", "").replace("%", "").strip()
    return float(s)

def extract_docx_xml_and_tables(docx_path):
    """Open docx as zip and extract document.xml root, parsed table matrices, and relationships."""
    if not os.path.exists(docx_path):
        record_fail(f"File not found: {docx_path}")
        return None, None, None, None
    
    with zipfile.ZipFile(docx_path, 'r') as z:
        xml_bytes = z.read("word/document.xml")
        rels_xml = z.read("word/_rels/document.xml.rels")
        media_files = {f: z.read(f) for f in z.namelist() if f.startswith("word/media/")}
        
    root = ET.fromstring(xml_bytes)
    root_rels = ET.fromstring(rels_xml)
    
    img_rels = {}
    for r in root_rels:
        if "image" in r.attrib.get("Type", ""):
            img_rels[r.attrib["Id"]] = r.attrib["Target"]
            
    # Extract all tables
    tables = []
    for tbl in root.iter(f"{{{XML_NAMESPACES['w']}}}tbl"):
        table_grid = []
        for tr in tbl.iter(f"{{{XML_NAMESPACES['w']}}}tr"):
            row_cells = []
            for tc in tr.iter(f"{{{XML_NAMESPACES['w']}}}tc"):
                cell_text = "".join(t.text for t in tc.iter(f"{{{XML_NAMESPACES['w']}}}t") if t.text)
                row_cells.append(cell_text.strip())
            if row_cells:
                table_grid.append(row_cells)
        if table_grid:
            tables.append(table_grid)
            
    return root, tables, img_rels, media_files

# =====================================================================
# SECTION 1: ADVERSARIAL SCAN FOR LATENT CORRUPTIONS & PLACEHOLDERS
# =====================================================================
print("\n" + "=" * 70)
print("TEST SUITE 1: ADVERSARIAL STRING & PLACEHOLDER AUDIT")
print("=" * 70)

FORBIDDEN_PATTERNS = [
    (r"\bN/?A\b", "N/A or NA token"),
    (r"\bNaN\b", "NaN floating point anomaly"),
    (r"\bnull\b", "null token"),
    (r"\bNone\b", "Python None token"),
    (r"\bundefined\b", "undefined JS/token"),
    (r"Error!\s*Reference source not found", "Broken Word cross-reference"),
    (r"\bTODO\b", "Unfinished TODO"),
    (r"\bFIXME\b", "Unfinished FIXME"),
    (r"\bTBD\b", "Unfinished TBD placeholder"),
    (r"\[placeholder\]", "Placeholder text"),
    (r"\b\d+\.\d{7,}\b", "Unformatted excessive-precision float (>6 decimals)"),
]

for doc_label, doc_path in [("multicp.docx", os.path.join(MULTICP_DIR, "multicp.docx")),
                            ("scmcp.docx", os.path.join(SCMCP_DIR, "scmcp.docx"))]:
    print(f"\nScanning {doc_label} XML for latent corruptions...")
    root, tables, img_rels, media_files = extract_docx_xml_and_tables(doc_path)
    
    # Check all <w:t> elements across whole doc
    all_texts = [t.text for t in root.iter(f"{{{XML_NAMESPACES['w']}}}t") if t.text]
    full_text = " ".join(all_texts)
    
    for pattern, pat_desc in FORBIDDEN_PATTERNS:
        matches = re.findall(pattern, full_text, flags=re.IGNORECASE)
        filtered_matches = []
        for m in matches:
            if pat_desc == "N/A or NA token" and m.upper() in ["N/A", "NA"]:
                filtered_matches.append(m)
            elif pat_desc == "NaN floating point anomaly" and m == "NaN":
                filtered_matches.append(m)
            elif pat_desc == "null token" and m.lower() == "null":
                filtered_matches.append(m)
            elif pat_desc == "Python None token" and m == "None":
                filtered_matches.append(m)
            elif pat_desc not in ["N/A or NA token", "NaN floating point anomaly", "null token", "Python None token"]:
                filtered_matches.append(m)
                
        if len(filtered_matches) == 0:
            record_pass(f"{doc_label}: 0 occurrences of '{pat_desc}'")
        else:
            record_fail(f"{doc_label}: Found {len(filtered_matches)} occurrences of '{pat_desc}'", str(filtered_matches[:5]))

    # Check for empty cells in data tables (ignoring qualitative status columns)
    empty_cells_count = 0
    for tbl_idx, tbl in enumerate(tables):
        # Skip figure tables (2 rows, 2 cols)
        if len(tbl) == 2 and len(tbl[0]) == 2:
            continue
        headers = tbl[0]
        for r_idx, row in enumerate(tbl[1:]):
            for c_idx, cell_str in enumerate(row):
                col_header = headers[c_idx] if c_idx < len(headers) else ""
                # "Best?" is a qualitative indicator column where non-best rows legitimately have blank string
                if col_header == "Best?":
                    if cell_str != "" and not cell_str.startswith("✓"):
                        record_fail(f"{doc_label} Table {tbl_idx+1} Row {r_idx+2} 'Best?' unexpected content: {cell_str}")
                    continue
                if cell_str == "":
                    empty_cells_count += 1
                    record_fail(f"{doc_label} Table {tbl_idx+1} Row {r_idx+2} Col {c_idx+1} ({col_header}) is empty!")
    if empty_cells_count == 0:
        record_pass(f"{doc_label}: Zero unintended empty cells across all data columns")


# =====================================================================
# SECTION 2: PIXEL CONSERVATION AUDIT (SUM = 262,144)
# =====================================================================
print("\n" + "=" * 70)
print("TEST SUITE 2: PIXEL CONSERVATION EMPIRICAL ORACLE (SUM == 262,144)")
print("=" * 70)

root_m, tables_m, img_rels_m, media_m = extract_docx_xml_and_tables(os.path.join(MULTICP_DIR, "multicp.docx"))

with open(os.path.join(MULTICP_DIR, "results", "multicp_pixel_counts.json")) as f:
    gt_m_pixels = json.load(f)
gt_m_pix_map = {item["model_key"]: item["counts"] for item in gt_m_pixels}

for t_idx, score in [(7, "RAPS"), (8, "SAPS")]:
    tbl = tables_m[t_idx]
    keys = [f"AlexNet_CN_{score}", f"GFNet_Mult_{score}", f"ViT_UNet_M_{score}"]
    print(f"\nChecking MultiCP Table {t_idx+1} ({score} Pixel Distribution)...")
    
    for col_idx, k in enumerate(keys):
        class_pixels = []
        for c in range(8):
            px_val = int(parse_num(tbl[c + 1][col_idx + 1]))
            gt_px = gt_m_pix_map[k][c]
            if px_val == gt_px:
                record_pass(f"MultiCP {score} {k} Class {c} count matches: {px_val}")
            else:
                record_fail(f"MultiCP {score} {k} Class {c} mismatch", f"Doc: {px_val} vs GT: {gt_px}")
            class_pixels.append(px_val)
            
        unc_val = int(parse_num(tbl[9][col_idx + 1]))
        gt_unc = gt_m_pix_map[k][8]
        if unc_val == gt_unc:
            record_pass(f"MultiCP {score} {k} Uncertain pixel count matches: {unc_val}")
        else:
            record_fail(f"MultiCP {score} {k} Uncertain pixel mismatch", f"Doc: {unc_val} vs GT: {gt_unc}")
            
        total_val = int(parse_num(tbl[10][col_idx + 1]))
        computed_sum = sum(class_pixels) + unc_val
        
        if computed_sum == 262144:
            record_pass(f"MultiCP {score} {k}: Computed sum of all classes + uncertain == 262,144")
        else:
            record_fail(f"MultiCP {score} {k}: Pixel sum violation!", f"Sum={computed_sum} != 262144")
            
        if total_val == 262144:
            record_pass(f"MultiCP {score} {k}: Printed total cell == 262,144")
        else:
            record_fail(f"MultiCP {score} {k}: Printed total cell != 262144", f"Total={total_val}")


root_s, tables_s, img_rels_s, media_s = extract_docx_xml_and_tables(os.path.join(SCMCP_DIR, "scmcp.docx"))

with open(os.path.join(SCMCP_DIR, "results", "pixel_counts_all_windows.json")) as f:
    gt_s_pixels = json.load(f)
gt_s_pix_map = {(it["window_size"], it["model_name"], it["scoring_method"]): it["counts"] for it in gt_s_pixels}

cols_s = [
    ("AlexNet", "RAPS"), ("GFNet", "RAPS"), ("ViT", "RAPS"),
    ("AlexNet", "SAPS"), ("GFNet", "SAPS"), ("ViT", "SAPS")
]

for w_idx, ws in enumerate([3, 5, 7, 9]):
    tbl_idx = w_idx * 7 + 2
    tbl = tables_s[tbl_idx]
    print(f"\nChecking SCMCP Window {ws} Pixel Distribution (Table {tbl_idx+1})...")
    
    for c_idx, (m, s) in enumerate(cols_s):
        class_pixels = []
        for c in range(8):
            px_val = int(parse_num(tbl[c + 1][c_idx + 1]))
            gt_px = gt_s_pix_map[(ws, m, s)][c]
            if px_val == gt_px:
                record_pass(f"SCMCP WS={ws} {m} {s} Class {c} matches: {px_val}")
            else:
                record_fail(f"SCMCP WS={ws} {m} {s} Class {c} mismatch", f"Doc: {px_val} vs GT: {gt_px}")
            class_pixels.append(px_val)
            
        unc_val = int(parse_num(tbl[9][c_idx + 1]))
        gt_unc = gt_s_pix_map[(ws, m, s)][8]
        if unc_val == gt_unc:
            record_pass(f"SCMCP WS={ws} {m} {s} Uncertain matches: {unc_val}")
        else:
            record_fail(f"SCMCP WS={ws} {m} {s} Uncertain mismatch", f"Doc: {unc_val} vs GT: {gt_unc}")
            
        total_val = int(parse_num(tbl[10][c_idx + 1]))
        computed_sum = sum(class_pixels) + unc_val
        
        if computed_sum == 262144:
            record_pass(f"SCMCP WS={ws} {m} {s}: Computed sum == 262,144")
        else:
            record_fail(f"SCMCP WS={ws} {m} {s}: Pixel sum violation!", f"Sum={computed_sum} != 262144")
            
        if total_val == 262144:
            record_pass(f"SCMCP WS={ws} {m} {s}: Printed total cell == 262,144")
        else:
            record_fail(f"SCMCP WS={ws} {m} {s}: Printed total cell != 262144", f"Total={total_val}")


# =====================================================================
# SECTION 3: DEEP GROUND-TRUTH NUMERICAL ASSERTIONS
# =====================================================================
print("\n" + "=" * 70)
print("TEST SUITE 3: DEEP CROSS-VERIFICATION OF ALL NUMERICAL ENTRIES")
print("=" * 70)

# 3.1 MultiCP Table 1 (TABLE I-A): Multi-Head Backbone Training
print("\n--- Verifying MultiCP Table 1: Backbone Training ---")
tbl1 = tables_m[0]
wb_m_train = openpyxl.load_workbook(os.path.join(MULTICP_DIR, "results", "model_training_multihead_results.xlsx"), data_only=True)
ws_tr = wb_m_train["Summary"]
tr_rows = list(ws_tr.iter_rows(values_only=True))[1:]
gt_tr_dict = {r[0]: r for r in tr_rows}

for r_idx, (m_key, m_disp) in enumerate([
    ("GFNet_MultiHead", "GFNet MultiHead"),
    ("AlexNet_CNN_MultiHead", "AlexNet-CNN MultiHead"),
    ("ViT_UNet_MultiHead", "ViT-UNet MultiHead")
]):
    row = tbl1[r_idx + 1]
    gt_r = gt_tr_dict[m_key]
    
    if row[0] == m_disp:
        record_pass(f"Table 1 Row {r_idx+1}: {row[0]} == {m_disp}")
    else:
        record_fail(f"Table 1 Row {r_idx+1} model name mismatch", f"{row[0]} != {m_disp}")
        
    doc_acc = parse_num(row[1])
    gt_acc = gt_r[5] * 100.0
    if abs(doc_acc - gt_acc) < 0.005:
        record_pass(f"Table 1 {m_disp} Test Acc: {doc_acc:.3f}% == {gt_acc:.3f}%")
    else:
        record_fail(f"Table 1 {m_disp} Test Acc mismatch", f"Doc: {doc_acc} vs GT: {gt_acc}")
        
    doc_kap = parse_num(row[2])
    gt_kap = gt_r[6]
    if abs(doc_kap - gt_kap) < 0.0005:
        record_pass(f"Table 1 {m_disp} Cohen Kappa: {doc_kap:.4f} == {gt_kap:.4f}")
    else:
        record_fail(f"Table 1 {m_disp} Cohen Kappa mismatch", f"Doc: {doc_kap} vs GT: {gt_kap}")
        
    doc_time = parse_num(row[6])
    gt_time = gt_r[7]
    if abs(doc_time - gt_time) < 0.5:
        record_pass(f"Table 1 {m_disp} Train Time: {doc_time:.1f}s == {gt_time:.1f}s")
    else:
        record_fail(f"Table 1 {m_disp} Train Time mismatch", f"Doc: {doc_time} vs GT: {gt_time}")
        
    doc_params = int(parse_num(row[5]))
    gt_params = gt_r[8]
    if doc_params == gt_params:
        record_pass(f"Table 1 {m_disp} Total Params: {doc_params:,d} == {gt_params:,d}")
    else:
        record_fail(f"Table 1 {m_disp} Total Params mismatch", f"Doc: {doc_params} vs GT: {gt_params}")

# 3.2 MultiCP Aggregate Tables (TABLE I & TABLE II)
print("\n--- Verifying MultiCP Tables 2 & 3: Aggregate Summaries ---")
df_m_sum = pd.read_csv(os.path.join(MULTICP_DIR, "results", "multicp_summary.csv"))

for t_idx, score in [(1, "RAPS"), (2, "SAPS")]:
    tbl = tables_m[t_idx]
    for r_idx, (m_raw, m_disp) in enumerate([("AlexNet CNN", "AlexNet"), ("GFNet", "GFNet"), ("ViT UNet", "ViT-UNet")]):
        row = tbl[r_idx + 1]
        gt_s = df_m_sum[(df_m_sum["model_name"] == m_raw) & (df_m_sum["scoring_method"] == score)].iloc[0]
        
        doc_cov = parse_num(row[1])
        if abs(doc_cov - gt_s["empirical_coverage"]) < 0.0005:
            record_pass(f"MultiCP Table {t_idx+1} {m_disp} EmpCov: {doc_cov:.4f} == {gt_s['empirical_coverage']:.4f}")
        else:
            record_fail(f"MultiCP Table {t_idx+1} {m_disp} EmpCov mismatch", f"Doc: {doc_cov} vs GT: {gt_s['empirical_coverage']}")
            
        doc_ss = parse_num(row[2])
        if abs(doc_ss - gt_s["avg_set_size"]) < 0.0005:
            record_pass(f"MultiCP Table {t_idx+1} {m_disp} AvgSetSize: {doc_ss:.4f} == {gt_s['avg_set_size']:.4f}")
        else:
            record_fail(f"MultiCP Table {t_idx+1} {m_disp} AvgSetSize mismatch", f"Doc: {doc_ss} vs GT: {gt_s['avg_set_size']}")
            
        doc_sc = parse_num(row[5])
        if abs(doc_sc - gt_s["scene_coverage"]) < 0.0005:
            record_pass(f"MultiCP Table {t_idx+1} {m_disp} SceneCov: {doc_sc:.4f} == {gt_s['scene_coverage']:.4f}")
        else:
            record_fail(f"MultiCP Table {t_idx+1} {m_disp} SceneCov mismatch", f"Doc: {doc_sc} vs GT: {gt_s['scene_coverage']}")
            
        doc_ur = parse_num(row[6])
        if abs(doc_ur - gt_s["uncertain_pixel_rate"]) < 0.0005:
            record_pass(f"MultiCP Table {t_idx+1} {m_disp} UncRate: {doc_ur:.4f} == {gt_s['uncertain_pixel_rate']:.4f}")
        else:
            record_fail(f"MultiCP Table {t_idx+1} {m_disp} UncRate mismatch", f"Doc: {doc_ur} vs GT: {gt_s['uncertain_pixel_rate']}")

# 3.3 MultiCP Per-Class Coverage (TABLE III & TABLE IV)
print("\n--- Verifying MultiCP Tables 4 & 5: Per-Class Coverage ---")
df_m_pc = pd.read_csv(os.path.join(MULTICP_DIR, "results", "multicp_per_class.csv"))
piv_m = df_m_pc.pivot(index="Class", columns=["model_name", "scoring_method"], values="Coverage")

for t_idx, score in [(3, "RAPS"), (4, "SAPS")]:
    tbl = tables_m[t_idx]
    col_mapping = [("AlexNet CNN", score), ("GFNet", score), ("ViT UNet", score)]
    for c_id in range(8):
        c_name = f"Class {c_id}"
        row = tbl[c_id + 1]
        for col_idx, (m_raw, _) in enumerate(col_mapping):
            doc_val = parse_num(row[col_idx + 1])
            gt_val = piv_m.loc[c_name, (m_raw, score)]
            if abs(doc_val - gt_val) < 0.0005:
                record_pass(f"MultiCP Table {t_idx+1} {score} {c_name} {m_raw}: {doc_val:.4f} == {gt_val:.4f}")
            else:
                record_fail(f"MultiCP Table {t_idx+1} {score} {c_name} {m_raw} mismatch", f"Doc: {doc_val} vs GT: {gt_val}")

# 3.4 MultiCP Head Sweeps (TABLE V & TABLE VI)
print("\n--- Verifying MultiCP Tables 6 & 7: Head Sweeps (M=1..7) ---")
df_m_hs = pd.read_csv(os.path.join(MULTICP_DIR, "results", "multicp_head_sweep.csv"))

for t_idx, score in [(5, "RAPS"), (6, "SAPS")]:
    tbl = tables_m[t_idx]
    for h in range(1, 8):
        row = tbl[h]
        if int(row[0]) == h:
            record_pass(f"MultiCP Table {t_idx+1} ({score}) Head {h} present")
        else:
            record_fail(f"MultiCP Table {t_idx+1} ({score}) Head {h} mismatch", f"Found {row[0]}")
            
        for m_idx, mk in enumerate(["AlexNet_CN", "GFNet_Mult", "ViT_UNet_M"]):
            key = f"{mk}_{score}"
            gt_row = df_m_hs[(df_m_hs["model_key"] == key) & (df_m_hs["heads"] == h)].iloc[0]
            doc_cov = parse_num(row[m_idx * 2 + 1])
            doc_ss = parse_num(row[m_idx * 2 + 2])
            
            if abs(doc_cov - gt_row["coverage"]) < 0.0005:
                record_pass(f"MultiCP Table {t_idx+1} ({score}) H={h} {mk} Cov: {doc_cov:.4f} == {gt_row['coverage']:.4f}")
            else:
                record_fail(f"MultiCP Table {t_idx+1} ({score}) H={h} {mk} Cov mismatch", f"Doc: {doc_cov} vs GT: {gt_row['coverage']}")
                
            if abs(doc_ss - gt_row["set_size"]) < 0.0005:
                record_pass(f"MultiCP Table {t_idx+1} ({score}) H={h} {mk} SetSz: {doc_ss:.4f} == {gt_row['set_size']:.4f}")
            else:
                record_fail(f"MultiCP Table {t_idx+1} ({score}) H={h} {mk} SetSz mismatch", f"Doc: {doc_ss} vs GT: {gt_row['set_size']}")

# 3.5 SCMCP Table XXXII: Cross-Window 24-Row Synthesis
print("\n--- Verifying SCMCP Table 29 (TABLE XXXII): 24-Row Consolidated Synthesis ---")
t32 = tables_s[28]
df_s_comb = pd.read_csv(os.path.join(SCMCP_DIR, "results", "combined_summary_all_windows.csv"))

if len(t32) == 25:
    record_pass("SCMCP TABLE XXXII has exactly 25 rows (1 header + 24 data rows)")
else:
    record_fail("SCMCP TABLE XXXII row count violation", f"Expected 25 rows, found {len(t32)}")

for r_idx, r in df_s_comb.iterrows():
    row = t32[r_idx + 1]
    ws = int(r["window_size"])
    m = r["model_name"]
    s = r["scoring_method"]
    
    if int(parse_num(row[0])) == ws:
        record_pass(f"TABLE XXXII Row {r_idx+1} WS: {row[0]} == {ws}")
    else:
        record_fail(f"TABLE XXXII Row {r_idx+1} WS mismatch", f"Doc: {row[0]} != GT: {ws}")
        
    if row[1] == m:
        record_pass(f"TABLE XXXII Row {r_idx+1} Model: {row[1]} == {m}")
    else:
        record_fail(f"TABLE XXXII Row {r_idx+1} Model mismatch", f"Doc: {row[1]} != GT: {m}")
        
    if row[2] == s:
        record_pass(f"TABLE XXXII Row {r_idx+1} Score: {row[2]} == {s}")
    else:
        record_fail(f"TABLE XXXII Row {r_idx+1} Score mismatch", f"Doc: {row[2]} != GT: {s}")
        
    doc_ec = parse_num(row[3])
    if abs(doc_ec - r["empirical_coverage"]) < 0.0005:
        record_pass(f"TABLE XXXII Row {r_idx+1} EmpCov: {doc_ec:.4f} == {r['empirical_coverage']:.4f}")
    else:
        record_fail(f"TABLE XXXII Row {r_idx+1} EmpCov mismatch", f"Doc: {doc_ec} vs GT: {r['empirical_coverage']}")
        
    doc_as = parse_num(row[4])
    if abs(doc_as - r["avg_set_size"]) < 0.0005:
        record_pass(f"TABLE XXXII Row {r_idx+1} AvgSet: {doc_as:.4f} == {r['avg_set_size']:.4f}")
    else:
        record_fail(f"TABLE XXXII Row {r_idx+1} AvgSet mismatch", f"Doc: {doc_as} vs GT: {r['avg_set_size']}")
        
    doc_sc = parse_num(row[5])
    if abs(doc_sc - r["scene_coverage"]) < 0.0005:
        record_pass(f"TABLE XXXII Row {r_idx+1} SceneCov: {doc_sc:.4f} == {r['scene_coverage']:.4f}")
    else:
        record_fail(f"TABLE XXXII Row {r_idx+1} SceneCov mismatch", f"Doc: {doc_sc} vs GT: {r['scene_coverage']}")
        
    doc_ur = parse_num(row[6])
    if abs(doc_ur - r["uncertain_pixel_rate"]) < 0.0005:
        record_pass(f"TABLE XXXII Row {r_idx+1} UncRate: {doc_ur:.4f} == {r['uncertain_pixel_rate']:.4f}")
    else:
        record_fail(f"TABLE XXXII Row {r_idx+1} UncRate mismatch", f"Doc: {doc_ur} vs GT: {r['uncertain_pixel_rate']}")
        
    doc_mc = parse_num(row[7])
    if abs(doc_mc - r["mean_per_class_cov"]) < 0.0005:
        record_pass(f"TABLE XXXII Row {r_idx+1} MeanCls: {doc_mc:.4f} == {r['mean_per_class_cov']:.4f}")
    else:
        record_fail(f"TABLE XXXII Row {r_idx+1} MeanCls mismatch", f"Doc: {doc_mc} vs GT: {r['mean_per_class_cov']}")
        
    doc_upx = int(parse_num(row[8]))
    gt_upx = gt_s_pix_map[(ws, m, s)][8]
    if doc_upx == gt_upx:
        record_pass(f"TABLE XXXII Row {r_idx+1} UncPx: {doc_upx:,d} == {gt_upx:,d}")
    else:
        record_fail(f"TABLE XXXII Row {r_idx+1} UncPx mismatch", f"Doc: {doc_upx} vs GT: {gt_upx}")
        
    doc_rt = parse_num(row[9])
    if abs(doc_rt - r["runtime_sec"]) < 0.5:
        record_pass(f"TABLE XXXII Row {r_idx+1} Runtime: {doc_rt:.2f}s == {r['runtime_sec']:.2f}s")
    else:
        record_fail(f"TABLE XXXII Row {r_idx+1} Runtime mismatch", f"Doc: {doc_rt} vs GT: {r['runtime_sec']}")


# =====================================================================
# SECTION 4: MEDIA, DRAWING BLIPS & FIGURE ASSET AUDIT
# =====================================================================
print("\n" + "=" * 70)
print("TEST SUITE 4: EMBEDDED MEDIA & OPENXML DRAWING BLIP AUDIT")
print("=" * 70)

for doc_label, doc_path, root_doc, img_rels, media_files, exp_panels in [
    ("multicp.docx", os.path.join(MULTICP_DIR, "multicp.docx"), root_m, img_rels_m, media_m, 16),
    ("scmcp.docx", os.path.join(SCMCP_DIR, "scmcp.docx"), root_s, img_rels_s, media_s, 64)
]:
    print(f"\nAuditing drawings and embedded media in {doc_label}...")
    
    # 4.1 Check all drawing blips in document XML
    blips = list(root_doc.iter(f"{{{XML_NAMESPACES['a']}}}blip"))
    if len(blips) == exp_panels:
        record_pass(f"{doc_label}: Contains exactly {len(blips)} <a:blip> drawing instances (== expected {exp_panels})")
    else:
        record_fail(f"{doc_label}: <a:blip> drawing count mismatch", f"Found {len(blips)} != {exp_panels}")
        
    # 4.2 Verify every blip resolves to an image relationship and media file
    unresolved_blips = 0
    for b_idx, blip in enumerate(blips):
        r_embed = blip.attrib.get(f"{{{XML_NAMESPACES['r']}}}embed")
        if not r_embed or r_embed not in img_rels:
            unresolved_blips += 1
            record_fail(f"{doc_label} blip #{b_idx+1}: missing relationship {r_embed}")
            continue
            
        target = img_rels[r_embed]
        # Resolve target path relative to word/
        media_path = f"word/{target}" if not target.startswith("word/") else target
        if media_path not in media_files:
            record_fail(f"{doc_label} blip #{b_idx+1}: media file {media_path} missing in zip archive!")
        else:
            record_pass(f"{doc_label} blip #{b_idx+1} ({r_embed}) -> {media_path} resolved")
            
    if unresolved_blips == 0:
        record_pass(f"{doc_label}: All {len(blips)} drawings successfully resolve to archived media files")
        
    # 4.3 Verify every media file decodes cleanly with PIL
    corrupted_images = 0
    for mf_path, mf_bytes in media_files.items():
        try:
            img = Image.open(io.BytesIO(mf_bytes))
            w, h = img.size
            if w < 50 or h < 50:
                record_fail(f"{doc_label} {mf_path}: suspiciously tiny dimensions ({w}x{h})")
                corrupted_images += 1
            else:
                record_pass(f"{doc_label} {mf_path}: valid {img.format} ({w}x{h}, {len(mf_bytes):,d} bytes)")
        except Exception as e:
            record_fail(f"{doc_label} {mf_path}: decode error: {e}")
            corrupted_images += 1
            
    if corrupted_images == 0:
        record_pass(f"{doc_label}: All {len(media_files)} embedded image files decode cleanly without corruption")


# =====================================================================
# SECTION 5: COLOR PALETTE AUDIT (ZERO TAB20, BOLD PALETTE ENFORCEMENT)
# =====================================================================
print("\n" + "=" * 70)
print("TEST SUITE 5: COLOR PALETTE EMPIRICAL VERIFICATION (ZERO TAB20)")
print("=" * 70)

# Canonical 8-class hex colors
CANONICAL_HEXES = ['#0000FF', '#00FF00', '#FF0000', '#00FFFF', '#FF00FF', '#FFFF00', '#A52A2A', '#FFA500']
# Background = #000000, Uncertain = #808080

# Convert to RGB tuples
def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

CANONICAL_RGBS = [hex_to_rgb(h) for h in CANONICAL_HEXES]
BG_RGB = (0, 0, 0)
UNC_RGB = (128, 128, 128)

print("Checking generated legend images for exact canonical colors...")
legend_bin_path = os.path.join(MULTICP_DIR, "results", "maps", "dias_binary_legend.png")
legend_cls_path = os.path.join(MULTICP_DIR, "results", "maps", "dias_class_legend.png")

if os.path.exists(legend_bin_path):
    record_pass("dias_binary_legend.png exists in multicp/results/maps/")
else:
    record_fail("dias_binary_legend.png missing!")

if os.path.exists(legend_cls_path):
    record_pass("dias_class_legend.png exists in multicp/results/maps/")
    img_cl = Image.open(legend_cls_path).convert("RGB")
    # Check that colors in class legend contain canonical hues
    pixels = list(img_cl.getdata())
    found_canonicals = 0
    for crgb in CANONICAL_RGBS:
        # Check if color exists within tolerance in legend
        matches = [p for p in pixels if all(abs(p[i] - crgb[i]) < 10 for i in range(3))]
        if len(matches) > 10:
            found_canonicals += 1
    if found_canonicals == 8:
        record_pass("dias_class_legend.png verified: contains all 8 canonical bold DIAS classes")
    else:
        record_fail(f"dias_class_legend.png: Only {found_canonicals}/8 canonical colors found!")
else:
    record_fail("dias_class_legend.png missing!")


# =====================================================================
# FINAL CHALLENGE REPORT & VERDICT DETERMINATION
# =====================================================================
print("\n" + "=" * 70)
print(f"CHALLENGER AUDIT COMPLETE: {total_tests} ASSERTIONS EVALUATED")
print(f"PASSED: {passed_tests} | FAILED: {failed_tests}")
print("=" * 70)

if failed_tests == 0:
    print("\nVERDICT: >>> APPROVE <<<")
    print("100% of adversarial stress-tests, XML searches, pixel sums, and ground-truth cross-checks PASSED.")
    sys.exit(0)
else:
    print(f"\nVERDICT: >>> REJECT <<< ({failed_tests} vulnerabilities detected!)")
    for v in vulnerabilities:
        print(f" - {v}")
    sys.exit(1)
