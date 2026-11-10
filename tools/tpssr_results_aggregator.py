#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate accepted TPSSR experiments into a traceable Excel workbook.

Python 3.9+, standard library only. Run from the project root after all tests.
Usage: python tpssr_results_aggregator.py --results-root results --output TPSSR_FINAL_RESULTS.xlsx

No historical TPSSR variants, previous RF, PCA-enhanced or test-result caches are imported.
Missing/incompatible inputs are reported, never silently substituted.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import math
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

# -----------------------------------------------------------------------------
# FINAL, EXPLICIT experiment selection. Edit ONLY if real output folder names differ.
# -----------------------------------------------------------------------------
FINAL_TPSSR_DIR = 'tpssr_learned_router3_lineaware_rankcv__minorityhybrid10pct__robustdonor'
FINAL_TPSSR_GAP_DIR = ('tpssr_learned_router3_lineaware_rankcv'
                       '__minorityhybridgap10pp_two10pct__robustdonor__lazycachev1')
TPSSR_FOLDER_BY_SPLIT = {
    'SWIR_8x2': FINAL_TPSSR_GAP_DIR,
    'VNIR_8x2': FINAL_TPSSR_GAP_DIR,
    'SWIR_A_to_B': FINAL_TPSSR_GAP_DIR,
    'SWIR_B_to_A': FINAL_TPSSR_GAP_DIR,
}
SPLITS = (
    ('SWIR_8x2', 'swir__train_1-2-3-4-6-7-8-9__test_5-10',
     ('SWIR5','SWIR10'), ('TPSSR','Spectral','Spatial','Spatio-spectral','PCA','RF','DDS2M','PFGIN'), 100),
    ('VNIR_8x2', 'vnir__train_1-2-3-4-6-7-8-9__test_5-10',
     ('VNIR5','VNIR10'), ('TPSSR','Spectral'), 100),
    ('SWIR_A_to_B', 'swir__train_1-2-3-4-5__test_6-7-8-9-10',
     ('SWIR6','SWIR7','SWIR8','SWIR9','SWIR10'), ('TPSSR','Spectral'), 40),
    ('SWIR_B_to_A', 'swir__train_6-7-8-9-10__test_1-2-3-4-5',
     ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR5'), ('TPSSR','Spectral'), 40),
)
METHOD_SOURCE = {
    'TPSSR': (FINAL_TPSSR_DIR, 'test_metrics.csv'),
    'Spectral': ('spectral','metrics.csv'),
    'Spatial': ('spatial','metrics.csv'),
    'Spatio-spectral': ('spatio_spectral','metrics.csv'),
    'PCA': ('pca','metrics.csv'),
    'RF': ('random_forest_multiscale','metrics.csv'),
    'DDS2M': ('dds2m_t3000_sp1500','metrics.csv'),
    'PFGIN': ('pfgin','metrics.csv'),
}
EXPECTED_ROUTER = 'learned_router3_lineaware_rankcv'
NA_TOKENS = {'','na','nan','none','null','inf','-inf','infinity','-infinity'}
METRICS = ('rmse','mae','r2','sam_deg')


def as_int(x, default=None):
    try:
        if str(x).strip().lower() in NA_TOKENS: return default
        return int(float(str(x).strip()))
    except (ValueError,TypeError,OverflowError): return default


def as_float(x):
    try:
        z=float(str(x).strip())
        return z if math.isfinite(z) else None
    except (ValueError,TypeError): return None


def truthy(x):
    return str(x).strip().lower() in ('1','true','yes','on')


def file_digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def metrics_mean(rows, key):
    vals=[r[key] for r in rows if r.get(key) is not None]
    return sum(vals)/len(vals) if vals else None


def summary(rows, expected=None):
    valid=[r for r in rows if r['valid_case']]
    full=[r for r in valid if r['full_coverage']]
    target=sum(r['target_n'] or 0 for r in rows)
    done=sum(r['valid_n'] or 0 for r in rows)
    rmse_parts=[(r['rmse'],r['valid_n']) for r in valid if r['valid_n'] and r['rmse'] is not None]
    pooled_rmse=(math.sqrt(sum(a*a*b for a,b in rmse_parts)/sum(b for a,b in rmse_parts))
                 if rmse_parts else None)
    return {
        'cases':len(rows), 'expected':expected, 'valid_cases':len(valid),
        'full_cases':len(full), 'partial_cases':len(valid)-len(full),
        'invalid_cases':len(rows)-len(valid),
        'coverage':done/target if target else None,
        'mean_rmse':metrics_mean(valid,'rmse'), 'pooled_rmse':pooled_rmse,
        'mean_mae':metrics_mean(valid,'mae'), 'mean_r2':metrics_mean(valid,'r2'),
        'mean_sam_deg':metrics_mean(valid,'sam_deg'),
        'full_mean_rmse':metrics_mean(full,'rmse'),
        'full_mean_mae':metrics_mean(full,'mae'),
        'full_mean_r2':metrics_mean(full,'r2'),
        'full_mean_sam_deg':metrics_mean(full,'sam_deg'),
    }


def strict_variant_check(raw, method, rel_path):
    """Fail closed for TPSSR; do not infer version from modification times."""
    if method != 'TPSSR':
        # Other benchmarks have no version marker in their metrics CSV. Filename
        # selection is explicit; report RF checkpoint provenance as a warning.
        if method == 'RF' and raw.get('method','').strip() != 'random_forest_multiscale':
            return 'RF method field is not random_forest_multiscale'
        return None
    required=('router_variant','fusion_donor_match',
              'candidate_spatial_consistency_enabled','candidate_fusion_guard_enabled',
              'candidate_global_minority_enabled','candidate_global_minority_threshold_percent',
              'candidate_global_minority_policy')
    missing=[field for field in required if field not in raw]
    if missing:
        return 'TPSSR result has no final-config metadata: '+','.join(missing)
    if raw['router_variant'].strip()!=EXPECTED_ROUTER:
        return 'Wrong TPSSR learned router'
    if raw['fusion_donor_match'].strip().lower() != 'robust':
        return 'Robust Donor is not enabled'
    if truthy(raw['candidate_spatial_consistency_enabled']):
        return 'Spatial Consistency must be OFF'
    if truthy(raw['candidate_fusion_guard_enabled']):
        return 'Fusion Guard must be OFF'
    if not truthy(raw['candidate_global_minority_enabled']):
        return 'Global Minority must be enabled'
    expected_policy = ('hybrid' if FINAL_TPSSR_DIR in str(rel_path) else 'hybrid_gap')
    if raw['candidate_global_minority_policy'].strip().lower() != expected_policy:
        return 'Wrong TPSSR minority policy: expected ' + expected_policy
    if expected_policy == 'hybrid_gap':
        gap = as_float(raw.get('candidate_global_minority_three_expert_gap_percent'))
        if gap is None or abs(gap - 10.0) > 1e-6:
            return 'Hybrid gap policy requires exactly 10 percentage points'
    percent = as_float(raw['candidate_global_minority_threshold_percent'])
    if percent is None or abs(percent - 10.0) > 1e-6:
        return 'Global Minority threshold must be exactly 10 percent'
    if raw.get('status','ok').strip().lower() not in ('ok','success','completed'):
        return 'TPSSR status is not ok'
    return None


def read_one_file(path, split, core, method, root):
    rows=[]; issues=[]; fieldnames=[]
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            rd=csv.DictReader(f)
            fieldnames=list(rd.fieldnames or [])
            required=('scenario','severity','repeat','rmse')
            if any(x not in fieldnames for x in required):
                return [], [f'Missing metric columns: {required}'], fieldnames
            for lineno,raw in enumerate(rd,2):
                if not raw or all(not str(v).strip() for v in raw.values() if v is not None): continue
                scenario=(raw.get('scenario') or '').strip()
                severity=as_int(raw.get('severity'))
                repeat=as_int(raw.get('repeat'))
                if not scenario or severity is None or repeat is None:
                    issues.append(f'Row {lineno}: missing scenario/severity/repeat')
                    continue
                error=strict_variant_check(raw,method,path)
                if error:
                    issues.append(f'Row {lineno}: {error}')
                    return [], issues, fieldnames
                target=as_int(raw.get('target_voxels',raw.get('target_n')))
                valid=as_int(raw.get('filled_voxels',raw.get('valid_n')))
                coverage=as_float(raw.get('coverage'))
                if coverage is None and target and valid is not None: coverage=valid/target
                measures={k:as_float(raw.get(k)) for k in METRICS}
                good=measures['rmse'] is not None and (coverage is None or coverage>0) and (valid is None or valid>0)
                if method=='TPSSR' and target is None: issues.append(f'Row {lineno}: missing TPSSR target_voxels')
                x={
                    'split':split,'dataset':core,'method':method,
                    'scenario':scenario,'severity':severity,'repeat':repeat,
                    'mask_file':raw.get('mask_file','') or f'{scenario}__s{severity:02d}__r{repeat:02d}.npz',
                    'target_n':target,'valid_n':valid,'coverage':coverage,
                    **measures,
                    'valid_case':good, 'full_coverage':good and coverage is not None and coverage>=1-1e-6,
                    'status':raw.get('status','') or ('ok' if good else 'invalid'),
                    'router_variant':raw.get('router_variant',''),
                    'fusion_donor_match':raw.get('fusion_donor_match',''),
                    'spatial_strength':as_float(raw.get('candidate_spatial_consistency_strength')),
                    'minority_percent':as_float(raw.get('candidate_global_minority_threshold_percent')),
                    'minority_changed_pixels':as_int(raw.get('candidate_global_minority_changed_pixels')),
                    'fusion_guard_on':raw.get('candidate_fusion_guard_enabled',''),
                    'eval_protocol':raw.get('eval_protocol',''),
                    'source':str(path.relative_to(root)),
                }
                rows.append(x)
    except (OSError, UnicodeError, csv.Error) as exc:
        issues.append(f'Cannot read CSV: {exc}')
    duplicates=[k for k,n in Counter((r['scenario'],r['severity'],r['repeat']) for r in rows).items() if n>1]
    if duplicates:
        issues.append(f'Duplicate mask identities in file ({len(duplicates)}): '+str(duplicates[:5]))
        return [], issues, fieldnames
    return rows,issues,fieldnames


def pairing(rows, methods=None):
    look=defaultdict(dict)
    for r in rows:
        if methods and r['method'] not in methods: continue
        key=(r['split'],r['dataset'],r['scenario'],r['severity'],r['repeat'])
        look[key][r['method']]=r
    return look


def norm_s(s):
    s=str(s or '')
    return ''.join(c if ord(c)>=32 or c in '\n\t' else ' ' for c in s)

# XLSX generation uses only Python standard library (no packages to install on a GPU server).
# This is a write-once OOXML report, with no modification of existing workbooks.
S_NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
R_NS='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL_NS='http://schemas.openxmlformats.org/package/2006/relationships'
CT_NS='http://schemas.openxmlformats.org/package/2006/content-types'
ET.register_namespace('',S_NS)
ET.register_namespace('r',R_NS)

def q(tag): return '{'+S_NS+'}'+tag

def col_name(n):
    o=''
    while n:
        n,r=divmod(n-1,26);o=chr(65+r)+o
    return o


def cell(root, r, c, val, style=0):
    if val is None: return
    name=col_name(c)+str(r)
    attr={'r':name,'s':str(style)}
    if isinstance(val,bool):
        ET.SubElement(root,q('c'),{**attr,'t':'b'}).append(ET.Element(q('v')))
        root[-1][0].text='1' if val else '0'
    elif isinstance(val,(int,float)) and (not isinstance(val,float) or math.isfinite(val)):
        e=ET.SubElement(root,q('c'),attr)
        ET.SubElement(e,q('v')).text=str(val)
    else:
        e=ET.SubElement(root,q('c'),{**attr,'t':'inlineStr'})
        si=ET.SubElement(e,q('is'))
        t=ET.SubElement(si,q('t'))
        t.text=norm_s(val)
        if t.text and (t.text[0].isspace() or t.text[-1].isspace()):
            t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')


def styles_bytes():
    root=ET.Element(q('styleSheet'))
    fmt=ET.SubElement(root,q('numFmts'),{'count':'3'})
    ET.SubElement(fmt,q('numFmt'),{'numFmtId':'164','formatCode':'0.000000'})
    ET.SubElement(fmt,q('numFmt'),{'numFmtId':'165','formatCode':'0.0%'})
    ET.SubElement(fmt,q('numFmt'),{'numFmtId':'166','formatCode':'0.0000'})
    fonts=ET.SubElement(root,q('fonts'),{'count':'3'})
    for bold,colour in [(False,'263445'),(True,'FFFFFF'),(True,'17375E')]:
        f=ET.SubElement(fonts,q('font'))
        ET.SubElement(f,q('sz'),{'val':'10'})
        if bold:ET.SubElement(f,q('b'))
        ET.SubElement(f,q('color'),{'rgb':'FF'+colour})
        ET.SubElement(f,q('name'),{'val':'Aptos'})
    fills=ET.SubElement(root,q('fills'),{'count':'4'})
    for kind,colour in [('none',None),('gray125',None),('solid','17375E'),('solid','EAF2FA')]:
        a=ET.SubElement(fills,q('fill'))
        p=ET.SubElement(a,q('patternFill'),{'patternType':kind})
        if colour:ET.SubElement(p,q('fgColor'),{'rgb':'FF'+colour})
    borders=ET.SubElement(root,q('borders'),{'count':'1'})
    b=ET.SubElement(borders,q('border'))
    for k in ('left','right','top','bottom','diagonal'): ET.SubElement(b,q(k))
    ET.SubElement(root,q('cellStyleXfs'),{'count':'1'}).append(ET.Element(q('xf'),{'numFmtId':'0','fontId':'0','fillId':'0','borderId':'0'}))
    xfs=ET.SubElement(root,q('cellXfs'),{'count':'7'})
    # base, header, title, metric six decimals, count integer, percent, metric 4 decimals
    for font,fill,num in [(0,0,0),(1,2,0),(2,3,0),(0,0,164),(0,0,1),(0,0,165),(0,0,166)]:
        ET.SubElement(xfs,q('xf'),{'numFmtId':str(num),'fontId':str(font),'fillId':str(fill),
                                      'borderId':'0','xfId':'0','applyNumberFormat':'1'})
    ET.SubElement(root,q('cellStyles'),{'count':'1'}).append(ET.Element(q('cellStyle'),{'name':'Normal','xfId':'0','builtinId':'0'}))
    return ET.tostring(root,encoding='utf-8',xml_declaration=True)


def sheet_xml(name,headers,data,subtitle):
    root=ET.Element(q('worksheet'))
    views=ET.SubElement(root,q('sheetViews'))
    v=ET.SubElement(views,q('sheetView'),{'workbookViewId':'0','showGridLines':'0'})
    ET.SubElement(v,q('pane'),{'ySplit':'4','topLeftCell':'A5','activePane':'bottomLeft','state':'frozen'})
    ET.SubElement(root,q('sheetFormatPr'),{'defaultRowHeight':'16'})
    cols=ET.SubElement(root,q('cols'))
    for i,h in enumerate(headers,1):
        hint=47 if any(t in h.lower() for t in ('path','source','sha256','note','issue','filename','details')) else 33 if h.lower() in ('method','scenario','experiment','split','dataset','status','topic') else 16
        ET.SubElement(cols,q('col'),{'min':str(i),'max':str(i),'width':str(hint),'customWidth':'1'})
    sd=ET.SubElement(root,q('sheetData'))
    r=ET.SubElement(sd,q('row'),{'r':'1','ht':'28','customHeight':'1'})
    cell(r,1,1,name,2)
    r=ET.SubElement(sd,q('row'),{'r':'2'})
    cell(r,2,1,subtitle,0)
    r=ET.SubElement(sd,q('row'),{'r':'4','ht':'23','customHeight':'1'})
    for c,h in enumerate(headers,1):cell(r,4,c,h,1)
    for index,record in enumerate(data,5):
        row=ET.SubElement(sd,q('row'),{'r':str(index)})
        for c,(h,v) in enumerate(zip(headers,record),1):
            if isinstance(v,bool):style=4
            elif isinstance(v,(int,float)) and not isinstance(v,bool):
                style=5 if 'coverage' in h.lower() or h.lower().endswith('_pct') else (4 if isinstance(v,int) else 3)
            else: style=0
            cell(row,index,c,v,style)
    if headers:
        ET.SubElement(root,q('autoFilter'),{'ref':f'A4:{col_name(len(headers))}{max(4,len(data)+4)}'})
    ET.SubElement(root,q('pageMargins'),{'left':'0.4','right':'0.4','top':'0.5','bottom':'0.5','header':'0.2','footer':'0.2'})
    return ET.tostring(root,encoding='utf-8',xml_declaration=True)


def write_xlsx(path,sheets):
    types=ET.Element('{'+CT_NS+'}Types')
    ET.SubElement(types,'{'+CT_NS+'}Default',{'Extension':'rels','ContentType':'application/vnd.openxmlformats-package.relationships+xml'})
    ET.SubElement(types,'{'+CT_NS+'}Default',{'Extension':'xml','ContentType':'application/xml'})
    ET.SubElement(types,'{'+CT_NS+'}Override',{'PartName':'/xl/workbook.xml','ContentType':'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml'})
    ET.SubElement(types,'{'+CT_NS+'}Override',{'PartName':'/xl/styles.xml','ContentType':'application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml'})
    for i in range(1,len(sheets)+1):
        ET.SubElement(types,'{'+CT_NS+'}Override',{'PartName':f'/xl/worksheets/sheet{i}.xml',
           'ContentType':'application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml'})
    rels=ET.Element('{'+REL_NS+'}Relationships')
    ET.SubElement(rels,'{'+REL_NS+'}Relationship',{'Id':'rId1','Type':R_NS+'/officeDocument','Target':'xl/workbook.xml'})
    book=ET.Element(q('workbook'))
    ls=ET.SubElement(book,q('sheets'))
    for i,(name,_,_,_) in enumerate(sheets,1):
        ET.SubElement(ls,q('sheet'),{'name':name,'sheetId':str(i),'{'+R_NS+'}id':f'rId{i}'})
    relbook=ET.Element('{'+REL_NS+'}Relationships')
    for i in range(1,len(sheets)+1):
        ET.SubElement(relbook,'{'+REL_NS+'}Relationship',{'Id':f'rId{i}','Type':R_NS+'/worksheet','Target':f'worksheets/sheet{i}.xml'})
    ET.SubElement(relbook,'{'+REL_NS+'}Relationship',{'Id':f'rId{len(sheets)+1}','Type':R_NS+'/styles','Target':'styles.xml'})
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        def put(name,data):z.writestr(name,data)
        def xml(e,ns):
            ET.register_namespace('',ns)
            return ET.tostring(e,encoding='utf-8',xml_declaration=True)
        put('[Content_Types].xml',xml(types,CT_NS))
        put('_rels/.rels',xml(rels,REL_NS))
        put('xl/workbook.xml',xml(book,S_NS))
        put('xl/_rels/workbook.xml.rels',xml(relbook,REL_NS))
        ET.register_namespace('',S_NS)
        put('xl/styles.xml',styles_bytes())
        for i,(name,header,rows,subtitle) in enumerate(sheets,1):
            put(f'xl/worksheets/sheet{i}.xml',sheet_xml(name,header,rows,subtitle))


def main():
    p=argparse.ArgumentParser(description='Read final TPSSR/benchmark results and export a traceable Excel workbook.')
    p.add_argument('--results-root',type=Path,default=Path('results'))
    p.add_argument('--output',type=Path,default=Path('TPSSR_FINAL_ALL_RESULTS.xlsx'))
    p.add_argument('--allow-incomplete',action='store_true',help='Export a draft even if required files are missing')
    args=p.parse_args()
    root=args.results_root.expanduser().resolve()
    if not root.is_dir():
        p.error('Results directory does not exist: '+str(root))
    all_rows=[]; audit=[]; sources=[]; fatal=[]
    for split,ex,cores,methods,n_per_core in SPLITS:
        for dataset in cores:
            for method in methods:
                folder,file=METHOD_SOURCE[method]
                if method == 'TPSSR':
                    folder = TPSSR_FOLDER_BY_SPLIT[split]
                f=root/ex/dataset/folder/file
                rel=f.relative_to(root)
                if not f.is_file():
                    message=f'MISSING expected file: {rel}'
                    audit.append([split,dataset,method,'MISSING',0,n_per_core,message])
                    fatal.append(message)
                    print('! '+message,flush=True)
                    continue
                rows,problems,fields=read_one_file(f,split,dataset,method,root)
                issues=' | '.join(problems)
                if problems or not rows:
                    fatal.append(f'{split}/{dataset}/{method}: {issues or "Empty file"}')
                if method=='TPSSR' and len(rows)!=n_per_core:
                    fatal.append(f'{split}/{dataset}/TPSSR must have {n_per_core} cases, found {len(rows)}')
                    problems.append('TPSSR mask count does not match expected')
                if len(rows)>n_per_core:
                    fatal.append(f'{split}/{dataset}/{method} has unexpected {len(rows)} > {n_per_core} cases')
                if method!='TPSSR' and len(rows)<n_per_core:
                    problems.append(f'Only {len(rows)}/{n_per_core} cases present: compare with TPSSR using shared full masks')
                all_rows.extend(rows)
                status='ERROR' if any('does not match' in z or 'Duplicate' in z or 'Wrong' in z or 'Cannot' in z for z in problems) or not rows else ('WARNING' if problems else 'OK')
                audit.append([split,dataset,method,status,len(rows),n_per_core,' | '.join(problems)])
                sources.append([split,dataset,method,str(rel),f.stat().st_size,
                                dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec='seconds'),
                                file_digest(f),len(rows),status])
    # TPSSR key sets are ground truth for expected fixed masks per split/core.
    groups=defaultdict(list)
    for r in all_rows:groups[(r['split'],r['dataset'],r['method'])].append(r)
    keys=defaultdict(set)
    for r in all_rows:
        if r['method']=='TPSSR':keys[(r['split'],r['dataset'])].add((r['scenario'],r['severity'],r['repeat']))
    for split,ex,cores,methods,n in SPLITS:
        for c in cores:
            for m in methods:
                theirs={(r['scenario'],r['severity'],r['repeat']) for r in groups[(split,c,m)]}
                reference=keys.get((split,c))
                if reference:
                    extra=theirs-reference
                    if extra:
                        audit.append([split,c,m,'ERROR',len(extra),0,'Masks not present in TPSSR: '+str(sorted(extra)[:6])]);fatal.append(f'Mask mismatch {split}/{c}/{m}')
                    missing=reference-theirs
                    if missing: audit.append([split,c,m,'WARNING',len(missing),n,'Missing mask identities versus TPSSR (first 6): '+str(sorted(missing)[:6])])

    overall=[]; per_image=[]; by_scenario=[]; by_severity=[]
    for split,ex,cores,methods,n in SPLITS:
        for m in methods:
            entire=[r for r in all_rows if r['split']==split and r['method']==m]
            if not entire:continue
            agg=summary(entire,len(cores)*n)
            overall.append([split,m,len(cores),agg['expected'],agg['cases'],agg['valid_cases'],agg['full_cases'],agg['partial_cases'],agg['invalid_cases'],agg['coverage'],agg['mean_rmse'],agg['pooled_rmse'],agg['mean_mae'],agg['mean_r2'],agg['mean_sam_deg'],agg['full_mean_rmse']])
            for c in cores:
                case=groups[(split,c,m)]
                if case:
                    a=summary(case,n)
                    per_image.append([split,c,m,n,a['cases'],a['full_cases'],a['partial_cases'],a['coverage'],a['mean_rmse'],a['pooled_rmse'],a['mean_mae'],a['mean_r2'],a['mean_sam_deg']])
            for s in sorted({r['scenario'] for r in entire}):
                case=[r for r in entire if r['scenario']==s]
                a=summary(case)
                by_scenario.append([split,m,s,a['cases'],a['valid_cases'],a['full_cases'],a['coverage'],a['mean_rmse'],a['pooled_rmse'],a['mean_mae'],a['mean_r2'],a['mean_sam_deg']])
                for severity in sorted({r['severity'] for r in case}):
                    grp=[r for r in case if r['severity']==severity]
                    v=summary(grp)
                    by_severity.append([split,m,s,severity,v['cases'],v['valid_cases'],v['full_cases'],v['coverage'],v['mean_rmse'],v['pooled_rmse'],v['mean_mae'],v['mean_r2'],v['mean_sam_deg']])
    per_mask=[]
    for r in sorted(all_rows,key=lambda v:(v['split'],v['dataset'],v['scenario'],v['severity'],v['repeat'],v['method'])):
        per_mask.append([r['split'],r['dataset'],r['method'],r['scenario'],r['severity'],r['repeat'],r['mask_file'],r['target_n'],r['valid_n'],r['coverage'],r['rmse'],r['mae'],r['r2'],r['sam_deg'],int(r['full_coverage']),r['status'],r['router_variant'],r['fusion_donor_match'],r['spatial_strength'],r['minority_percent'],r['minority_changed_pixels'],r['fusion_guard_on'],r['source']])
    paired=[]; paired_summary=[]
    look=pairing(all_rows)
    for key,record in sorted(look.items()):
        left=record.get('TPSSR')
        if not left:continue
        for m,v in sorted(record.items()):
            if m=='TPSSR':continue
            matched=bool(left['full_coverage'] and v['full_coverage'] and left['rmse'] is not None and v['rmse'] is not None)
            paired.append([*key,m,int(matched),left['rmse'],v['rmse'],(v['rmse']-left['rmse']) if matched else None,
                           int(left['rmse']<v['rmse']) if matched else None,
                           left['coverage'],v['coverage']])
    for split,ex,cores,methods,n in SPLITS:
        for m in methods:
            if m=='TPSSR':continue
            pp=[p for p in paired if p[0]==split and p[5]==m]
            valid=[p for p in pp if p[6]]
            paired_summary.append([split,m,len(pp),len(valid),sum(p[10] for p in valid),
               metrics_mean([{'left':p[7]} for p in valid],'left'),
               metrics_mean([{'right':p[8]} for p in valid],'right'),
               sum(p[9] for p in valid)/len(valid) if valid else None])
    common=[];common_keys=[]
    for split,ex,cores,methods,n in SPLITS:
        item_keys=sorted(k for k in look if k[0]==split)
        qualifying=[]
        for key in item_keys:
            rec=look[key]
            if all(m in rec and rec[m]['full_coverage'] and rec[m]['rmse'] is not None for m in methods):qualifying.append(key)
        common_keys.extend([[*key,len(methods)] for key in qualifying])
        for m in methods:
            rows=[look[key][m] for key in qualifying]
            x=summary(rows)
            common.append([split,m,len(qualifying),len(item_keys),x['mean_rmse'],x['mean_mae'],x['mean_r2'],x['mean_sam_deg']])
    metadata=[
      ['Dataset policy','Only four explicitly named split directories are scanned'],
      ['Model policy','RankCV + Hybrid Minority (SWIR 8x2) or Hybrid Gap (other splits; with three experts, second-smallest minus smallest >= 10 percentage points); two-expert minority below 10%; Spatial Consistency OFF; Robust Donor ON; Fusion Guard OFF'],
      ['TPSSR folder for SWIR 8x2',FINAL_TPSSR_DIR],
      ['TPSSR folder for other splits',FINAL_TPSSR_GAP_DIR],
      ['RF folder','random_forest_multiscale (RF direct observed v3; verify checkpoint payload variant externally)'],
      ['PCA','pca only (enhanced PCA excluded)'],
      ['DDS2M','dds2m_t3000_sp1500 existing final run, no rerun'],
      ['Methods SWIR 8x2',', '.join(SPLITS[0][3])],
      ['Methods other splits','TPSSR, Spectral'],
      ['Total expected masks per split','200, not per core'],
      ['Missing targets policy','Do not synthesize data or replace missing final version with legacy'],
      ['Mean_RMSE','Arithmetic mean of valid mask-level RMSE; the principal paper metric used historically'],
      ['Pooled_RMSE','sqrt(sum(RMSE_mask^2 * valid_n) / sum(valid_n)) over cases with valid counts; NOT mean_RMSE'],
      ['Full_coverage','coverage >= 0.999999 and finite RMSE'],
      ['Common_masks','Intersection of FULL coverage, with valid metric for every requested method within a split'],
      ['Paired full','TPSSR and comparator both full coverage on identical dataset+scenario+severity+repeat'],
      ['Partial method caveat','Overall unpaired averages are descriptive; use paired/common subsets for fair comparisons'],
      ['SSIM','Not reported in final paper or summary tables'],
      ['Fixed masks','Matched by dataset, scenario, severity, repeat. Numerical coordinates are not rechecked here.'],
      ['Versions','No old TPSSR variant, RF previous, old RF, enhanced PCA, experimental deep branches'],
      ['RF provenance','Metrics CSV may not encode model variant: verify trained checkpoint variant=multiscale_rf_v3_observed_donors41, feature_dim=773 before publication'],
      ['Generated',dt.datetime.now().isoformat(timespec='seconds')],
      ['Results root',str(root)],
      ['Blocking issues',len(fatal)],
      ['Status','INCOMPLETE' if fatal else 'COMPLETE'],
    ]
    sheets=[
      ('README',['Topic','Details'],metadata,'Read this first. Provenance, aggregation definitions and interpretation cautions.'),
      ('Overall',['Split','Method','Datasets','Expected_masks','Cases_found','Valid','Fully_covered','Partially_covered','Invalid','Voxel_coverage','Mean_RMSE','Pooled_RMSE','Mean_MAE','Mean_R2','Mean_SAM_deg','Full_only_Mean_RMSE'],overall,'Arithmetic means over valid masks. Check coverage and fair matched subsets.'),
      ('By_Image',['Split','Dataset','Method','Expected','Cases','Full_cases','Partial_cases','Voxel_coverage','Mean_RMSE','Pooled_RMSE','Mean_MAE','Mean_R2','Mean_SAM_deg'],per_image,'Separate holdout cores, matched to one experiment split.'),
      ('By_Scenario',['Split','Method','Scenario','Cases','Valid','Full','Voxel_coverage','Mean_RMSE','Pooled_RMSE','Mean_MAE','Mean_R2','Mean_SAM_deg'],by_scenario,'Scenario average across all available severities and test cores.'),
      ('By_Severity',['Split','Method','Scenario','Severity','Cases','Valid','Full','Voxel_coverage','Mean_RMSE','Pooled_RMSE','Mean_MAE','Mean_R2','Mean_SAM_deg'],by_severity,'Scenario x severity, for main paper plots and S5 comparisons.'),
      ('All_Masks',['Split','Dataset','Method','Scenario','Severity','Repeat','Mask_filename','Target_n','Valid_n','Coverage','RMSE','MAE','R2','SAM_deg','Full_coverage','Status','Router','Donor_match','Spatial_strength','Global_minority_pct','Global_minority_changed_pixels','Guard_enabled','Source_path'],per_mask,'All raw per-mask results, one row per method. SSIM intentionally excluded.'),
      ('Paired_Detail',['Split','Dataset','Scenario','Severity','Repeat','Comparator','Full_pair','TPSSR_RMSE','Comparator_RMSE','Comparator_minus_TPSSR','TPSSR_wins','TPSSR_coverage','Comparator_coverage'],paired,'RMSE differences only shown for FULL-coverage matched pairs.'),
      ('Paired_Summary',['Split','Comparator','Shared_mask_ids','Full_pairs','TPSSR_wins','TPSSR_Mean_RMSE_on_pairs','Comparator_Mean_RMSE_on_pairs','Mean_RMSE_gap'],paired_summary,'Fair per-comparator full-coverage common masks. Positive gap favours TPSSR.'),
      ('Common_Masks',['Split','Method','Common_full_cases','Mask_ids_in_TPSSR','Mean_RMSE','Mean_MAE','Mean_R2','Mean_SAM_deg'],common,'All methods within each split on identical full-coverage masks.'),
      ('Common_Keys',['Split','Dataset','Scenario','Severity','Repeat','Methods'],common_keys,'Mask identities retained in all-method common subset.'),
      ('Input_Audit',['Split','Dataset','Method','Status','Rows_found','Expected_per_core','Details'],audit,'An ERROR/MISSING means workbook is incomplete; consult Sources and run reports.'),
      ('Sources',['Split','Dataset','Method','Source_path','File_bytes','Modified_local','SHA256','Rows_accepted','Status'],sources,'Exact CSV provenance and SHA256 for traceability.'),
    ]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_xlsx(args.output,sheets)
    print(f'Workbook: {args.output.resolve()}',flush=True)
    print(f'Final mask rows: {len(all_rows)}; files: {len(sources)}; blocking issues: {len(fatal)}',flush=True)
    for issue in fatal[:30]:print('BLOCKING:',issue,flush=True)
    if fatal and not args.allow_incomplete:
        print('Incomplete result set. Workbook is a DRAFT. Exit status 2. Use --allow-incomplete to allow drafts.',file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':
    sys.exit(main())
