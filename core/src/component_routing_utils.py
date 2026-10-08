from __future__ import annotations
import math
import numpy as np


def connected_components_4(binary):
    m=np.asarray(binary,dtype=bool); h,w=m.shape; seen=np.zeros_like(m); out=[]
    for sy,sx in np.argwhere(m):
        sy=int(sy); sx=int(sx)
        if seen[sy,sx]: continue
        stack=[(sy,sx)]; seen[sy,sx]=True; comp=[]
        while stack:
            y,x=stack.pop(); comp.append((y,x))
            for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0<=yy<h and 0<=xx<w and m[yy,xx] and not seen[yy,xx]:
                    seen[yy,xx]=True; stack.append((yy,xx))
        out.append(np.asarray(comp,dtype=np.int64))
    return out


def component_mask(shape, coords):
    out=np.zeros(shape,dtype=bool); c=np.asarray(coords,dtype=np.int64)
    if len(c): out[c[:,0],c[:,1]]=True
    return out


def component_features(mask, component):
    mask=np.asarray(mask,dtype=bool); comp=np.asarray(component,dtype=bool); coords=np.argwhere(comp); area=max(len(coords),1); h,w,_=mask.shape
    y0,x0=coords.min(axis=0); y1,x1=coords.max(axis=0); bh=int(y1-y0+1); bw=int(x1-x0+1); fill=area/max(bh*bw,1)
    boundary=0
    for y,x in coords:
        y=int(y); x=int(x)
        if any(not (0<=yy<h and 0<=xx<w and comp[yy,xx]) for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1))): boundary+=1
    missfrac=mask[comp,:].mean(axis=1) if area else np.asarray([0.0]); aspect=max(bh,bw)/max(min(bh,bw),1)
    return np.asarray([
        math.log1p(area)/math.log1p(max(h*w,2)), bh/max(h,1), bw/max(w,1), fill,
        math.log1p(aspect)/math.log1p(max(h,w,2)), boundary/max(area,1),
        float(mask[comp,:].mean()) if area else 0.0, float(np.mean(missfrac)),
        float(np.std(missfrac)), float(np.max(missfrac)), float(np.mean(missfrac>=0.999)),
        min(1.0,area/225.0),
    ],dtype=np.float32)


def component_band_stats(mask, component):
    mask=np.asarray(mask,dtype=bool); comp=np.asarray(component,dtype=bool); area=max(int(comp.sum()),1)
    counts=mask[comp,:].sum(axis=0).astype(np.float32); frac=counts/float(area); count_norm=np.log1p(counts)/math.log1p(float(area))
    return frac.astype(np.float32),count_norm.astype(np.float32),counts.astype(np.float32)


def representative_pixels(component, max_pixels=32):
    coords=np.argwhere(np.asarray(component,dtype=bool))
    if len(coords)<=int(max_pixels): return coords.astype(np.int64)
    ids=np.linspace(0,len(coords)-1,int(max_pixels)).round().astype(np.int64)
    return coords[ids].astype(np.int64)
