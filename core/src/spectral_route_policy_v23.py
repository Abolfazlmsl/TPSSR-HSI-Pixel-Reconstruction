"""TPSSR V2.3 observed-only geometric long-gap routing and local median fallback.

Extracted verbatim from evaluated script evaluate_tpssr_two_expert_v23.py.
Do not modify thresholds using SWIR5 or SWIR10 test outcomes.
"""
from __future__ import annotations
import numpy as np

VERSION = 'v23_predeclared_geometry_and_neighbor_median_20261014'

# Frozen engineering hypotheses: do NOT select values from SWIR5/10 outcomes.
# Extra Region route for contiguous bands of a same-pixel true-mask gap, when
# almost all surrounding observable neighbors provide the missing wavelengths.
MIN_EXTRA_RUN_BANDS = 10
MIN_EXTRA_RUN_FRACTION = 0.035
MIN_RUN_TARGET_PURITY = 0.92
MIN_DOMINANT_TARGET_SHARE = 0.90
MIN_EXTRA_OBSERVED_NEIGHBORS = 5
NEIGHBOR_GAP_OBS_FRACTION = 0.95
MIN_SUPPORTED_NEIGHBOR_FRACTION = 0.75

# Locally supported Region replacement, limited to completely masked pixels
# that are spatially isolated and surrounded by reliable measured spectra.
MIN_LOCAL_FULL_SPECTRA = 6
MIN_LOCAL_GLOBAL_OBSERVED_FRACTION = 0.95
MAX_NEIGHBORS_WITH_ANY_MASK = 1
MIN_LOCAL_BAND_OBSERVATIONS = 6
LOCAL_MAD_ABS = 0.006
LOCAL_MAD_REL = 0.10


def policy_spec():
    return dict(version=VERSION,min_extra_run_bands=MIN_EXTRA_RUN_BANDS,
                min_extra_run_fraction=MIN_EXTRA_RUN_FRACTION,
                min_run_target_purity=MIN_RUN_TARGET_PURITY,
                min_dominant_target_share=MIN_DOMINANT_TARGET_SHARE,
                min_extra_observed_neighbors=MIN_EXTRA_OBSERVED_NEIGHBORS,
                neighbor_gap_obs_fraction=NEIGHBOR_GAP_OBS_FRACTION,
                min_supported_neighbor_fraction=MIN_SUPPORTED_NEIGHBOR_FRACTION,
                min_local_full_spectra=MIN_LOCAL_FULL_SPECTRA,
                min_local_global_observed_fraction=MIN_LOCAL_GLOBAL_OBSERVED_FRACTION,
                max_neighbors_with_any_mask=MAX_NEIGHBORS_WITH_ANY_MASK,
                min_local_band_observations=MIN_LOCAL_BAND_OBSERVATIONS,
                local_mad_abs=LOCAL_MAD_ABS,local_mad_rel=LOCAL_MAD_REL)


def contiguous_true_run(masked):
    arr=np.asarray(masked,bool)
    edges=np.flatnonzero(np.diff(np.pad(arr.astype(np.int8),(1,1))))
    if len(edges)==0:return 0,0
    starts=edges[::2]; ends=edges[1::2]
    i=np.argmax(ends-starts)
    return int(starts[i]),int(ends[i])


def route_extra_region(cube,mask,base_routes,eval_coords):
    """Return extra-region *pixel* map, never using masked cube values."""
    cube=np.asarray(cube);mask=np.asarray(mask,bool)
    h,w,b=cube.shape
    observed=(~mask)&np.isfinite(cube)
    base_hybrid=np.asarray(base_routes,bool)
    if base_hybrid.shape!=(h,w):raise ValueError('route map shape')
    extra=np.zeros((h,w),bool)
    n_gap=0;n_support=0
    min_len=max(MIN_EXTRA_RUN_BANDS,int(np.ceil(MIN_EXTRA_RUN_FRACTION*b)))
    for y,x in np.argwhere(base_hybrid):
        y=int(y);x=int(x)
        row=np.asarray(mask[y,x],bool)
        a,z=contiguous_true_run(row)
        n=int(row.sum());gap_len=z-a
        if gap_len<min_len or n<=0 or gap_len/n<MIN_DOMINANT_TARGET_SHARE:
            continue
        # Require target-masked fraction in the *longest unavailable run*;
        # guards against underlying nonfinite measurements making artificial gaps.
        j=a
        while j>0 and not observed[y,x,j-1]:j-=1
        k=z
        while k<b and not observed[y,x,k]:k+=1
        if gap_len/max(k-j,1)<MIN_RUN_TARGET_PURITY:
            continue
        n_gap+=1
        total=good=0
        for yy in range(max(0,y-1),min(h,y+2)):
            for xx in range(max(0,x-1),min(w,x+2)):
                if yy==y and xx==x:continue
                total+=1
                if np.mean(observed[yy,xx,a:z])>=NEIGHBOR_GAP_OBS_FRACTION:
                    good+=1
        if total>=MIN_EXTRA_OBSERVED_NEIGHBORS and good>=MIN_EXTRA_OBSERVED_NEIGHBORS and good/total>=MIN_SUPPORTED_NEIGHBOR_FRACTION:
            extra[y,x]=True;n_support+=1
    return extra,dict(gap_candidates=n_gap,supported_gap_pixels=n_support)


def local_neighbor_median(cube,mask,coords,route):
    """Measured-neighbor median for spatially isolated, fully missing pixels.

    Bandwise confidence checks depend only on unmasked observed neighbors.
    No masked target is ever read, including when a neighbor is itself masked.
    """
    coords=np.asarray(coords,np.int64);route=np.asarray(route,np.int64)
    candidate=np.full(len(coords),np.nan,np.float32)
    h,w,b=cube.shape
    num_pixels=0
    number_admissible=0
    if not np.any(route==2):return candidate,dict(local_candidate_pixels=0,local_accepted_voxels=0)
    pairs=np.unique(coords[route==2,:2],axis=0)
    accepted_positions={(int(y),int(x)):[] for y,x in pairs}
    pixel_options={}
    for y,x in pairs:
        y=int(y);x=int(x)
        if not np.all(mask[y,x]):continue
        neighbors=[(yy,xx) for yy in range(max(0,y-1),min(h,y+2))
                   for xx in range(max(0,x-1),min(w,x+2)) if not(yy==y and xx==x)]
        if len(neighbors)<MIN_LOCAL_FULL_SPECTRA:continue
        if sum(bool(np.any(mask[yy,xx])) for yy,xx in neighbors)>MAX_NEIGHBORS_WITH_ANY_MASK:
            continue
        full=[(yy,xx) for yy,xx in neighbors
              if np.mean((~mask[yy,xx])&np.isfinite(cube[yy,xx]))>=MIN_LOCAL_GLOBAL_OBSERVED_FRACTION]
        if len(full)<MIN_LOCAL_FULL_SPECTRA:continue
        pix=np.stack([np.where((~mask[yy,xx])&np.isfinite(cube[yy,xx]),cube[yy,xx],np.nan)
                     for yy,xx in full]).astype(np.float64)
        counts=np.isfinite(pix).sum(axis=0)
        # np.nanmedian ignores invalid or masked neighbor measurements only.
        with np.errstate(invalid='ignore'):
            median=np.nanmedian(pix,axis=0)
            mad=np.nanmedian(np.abs(pix-median[None,:]),axis=0)
        good=(counts>=MIN_LOCAL_BAND_OBSERVATIONS)&np.isfinite(median)&np.isfinite(mad)&(mad<=(LOCAL_MAD_ABS+LOCAL_MAD_REL*np.abs(median)))
        pixel_options[(y,x)]=(median,good)
        num_pixels+=1
    for i,(y,x,band) in enumerate(coords):
        if route[i]!=2:continue
        obj=pixel_options.get((int(y),int(x)))
        if obj is None:continue
        median,good=obj
        if good[int(band)]:
            candidate[i]=np.float32(median[int(band)])
            number_admissible+=1
    return candidate,dict(local_candidate_pixels=num_pixels,local_accepted_voxels=number_admissible)

