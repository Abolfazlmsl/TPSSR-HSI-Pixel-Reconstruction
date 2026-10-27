import numpy as np
import torch

from spectral_support_stress import masked_patch, neighbor_features, ZeroAnchorNeighbor, PERCENTAGES, PATCHES


def make_cube(bands=200):
    yy,xx=np.mgrid[:65,:65]
    wave=np.linspace(0.1,0.8,bands,dtype=np.float32)
    cube=np.broadcast_to(wave,(65,65,bands)).copy()+yy[:,:,None]*.002+xx[:,:,None]*.001
    invalid=np.zeros(cube.shape,bool)
    return cube,invalid


def test_exact_center_band_counts_and_spatial_patch():
    cube,invalid=make_cube()
    for pct in PERCENTAGES:
        for size in PATCHES:
            values,mask,y,x,truth,target=masked_patch(cube,invalid,32,32,pct,size,
                              'random_bands',np.random.default_rng(120))
            expected=round(200*pct/100)
            assert int(target.sum())==expected
            assert np.isnan(values[y,x,target]).all()
            assert np.allclose(truth,cube[32,32])
            square=mask[y-size//2:y+size//2+1,x-size//2:x+size//2+1]
            assert np.all(square[:,:,target])
            assert not np.isfinite(values[mask]).any()
            assert not np.any(mask[y,x,~target])


def test_contiguous_and_missing_target_protection():
    cube,invalid=make_cube()
    v,b,y,x,t,target=masked_patch(cube,invalid,32,32,90,1,
                             'contiguous_bands',np.random.default_rng(6))
    target_indices=np.flatnonzero(target)
    assert len(target_indices)==180 and np.all(np.diff(target_indices)==1)
    mean=np.zeros(200,np.float32)
    std=np.ones(200,np.float32)
    wave=np.arange(200,dtype=np.float32)
    f,a,p=neighbor_features(v,b,y,x,mean,std,wave)
    other=v.copy()
    other[y,x,:]=98765.0  # A stray unmasked center truth must have no effect.
    f2,a2,p2=neighbor_features(other,b,y,x,mean,std,wave)
    assert np.array_equal(a,a2)
    np.testing.assert_allclose(f,f2)
    np.testing.assert_allclose(p,p2)


def test_zero_neighbor_support_is_reported_not_fabricated():
    cube,invalid=make_cube()
    v,b,y,x,t,target=masked_patch(cube,invalid,32,32,100,19,
                             'random_bands',np.random.default_rng(6))
    b[:]=True
    v[:]=np.nan
    f,a,p=neighbor_features(v,b,y,x,np.zeros(200,np.float32),
                            np.ones(200,np.float32),np.arange(200,dtype=np.float32))
    assert not a.any()
    assert np.isnan(p).all()
    assert np.isfinite(f).all()


def test_model_identity_initialization_and_shape():
    model=ZeroAnchorNeighbor()
    x=torch.randn(3,200,6)
    y=model(x)
    torch.testing.assert_close(y,x[:,:,0])


def test_per_band_donors_reach_outside_partially_missing_large_patch():
    # Regression: the old 96-pixel shared donor limit incorrectly made the
    # target-band donor count zero in 19x19 partial spectral patches.
    cube, invalid = make_cube(bands=200)
    mean = np.zeros(200, np.float32)
    std = np.ones(200, np.float32)
    wave = np.arange(200, dtype=np.float32)
    for pct in (30, 80, 90, 99.5):
        for pattern in ('random_bands', 'contiguous_bands'):
            values, bad, cy, cx, _, target = masked_patch(
                cube, invalid, 32, 32, pct, 19, pattern,
                np.random.default_rng(19))
            feature, support, baseline = neighbor_features(
                values, bad, cy, cx, mean, std, wave)
            assert np.all(support[target])
            assert np.all(np.isfinite(baseline[target]))
            assert np.all(np.isfinite(feature))


def test_no_fabricated_support_when_band_is_missing_everywhere():
    cube, invalid = make_cube(bands=20)
    invalid[:, :, 5] = True
    cube[invalid] = np.nan
    values, bad, cy, cx, _, target = masked_patch(
        cube, invalid, 32, 32, 100, 19, 'random_bands',
        np.random.default_rng(20))
    f, support, baseline = neighbor_features(
        values, bad, cy, cx, np.zeros(20, np.float32),
        np.ones(20, np.float32), np.arange(20, dtype=np.float32))
    assert not support[5]
    assert np.isnan(baseline[5])
    assert np.isfinite(f).all()
