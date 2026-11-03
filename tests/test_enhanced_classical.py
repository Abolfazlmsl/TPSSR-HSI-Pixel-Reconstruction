"""Sanity checks: hidden target reflectances must never leak into baselines."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from src.enhanced_classical import (FEATURE_DIM, extract_rf_patch,
    rf_multiscale_feature, reconstruct_spatial_pca, predict_multiscale_rf)


def test_enhanced_estimators_ignore_masked_ground_truth():
    rng = np.random.default_rng(39)
    h, w, b = 65, 65, 45
    cube = (.3 + .05*np.sin(np.arange(b)/9)[None,None,:]
            + .02*rng.normal(size=(h,w,1))).astype('float32')
    missing = np.zeros_like(cube, bool)
    missing[26:41, 26:41] = True
    changed = cube.copy()
    changed[missing] = 999.0
    mean = np.mean(cube,axis=(0,1))
    coords = np.array([[32,32,21],[35,35,22]],dtype=np.int64)
    model = PCA(n_components=4).fit(cube.reshape(-1,b))
    a = reconstruct_spatial_pca(cube,missing,coords,model,mean)
    z = reconstruct_spatial_pca(changed,missing,coords,model,mean)
    np.testing.assert_allclose(a,z,rtol=0,atol=0)
    v,m=extract_rf_patch(cube,missing,32,32,21)
    v2,m2=extract_rf_patch(changed,missing,32,32,21)
    f=rf_multiscale_feature(v,m,mean,21)
    f2=rf_multiscale_feature(v2,m2,mean,21)
    assert f.shape==(FEATURE_DIM,)
    np.testing.assert_array_equal(f,f2)
    forest=RandomForestRegressor(n_estimators=5,random_state=7).fit(
        np.vstack([f,f+.01,f-.01,f+.02]),np.array([.2,.25,.15,.3]))
    pred=predict_multiscale_rf(forest,mean,cube,missing,coords)
    pred2=predict_multiscale_rf(forest,mean,changed,missing,coords)
    np.testing.assert_array_equal(pred,pred2)
