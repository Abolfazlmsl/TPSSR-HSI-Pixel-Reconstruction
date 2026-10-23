from experiment_config import ExperimentSpec, selected_lineaware_region, selected_results_folder


def _spec(preset):
    if preset == 'swir_8x2':
        return ExperimentSpec(('SWIR1','SWIR2','SWIR3','SWIR4','SWIR6','SWIR7','SWIR8','SWIR9'),
                              ('SWIR5','SWIR10'), 'SWIR5', 'swir_8x2', 'SWIR')
    if preset == 'vnir_8x2':
        return ExperimentSpec(('VNIR1','VNIR2','VNIR3','VNIR4','VNIR6','VNIR7','VNIR8','VNIR9'),
                              ('VNIR5','VNIR10'), 'VNIR5', 'vnir_8x2', 'VNIR')
    return ExperimentSpec(('SWIR1','SWIR2','SWIR3','SWIR4','SWIR5'),
                          ('SWIR6','SWIR7','SWIR8','SWIR9','SWIR10'), 'SWIR6', 'a_to_b', 'SWIR')


def test_selected_lineaware_variant_and_result_isolation():
    spec = _spec('swir_8x2')
    assert selected_lineaware_region(spec, 'learned_router3_aligned')
    assert selected_results_folder(spec, 'learned_router3_aligned') == 'tpssr_learned_router3_aligned_lineaware'
    assert selected_results_folder(spec, 'learned_router3_rmse') == 'tpssr_learned_router3_rmse'


def test_other_presets_keep_their_own_region():
    for key in ('vnir_8x2','a_to_b'):
        spec = _spec(key)
        assert not selected_lineaware_region(spec, 'learned_router3_aligned')
        assert selected_results_folder(spec, 'learned_router3_aligned') == 'tpssr_learned_router3_aligned'
