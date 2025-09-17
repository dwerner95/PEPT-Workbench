import numpy as np

from pept_gui.io import TimeMask, concatenate_and_mask, decimate, mask_by_time, stitch_time_axes


def test_time_mask_apply_filters_rows():
    timestamps = np.linspace(0, 1.0, 11)
    array = np.column_stack([timestamps, np.arange(11)])
    mask = TimeMask(0.2, 0.6)
    filtered = mask.apply(array, time_column=0)
    assert filtered[0, 0] >= 0.2
    assert filtered[-1, 0] < 0.6
    assert filtered.shape[0] == 4


def test_mask_by_time_matches_class():
    array = np.column_stack([np.linspace(0, 2, 5), np.arange(5)])
    masked = mask_by_time(array, 0.4, 1.6)
    direct = TimeMask(0.4, 1.6).apply(array)
    assert np.array_equal(masked, direct)


def test_stitch_time_axes_adjusts_offsets():
    chunk_a = np.array([[0.0, 1.0], [0.5, 2.0]])
    chunk_b = np.array([[0.0, 3.0], [0.5, 4.0]])
    stitched = stitch_time_axes([chunk_a, chunk_b])
    assert np.allclose(stitched[1][:, 0], np.array([0.5, 1.0]))


def test_concatenate_and_mask_joins_chunks():
    chunk_a = np.array([[0.0, 1.0], [0.5, 2.0]])
    chunk_b = np.array([[0.0, 3.0], [0.5, 4.0]])
    mask = TimeMask(0.25, 1.25)
    result = concatenate_and_mask([chunk_a, chunk_b], mask)
    assert result.shape[0] == 3
    assert np.all((result[:, 0] >= 0.25) & (result[:, 0] < 1.25))


def test_decimate_every_third():
    array = np.arange(30).reshape(10, 3)
    decimated = decimate(array, every=3)
    assert decimated.shape[0] == 4
    assert np.array_equal(decimated[:, 0], np.array([0, 9, 18, 27]))
