import librosa
import random
import numpy as np
import numpy.testing as npt
import os
import pytest

REPO_PATH = os.path.abspath('../..')
import sys
sys.path.append(REPO_PATH)
import create_training_data.code.augmentations as aug 

@pytest.fixture
def audio():
    # For audio in this directory
    def _intake_func(rel_path):
        return os.path.join(os.path.abspath('.'), rel_path)
    return _intake_func

def test_wraparound_extract():
    # test zero beginning, not getting to end of original array
    npt.assert_array_equal(aug.wraparound_extract(original = np.array([0, 1]), begin = 0, length = 1), np.array([0]))

    # test zero beginning, not getting to end of original array
    npt.assert_array_equal(aug.wraparound_extract(original = np.array([0, 1]), begin = 0, length = 2), np.array([0, 1]))

    # test zero beginning, not wrapping
    npt.assert_array_equal(aug.wraparound_extract(original = np.array([0, 1]), begin = 0, length = 2), np.array([0, 1]))

    # test zero beginning, wrapping around
    npt.assert_array_equal(aug.wraparound_extract(original = np.array([0, 1]), begin = 0, length = 3), np.array([0, 1, 0]))

    # test nonzero beginning, not wrapping
    npt.assert_array_equal(aug.wraparound_extract(original = np.array([0, 1]), begin = 1, length = 1), np.array([1]))

    # test nonzero beginning, wrapping around
    npt.assert_array_equal(aug.wraparound_extract(original = np.array([0, 1]), begin = 1, length = 3), np.array([1, 0, 1]))

    # test multiwrap
    npt.assert_array_equal(aug.wraparound_extract(original = np.array([0, 1]), begin = 1, length = 10), np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]))

    # test wrapping around beginning
    npt.assert_array_equal(aug.wraparound_extract(original = np.array([0, 1]), begin = 5, length = 3), np.array([1, 0, 1]))


    
def test_get_chunk():
    # TODO
    pass



def test_cyclic_shift():
    # Test random splitting
    random.seed(100)
    npt.assert_array_equal(aug.cyclic_shift(np.array((0, 1, 2, 3, 4, 5, 6, 7))), np.array([2, 3, 4, 5, 6, 7, 0, 1]))

    # Test deterministic splitting
    npt.assert_array_equal(aug.cyclic_shift(np.array([0, 1, 2]), split_point=0.5), np.array([1, 2, 0]))

    # Test deterministic splitting
    npt.assert_array_equal(aug.cyclic_shift(np.array([0, 1, 2, 3]), split_point=0.5), np.array([2, 3, 0, 1]))
    
    
def test_divide_chunks():
    # Test chunk division at set amount
    array0 = np.array([0, 0, 0])
    array1 = np.array([1, 1, 1])
    array2 = np.array([2])
    all_arrays = (array0, array1, array2)
    cat_arrays = np.concatenate(all_arrays)
    results = aug.divide_samples(cat_arrays, sample_rate=1, low_duration=3, high_duration=3)
    for idx, result in enumerate(results):
        npt.assert_array_equal(result, all_arrays[idx])

    # Test random chunk division
    random.seed(333)
    # Predetermined results with random.seed(333)
    predetermined = [np.array([0, 1, 2, 3, 4, 5, 6, 7]), np.array([8, 9])]
    results = aug.divide_samples(np.array(range(10)), sample_rate=1, low_duration=0, high_duration=10)
    for idx, result in enumerate(results):
        npt.assert_array_equal(result, predetermined[idx])
        
def test_combine_chunks(audio):
    # Test that divided samples can be recombined successfully
    samples, sr = librosa.load(audio('silence_10s.mp3'))
    divided = aug.divide_samples(samples, sample_rate=sr, low_duration=0.5, high_duration=4)
    npt.assert_array_equal(aug.combine_samples(divided), samples)
    
def test_time_stretch():
    # Predetermined results with random.seed(333)
    #predetermined = [np.array([0., 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 7]), np.array([8, 9])]
    random.seed(3)
    divs = aug.divide_samples(np.linspace(0, 1, 10), sample_rate=1, low_duration=0, high_duration=10)
    np.random.seed(111)
    results = aug.time_stretch_divisions(divs)

    # predetermined results for random.seed == 3 and np.random.seed == 111
    # np.random.seed must be set because randomness in time_stretch_divisions
    # comes from np.random.normal
    predetermined = [
        np.array([0.        , 0.11111111, 0.22222222]),
        np.array([0.27306482, 0.36267295, 0.45205913, 0.54126718,
               0.63025186, 0.71660051, 0.80273002])
    ]

    for idx, result in enumerate(results):
        npt.assert_array_almost_equal(result, predetermined[idx])
        
def test_pitch_shift():
    # TODO
    pass

def test_random_filter(audio):
    unfiltered, sample_rate = librosa.load(audio('1min.wav'))

    # This filter will produce an invalid output 
    # i.e., the array will contain values above 1
    filtered_not_checked = aug.random_filter(
        unfiltered, sample_rate, percent_chance=1,
        filter_type = 'highpass',
        filter_order = 5,
        filter_low = 2690,
        filter_high = 11024.0,
        error_check = False
    )
    assert((filtered_not_checked > 1).any())

    # The same filter as above, but with error checking: 
    # the error check should flag the invalid content
    # in the filtered result and return the original array
    filtered_checked = aug.random_filter(
        unfiltered, sample_rate, percent_chance=1,
        filter_type = 'highpass',
        filter_order = 5,
        filter_low = 2690,
        filter_high = 11024.0,
        #error_check = True # Error checking by default
    )
    assert(~(filtered_checked > 1).any())
    assert(filtered_checked is unfiltered)
    
    
def test_fade():
    # Assert that fading out doesn't work if fade_len is too long
    with pytest.raises(IndexError):
        aug.fade(array = np.array((1, 1, 1, 1, 1)), fade_len=6, start_amp=1)

    # Assert that can only provide 0 or 1 as start_amp
    with pytest.raises(ValueError):
        aug.fade(array = np.array((1, 1, 1)), fade_len=3, start_amp=1.0)
    with pytest.raises(ValueError):
        aug.fade(array = np.array((1, 1, 1)), fade_len=3, start_amp=True)

    # Fade in on exactly correct length array
    fade_in = aug.fade(array = np.array((1, 1, 1, 1, 1)), fade_len=5, start_amp=0)
    npt.assert_array_equal(fade_in, np.array([0., 0.25, 0.5, 0.75, 1.]))

    # Fade out on long array
    fade_out = aug.fade(array = np.array((1, 1, 1, 1, 1, 1, 1)), fade_len=5, start_amp=1)
    npt.assert_array_equal(fade_out, np.array([1., 1., 1., 0.75, 0.5, 0.25, 0.]))
    
def test_sum_samples(audio):
    # Test fade & wraparound on audio-like numpy arrays
    nowrap_nofade = aug.sum_samples(
        samples_original = np.array((1., 1., 500.)),
        samples_new = np.array((10., 11.)),
        sample_rate = 1,
        wraparound_fill = False,
        fade_out = False
    )
    npt.assert_array_equal(nowrap_nofade, np.array([11., 12.,  500.]))

    nowrap_fade = aug.sum_samples(
        samples_original = np.array((1., 1., 1., 500.)),
        samples_new = np.array((10., 10.)),
        sample_rate = 1,
        wraparound_fill = False,
        fade_out = True
    )
    npt.assert_array_equal(nowrap_fade, np.array([ 11.,   1.,   1., 500.]))

    wrap_nofade = aug.sum_samples(
        samples_original = np.array((1., 1., 500.)),
        samples_new = np.array((10., 11.)),
        sample_rate = 1,
        wraparound_fill = True,
        fade_out = False
    )
    npt.assert_array_equal(wrap_nofade, np.array([11., 12.,  510.]))

    # Same behavior as wrap_nofade
    wrap_fade = aug.sum_samples(
        samples_original = np.array((1., 1., 500.)),
        samples_new = np.array((10., 11.)),
        sample_rate = 1,
        wraparound_fill = True,
        fade_out = True
    )
    npt.assert_array_equal(wrap_nofade, np.array([11., 12.,  510.]))


    # Test on actual audio without fade or wraparound
    samples, sample_rate = librosa.load(audio('1min.wav'))
    samples_original = samples[:22050]
    samples_new = aug.cyclic_shift(samples_original)[:11025]

    summed = aug.sum_samples(
        samples_original = samples_original,
        samples_new = samples_new,
        sample_rate = sample_rate,
        wraparound_fill = False,
        fade_out = False)

    true_summed = np.add(samples_original, np.pad(samples_new, (0, 11025), constant_values=0, mode='constant'))
    npt.assert_array_equal(summed, true_summed)