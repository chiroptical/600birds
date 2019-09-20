import os
import pytest
import numpy.testing as npt
import inspect

# Main functions
from ..code.audio_aug import *

# Private helper functions
from ..code.audio_aug import (_audio_manipulation,
    _wraparound_extract, _shift_array, _divide_samples,
    _combine_samples, _fade, _sum_samples, _select_chunk
)

TEST_PATH = os.path.abspath(os.path.dirname(__file__))

####################################################
#################### Fixtures ######################
####################################################

@pytest.fixture
def audio_ex():
    '''
    Return Audio object for file in
    current directory'''
    
    def _select_path(path = 'veryshort.wav'):
        abs_path = os.path.join(TEST_PATH, path)
        return Audio(path = abs_path, label='test')

    return _select_path
    



####################################################
############### Tests of Audio class ###############
####################################################

def test_Audio_path_loading_error_checking():
    # Test handling of uncouth files
    with pytest.raises(FileNotFoundError):
        Audio(path = "SirNotAppearingOnThisFilesystem.wav", label = 'test_label')


def test_Audio_set_samples_error_checking(audio_ex):
    # Test handling of bad sample setting
    with pytest.raises(ValueError):
        chunk = audio_ex()
        samples = chunk.samples
        chunk.set_samples('1')
    assert(chunk.samples is samples)
    

def test_Audio_add_source_input_checking_not_all_args(audio_ex):
    chunk =  audio_ex()
    source_path = chunk.original_path

    # Not all arguments provided
    with pytest.raises(ValueError):
        chunk.add_source(path = 'me')
    with pytest.raises(ValueError):
        chunk.add_source(path = 'me', start_time = 1)    
    with pytest.raises(ValueError): 
        chunk.add_source(source = (source_path, ('a')) )

def test_Audio_add_source_input_checking_bad_path(audio_ex):
    chunk =  audio_ex()

    # Bad source path
    with pytest.raises(FileNotFoundError):
        chunk.add_source(source = ('me', (1, 2)))
    with pytest.raises(FileNotFoundError):
        chunk.add_source(path = 'me', start_time = 1, duration = 2)


def test_Audio_add_source_input_checking_bad_timing(audio_ex):
    chunk =  audio_ex()
    source_path = chunk.original_path

    # Bad start time or duration
    with pytest.raises(ValueError): 
        chunk.add_source(source = (source_path, ('a', 1)))
        chunk.add_source(source = (source_path, (1, 'a')))
        chunk.add_source(path = source_path, start_time = 'a', duration = 1) 
        chunk.add_source(path = source_path, start_time = 1, duration = 'a')

def test_Audio_add_source_actually_works_kwords(audio_ex):
    chunk = audio_ex()
    original_path = chunk.original_path
    chunk.add_source(path = original_path, start_time = 1, duration = 1)
    assert(chunk.sources) == [(original_path, (1, 1))]

def test_Audio_add_source_actually_works_tuple(audio_ex):
    chunk = audio_ex()
    original_path = chunk.original_path
    chunk.add_source(source = (original_path, (1, 1)))
    assert(chunk.sources) == [(original_path, (1, 1))]


####################################################
### Tests for wrapper of manipulation functions ####
####################################################

def test_audio_wrapper_arg_checking():
    # Raise error if kwarg is not "audio" type
    @_audio_manipulation
    def function_with_good_kwarg(audio = None):
        return None
    with pytest.raises(ValueError):
        function_with_good_kwarg(audio = 'nope')

    @_audio_manipulation 
    def function_with_bad_kwarg(notaudio = None):
        return None
    with pytest.raises(ValueError):
        function_with_bad_kwarg(notaudio = 'not')

        

####################################################
###### Tests for every manipulation function #######
####################################################

functional_audio_manipulations = [eval(func_string) for func_string in audio_manipulations]
@pytest.mark.parametrize('function', functional_audio_manipulations)
def test_audio_manipulation_audio_is_arg(function):
    # Throws a KeyError if 'audio' is not an argument
    inspect.signature(function).parameters['audio']
    
@pytest.mark.parametrize('function', functional_audio_manipulations)
def test_audio_manipulation_returns_Audio(function, audio_ex):
    
    my_audio = audio_ex()
    returned_audio = function(audio = my_audio)
    
    # Ensure function gave us the correct return
    assert isinstance(returned_audio, Audio)

@pytest.mark.parametrize('function', functional_audio_manipulations)
def test_audio_manipulation_adds_manipulation(function, audio_ex):    
    my_audio = audio_ex()
    returned_audio = function(audio = my_audio)
    manipulation_1 = returned_audio.manipulations[0]
    
    # Create a desired dictionary of default values
    default = inspect.signature(function)
    sig_dict = dict(default.parameters)
    for key in sig_dict:
        sig_dict[key] = sig_dict[key].default
    sig_dict.pop('audio')
    
    # Ensure function added the correct entry to the manipulation list
    assert manipulation_1 == (function.__name__, sig_dict)
    
    
# Tests of the above tests
def test_audio_manipulation_test_catches_no_audio_arg():
    # Audio is not a kwarg
    def function_without_audio_arg(not_audio):
        return None
    with pytest.raises(KeyError):
        test_audio_manipulation_audio_is_arg(function_without_audio_arg)

def test_audio_manipulation_test_catches_wrong_return_format(audio_ex):
    # Manipulation does not return correct type
    def function_returning_wrong_type(audio = None):
        return True
    with pytest.raises(AssertionError):
        test_audio_manipulation_returns_Audio(function_returning_wrong_type, audio_ex)
    
    def function_with_two_returns(audio = None):
        return 'a', 'b'
    with pytest.raises(AssertionError):
        test_audio_manipulation_returns_Audio(function_with_two_returns, audio_ex)
        
def test_audio_manipulation_test_catches_audio_object_not_removed_from_manips(audio_ex):
    # Manipulation does not delete 'audio' object from arguments
    def function_that_does_not_remove_audio_from_options(audio = None):
        arguments = locals()
        audio.set_possible_manipulations(['function_that_does_not_remove_audio_from_options'])
        audio.add_manipulation('function_that_does_not_remove_audio_from_options', arguments)
        return audio
    with pytest.raises(AssertionError):
        my_audio = audio_ex()
        test_audio_manipulation_adds_manipulation(function_that_does_not_remove_audio_from_options, audio_ex)   
        
# Manipulation works exactly as it's supposed to
def test_audio_manipulation_test_passes_good_manipulation_addition(audio_ex):
    possible_manipulations = ['function_that_works']
    def function_that_works(audio = None, another = 'default'):
        arguments = locals()
        del arguments['audio_ex']
        audio = audio_ex()
        audio.set_possible_manipulations(['function_that_works'])
        
        del arguments['audio']
        audio.add_manipulation('function_that_works', arguments)
        return audio
    test_audio_manipulation_adds_manipulation(function_that_works, audio_ex)


    
####################################################
###### Tests for individual manipulation fns #######
####################################################

# Chunk extraction
    
def test_wraparound_extract():
    # test zero beginning, not getting to end of original array
    npt.assert_array_equal(_wraparound_extract(original = np.array([0, 1]), begin = 0, length = 1), np.array([0]))

    # test zero beginning, not getting to end of original array
    npt.assert_array_equal(_wraparound_extract(original = np.array([0, 1]), begin = 0, length = 2), np.array([0, 1]))

    # test zero beginning, not wrapping
    npt.assert_array_equal(_wraparound_extract(original = np.array([0, 1]), begin = 0, length = 2), np.array([0, 1]))

    # test zero beginning, wrapping around
    npt.assert_array_equal(_wraparound_extract(original = np.array([0, 1]), begin = 0, length = 3), np.array([0, 1, 0]))

    # test nonzero beginning, not wrapping
    npt.assert_array_equal(_wraparound_extract(original = np.array([0, 1]), begin = 1, length = 1), np.array([1]))

    # test nonzero beginning, wrapping around
    npt.assert_array_equal(_wraparound_extract(original = np.array([0, 1]), begin = 1, length = 3), np.array([1, 0, 1]))

    # test multiwrap
    npt.assert_array_equal(_wraparound_extract(original = np.array([0, 1]), begin = 1, length = 10), np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]))

    # test wrapping around beginning
    npt.assert_array_equal(_wraparound_extract(original = np.array([0, 1]), begin = 5, length = 3), np.array([1, 0, 1]))


# Get chunk

def test_get_chunk_skips_correctly(audio_ex):
    # Implemented due to get_chunk returning huge lengths
    # of data whenever it randomly skipped
    
    # Without skipping, desired and actual are equal to 3 decimals
    audio = audio_ex('1min.wav')
    chunked_audio = get_chunk(audio, chance_random_skip = 0) # force no skip
    desired_seconds_long = chunked_audio.sources[0][1][1]
    actual_seconds_long = chunked_audio.samples.shape[0]/chunked_audio.sample_rate
    npt.assert_almost_equal(desired_seconds_long, actual_seconds_long, decimal = 3)
    
    
    # With skipping, desired and actual should be equal to 3 decimals
    audio = audio_ex('1min.wav')
    chunked_audio = get_chunk(audio, chance_random_skip = 1) # force skip
    desired_seconds_long = chunked_audio.sources[0][1][1]
    actual_seconds_long = chunked_audio.samples.shape[0]/chunked_audio.sample_rate
    npt.assert_almost_equal(desired_seconds_long, actual_seconds_long, decimal = 3)


# Cyclic shift

def test_array_shifting():
    # Test random splitting
    random.seed(100)
    npt.assert_array_equal(_shift_array(np.array((0, 1, 2, 3, 4, 5, 6, 7))), np.array([2, 3, 4, 5, 6, 7, 0, 1]))

    # Test deterministic splitting
    npt.assert_array_equal(_shift_array(np.array([0, 1, 2]), split_point=0.5), np.array([1, 2, 0]))

    # Test deterministic splitting
    npt.assert_array_equal(_shift_array(np.array([0, 1, 2, 3]), split_point=0.5), np.array([2, 3, 0, 1]))
test_array_shifting()


# Pitch shift/time stretch

def test_divide_samples_at_set_amount():
    # Test chunk division at set amount
    array0 = np.array([0, 0, 0])
    array1 = np.array([1, 1, 1])
    array2 = np.array([2])
    all_arrays = (array0, array1, array2)
    cat_arrays = np.concatenate(all_arrays)
    divisions = _divide_samples(samples=cat_arrays, sample_rate=1, low_duration=3, high_duration=3)
    
    for idx, division in enumerate(divisions):
        npt.assert_array_equal(division, all_arrays[idx])
        

def test_divide_samples_at_random_position():
    # Test random chunk division
    random.seed(333)
    
    # Predetermined results with random.seed(333)
    predetermined = [np.array([0, 1, 2, 3, 4, 5, 6, 7]), np.array([8, 9])]

    range_10 = np.array(range(10))
    divisions = _divide_samples(samples=range_10, sample_rate=1, low_duration=0, high_duration=10)
    
    for idx, result in enumerate(divisions):
        npt.assert_array_equal(result, predetermined[idx])
        
def test_combine_samples():
    # Test that divided samples can be recombined successfully
    
    samples, sr = librosa.load(os.path.join(TEST_PATH, 'silence_10s.mp3'))
    divided = _divide_samples(samples, sample_rate=sr, low_duration=0.5, high_duration=4)
    npt.assert_array_equal(_combine_samples(divided), samples)
    
def test_random_time_stretching():
    audio = Audio(samples = np.linspace(0, 1, 10), sample_rate=1, label = 'test')
    random.seed(33)
    np.random.seed(99)
    results = time_stretch_divisions(audio)

    # predetermined results for random.seed == 3 and np.random.seed == 111
    # np.random.seed must be set because randomness in time_stretch_divisions
    # comes from np.random.normal
    predetermined = np.array([0., 0.11111111, 0.22222222, 0.33333333, 0.44444444,
                  0.55555556, 0.53444054, 0.62240575, 0.88888889, 1.])
    
    npt.assert_array_almost_equal(results.samples, predetermined)


# Random audio filtering

def test_filter_err_checking(audio_ex):
    # This audio contains values above 1 naturally,
    # and will cause errors in the filters:
    audio = audio_ex()
    original_samples = audio.samples
    assert(~(audio.samples > 1).any())

    # This filter will produce an invalid output 
    # i.e., the array will contain values above 1
    filtered_not_checked = random_filter(
        audio,
        percent_chance = 1,
        filter_type = 'highpass',
        filter_order = 5,
        filter_low = 20,
        filter_high = 30,
        error_check = False
    )
    assert(filtered_not_checked.samples is not original_samples)

    
    audio = audio_ex()
    original_samples = audio.samples
    assert(~(audio.samples > 1).any())
    # The same filter as above, but with error checking: 
    # the error check should flag the invalid content
    # in the filtered result and return the original array
    filtered_checked = random_filter(
        audio,
        percent_chance = 1,
        filter_type = 'highpass',
        filter_order = 5,
        filter_low = 20,
        filter_high = 30,
        #error_check = True # Error checking by default
    )
    assert(filtered_checked.samples is original_samples)
    
# Add audio chunks

def test_only_binary_start_amp():
    with pytest.raises(ValueError):
        _fade(array = np.array((1, 1, 1)), fade_len=3, start_amp=1.0)
    with pytest.raises(ValueError):
        _fade(array = np.array((1, 1, 1)), fade_len=3, start_amp=True)

# Assert that fading out doesn't work if fade_len is too long
def test_fade_too_long():
    with pytest.raises(IndexError):
        _fade(array = np.array((1, 1, 1, 1, 1)), fade_len=6, start_amp=1)
        
# Fade in on array exactly the same length as fade_len
def test_fade_on_exact_length_array():
    fade_in = _fade(array = np.array((1, 1, 1, 1, 1)), fade_len=5, start_amp=0)
    npt.assert_array_equal(fade_in, np.array([0., 0.25, 0.5, 0.75, 1.]))

# Fade out array longer than fade_len
def test_fade_on_long_array():
    fade_out = _fade(array = np.array((1, 1, 1, 1, 1, 1, 1)), fade_len=5, start_amp=1)
    npt.assert_array_equal(fade_out, np.array([1., 1., 1., 0.75, 0.5, 0.25, 0.]))
    
def test_wrap_fade_combos():
    # Test fade & wraparound on audio-like numpy arrays
    nowrap_nofade = _sum_samples(
        samples_original = np.array((1., 1., 500.)),
        samples_new = np.array((10., 11.)),
        sample_rate = 1,
        wraparound_fill = False,
        fade_out = False
    )
    npt.assert_array_equal(nowrap_nofade, np.array([11., 12.,  500.]))

    nowrap_fade = _sum_samples(
        samples_original = np.array((1., 1., 1., 500.)),
        samples_new = np.array((10., 10.)),
        sample_rate = 1,
        wraparound_fill = False,
        fade_out = True
    )
    npt.assert_array_equal(nowrap_fade, np.array([ 11.,   1.,   1., 500.]))

    wrap_nofade = _sum_samples(
        samples_original = np.array((1., 1., 500.)),
        samples_new = np.array((10., 11.)),
        sample_rate = 1,
        wraparound_fill = True,
        fade_out = False
    )
    npt.assert_array_equal(wrap_nofade, np.array([11., 12.,  510.]))

    # Same behavior as wrap_nofade
    wrap_fade = _sum_samples(
        samples_original = np.array((1., 1., 500.)),
        samples_new = np.array((10., 11.)),
        sample_rate = 1,
        wraparound_fill = True,
        fade_out = True
    )
    npt.assert_array_equal(wrap_nofade, np.array([11., 12.,  510.]))
    

def test_fade_on_actual_audio(audio_ex):
    # Test on actual audio without fade or wraparound
    audio_original = audio_ex('1min.wav')
    audio_new = cyclic_shift(audio_original)

    samples_original = audio_original.samples[:22050]
    samples_new = audio_new.samples[:11025]

    summed = _sum_samples(
        samples_original = samples_original,
        samples_new = samples_new,
        sample_rate = audio_original.sample_rate,
        wraparound_fill = False,
        fade_out = False)

    true_summed = np.add(samples_original, np.pad(samples_new, (0, 11025), constant_values=0, mode='constant'))
    npt.assert_array_equal(summed, true_summed)
    