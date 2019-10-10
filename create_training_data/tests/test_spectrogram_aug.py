# TODO: remove ability for spectrogram to be an arg
# Testing imports
import os
import pytest
import numpy.testing as npt
import inspect
from ..code.audio_aug import Audio
from ..code.spectrogram_aug import *

# Helper functions
from ..code.spectrogram_aug import (
    _spectrogram_manipulation,
    _remove_bands, _resize_bands)

####################################################
#################### Helpers ######################
####################################################



def get_abs_path(file):
    '''
    Get absolute path for file in same dir as this test file
    
    '''
    test_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(test_path, file)

####################################################
#################### Fixtures ######################
####################################################

@pytest.fixture
def audio_ex():
    '''
    Return Audio object for file in
    current directory
    '''

    abs_path = get_abs_path(file = 'veryshort.wav')
    
    return Audio(path = abs_path, label='test')


@pytest.fixture
def spect_ex():
    
    '''
    Return Spectrogram object for file in
    current directory
    '''
    
    def _select_options(mel = None, img = False, file = 'veryshort.wav'):
        '''
        Args:
            mel (bool or None): whether or not to compute a spectrogram.
                None: do not compute spect
                True: compute mel spect
                False: compute linear spect
            img (bool): whether or not to convert spect to image
                False: don't convert
                True: convert
            file (str): filename, for file in same dir
                as this test file, of the audio file
                to be used in creating the Spectrogram instance
        '''
        
        abs_path = get_abs_path(file = file)
        audio = Audio(path = abs_path, label='test')
        spect = Spectrogram(audio = audio)
        
        if mel is None:
            return spect
        elif mel is True:
            computed = make_mel_spectrogram(spect)
        else:
            computed = make_linear_spectrogram(spect)
        
        if not img:
            return computed
        else:
            return resize_spect_random_interpolation(computed)

    return _select_options

'''

@pytest.fixture
def spect_mel_ex():
    '#''
    Return Spectrogram object for file in
    current directory, with mel spectrogram
    already computed
    '#''
    
    path = 'veryshort.wav'
    abs_path = os.path.join(TEST_PATH, path)
    audio = Audio(path = abs_path, label='test')
    spectrogram = Spectrogram(audio = audio)
    return make_mel_spectrogram(spectrogram)


@pytest.fixture
def spect_lin_ex():
    '#''
    Return Spectrogram object for file in
    current directory, with linear spectrogram
    already computed
    ''#'
    
    path = 'veryshort.wav'
    abs_path = os.path.join(TEST_PATH, path)
    audio = Audio(path = abs_path, label='test')
    spectrogram = Spectrogram(audio = audio)
    return make_linear_spectrogram(spectrogram)



@pytest.fixture
def spect_mel_img_ex():
    ''#'
    Return Spectrogram object for file in
    current directory, with mel spectrogram
    already computed and converted to image
    '#''
    
    path = 'veryshort.wav'
    abs_path = os.path.join(TEST_PATH, path)
    audio = Audio(path = abs_path, label='test')
    spectrogram = Spectrogram(audio = audio)
    spectrogram = make_mel_spectrogram(spectrogram)
    return resize_spect_random_interpolation(spectrogram)

    
@pytest.fixture
def spect_lin_img_ex():
    '#''
    Return Spectrogram object for file in
    current directory, with linear spectrogram
    already computed and converted to image
    ''#'
    
    path = 'veryshort.wav'
    abs_path = os.path.join(TEST_PATH, path)
    audio = Audio(path = abs_path, label='test')
    spectrogram = Spectrogram(audio = audio)
    spectrogram = make_linear_spectrogram(spectrogram)
    return resize_spect_random_interpolation(spectrogram)

@pytest.fixture
def all_fixtures():
    return [audio_ex,
    spect_ex, 
    spect_mel_ex, spect_lin_ex,
    spect_mel_img_ex, spect_lin_img_ex,]'''
####################################################
########### Tests for Spectrogram class ############
####################################################

def test_Spectrogram_requires_Audio_object():
    # Must use an Audio object
    with pytest.raises(ValueError):
        spect = Spectrogram(audio = 'this is a string')
    
    
def test_Spectrogram_add_source_not_all_arguments_provided(spect_ex):

    spect = spect_ex()
    source_path = spect.audio.original_path

    # Not all arguments provided
    with pytest.raises(ValueError):
        spect.add_source(path = 'me')
    with pytest.raises(ValueError):
        spect.add_source(path = 'me', start_time = 1)    
    with pytest.raises(ValueError): 
        spect.add_source(source = (source_path, ('a')) )


def test_Spectrogram_add_source_bad_path(spect_ex):
    spect = spect_ex()
    
    # Bad source path
    with pytest.raises(FileNotFoundError):
        spect.add_source(source = ('me', (1, 2)))
    with pytest.raises(FileNotFoundError):
        spect.add_source(path = 'me', start_time = 1, duration = 2)
        
        
def test_Spectrogram_add_source_bad_duration(spect_ex):
    spect = spect_ex()
    source_path = spect.audio.original_path
        
    # Bad start time or duration
    with pytest.raises(ValueError): 
        spect.add_source(source = (source_path, ('a', 1)))
        spect.add_source(source = (source_path, (1, 'a')))
        spect.add_source(path = source_path, start_time = 'a', duration = 1) 
        spect.add_source(path = source_path, start_time = 1, duration = 'a')
        
        

def test_Spectrogram_add_source_works_correctly_kword(spect_ex):

    spect = spect_ex()
    source_path = spect.audio.original_path
    spect.add_source(path = source_path, start_time = 1, duration = 1)
    assert(spect.sources) == [(source_path, (1, 1))]

def test_Spectrogram_add_source_works_correctly_tuple(spect_ex):

    spect = spect_ex()
    source_path = spect.audio.original_path
    spect.add_source(source = (source_path, (1, 1)))
    assert(spect.sources) == [(source_path, (1, 1))]
    
    
####################################################
######## Tests of spectrogram manip wrapper ########
####################################################
    
def test_spectrogram_wrapper_accepts_spectrogram_arg(spect_ex):
    @_spectrogram_manipulation
    def function_with_spectrogram_arg(spectrogram):
        return spectrogram
    
    function_with_spectrogram_arg(spect_ex())
    
def test_spectrogram_wrapper_spectrogram_arg_cannot_be_kwarg(spect_ex):
    @_spectrogram_manipulation
    def function_with_spectrogram_kwarg(spectrogram = None, spect_ex = spect_ex):
        return spect_ex
    with pytest.raises(ValueError):
        function_with_spectrogram_kwarg(spectrogram = spect_ex)
        
####################################################
######### Tests for all manipulation funcs #########
####################################################


functional_spectrogram_manipulations = [eval(func_string) for func_string in spectrogram_manipulations]
@pytest.mark.parametrize('function', functional_spectrogram_manipulations)
def test_spectrogram_manipulation_spectrogram_is_arg(function):
    # Throws a KeyError if 'spectrogram' is not an argument
    inspect.signature(function).parameters['spectrogram']
    
@pytest.mark.parametrize('function', functional_spectrogram_manipulations)
def test_spectrogram_manipulation_returns_Spectrogram(
    function,
    spect_ex):
    
    def _test_returns(mel, img):
        spect = spect_ex(mel = mel, img = img)
        spect = function(spect)
        assert isinstance(spect, Spectrogram)
    
    try:
        # Fails for functions that require spectrogram to be computed
        _test_returns(mel = None, img = False)

    except SpectrogramNotComputedError:
        
        try:
             # Fails for functions that require spect to be converted to img
            _test_returns(mel = True, img = False)
            _test_returns(mel = False, img = False)
        
        except ImageNotComputedError:
            _test_returns(mel = True, img = True)
            _test_returns(mel = False, img = True)
            
        
@pytest.mark.parametrize('function', functional_spectrogram_manipulations)
def test_spectrogram_manipulation_adds_manipulation(
    function, 
    spect_ex
):
    '''
    '''

    spect = spect_ex()

    def _test_adds_manip(mel = None, img = False):
        '''
        Test a spectrogram returned from the func of interest
        
        Args:
            mel: whether or not to compute a spectrogram.
                None: do not compute spect
                True: compute mel spect
                False: compute linear spect
            img: whether or not to convert spect to image
                False: don't convert
                True: convert
        '''

        if mel is None:
            manip_num = 0
        elif img is False:
            manip_num = 1
        else:
            manip_num = 2

        spect = spect_ex(mel = mel, img = img)
        manip = function(spect).manipulations[manip_num]
        

        # Create a desired dictionary of default values
        default = inspect.signature(function)
        sig_dict = dict(default.parameters)
        for key in sig_dict:
            sig_dict[key] = sig_dict[key].default
        sig_dict.pop('spectrogram')
        default_addition = (function.__name__, sig_dict)
        
        # Ensure function added the correct entry to the manipulation list
        if manip != default_addition:
            raise AssertionError(f'function {function.__name__} adds bad manipulation.\n'
                                f'Bad manipulation: \n{manip}\n'
                                f'Should have been: \n{default_addition}')
    
    try:
        _test_adds_manip(mel = None, img = False)
        
    except SpectrogramNotComputedError:
        try:
            _test_adds_manip(mel = True, img = False)
            _test_adds_manip(mel = False, img = False)
            
        except ImageNotComputedError:
            _test_adds_manip(mel = True, img = True)
            _test_adds_manip(mel = False, img = True)
            
####################################################
###### Meta-tests: tests of spect manip tests ######
####################################################

def test_spect_manipulation_test_catches_no_spectrogram_arg():
    # spectrogram is not an arg or kwarg
    def function_without_spect_arg(not_spectrogram):
        return None
    with pytest.raises(KeyError):
        test_spectrogram_manipulation_spectrogram_is_arg(function_without_spect_arg)
        
def test_spect_manipulation_test_catches_wrong_return_format(spect_ex):
    # Manipulation does not return correct type
    def function_returning_wrong_type(spectrogram):
        return True
    with pytest.raises(AssertionError):
        test_spectrogram_manipulation_returns_Spectrogram(
            function_returning_wrong_type, spect_ex)
    
    def function_with_two_returns(spectrogram):
        return 'a', 'b'
    with pytest.raises(AssertionError):
        test_spectrogram_manipulation_returns_Spectrogram(function_with_two_returns, spect_ex)
        
        
def test_spect_manipulation_test_catches_Spectrogram_object_not_removed_from_manips(spect_ex):
    # Manipulation does not delete 'spectrogram' object from arguments
    def function_that_does_not_remove_spectrogram_from_options(spectrogram):
        arguments = locals()
        spectrogram.set_possible_manipulations(['function_that_does_not_remove_spectrogram_from_options'])
        spectrogram.add_manipulation('function_that_does_not_remove_spectrogram_from_options', arguments)
        return spectrogram
    
    with pytest.raises(AssertionError):
        test_spectrogram_manipulation_adds_manipulation(function_that_does_not_remove_spectrogram_from_options, spect_ex)

        
def test_spectrogram_test_passes_good_manipulation_addition(spect_ex):
    # Manipulation works exactly as it's supposed to
    possible_manipulations = ['function_that_works']
    def function_that_works(spectrogram, another = 'default'):
        arguments = locals()
        path = get_abs_path('silence_10s.mp3')
        audio = Audio(path = path, label='silence')
        spectrogram = Spectrogram(audio)
        spectrogram.set_possible_manipulations(['function_that_works'])
        
        del arguments['spectrogram']
        spectrogram.add_manipulation('function_that_works', arguments)
        
        return spectrogram
    test_spectrogram_manipulation_adds_manipulation(function_that_works, spect_ex)

def test_remove_bands_first_hi_and_last_two():
    # Remove the first band and the last two
    
    removed = _remove_bands(
        np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4]]
        ),
        min_lo = 1,
        max_lo = 1,
        min_hi = 2,
        max_hi = 2
        )
    npt.assert_array_equal(removed, np.array([
        [1, 1, 1, 1],
        [2, 2, 2, 2]]))
test_remove_bands_first_hi_and_last_two()
  

# Other tests of spectrogram manipulation funcs
def test_remove_bands_check_vals():
    # Ensure value checking of min/max bands to remove
    # Min should be less than max
    with pytest.raises(ValueError):
        test = _remove_bands(
            np.array([[0, 0, 0, 0]]),
            min_lo = 2,
            max_lo = 1,
            # These work
            min_hi = 0,
            max_hi = 1
        )

    # Can't remove negative bands
    with pytest.raises(ValueError):
        test = _remove_bands(
            np.array([[0, 0, 0, 0]]),
            min_hi = -2,
            max_hi = 2,
            # These work
            min_lo = 0,
            max_lo = 1
        )

    # Can't remove more bands than exist in spectrogram
    with pytest.raises(ValueError): 
        # Attempt to remove exactly two bands
        test = _remove_bands(
            np.array([[0, 0, 0, 0]]),
            min_lo = 1,
            min_hi = 1,
            max_lo = 1,
            max_hi = 1
        )
        
    # Can't remove all bands in spectrogram
    with pytest.raises(ValueError): 
        # Attempt to remove exactly one band
        test = _remove_bands(
            np.array([[0, 0, 0, 0]]),
            min_lo = 0,
            min_hi = 0,
            max_lo = 1,
            max_hi = 1
        )
        
def test_remove_bands_remove_no_bands():
    array = np.array([1, 2, 3, 4])
    test = _remove_bands(
        array = array,
        min_lo = 0,
        min_hi = 0,
        max_lo = 0,
        max_hi = 0
    )
    npt.assert_array_equal(test, array)
    


def test_resize_bands_cols():
    # Test resizing random columns
    test_arr = np.array([
        [0, 1],
        [1, 1],
    ])
    true_2x_stretched = np.array([
        [0, 0, 1, 1],
        [1, 1, 1, 1]])

    stretched_2x = np.rint(_resize_bands(
        array = test_arr,
        rows_or_cols = 'cols',
        chance_resize = 1,
        min_division_size = 1,
        max_division_size = 1,
        min_stretch_factor = 2,
        max_stretch_factor = 2))

    npt.assert_array_equal(stretched_2x, true_2x_stretched)


def test_resize_bands_rows():
    # Test resizing random rows
    test_arr = np.array([
        [0, 0],
        [1, 1],
    ])
    true_2x_stretched = np.array([
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1]])

    stretched_2x = np.rint(_resize_bands(
        array = test_arr,
        rows_or_cols = 'rows',
        chance_resize = 1,
        min_division_size = 1,
        max_division_size = 1,
        min_stretch_factor = 2,
        max_stretch_factor = 2))

    npt.assert_array_equal(stretched_2x, true_2x_stretched)
