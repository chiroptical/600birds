# Image module imports
import librosa
import numpy as np
import skimage.transform
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
from scipy import signal
from ..code.audio_aug import Audio
import os
import random
from functools import wraps

spectrogram_manipulations = []

####################################################
################ Spectrogram class #################
####################################################

class Spectrogram():
    def __init__(self, audio = None, mel = False, spect_settings = 'default'):
        '''
        Set up a Spectrogram object but do not create spectrogram
        '''
        global spectrogram_manipulations

        # From audio object, filled by class method _set_audio
        self.audio = None
        self.sample_rate = None
        self.samples = None
        self.labels = None
        
        # From other arguments, filled by class method _set_settings
        self.mel = None
        self.spect_settings = {}
        
        # Filled by _make_spectrogram
        self.spect = None
        self.times = None
        self.freqs = None
        self.possible_manipulations = spectrogram_manipulations
        
        # Filled by class methods
        self.manipulations = []
        self.sources = []
    
        # Set self.audio, self.samples, self.sample_rate, self.labels
        self._set_audio(audio)
 

    def __repr__(self):
        sources = [source[0] for source in self.audio.sources]
        return f'Spectrogram({sources})'
        
    def _set_audio(self, audio):
         
        if not isinstance(audio, Audio):
            raise ValueError('must pass an instance of Audio class')
        
        self.audio = audio
        self.samples = audio.samples
        self.sample_rate = audio.sample_rate
        self.labels = audio.labels
        
        for source in audio.sources:
            self.add_source(source = source)


    def set_spect_attrs(self, mel, spect, freqs = None, times = None):
        '''
        Set self.mel, self.spect, self.freqs, and self.times
        '''
        
        # Validate inputs
        if not isinstance(mel, bool):
            raise ValueError('mel must be bool')
        if not isinstance(spect, np.ndarray):
            raise ValueError('spect must be np.ndarray')
        if mel:
            if not ((freqs is None) and (times is None)):
                raise ValueError('for mel spectrogram, freqs and times are both None')     
        else:
            if not (isinstance(freqs, np.ndarray)) and (isinstance(times, np.ndarray)):
                raise ValueError('freqs and times must be np.ndarrays')
            
        self.mel = mel
        self.spect = spect
        self.freqs = freqs
        self.times = times
        
    
    def set_possible_manipulations(self, manip_list):
        '''
        Set self.possible_manipulations
        
        This determines what manipulations will be 
        considered valid in the add_manipulation step
        
        Inputs:
            manip_list: list of strings of names of
                possible manipulations
        '''
        
        self.possible_manipulations = manip_list
    
        
    def add_manipulation(self, manip, manip_kwargs):
        '''
        Add a manipulation to the list of manipulations
        
        Appends a tuple, (manip, manip_kwargs), to the
        list of manipulations for this audio, 
        self.manipulations.
        
        Inputs: 
            manip: function name of the manipulation
            manip_kwargs: keyword arguments used to
                call the manipulation function
        '''
        
        try:
            assert manip in self.possible_manipulations
        except:
            raise ValueError(f'Invalid input to Spectrogram.add_manipulation(): {manip} not in list of valid manipulations')
            
        try:
            assert isinstance(manip_kwargs, dict)
        except:
            raise ValueError(f'Invalid inputs to Spectrogram.add_manipulation(): {manip_kwargs} must be dict')
        
        self.manipulations.append((manip, manip_kwargs))
        return True
    
    
    def add_source(self, path = None, start_time = None, duration = None, source = None):
        '''
        Add a source to the list of sources
        
        Appends a tuple, (path, dur_tuple),
        to the list of sources for this spectrogram,
        self.sources. Can be called either with each 
        individual component (path, start_time, duration)
        or with a `source` tuple straight from another 
        Audio object.
        
        Inputs:
            path (string): path to the source file
            start_time (float): start time in seconds
                within the source file
            duration (float): duration in seconds
                within the source file
            source (tuple): a tuple containing
                the above information:
                (path, (start_time, duration))
        '''
        
        # Validate all inputs were given
        
        # If using source
        if source:
            if (not isinstance(source, tuple)) or len(source) != 2:
                raise ValueError(f'Spectrogram.set_sources() input source must be a tuple of len 2. Got: {source}')
            path = source[0]
            dur_tuple = source[1]
            if (not isinstance(dur_tuple, tuple)) or len(dur_tuple) != 2:
                raise ValueError(f'Second element of input source must be a tuple of len 2. Got: {dur_tuple}')
            start_time = dur_tuple[0]
            duration = dur_tuple[1]
        # If using path, start_time, and duration:
        else:
            if (not path) or (not start_time) or (not duration):
                raise ValueError(f'If not calling Spectrogram.set_sources() with a source, must provide all three of: path, start_time, duration')
            source = (path, (start_time, duration))
        
        # Check inputs for correctness
        if not os.path.exists(path):
            raise FileNotFoundError(f'Source file does not exist: {path}')
        try:
            start_time = float(start_time)
            duration = float(duration)
        except ValueError:
            raise ValueError(f'start time and duration must be floats. Given type(start_time) == {type(start_time)}, type(duration) == {type(duration)}')
        
        # Append to list of sources
        self.sources.append(source)
        
        return True


####################################################
##### Wrapper for spectrogram manipulation fns #####
####################################################

def spectrogram_manipulation(func):
    '''
    Functionality for spectrogram manipulation functions
    
    Wrapper for spectrogram manipulation that ensures the
    input to the function is of class Spectrogram and 
    adds a record of the manipulation to the 
    Spectrogram object's `manipulations` attribute.
    
    When a manipulation function is first defined
    with this wrapper, this function appends the
    manipulation to a list of valid manipulations,
    which will be tested to ensure the manipulation
    function has the attributes required (described
    below).
    
    Inputs:
      - func (function): a function with all arguments
        provided as kwargs, except one argument called `spectrogram`.
        Function must return an Spectrogram object and a list of
        keyword arguments that does not include the  
        reference to the manipulated Spectrogram object
    
    Returns:
      - wrapped version of the function that returns only 
        the Spectrogram object.
    '''
    
    global spectrogram_manipulations
    spectrogram_manipulations.append(func.__name__)
    
    @wraps(func) #Allows us to call help(func)
    def validate_spectrogram(*args, **kwargs):
        try:
            spect_arg = kwargs['spectrogram']
        except KeyError:
            try:
                spect_arg = args[0]
            except IndexError:
                raise ValueError(f"no first argument given. If spectrogram kwarg is not used, a Spectrogram object must be provided as first argument.")
            
        if not isinstance(spect_arg, Spectrogram):
            raise ValueError(f"a Spectrogram object must be provided as keyword argument spectrogram. Got {type(spect_arg)}")
            
        # Run manipulation
        manipulated_spect, arguments = func(*args, **kwargs)
        
        # Add manipulation to list of manipulations
        manipulated_spect.add_manipulation(func.__name__, arguments)
        
        return manipulated_spect
    
    return validate_spectrogram



####################################################
####### Spectrogram manipulation functions #########
####################################################

class SpectrogramNotComputedError(BaseException):
    pass

####################################################
# Functions for computing spectrograms

@spectrogram_manipulation
def make_mel_spectrogram(
    spectrogram,
    fmax = None,
    fmin = 0,
    n_mels = 128,
    S = None,
    n_fft = 2048,
    hop_length = 360,
    win_length = 1536,
    window = 'hann',
    center = True,
    power = 2.0,
):

    '''
    Make spectrogram in decibels

    Create a mel spectrogram in decibels between
    a certain band of frequencies (fmax and fmin).

    Args:
        fmax: maximum frequency to include in spectrogram. 
            If not provided, automatically set to 1/2 sample rate.
        fmin: minimum frequency to include in spectrogram
        other arguments: see the librosa documentation:
            https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
            https://librosa.github.io/librosa/generated/librosa.filters.mel.html

    Returns:
        decibel-formatted spectrogram in the form of an np.array
    '''
    options = locals()
    del options['spectrogram']
    
    y = spectrogram.samples
    sr = spectrogram.sample_rate
    if not fmax:
        fmax = sr/2

    spect = librosa.feature.melspectrogram(
        y = y, 
        sr = sr, 
        fmax = fmax,
        fmin = fmin,
        S = S,
        n_fft = n_fft,
        hop_length = hop_length,
        win_length = win_length,
        window = window,
        center = center,
        power = power,
        n_mels = n_mels
    )

    spectrogram.set_spect_attrs(
        mel = True,
        spect = spect,
        freqs = None,
        times = None)

    return spectrogram, options


@spectrogram_manipulation
def make_linear_spectrogram(
    spectrogram,
    window = "hann",
    nperseg = 512,
    noverlap = 256,
    nfft = 512,
    scaling = "spectrum"
):
    options = locals()
    del options['spectrogram']
    
    freqs, times, spect = signal.spectrogram(
        spectrogram.samples,
        spectrogram.sample_rate,
        window = window,
        nperseg = nperseg,
        noverlap = noverlap,
        nfft = nfft,
        scaling = "spectrum",
    )
    
    spectrogram.set_spect_attrs(
        mel = False,
        spect = spect,
        freqs = freqs,
        times = times)
    
    return spectrogram, options

####################################################
# Removing high/low spectrogram rows

def _remove_bands(array, min_lo, max_lo, min_hi, max_hi):
    
    # Ensure sensible hi/lo bands
    values = [min_lo, max_lo, min_hi, max_hi]
    for value in values: 
        if value < 0: 
            raise ValueError('Number of bands to remove must be positive')
    if (min_lo > max_lo) or (min_hi > max_hi):
        raise ValueError('Minimum number of bands to remove must be less than or equal to maximum')
    if max_lo + max_hi >= array.shape[0]:
        raise ValueError('Maximum number of bands to remove cannot be greater than or equal to number of bands in spectrogram')

    
    # Note the high-frequency bands are the last bands in the spectrogram
    hi_remove = random.randint(min_hi, max_hi)
    lo_remove = random.randint(min_lo, max_lo)

    if hi_remove == 0:
        return array[lo_remove:]
    else:
        return array[lo_remove:-hi_remove]

    
@spectrogram_manipulation
def remove_random_hi_lo_bands(
    spectrogram,
    min_lo = 0,
    max_lo = 10,
    min_hi = 0,
    max_hi = 6,
):
    '''
    Remove random bands at top and bottom of spectrogram
    '''
    options = locals()
    del options['spectrogram']
    
    if not isinstance(spectrogram.spect, np.ndarray):
        raise SpectrogramNotComputedError('spectrogram.spect has not been computed.'
                        ' Use make_mel_spectrogram or make_linear_spectrogram')
    
    spectrogram.spect = _remove_bands(spectrogram.spect, **options)
    
    return spectrogram, options

####################################################
# Resize random columns/rows

def _resize_bands(
    array,
    rows_or_cols,
    chance_resize,
    min_division_size,
    max_division_size,
    min_stretch_factor,
    max_stretch_factor
):
    
    # Probabilistically don't resize
    if random.random() > chance_resize:
        return array
    
    
    
    if rows_or_cols == 'rows': axis = 0
    else: axis = 1
        
    len_array = array.shape[axis]
    len_divisions = 0
    stretched_divisions = []
    
    # Check sensibility of division sizes and stretch factors
    factors = [
        chance_resize,
        min_division_size,
        max_division_size,
        min_stretch_factor,
        max_stretch_factor
    ]
    for factor in factors: 
        if factor < 0: 
            raise ValueError('Division and stretch sizes must be > 0')
    if (min_division_size > max_division_size) or \
        (min_stretch_factor > max_stretch_factor):
            raise ValueError('Minimum division and stretch sizes must be smaller than maximum')

    
    if rows_or_cols not in ['rows', 'cols']:
        raise ValueError("Parameter rows_or_cols must be either 'rows' or 'cols'")
 
            
            
    # Incrementally divide spectrogram and stretch portions

    current_position = 0
    while current_position < len_array:
        # Grab randomly sized portion of spectrogram
        size_division = random.randint(min_division_size, max_division_size)
        if rows_or_cols == 'rows':
            division = array[current_position:current_position+size_division, :]
        else:
            division = array[:, current_position:current_position+size_division]

        # Select stretch factor
        stretch_factor = random.uniform(min_stretch_factor, max_stretch_factor)

        # Stretch portion of spectrogram
        if rows_or_cols == 'cols': multiplier = (1, stretch_factor)
        else: multiplier = (stretch_factor, 1)
        stretched = skimage.transform.rescale(
            division,
            multiplier,
            preserve_range = True,
            multichannel = False
        )
        stretched_divisions.append(stretched)
       
        current_position += size_division
    
    
    # Concatenate all stretched divisions
    return np.concatenate(stretched_divisions, axis = axis)



@spectrogram_manipulation
def resize_random_bands(
    spectrogram,
    rows_or_cols = 'rows',
    chance_resize = 0.5,
    min_division_size = 10,
    max_division_size = 100,
    min_stretch_factor = 0.9,
    max_stretch_factor = 1.1
):
    '''
    Resize random row or column chunks of spectrogram
    
    With a certain percentage chance, divide 
    spectrogram into random chunks along one axis 
    (row-wise or column-wise chunks) and resize
    each chunk with a randomly chosen scaling factor.
    
    Args:
        spectrogram (np.array): the spectrogram image
        rows_or_cols (string, 'rows' or 'cols'): whether
            to divide spectrogram into chunks by rows 
            (horizontal chunks spanning all times of the
            spectrogram) or columns (vertical chunks
            spanning the whole frequency range of the 
            spectrogram)
        chance_resize (float between 0 and 1): percent
            chance of dividing up the spectrogram and
            performing resizing operations
        min_division_size (int > 0): minimum size in pixels
            for each spectrogram division
        max_division_size (int): maximum size in pixels 
            for each spectrogram division.
        min_stretch_factor (float): minimum scaling factor.
            values < 1 allow spectrogram to shrink
        max_stretch_factor (float): maximum scaling factor.
            values > 1 allow spectrogram to stretch
    
    Returns:
        either the rescaled spectrogram 
            (with probability = chance_resize) or the 
            original spectrogram (prob = 1 - chance_resize)
    '''
    options = locals()
    del options['spectrogram']
    
    if spectrogram.spect is None:
        raise SpectrogramNotComputedError('spectrogram.spect is not computed yet.'
                                         ' Use make_mel_spectrogram or make_linear_spectrogram')
    
    
    spectrogram.spect = _resize_bands(array = spectrogram.spect, **options)
    
    return spectrogram, options


####################################################
# Resize spectrogram to network dimensions

class ImageNotComputedError(Exception):
    pass

@spectrogram_manipulation
def resize_spect_random_interpolation(
    spectrogram,
    width = None,
    height = None,
    chance_random_interpolation = 0.15
):
    '''
    Resize spectrogram with random interpolation
    
    Convert np.ndarray spectrogram into a PIL Image, and
    resize it using either a Lanczos filter or a randomly
    selected filter.
    
    Args:
        spectrogram (np.array): spectrogram np array
        width (int): width to resize to
        height (int): height to resize to
        chance_random_interpolation (float between 0 and 1):
            the percent chance that instead of a Lanczos
            filter, a different filter will be used to 
            resize. Filter choices are Box, Nearest, Bilinear,
            Hamming, and Bicubic, as implemented in PIL.
    
    Returns:
        a resized PIL image, in RGB
    '''
    options = locals()
    del options['spectrogram']
    
    if spectrogram.spect is None:
        raise SpectrogramNotComputedError('spectrogram.spect is not computed yet.'
                                         ' Use make_mel_spectrogram or make_linear_spectrogram')
    
    spect = spectrogram.spect

    if width is None:
        width = spect.shape[0]
    if height is None:
        height = spect.shape[1]
    
    if not isinstance(width, int) and isinstance(height, int):
        raise ValueError('Height and width must be given in integers')
    

    # Convert spectrogram to image:
    spect = spect[::-1, ...] # Flip array
    spect_image = PIL.Image.fromarray(spect.astype(np.uint8))
    spect_image = PIL.ImageOps.invert(spect_image) # Invert colors

    # Randomly choose interpolation
    if random.random() > chance_random_interpolation:
        interpolation = PIL.Image.LANCZOS
    else:
        interpolation = random.choice([
            PIL.Image.BOX,
            PIL.Image.NEAREST,
            PIL.Image.BILINEAR,
            PIL.Image.HAMMING,
            PIL.Image.BICUBIC
        ])

    resized = spect_image.resize((width, height), interpolation) 
    rgb = resized.convert('RGB')
    spectrogram.spect = rgb
    
    return spectrogram, options

####################################################
# Jitter brightness, contrast, saturation

@spectrogram_manipulation
def color_jitter(
    spectrogram,
    brightness = 0.3,
    contrast = 0.3,
    saturation = 0.3,
    hue = 0.01
):
    '''
    Randomly change colors of image
    
    Randomly change the brightness, contrast,
    saturation, and hue of an image. Random
    choices are chosen uniformly from a distribution
    that includes the contrast value. This function's
    behavior mimics the behavior of the ColorJitter
    object in pytorch: 
        https://github.com/pytorch/vision/blob/3483342733673c3182bd5f8a4de3723a74ce5845/torchvision/transforms/transforms.py
    by centering brightness, contrast, and saturation
    jitters around 1, and centering hue jitters around 0.
    
    Args:
        image (PIL image):
        brightness (float > 0): how much to jitter
            brightness. Jitter amount is chosen uniformly
            from [max(0, 1 - brightness), 1 + brightness].
        contrast (float > 0): how much to jitter
            contrast. Jitter amount is chosen uniformly
            from [max(0, 1 - contrast), 1 + contrast].
        saturation (float > 0): how much to jitter
            saturation. Jitter amount is chosen uniformly
            from [max(0, 1 - saturation), 1 + saturation].
        hue (float, 0 <= hue <= 0.5): how much to jitter hue.
            Jitter amount is chosen uniformly from [-hue, hue].
    '''
    options = locals()
    del options['spectrogram']
    
    spect = spectrogram.spect
    
    if spect is None:
        raise SpectrogramNotComputedError('spectrogram.spect is not computed yet.'
                                         ' Use make_mel_spectrogram or make_linear_spectrogram')
    if not isinstance(spect, PIL.Image.Image):
        raise ImageNotComputedError(f"spectrogram should be PIL Image. Use resize_spect_random_interpolation")
    if spect.mode != 'RGB':
        raise ValueError(f"image.mode should be 'RGB'. Got {image.mode}")
    
    if brightness is not None:
        brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        enhancer = PIL.ImageEnhance.Brightness(spect)
        spect = enhancer.enhance(brightness_factor)

    if contrast is not None:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        enhancer = PIL.ImageEnhance.Contrast(spect)
        spect = enhancer.enhance(contrast_factor)

    if saturation is not None:
        saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        enhancer = PIL.ImageEnhance.Color(spect)
        spect = enhancer.enhance(saturation_factor)
    
    if hue is not None: 
        if (hue > 0.5) or (hue < -0.5):
            raise ValueError(f"hue should be between -0.5 and 0.5. Got {hue}")            
        hue_factor = random.uniform(-hue, hue)

        h, s, v = spect.convert('HSV').split()
        
        np_h = np.array(h, dtype=np.uint8)
        # uint8 addition take cares of rotation across boundaries
        with np.errstate(over='ignore'):
            np_h += np.uint8(hue_factor * 255)
        h = PIL.Image.fromarray(np_h, 'L')

        spect = PIL.Image.merge('HSV', (h, s, v)).convert('RGB')
    
    spectrogram.spect = spect
    
    return spectrogram, options

####################################################

