# Audio module imports
import inspect
import librosa
import os
import numpy as np
import random
import math

audio_manipulations = []

####################################################
################### Audio class ####################
####################################################

class Audio():
    '''
    A class to hold audio
    
    This class is a mutable container for
    information about an approx. 5s chunk
    of audio. It has the following attributes:
    
        self.samples (np.ndarray), audio samples
        self.sample_rate (float or int), samples/sec
            of self.samples
        self.manipulations (list of tuples), a list of 
            manipulations that have been performed on 
            this audio. Each tuple has the name of the
            function used to manipulate the 
        self.sources (list of tuples), a list of tuples
            where each tuple contains a path to the source,
            the start time of the samples, and the 
            duration of the samples
        self.labels (list of strings), a list of class 
            labels for all the contributing audio sources
    
    It may also have these attributes:
        self.original_path (string): path to original object.
            This is optional for convenience in creating/testing
            Audio objects from samples.
    '''
    
    global audio_manipulations
    
    __slots__ = ['original_path', 'samples', 'sample_rate', 
                 'possible_manipulations', 'manipulations',
                 'sources', 'labels'
                ]
        
    def __repr__(self):
        sources = [source[0] for source in self.sources]
        return f'Audio({sources})'
    
    def __init__(self, label, path = None, samples = None, sample_rate = None):

        self.original_path = None
        self.samples = None
        self.sample_rate = None
        self.possible_manipulations = []
        self.manipulations = []
        self.sources = []
        self.labels = set()
        
        # Obtain samples and sample rate
        if path:
            if samples:
                raise ValueError('Only one of `path` or `samples` can be provided')
            else: 
                self.original_path = path
                
                # Load all samples
                samples, sample_rate = self._load_audio(path)
                self.set_samples(samples)
                self.sample_rate = sample_rate   
        elif samples.any():
            if not sample_rate:
                raise ValueError('If `samples` are used, must provide `sample_rate`')
            else:
                self.samples = samples
                self.sample_rate = sample_rate
        else:
            raise ValueError('A non-empty `path` or `samples` must be provided')
        
        # Add label or labels
        self.add_label(label)
        
        # Set possible manipulations to global variable by default
        self.possible_manipulations = audio_manipulations
        
    def _load_audio(self, path):  
        # Load file
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")
        return librosa.load(path) #load samples from path

    def set_samples(self, samples):
        if not isinstance(samples, np.ndarray):
            raise ValueError(f'Samples must be numpy.ndarray')
        if samples.ndim != 1:
            raise ValueError(f'Multi-channel sample array provided. (Dimensions: {samples.shape})')
        
        self.samples = samples
    
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
        
        if manip not in self.possible_manipulations:
            raise ValueError(f'Invalid input to Audio.add_manipulation(): {manip} not in list of valid manipulations')
            
        if not isinstance(manip_kwargs, dict):
            raise ValueError(f'Invalid inputs to Audio.add_manipulation(): '
                             f'manip_kwargs must be dict. Got {type(manip_kwargs)}')
        
        self.manipulations.append((manip, manip_kwargs))
        return True
    
    
    def add_label(self, label):
        '''
        Add label or set of labels to self.labels
        
        Inputs:
            label (string, list of strings, or
                set of strings): labels to add
        '''
        if isinstance(label, str):
            self.labels.add(label)
        elif isinstance(label, list) or isinstance(label, set):
            for l in label:
                self.labels.add(label)
    
    
    def add_source(self, path = None, start_time = None, duration = None, source = None):
        '''
        Add a source to the list of sources
        
        Appends a tuple, (path, dur_tuple),
        to the list of sources for this audio,
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
                raise ValueError(f'Audio.set_sources() input source must be a tuple of len 2. Got: {source}')
            path = source[0]
            dur_tuple = source[1]
            if (not isinstance(dur_tuple, tuple)) or len(dur_tuple) != 2:
                raise ValueError(f'Second element of input source must be a tuple of len 2. Got: {dur_tuple}')
            start_time = dur_tuple[0]
            duration = dur_tuple[1]
        # If using path, start_time, and duration:
        else:
            if (not path) or (not start_time) or (not duration):
                raise ValueError(f'If not calling Audio.set_sources() with a source, must provide all three of: path, start_time, duration')
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
######## Wrapper for manipulation functions ########
####################################################

def _audio_manipulation(func):
    '''
    Functionality for audio manipulation functions
    
    Wrapper for audio manipulation that ensures the
    input to the function is of class Audio and 
    adds a record of the manipulation to the 
    Audio object's `manipulations` attribute.
    
    When a manipulation function is first defined
    with this wrapper, this function appends the
    manipulation to a list of valid manipulations,
    which will be tested to ensure the manipulation
    function has the attributes required (described
    below).
    
    Inputs:
      - func (function): a function with all arguments
        provided as kwargs, except one positional argument 
        called `spectrogram`. This function must return 
        a Spectrogram object.
    
    Returns:
      - wrapped version of the function that returns only 
        the Audio object.
    '''
    from functools import wraps
    
    global audio_manipulations
    audio_manipulations.append(func.__name__)
    
    @wraps(func) #Allows us to call help(func)
    def validate_audio(*args, **kwargs):
        if 'audio' in kwargs.keys():
            raise ValueError('audio must not be given as a keyword argument')
        else:
            audio_arg = args[0]
        
        if not isinstance(audio_arg, Audio):
            raise ValueError("an Audio object must be provided as first argument or keyword argument 'audio'")
        
        if len(args) > 1:
            raise ValueError('only one argument, audio, can be positional.'
                            'Others must be called as keyword arguments.'
                            f'Got {len(args)} positional arguments.')

        # Run manipulation
        manipulated_audio = func(audio_arg, **kwargs)
        
        # Create a dictionary of kwargs function was called with,
        # including default kwargs if the default was used
        param_dict = kwargs
        for param in inspect.signature(func).parameters.values():
            if param.name not in param_dict.keys():
                if param.default is not param.empty:
                    param_dict[param.name] = param.default
        
        # Add manipulation to list of manipulations
        manipulated_audio.add_manipulation(func.__name__, param_dict)
        
        return manipulated_audio
    
    return validate_audio



####################################################
############### Manipulation functions #############
####################################################

####################################################
################# Chunk extraction #################

def _wraparound_extract(original, begin, length):
    '''
    Extracts elements from numpy.array in a "wraparound" fashion
    
    Extracts a certain number of elements from 
    a numpy.array starting at a certain position.
    If the chosen position and length go
    past the end of the array, the extraction
    "wraps around" to the beginning of the numpy.array
    as many times as necessary. For instance:
    
    _wraparound_extract(
        original = [0, 5, 10],
        begin = 1, 
        length = 7) -> [5, 10, 0, 5, 10, 0, 5]
    
    Args:
        original (np.array): the original array 
        begin (int): beginning position to extract
        length (int): number of elements to extract
    '''

    # Get `head`: the array after the beginning position
    assert isinstance(original, np.ndarray)
    len_original = original.shape[0]
    begin = begin % len_original
    head = original[begin:]
    len_head = head.shape[0]

    # Number of elements we require for full wrap-around
    wrap_needed = length - len_head

    # Generate the desired list, wrapped if necessary
    if wrap_needed > 0:
        repeats = np.tile(original, int(wrap_needed/len_original))
        tail = np.array(original[ : (wrap_needed % len_original)])
        desired_list = np.concatenate((head, repeats, tail))
    else:
        desired_list = original[begin:begin+length]
    
    return desired_list

@_audio_manipulation
def get_chunk(
    audio,
    start_position = None, # randomize start position
    duration = 5, # 5 seconds
    duration_jitter = 0.5, #jitter duration +- 0.5s
    chance_random_skip = 0.3 #randomly skip 30% of the time
):
    '''
    Extracts chunk of audio with some augmentation
    
    Extracts samples of audio from a master list
    of samples. 
    
    Available data augmentation options include:
        - selecting a position to start extracting from
          or allowing function to randomly choose start
        - selecting duration of chunk and allowing
          for random jitter of duration
        - randomly skipping some number of samples from
          0 to the length of the chunk
    
    If the chunk to be extracted reaches the end of the
    samples, the chunk will "wrap around" and start
    reading from the beginning of the samples.
    
    Args:
        audio (instance of class Audio): Audio object to remove chunk from
        start_position (int): position in the file to start
            extracting samples from. If None, the start position 
            is chosen randomly
        duration (float): desired duration, in seconds, 
            of chunk to extract
        duration_jitter (float): if this value is not 0,
            the duration of the chunk extracted will 
            be randomly selected from the range 
            (duration - duration_jitter, duration + duration_jitter)
        chance_random_skip (float between 0 and 1):
            percent chance of random skipping. In a random skip,
            a position within the chunk will be randomly
            selected, and from that position in the 
            audio file, a random number of samples will 
            be skipped. The number of samples skipped is between
            0 and the number of samples in the entire chunk
    
    Returns to wrapper:
        audio (Audio): manipulated audio object
        options (dict): options the function was called with
    
    Returns when wrapped:
        audio (Audio): manipulated audio object
        
    '''
    
    # Get a random start position
    num_samples = len(audio.samples)
    if not start_position:
        start_position = random.randint(0, num_samples)

    # Convert seconds to samples
    seconds_to_extract = duration + random.uniform(-duration_jitter, duration_jitter)
    samples_to_extract = int(seconds_to_extract * audio.sample_rate)
    
    # Get chunks with skip in the middle with probability = chance_random_skip
    if random.random() < chance_random_skip:
        len_before_skip = random.randint(0, samples_to_extract)
        len_of_skip = random.randint(0, samples_to_extract)
        
        chunk_1_start = start_position
        chunk_1_len = len_before_skip
        chunk_2_start = chunk_1_start + chunk_1_len + len_of_skip
        chunk_2_len = samples_to_extract - len_before_skip
        
        chunk_1 = _wraparound_extract(
            audio.samples,
            begin = chunk_1_start,
            length = chunk_1_len)
        chunk_2 = _wraparound_extract(
            audio.samples,
            begin = chunk_2_start,
            length = chunk_2_len)
        
        chunk = np.concatenate((chunk_1, chunk_2))
    
    # Otherwise get contiguous chunk
    else:
        chunk = _wraparound_extract(audio.samples, start_position, samples_to_extract) 
        
    start_position_seconds = start_position / audio.sample_rate
    start_and_len = (start_position_seconds, seconds_to_extract)
    
    
    # Update attributes of audio function
    audio.set_samples(samples = chunk)
    audio.add_source(
        path = audio.original_path,
        start_time = start_position_seconds,
        duration = seconds_to_extract)
    
    return audio


####################################################
##################### Cyclic shift #################

def _shift_array(array, split_point = None):
    '''
    Shift array cyclicly by a random amount
    
    Shift array cyclicly by a random amount. Equivalent to
    splitting array into two parts at a random element, then
    switching the order of the parts.
    
    Args: 
        array (np.array): 1D-array to be split
        split_point (float): percentage from (0, 1) describing
            where in array to split -- for testing purposes.
            For stochastic splitting, leave as None.
    
    Returns:
        shifted_array: shifted array
    '''
    
    if not isinstance(array, np.ndarray):
        raise ValueError('array must be np.ndarray')
    length = array.shape[0]
    
    # Stochastic split point, or split point by floor of split_point * length of array
    if not split_point: split_point = random.randint(0, length)
    else: split_point = int(split_point * length)
    
    return np.concatenate((array[split_point:], array[:split_point]))


@_audio_manipulation
def cyclic_shift(audio, split_point = None):
    '''
    Shift audio samples by a random amount
    
    Inputs: 
        audio (Audio object)
        split_point: where to split the things
    '''
    
    new_samples = _shift_array(audio.samples, split_point = split_point)
    
    audio.set_samples(new_samples)
    
    return audio


####################################################
###### Divided pitch shift/time stretch ############


def _divide_samples(
    samples,
    sample_rate,
    low_duration = 0.5,
    high_duration = 5
):
    '''
    Divide audio samples into random-sized segments
    
    Divide audio samples into random-sized segments
    between the desired durations. The number
    of segments is not deterministic.
    
    Args:
        samples (np.ndarray): 1d array of samples
        sample_rate (int or float): sample rate of samples
        low_duration (float): minimum duration
            in seconds of any segment
        high_duration (float): maximum duration
            in seconds of any segment
    
    Returns:
        segments, list of sample lists
    '''

    min_chunk = int(low_duration * sample_rate)
    max_chunk = int(high_duration * sample_rate)
    
    samples_to_take = samples.copy()
    
    segments = []
    
    while samples_to_take.shape[0]:
        seg_size = random.randint(min_chunk, max_chunk)
        segment, samples_to_take = np.split(samples_to_take, [seg_size])
        segments.append(segment)
    
    return segments


def _combine_samples(divided):
    '''
    Recombine divided sample arrays
    
    Combine divided sample arrays back into a 
    single array, perhaps after each division
    has been modified by pitch shifting, time stretching, etc.
    
    Args:
        divided (list of np.ndarrays): list of sample arrays
            divided by _divide_samples()
    
    Returns:
        sample arrays concatenated back into a single array
    '''
    
    return np.concatenate(divided)

@_audio_manipulation
def time_stretch_divisions(
    audio,
    low_division_duration = 0.5,
    high_division_duration = 4,
    chance_per_division = 0.50,
    mean_stretch = 1,
    sd_stretch = 0.05
):
    '''
    Time stretch divisions
    
    Given an Audio object, divide its samples and
    time stretch each division with some probability. 
    
    Args:
        audio (Audio object): audio object to
            be divided and time-stretched
        low_division_duration (float): minimum duration
            in seconds of any segment
        high_division_duration (float): maximum duration
            in seconds of any segment
        chance_per_division (float between 0 and 1): for
            each division, the chance it will be time-stretched
        mean_stretch (float): the mean stretch multiplier.
            == 1 is no stretch; > 1 is sped up, < 1 is slowed down
        sd_stretch (float > 0): the sd of the stretch 
            distribution. 
    
    Returns:
        stretched_divisions, time-stretched divisions
    '''
    
    samples = audio.samples
    sample_rate = audio.sample_rate
    divisions = _divide_samples(
        samples,
        sample_rate = sample_rate, 
        low_duration = low_division_duration,
        high_duration = high_division_duration)
    
    stretched_divisions = []

    for d in divisions:
        stretched_d = d
        # Stretch with some chance
        if random.random() < chance_per_division:
            stretch_factor = np.random.normal(
                loc = mean_stretch,
                scale = sd_stretch)
            if len(stretched_d) > 1:
                stretched_d = librosa.effects.time_stretch(y = stretched_d, rate = stretch_factor)
        stretched_divisions.append(stretched_d)
    
    recombined = _combine_samples(stretched_divisions)
    audio.set_samples(recombined)
    
    return audio

@_audio_manipulation
def pitch_shift_divisions(
    audio,
    low_division_duration = 0.5,
    high_division_duration = 4,
    chance_per_division = 0.40,
    mean_shift = 0,
    sd_shift = 0.25
):
    '''
    Time stretch divisions
    
    Given an Audio object, divide its samples and
    pitch-shift each division with some probability. 
    The mean_shift and sd_shift should be given in "fractional
    half-steps," e.g. 0.25 = 1/4th of a half-step = 25 cents.
    
    Args:
        audio (Audio object): audio object to
            be divided and time-stretched
        low_division_duration (float): minimum duration
            in seconds of any segment
        high_division_duration (float): maximum duration
            in seconds of any segment
        chance_per_division (float between 0 and 1): for
            each division, the chance it will be time-stretched
        mean_shift (float): the mean pitch shift in (fractional) half-steps
            == 0 is no shift; > 0 is shift up; < 1 is shift down
        sd_shift (float > 0): the sd of the shift 
            distribution in cents
    
    Returns:
        shifted_divisions, pitch-shifted divisions
    '''
    
    samples = audio.samples
    sample_rate = audio.sample_rate
    divisions = _divide_samples(
        samples,
        sample_rate = sample_rate, 
        low_duration = low_division_duration,
        high_duration = high_division_duration)
    
    shifted_divisions = []
    
    for d in divisions:
        shifted_d = d
        if random.random() < chance_per_division:
            shift_factor = np.random.normal(
                loc = mean_shift,
                scale = sd_shift)
            shifted_d = librosa.effects.pitch_shift(
                y = shifted_d,
                sr = sample_rate,
                n_steps = shift_factor)
            
        shifted_divisions.append(shifted_d)
        
    recombined = _combine_samples(shifted_divisions)
    audio.set_samples(recombined)
    
    return audio

####################################################
################ Random audio filtering ############

@_audio_manipulation
def random_filter(
    audio,
    percent_chance = 0.20,
    filter_type = None,
    filter_order = None,
    filter_low = None,
    filter_high = None,
    error_check = True
):
    '''
    Randomly filter audio samples
    
    With some probability, apply a filter to `samples`. 
    Some or all of the filter's characteristics can be 
    provided by the user; otherwise, they are
    are randomly selected from the following options:
    
    Type: lowpass, highpass, bandpass, bandstop
    Order: 1-5
    Low cutoff frequency: from 1Hz to (sample_rate/2) - 1 Hz
    High cutoff frequency (bandpass 
        and bandstop filters): from low_freq+1 
        to (sample_rate/2) - 1 Hz
        
    If filter output contains values not between -1.0 and 1.0,
    the original signal is returned to avoid glitchy filters.
    '''
    
    from scipy.signal import butter, lfilter
    
    samples = audio.samples
    sample_rate = audio.sample_rate
    
    if random.random() < percent_chance:
        
        # Nyquist frequency
        nyq = 0.5 * sample_rate
        
        # Select random filter choices
        if not filter_type: filter_type = random.choice(
            ['lowpass', 'highpass', 'bandpass', 'bandstop'])
        if not filter_order: filter_order = random.randint(1, 5)
        if not filter_low: filter_low = random.randint(1, (nyq - 1))
        if not filter_high:
            if filter_type in ['bandpass', 'bandstop']:
                filter_high = random.randint(filter_low, nyq - 1)
            else:
                filter_high = nyq - 1
        

        # Filter the audio
        low = filter_low / nyq
        high = filter_high / nyq
        b, a = butter(filter_order, [low, high], btype='band')
        filtered = lfilter(b, a, samples)

         # Set samples to filtered if not error checking, or if passes error check
        if not error_check:
            audio.set_samples(filtered)
        elif error_check:
            if ~(np.less(filtered, -1, where=~np.isnan(filtered)).any()) and \
               ~(np.greater(filtered, 1, where=~np.isnan(filtered)).any()):
                audio.set_samples(filtered)
    
    return audio



####################################################
################ Adding audio chunks ###############

def _fade(array, fade_len, start_amp=1):
    '''
    Fade audio in or out
    
    Args:
        array (np.array): 1d audio array to fade
            in or out
        fade_len (int): the number of samples over which
            the fade should occur; must be smaller than 
            array.shape[0]
        start_amp (int, 1 or 0): whether to start at full 
            volume and fade out (1) or start at
            0 volume and fade in (0)
        
    '''
    
    if not ((start_amp is 0) or (start_amp is 1)):
        raise ValueError(f'start_amp must be either 0 or 1. Got {start_amp}')
    
    pad_len = int(array.shape[0] - fade_len)
    if pad_len < 0:
        raise IndexError(f'Given value of fade_len ({fade_len}) is longer than the number of samples in array ({array.shape[0]})')
    
    # Construct fade filter
    #fade_filter = np.linspace(start_amp, int(not start_amp), fade_len)
    # If fade_len is 1 and start_amp is 1, the above code results in 
    # a fade_filter = np.array([1.]), i.e. no fading. The below code
    # ensures that the end amplitude is included
    fade_filter = np.flip(np.linspace(int(not start_amp), start_amp, fade_len))
    
    # Pad filter for array length
    if start_amp == 0: # fade in at start
        fade_filter_padded = np.pad(
            fade_filter,
            (0, pad_len), # pad right side
            constant_values = 1, # with 1s
            mode = 'constant'
        )
    else: # start_amp == 1, fade out at end
        fade_filter_padded = np.pad(
            fade_filter,
            (pad_len, 0), # pad left side
            constant_values = 1, # with 1s
            mode = 'constant'
        )
    return np.multiply(array, fade_filter_padded)

def _sum_samples(
    samples_original,
    samples_new,
    sample_rate,
    wraparound_fill = False,
    fade_out = True
):
    '''
    Sums audio samples and updates labels
    
    Combines audio samples, samples_new, on top
    of samples_original, overlaying samples_new
    so it begins at the same time as samples_original.
    
    Args:
        samples_original (np.array): samples to 
            overlay new samples on
        samples_new (np.array): samples to be
            overlayed on original samples. If shorter
            than samples_original, can either be repeated/
            wrapped around to reach length of
            samples_original, or can be faded out
        sample_rate (int or float): mutual sample rate
            of both samples_original and samples_new
        wraparound_fill (bool): whether or not to 
            fill in short samples_new by wrapping around
        fade_out (bool): whether or not to fade out 
            short samples_new. If wraparound_fill == True,
            this option does not apply.
            
    Returns:
        summed samples
    '''
    
    original_len = samples_original.shape[0]
    new_len = samples_new.shape[0]
    discrepancy = original_len - new_len
    
    # Add new samples to original samples, possibly applying 
    # fade-out, filling, etc.
    if discrepancy > 0: # if new_len shorter than original_len
        # Make up length by repeating/"wrapping around"
        if wraparound_fill:
            samples_to_add = _wraparound_extract(
                original = samples_new,
                begin = 0,
                length = original_len)
        
        # Make up length with zero-padding
        else:
            samples_to_add = samples_new.copy()
            if fade_out:
                # Number of samples used in fade should be about 0.5s
                fade_samples = math.ceil(0.1 * sample_rate)
                if fade_samples > new_len: fade_samples = new_len

                # Apply fade
                samples_to_add = _fade(
                    array = samples_to_add,
                    fade_len = fade_samples,
                    start_amp = 1,
                )
            
            # Zero pad
            samples_to_add = np.pad(
                samples_to_add,
                (0, discrepancy),
                constant_values = 0,
                mode='constant'
            )
    else:
        samples_to_add = samples_new[:original_len]
        
    return np.add(samples_original, samples_to_add)

def _select_chunk(
    chunk_source,
    label,
    start_position = None,
    duration = 6, # should almost always be longer than source chunk
    duration_jitter = 0,
    chance_random_skip = 0.3,
    seed = None
):
    
    # Randomly choose and open a source audio file
    random.seed(seed)
    wavs = [f for f in os.listdir(chunk_source) if f[-4:].lower() == '.wav']
    mp3s = [f for f in os.listdir(chunk_source) if f[-4:].lower() == '.mp3']
    desired_path = os.path.join(chunk_source, random.choice(wavs+mp3s))
    source_audio = Audio(path = desired_path, label = label)
    
    return get_chunk(
        source_audio,
        start_position = start_position,
        duration = duration, 
        duration_jitter = duration_jitter, 
        chance_random_skip = chance_random_skip
    )


@_audio_manipulation
def sum_chunks(
    audio,
    label_dict = None,
    new_chunk_labels = ['random']*4,
    start_position = None,
    duration = 6, 
    duration_jitter = 0,
    chance_random_skip = 0.3,
    seed = None
):
    '''
    Add a random chunk to audio
    
    Grab a random number of chunks, from 0 to 4, 
    randomize their signal amplitude (multiply
    by a random number from 0 to 1), and add 
    the chunks to the audio. 
    
    Args:
        audio (Audio instance): original chunk 
        label_dict (dict): dictionary associating
            labels (keys) with paths (values). Each
            path is the place on the filesystem where 
            files of the given label can be found.
        new_chunk_labels (list of strings): list of 
            labels for new chunks, in order of potential
            addition. 
            
            Labels should be strings. Options:
            
                'original': same as original
                'different': different from original
                'random': any
                any key in label_dict
            
            New chunks are added with the 
            following probabilities:
                first chunk: 50%
                second chunk: 
                    if first chunk added: 40%
                    else: 0%
                third chunk:
                    if second chunk added: 30%
                    else: 0%
                fourth chunk: 
                    if third chunk added: 20%
                    else: 0%
                    
       start_position (int): position in the file to start
            extracting samples from. If None, the start position 
            is chosen randomly
        duration (float): desired duration, in seconds, 
            of chunk to extract
        duration_jitter (float): if this value is not 0,
            the duration of the chunk extracted will 
            be randomly selected from the range 
            (duration - duration_jitter, duration + duration_jitter)
        chance_random_skip (float between 0 and 1):
            percent chance of random skipping. In a random skip,
            a position within the chunk will be randomly
            selected, and from that position in the 
            audio file, a random number of samples will 
            be skipped. The number of samples skipped is between
            0 and the number of samples in the entire chunk
    '''
    
    random.seed(seed)
    
    if not label_dict:
        label_dict = {'test': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests/')}
    
    if (not isinstance(new_chunk_labels, list)) or (len(new_chunk_labels) != 4):
        raise ValueError("`new_chunk_labels` must be a list of four labels")
    for label in new_chunk_labels:
        if label not in ['original', 'different', 'random'] + list(label_dict.keys()):
            raise ValueError("Labels must be in label_dict.keys() or"
                            " in ['original', 'different', 'random']")
    
    sample_rate = audio.sample_rate
    
    chunks_to_add = 0
    if random.random() < 0.5:
        chunks_to_add += 1
        if random.random() < 0.4:
            chunks_to_add += 1
            if random.random() < 0.3:
                chunks_to_add += 1
                if random.random() < 0.2:
                    chunks_to_add += 1


    audio_max = np.max(audio.samples)
    # Iteratively combine chunks and labels
    for idx in range(chunks_to_add):
        # Select a new label if necessary
        label = new_chunk_labels[idx]
        if label == 'random':
            label = random.choice(list(label_dict.keys()))
            chunk_source = label_dict[label]
        elif label == 'different':
            possible = list(label_dict.keys())
            for l in audio.labels:
                possible.pop(l)
            label = random.choice(possible)
        elif label == 'same':
            label = random.choice(list(audio.labels))
            

        # Randomly grab chunk from source
        chunk_source = label_dict[label]
        new_chunk = _select_chunk(
            chunk_source = chunk_source,
            label = label,
            start_position = start_position,
            duration = duration,
            duration_jitter = duration_jitter,
            chance_random_skip = chance_random_skip
        )
        
        # Randomly change amplitude of chunk; can only be 2x amplitude of audio.samples

        # How many times softer or louder the current audio is than new chunk
        loudness_ratio = audio_max / np.max(new_chunk.samples)
        # How much softer or louder louder the new audio should be
        amplifier = random.uniform(0.5, 1.1)
        # Find amplifier with respect to loudness ratio
        amp_modifier = loudness_ratio * amplifier

        new_chunk.set_samples(np.multiply(new_chunk.samples, amp_modifier))
        
        # Add chunks together
        summed_samples = _sum_samples(
            samples_original = audio.samples,
            samples_new = new_chunk.samples,
            sample_rate = sample_rate,
            wraparound_fill = False,
            fade_out = False
        )
        
        audio.set_samples(summed_samples)
        for new_label in new_chunk.labels:
            audio.add_label(new_label)
        for new_source in new_chunk.sources:
            audio.add_source(source = new_source)
    
    return audio
