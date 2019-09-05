import random
import numpy as np
import librosa
from scipy.signal import butter, lfilter
import math

########################################################
################### CHUNK EXTRACTION ###################
########################################################

def wraparound_extract(original, begin, length):
    '''
    Extracts elements from numpy.array in a "wraparound" fashion
    
    Extracts a certain number of elements from 
    a numpy.array starting at a certain position.
    If the chosen position and length go
    past the end of the array, the extraction
    "wraps around" to the beginning of the numpy.array
    as many times as necessary. For instance:
    
    wraparound_extract(
        original = [0, 5, 10],
        begin = 1, 
        length = 7) -> [5, 10, 0, 5, 10, 0, 5]
    
    Args:
        original (np.array): the original array 
        begin (int): beginning position to extract
        length (int): number of elements to extract
    '''

    # Get `head`: the array after the beginning position
    assert(type(original) == np.ndarray)
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
    
    #print(desired_list)
    return desired_list

def get_chunk(
    samples, 
    sample_rate,
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
        samples (numpy.array): audio samples loaded
            by librosa.load or audio.load
        sample_rate (int or float): sample rate of `samples`
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
    
    Returns:
        samples
    '''
    
    # Get a random start position
    num_samples = len(samples)
    if not start_position:
        start_position = random.randint(0, num_samples)

    # Convert seconds to samples
    seconds_to_extract = duration + random.uniform(-duration_jitter, duration_jitter)
    samples_to_extract = int(seconds_to_extract * sample_rate)
    
    # Get chunks with skip in the middle with probability = chance_random_skip
    if random.random() < chance_random_skip:
        position_to_skip = random.randint(0, samples_to_extract)
        amount_to_skip = random.randint(0, samples_to_extract)

        chunk_1_start = start_position
        chunk_1_end = chunk_1_start + position_to_skip
        chunk_2_start = chunk_1_end + amount_to_skip
        chunk_2_end = chunk_1_start + (samples_to_extract - position_to_skip)
        
        chunk_1 = wraparound_extract(samples, chunk_1_start, chunk_1_end)
        chunk_2 = wraparound_extract(samples, chunk_2_start, chunk_2_end)
        chunk = np.concatenate((chunk_1, chunk_2))
    
    # Otherwise get contiguous chunk
    else:
        chunk = wraparound_extract(samples, start_position, samples_to_extract) 
        
    
    return chunk
    



########################################################
################### CYCLIC SHIFT #######################
########################################################

def cyclic_shift(array, split_point = None):
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
    
    assert(type(array) == np.ndarray)
    length = array.shape[0]
    
    # Stochastic split point, or split point by floor of split_point * length of array
    if not split_point: split_point = random.randint(0, length)
    else: split_point = int(split_point * length)
    
    return np.concatenate((array[split_point:], array[:split_point]))




########################################################
############ STRETCH AND SHIFT DIVISIONS ###############
########################################################

def divide_samples(
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
    
def combine_samples(divided):
    '''
    Recombine divided sample arrays
    
    Combine divided sample arrays back into a 
    single array, perhaps after each division
    has been modified by pitch shifting, time stretching, etc.
    
    Args:
        divided (list of np.ndarrays): list of sample arrays
            divided by divide_samples()
    
    Returns:
        sample arrays concatenated back into a single array
    '''
    
    return np.concatenate(divided)


def time_stretch_divisions(
    divisions,
    chance_per_division = 0.50,
    mean_stretch = 1,
    sd_stretch = 0.05
):
    '''
    Time stretch divisions
    
    Given a list of np.ndarrays, each np.ndarray representing
    audio samples, time stretch each array with some probability. 
    
    Args"
        divisions (list of np.ndarrays): list of np.ndarrays
            where each element of the list is samples from
            an audio file. A list of divisions can be generated 
            with helper functions in this module
        chance_per_division (float between 0 and 1): for
            each division, the chance it will be time-stretched
        mean_stretch (float): the mean stretch multiplier.
            == 1 is no stretch; > 1 is sped up, < 1 is slowed down
        sd_stretch (float > 0): the sd of the stretch 
            distribution. 
    
    Returns:
        stretched_divisions, time-stretched divisions
    '''
    stretched_divisions = []
    
    for d in divisions:
        if random.random() < chance_per_division:
            stretch_factor = np.random.normal(
                loc = mean_stretch,
                scale = sd_stretch)
            stretched_d = librosa.effects.time_stretch(y = d, rate = stretch_factor)
            stretched_divisions.append(stretched_d)
        else:
            stretched_divisions.append(d)
    
    return stretched_divisions




def pitch_shift_divisions(
    divisions,
    sample_rate,
    chance_per_division = 0.40,
    mean_shift = 0,
    sd_shift = 0.25
):
    '''
    Pitch shift divisions
    
    Given a list of np.ndarrays, each np.ndarray representing
    audio samples, pitch-shift each array with some probability. 
    The mean_shift and sd_shift should be given in "fractional
    half-steps," e.g. 0.25 = 1/4th of a half-step = 25 cents.
    
    Args:
        divisions (list of np.ndarrays): list of np.ndarrays
            where each element of the list is samples from
            an audio file. A list of divisions can be generated 
            with helper functions in this module
        sample_rate (int or float): sample rate of all divisions
        chance_per_division (float between 0 and 1): for
            each division, the chance it will be time-stretched
        mean_shift (float): the mean pitch shift in (fractional) half-steps
            == 0 is no shift; > 0 is shift up; < 1 is shift down
        sd_shift (float > 0): the sd of the shift 
            distribution in cents
    
    Returns:
        shifted_divisions, pitch-shifted divisions
    '''
    shifted_divisions = []
    
    for d in divisions:
        if random.random() < chance_per_division:
            shift_factor = np.random.normal(
                loc = mean_shift,
                scale = sd_shift)
            shifted_d = librosa.effects.pitch_shift(
                y = d,
                sr = sample_rate,
                n_steps = shift_factor)
            shifted_divisions.append(shifted_d)
        else:
            shifted_divisions.append(d)
    
    return shifted_divisions




########################################################
############## APPLY RANDOM AUDIO FILTER ###############
########################################################
def random_filter(
    samples,
    sample_rate,
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

         # Error check filtered audio
        if error_check:
            if  np.less(filtered, -1, where=~np.isnan(filtered)).any() or \
                np.greater(filtered, 1, where=~np.isnan(filtered)).any():
                return samples
                # For debugging
                #return samples, filtered, [filter_type, filter_order, filter_low, filter_high]
            else:
                return filtered
                # For debugging
                #return filtered, filtered, [filter_type, filter_order, filter_low, filter_high]
        else:
            return filtered
    
    else: return samples
    

########################################################
############## ADD AUDIO CHUNKS TOGETHER ###############
########################################################
def fade(array, fade_len, start_amp=1):
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


def sum_samples(
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
            samples_to_add = wraparound_extract(
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
                samples_to_add = fade(
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