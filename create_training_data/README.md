# Create training data

Create training data (spectrograms containing ~5s of labeled audio). For an example pipeline loading these modules and using them to augment data, see [`notebooks/example_augmentation.ipynb`](https://github.com/rhine3/600birds/blob/master/create_training_data/notebooks/example_augmentation.ipynb).

## About these modules

In machine learning, *augmentation* refers to creating or modifying data to supplement prexisting training data. Augmentation "messes up" the given training data to increase the size of the dataset, as well as to give more breadth in the hope of making the machine learning more robust. 

To augment data for machine learning on spectrograms, we perform "manipulations" on audio samples or spectrogram images. This example augmentation workflow uses code from two modules, `code/audio_aug.py` and `code/spectrogram_aug.py`. The modules are parallel: both define a class, `Audio` or `Spectrogram`. The class attributes include, among other things, a list of which manipulation functions have been called on the object. 

All public manipulation functions have a similar structure. They are each called with one required argument (which should be either an `Audio` or a `Spectrogram` object) and several optional keyword arguments. The public functions are wrapped with a wrapper which performs input validation and updates some aspects of the modules.  Some functions in the module start with an underscore, including the wrapper. These are helper functions that are meant to be used only within the module itself; they may or may not operate on instances of the class.

## Background

These augmentation procedures are based on [data augmentation procedures](http://ceur-ws.org/Vol-2380/paper_86.pdf) developed by Mario Lasseck.

The 2018 challenge took a very different approach to the "audio chunk summation" part of the audio augmentation.  This approach was not used in the 2019 challenge because of the availability of annotated noise/background data. The augmentations included:
* Different sources of summed audio chunks: instead of using annotated noise/background segments from validation files, use image processing to roughly segment training files into "background" and "foreground"
   * Form "BirdsOnly" dataset by concatenating all bird sounds
   * Form "NoiseOnly" dataset by concatenating all background sounds
   * From "BackgroundOnly" dataset by concatenating all longer sequences of background sounds 
* Reconstruct audio signal by summing up the segmented elements described above



## Audio augmentations
All functions operate on instances of `Audio` and include optional keyword arguments.

* `get_chunk()`: Extract chunk of audio at random position in file
     * Wrap around to beginning of file if necessary
     * With 30% chance, select a random position in the chunk, and from that position skip a random number of samples from file, between 0 and the duration of the whole chunk
     * Jitter duration of audio chunk extracted +- 0.5s (results in global stretching when spectrogram is resized to fixed width)

* `cyclic_shift()`: Random cyclic shift (equivalent to splitting sample array in two at a random position and switching positions of the two pieces)

* time shifting & pitch stretching:

    * `time_stretch_divisions()`: Local time stretching on divisions: 50% chance of applying time stretching factor randomly chosen from Gaussian(mean = 1, sd=0.05) 

    * `pitch_shift_divisions()`: Local pitch shifting on divisions: 40% chance of applying pitch shift offset randomly chosen from Gaussian(mean = 0, sd=25 cents [1/8th of a tone])


* `random_filter()`: with 20% chance, filter audio in time domain with the following options chosen randomly: 

    * Type: lowpass, highpass, bandpass, bandstop 
    * Order: 1-5
    * Low cutoff frequency: 1 to sample_rate-3 Hz
    * High cutoff frequency (bandpass and bandstop filters): low_freq+1 to sample_rate-1 Hz
    * If filter output contains anything not between -1.0 and 1.0, return original signal
    
* ```sum_chunks()```: Add in noise from other audio chunks & randomize signal amplitude of chunks before summation

    * Background: noise from validation files and BAD files
    * Files containing same bird species
    * Files containing different bird species (up to 4 chunks added, along with adding their labels, with conditional probabilities of 50, 40, 30, and 20%)
    
   
## Image augmentations
All functions operate on instances of `Spectrogram` and include optional keyword arguments.

* Spectrogram creation functions, `make_linear_spectrogram()` or `make_mel_spectrogram()`:

    * Create linear or mel spectrogram  (window size = 1536, hop length = 360)
    * Transform spectrogram to decibel units using logarithm
    * Remove low and high frequencies (160Hz < x < 10300Hz)

* [X] `remove_random_hi_lo_bands()`: Global frequency shifting/stretching by removing additional high and low frequency bands (remove random number of first 10 and last 6 rows)

* [X] `resize_random_bands()`: Implemented as one function
   * Resize random columns: 50% chance of piecewise time stretching by resizing random number of columns at random position. 

      * With 50% chance, randomly divide spectrogram into vertical pieces of size between 10 and 100 pixels.
      * Resize all pieces individually by factor chosen between 0.9 and 1.1

   * Resize random rows: 40% chance of piecewise frequency stretching by resizing random number of rows at random position

        * With 40% chance, randomly divide spectrogram into horizontal pieces of size between 10 and 100 pixels
        * Resize all pieces individually by a factor between 0.95 and 1.15

* [X] `resize_spect_random_interpolation()`: Different interpolation filters for spectrogram resizing: 85% chance of using Lanczos filter; 15% chance of using a different resampling filter from the python imaging library (Nearest, Box, Bilinear, Hamming, Bicubic)

* [X] `color_jitter()`: Color jitter (brightness, contrast, saturation: factor 0.3; hue: factor 0.01)
