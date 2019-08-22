# sound-augmentation
Modules for augmentation of wildlife recording data. These augmentation techniques are used in Mario Lasseck's entries to the BirdCLEF machine learning competition.

Functions to use:
* Apply short-time Fourier transform
* Transform spectrogram to decibel units using logarithm (window size = 1536, hop length = 360)
* Convert to mel spectrogram
* Remove low and high frequencies (160Hz < x < 10300Hz)
* Resize spectrogram to network dimensions
* Convert grayscale image to RGB image

Audio augmentations:

* [X] ```get_chunk()```: Extract chunk of audio at random position in file
     * Wrap around to beginning of file if necessary
     * With 30% chance, select a random position in the chunk, and from that position skip a random number of samples from file, between 0 and the duration of the whole chunk
     * Jitter duration of audio chunk extracted +- 0.5s (results in global stretching when spectrogram is resized to fixed width)
     * Helper function: ```wraparound_extract()```

* [X]  ```cyclic_shift()```: Random cyclic shift (equivalent to splitting sample array in two at a random position and switching positions of the two pieces)

* [X] time shifting & pitch stretching:

    * [X]  ```time_stretch_divisions()```: Local time stretching on divisions: 50% chance of applying time stretching factor randomly chosen from Gaussian(mean = 1, sd=0.05) 

    * [X]  ```pitch_shift_divisions()```: Local pitch shifting on divisions: 40% chance of applying pitch shift offset randomly chosen from Gaussian(mean = 0, sd=25 cents [1/8th of a tone])


    * [X] Helper functions:
    * ```divide_samples()```:  Randomly divide samples into segments of duration between 0.5 and 4s

    * ```combine_samples()```:  Simple wrapper function to recombine divided samples


* [X]  ```random_filter()```: with 20% chance, filter audio in time domain with the following options chosen randomly: 

    * Type: lowpass, highpass, bandpass, bandstop 
    * Order: 1-5
    * Low cutoff frequency: 1 to sample_rate-3 Hz
    * High cutoff frequency (bandpass and bandstop filters): low_freq+1 to sample_rate-1 Hz
    * If filter output contains anything not between -1.0 and 1.0, return original signal
    
* [ ] ```sum_chunks()```: Add in noise from other audio chunks & randomize signal amplitude of chunks before summation

    * Background: noise from validation files and BAD files
    * Files containing same bird species
    * Files containing different bird species (up to 4 chunks added, along with adding their labels, with conditional probabilities of 50, 40, 30, and 20%)
    * Helper functions:
        
        * [X] `fade()`: fade audio in or out
        * [X] `sum_samples()`: perform actual addition of samples
        * [ ] `sum_labels()`: combine labels
        * [ ] `select_chunk()`: select chunk from a particular label
   
   
    
Image augmentations:
* [ ] ```remove_random_hi_lo_bands()```: Global frequency shifting/stretching by removing additional high and low frequency bands (remove random number of first 10 and last 6 rows)

* [ ] ```resize_random_cols()```: Piecewise time stretching by resizing random number of columns at random position

* [ ] ```resize_random_rows()```: Piecewise frequency stretching by resizing random number of rows at random position

* [ ] ```resize_spect_random_filter()```: Different interpolation filters for spectrogram resizing: 85% chance of using Lanczos filter; 15% chance of using a different resampling filter from the python imaging library (Nearest, Box, Bilinear, Hamming, Bicubic)

* [ ] ```color_jitter()```: Color jitter (brightness, contrast, saturation: factor 0.3; hue: factor 0.01)

The 2018 challenge took a very different approach to the "audio chunk summation" part of the audio augmentation.  This approach was not used in the 2019 challenge because of the availability of annotated noise/background data. The augmentations included:
* Different sources of summed audio chunks: instead of using annotated noise/background segments from validation files, use image processing to roughly segment training files into "background" and "foreground"
   * Form "BirdsOnly" dataset by concatenating all bird sounds
   * Form "NoiseOnly" dataset by concatenating all background sounds
   * From "BackgroundOnly" dataset by concatenating all longer sequences of background sounds 
* Reconstruct audio signal by summing up the segmented elements described above
