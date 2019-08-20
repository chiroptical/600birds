# sound-augmentation
Modules for augmentation of wildlife recording data. These augmentation techniques are used in Mario Lasseck's entries to the BirdCLEF machine learning competition.

Functions to use:
* Apply short-time Fourier transform
* Transform spectrogram to decibel units using logarithm
* Convert to mel spectrogram
* Remove low and high frequencies
* Resize spectrogram to network dimensions
* Convert grayscale image to RGB image

Audio functions:
* Extract chunk of audio at random position in file
* Jitter duration of audio chunk
* Randomly divide chunk into segments of duration between 0.5 and 4s
* Local time stretching on divisions: time stretching factor randomly chosen from Gaussian(mean = 1, sd=0.05) 
* Local pitch shifting on divisions: pitch shift offset randomly chosen from Gaussian(mean = 0, sd=25 cents [1/8th of a tone])
* With 20% chance, filter audio in time domain with the following options chosen randomly: 
    * Type: lowpass, highpass, bandpass, bandstop 
    * Order: 1-5
    * Low cutoff frequency: 1 to sample_rate-3 Hz
    * High cutoff frequency (bandpass and bandstop filters): low_freq+1 to sample_rate-1 Hz
    * If filter output contains anything not between -1.0 and 1.0, return original signal
* Random cyclic shift
* Add audio chunks:
    * Background
    * Files containing same bird species
    * Files containing different bird species (up to 4 chunks added, 
   
    


Image functions to create:
* Create noise only file

