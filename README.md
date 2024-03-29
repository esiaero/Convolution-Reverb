# Convolution Reverb

Reverberation makes audio effects more immersive, but access to spaces with adequate reverberation is limited and expensive.  
Put simply, convolution reverb is an artificial method of achieving reverberation that “convolves” two sound files to produce a new file that sounds like a mixture of the two.

For example, the echoes of a cathedral can be convolved with a zero-reverb recording of someone saying “Hello” to produce an audio file where the “Hello” sounds “echo”-y, as if it was said in the cathedral.

## What is this?
This "main" branch takes in dry and IR files to convolve and saves it into a new file. There is an option to use a GPU, but it is not currently recommended.

Contrast this with the [live](https://github.com/esiaero/Convolution-Reverb/tree/live) branch.

## How to use this?
IR files can be downloaded from a variety of places - see below; the dry file can be anything. Insert the paths where necessary in the code and run.

## Disclaimer
Do not use for critical programs. This is a learning exercise.

# Attributions
IR files:
www.openairlib.net
Audiolab, University of York
Dr. Damian T. Murphy

I/O Audio:
https://github.com/adamstark/AudioFile