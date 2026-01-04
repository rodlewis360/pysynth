This is pysynth, a deeply customizable FM synth written in python.

## How to Use
Sounds can be programmed in sound_library.py by putting together Waves and Modulators, then adding Waves together into a Sound. Here's an example:

```
fm_ramp_modulator = FrequencyModulator(generate_ramp, frequency=1, amplitude=0.005, name="fm_ramp_modulator")
vibrato = FrequencyModulator(generate_sin, amplitude=0.00028, frequency=1, sub_modulator=fm_ramp_modulator, name="vibrato")

# ...

base_sin_wave = Wave(generate_sin, name="sin_wave")
vibrato_square_wave = Wave(generate_square, modulator=vibrato, name="vibrato_square_wave")

# ...

ringing_sin = Sound({
    base_sin_wave : (lambda freq : 1),
    vibrato_square_wave : (lambda freq : 1),
}, modulator=ChainModulator([EnvelopeModulator(0.1, 0.1, 0.5, 0.5, name="ringing_sin")]))
```

Note that this process is very different to most FM synths because of 3 things:
1. There is no GUI for pysynth, so everything is entered as a python object.
2. Modulators accept wave functions as inputs, and can be affected by other modulators.
3. Sounds are composed of multiple Wave objects whose amplitudes are determined as a function of frequency.

Then, to generate a sample from a midi file, use `generate_audio_from_midi` from synth.py and plug in the Sound you want to use.

### Sounds
Let's break this example down. Starting with the ringing_sin Sound:
```
ringing_sin = Sound({
    base_sin_wave : (lambda freq : 1),
    vibrato_square_wave : (lambda freq : 1),
}, modulator=ChainModulator([EnvelopeModulator(0.1, 0.1, 0.5, 0.5, name="ringing_sin")]))
```
This Sound is composed of a `base_sin_wave` and a `vibrato_square_wave` blended 1:1 regardless of the frequency. These are wave objects declared earlier in the code, which we'll look at later. 

The Sound also features a modulator, specifically the `ChainModulator`. A `ChainModulator` allows multiple modulators to be applied to a single `Wave` or `Sound`, and this `ChainModulator` applies an `EnvelopeModulator`, which applies an ADSR envelope to the wave's amplitude.

Each Sound is also given a name for debugging purposes - this will be displayed in the wave visualizer when it is rendered in debug mode.

### Waves
Next is the Waves:
```
base_sin_wave = Wave(generate_sin, name="sin_wave")
vibrato_square_wave = Wave(generate_square, modulator=vibrato, name="vibrato_square_wave")
```
Each `Wave` accepts a function that takes a frequency, duration, and sample rate and returns a list of floats representing a wave. This is the wave function.
In the case of `base_sin_wave`, this wave function is a simple sine wave.
`vibrato_square_wave` uses a simple square wave.

Then, each wave can accept a modulator. `base_sin_wave` has no modulator, but `vibrato_square_wave` uses the `vibrato` modulator, declared earlier in the code.
Finally, each wave has a name for debugging purposes, just like `Sound`s.

### Modulators
Modulators are the meat of an FM synth - they provide the unique sounds found in FM synths. Here's the example code for a vibrato modulator:
```
fm_ramp_modulator = FrequencyModulator(generate_ramp, frequency=1, amplitude=0.005, name="fm_ramp_modulator")
vibrato = FrequencyModulator(generate_sin, amplitude=0.00028, frequency=1, sub_modulator=fm_ramp_modulator, name="vibrato")
```
There are two modulators in this example. We'll start with the `vibrato` modulator. 

Since not all `Modulator`s do the same thing, there are different `Modulator`s accepting different parameters. `vibrato` is a `FrequencyModulator`, meaning it changes the frequency of the Wave it modulates. 

All `Modulator`s accept a wave function as their input, just like `Wave`s. Vibrato in the real world rapidly shifts the pitch up and down continuously, so the `vibrato` modulator will use a simple sine wave as its wave function.

Then, the base amplitude and frequency of the modulator are put in. For a `FrequencyModulator`, the frequency is the frequency (in Hz) that will be fed to the wave function to change the pitch of the `Wave` it is modulating. The amplitude of a `FrequencyModulator` is how much the original pitch is changed by (in Hz).

Next, each `Modulator` can accept a sub-modulator, which is another `Modulator` object that modulates the output of the wave function used to modulate the Wave. `vibrato` uses the `fm_ramp_modulator`.

Finally, each `Modulator` has a name for debugging purposes, just like Sounds and Waves.

`fm_ramp_modulator` is a FrequencyModulator that uses generate_ramp as its wave function, meaning it changes the frequency of whatever Wave it modulates more over time.

---

Putting all of that together, `ringing_sin` is a 1 : 1 mix of a sine wave and a square wave that has a mild vibrato applied over time, and the whole `Sound` has an ADSR envelope applied to it.

## Debug
To enter Debug mode, change the debug flag at the top of synth.py to True. This will graph all of the waves (including those of Modulators and Sounds) as they are generated and mixed using matplotlib and Tkinter.

---

Please note that Github Copilot with GPT-4 was used to generate parts of this code and research this concept during development.
