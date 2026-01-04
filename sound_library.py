import numpy as np
import visualizer
from scipy.signal import butter, lfilter

debug_modulator = False
graph_debug_modulator = False

class Note:
    def __init__(self, frequency : float, amplitude : float, sample_rate : int = 44100):
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.amplitude = amplitude

def butter_lowpass(cutoff, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, sample_rate, order=5):
    b, a = butter_lowpass(cutoff, sample_rate, order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, sample_rate, order=5):
    b, a = butter_highpass(cutoff, sample_rate, order)
    return lfilter(b, a, data)

def generate_sin(frequency : float, duration : float, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (np.sin(2 * np.pi * frequency * t))

def generate_square(frequency : float, duration : float, sample_rate=44100):
    # Create a time vector from 0 to the duration
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (np.sign(np.sin(2 * np.pi * frequency * t)))

def generate_trapezoid(frequency: float, duration: float, sample_rate=44100, slope=0.2):
    """
    Generate a trapezoidal wave (smoothed square wave).
    slope: fraction of the period used for the rising/falling edge (0 < slope < 0.5)
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    period = 1.0 / frequency
    phase = np.mod(t, period) / period
    s = np.clip(slope, 1e-6, 0.499)  # avoid degenerate cases

    # Trapezoid: ramp up, high, ramp down, low
    wave = np.zeros_like(phase)
    # Rising edge
    rising = (phase < s)
    wave[rising] = (phase[rising] / s) * 2 - 1
    # High plateau
    high = (phase >= s) & (phase < 0.5 - s)
    wave[high] = 1
    # Falling edge
    falling = (phase >= 0.5 - s) & (phase < 0.5 + s)
    wave[falling] = 1 - ((phase[falling] - (0.5 - s)) / (2 * s)) * 2
    # Low plateau
    low = (phase >= 0.5 + s) & (phase < 1 - s)
    wave[low] = -1
    # Rising edge (wrap around)
    rising2 = (phase >= 1 - s)
    wave[rising2] = ((phase[rising2] - 1) / s) * 2 + 1

    return wave

def generate_smooth_trapezoid(frequency: float, duration: float, sample_rate=44100, slope=0.2, curve=0.1):
    """
    Generate a trapezoidal wave with smooth, curved corners.
    slope: fraction of the period used for the rising/falling edge (0 < slope < 0.5)
    curve: controls the roundness of the corners (0 = sharp, higher = rounder)
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    period = 1.0 / frequency
    phase = np.mod(t, period) / period
    s = np.clip(slope, 1e-6, 0.499)
    c = np.clip(curve, 1e-6, 0.499)

    # Helper for smoothstep (cubic Hermite interpolation)
    def smoothstep(x):
        return 3 * x**2 - 2 * x**3

    wave = np.zeros_like(phase)
    # Rising edge (curved)
    rising = (phase < s)
    x = phase[rising] / s
    wave[rising] = -1 + 2 * smoothstep(x)

    # High plateau
    high = (phase >= s) & (phase < 0.5 - s)
    wave[high] = 1

    # Falling edge (curved)
    falling = (phase >= 0.5 - s) & (phase < 0.5 + s)
    x = (phase[falling] - (0.5 - s)) / (2 * s)
    wave[falling] = 1 - 2 * smoothstep(x)

    # Low plateau
    low = (phase >= 0.5 + s) & (phase < 1 - s)
    wave[low] = -1

    # Wrap-around rising edge (curved)
    rising2 = (phase >= 1 - s)
    x = (phase[rising2] - (1 - s)) / s
    wave[rising2] = -1 + 2 * smoothstep(x)

    return wave

def generate_sawtooth(frequency : float, duration : float, sample_rate=44100):
    # Create a time vector from 0 to the duration
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (t * frequency - np.floor(t * frequency + 0.5))

def generate_noise(frequency : float, duration : float, sample_rate=44100):
    # Create a time vector from 0 to the duration
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave_data = (butter_lowpass_filter(np.random.uniform(-1, 1, len(t)), min(frequency * 50, sample_rate * 0.5 - 1), sample_rate))
    # Noise has issues with amplitude so we normalize it.
    # This means noise will fade a bit as the pitch goes higher though.
    return wave_data / wave_data.max()

def generate_ramp(frequency, duration: float, sample_rate=44100):
    # Create a time vector from 0 to the duration
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Ramp wave increases linearly from 0 to 1
    ramp_wave = np.clip(t / duration, 0, 1)
    return ramp_wave

def generate_downramp(frequency, duration: float, sample_rate=44100):
    # Create a time vector from 0 to the duration
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Ramp wave decreases linearly from 1 to 0
    ramp_wave = np.clip(t / duration, 1, 0)
    return ramp_wave
    
class Modulator:
    def __init__(self, generate_wave, frequency, amplitude=1, sub_modulator=None, name=""):
        self.generate_wave = generate_wave
        self.frequency = frequency
        self.amplitude = amplitude
        self.sub_modulator = sub_modulator  # Optionally, a modulator to modulate this modulator
        self.name = name
        if sub_modulator:
            sub_modulator.assign_wave(self)

    def assign_wave(self, wave):
        self.wave = wave

    def apply_modulation(self, wave, duration, sample_rate, base_frequency, base_amplitude):
        if not self.wave:
            return wave
        if debug_modulator:
            print(self.name, ": empty modulator -> ", self.wave.name)
            print()
        if graph_debug_modulator:
            visualizer.visualize_wave_with_app(wave, self.wave.name, sample_rate)
        
        # Generate this modulator's wave
        mod_wave = self.generate_wave(self.frequency, duration, sample_rate)

        if graph_debug_modulator:
            visualizer.visualize_wave_with_app(mod_wave, f"Mod for {self.name} -> {self.wave.name}", sample_rate)
        return wave

class FrequencyModulator(Modulator):
    def __init__(self, generate_wave, frequency, amplitude=1, sub_modulator=None, name=""):
        self.generate_wave = generate_wave
        self.frequency = frequency
        self.amplitude = amplitude
        self.sub_modulator = sub_modulator
        self.name = name or "FrequencyModulator"
        if sub_modulator:
            sub_modulator.assign_wave(self)

    def apply_modulation(self, wave, duration, sample_rate, base_frequency, base_amplitude):
        assert self.wave, "Frequency modulator must be assigned a wave to modulate."
        if debug_modulator:
            print(self.name, " -> ", self.wave.name)
        if graph_debug_modulator:
            visualizer.visualize_wave_with_app(wave, self.wave.name, sample_rate)
        
        # Generate this modulator's wave
        mod_wave = self.generate_wave(self.frequency, duration, sample_rate)

        if graph_debug_modulator:
            visualizer.visualize_wave_with_app(mod_wave, f"Mod for {self.name} -> {self.wave.name}", sample_rate)
        
        # If there's a sub-modulator, apply its modulation recursively
        if self.sub_modulator:
            mod_wave = self.sub_modulator.apply_modulation(mod_wave, duration, sample_rate, base_frequency, base_amplitude) * self.amplitude
        else:
            mod_wave *= self.amplitude
            
        # FM can only occur when the modulator has been assigned a wave.
        # Instantaneous frequency at each point in time (modulation is added to base frequency)
        if debug_modulator:
            print("orig freq", base_frequency)
        instantaneous_frequency = base_frequency + mod_wave * base_frequency  # mod_wave scales the base frequency
        if debug_modulator:
            print("max mod wave", mod_wave.max(), "max amp", self.amplitude)
            print("max end freq", instantaneous_frequency.max())
        # Generate the wave using the instantaneous frequency at each time step
        return self.wave.generate_wave(instantaneous_frequency, duration, sample_rate)

class AmplitudeModulator(Modulator):
    def __init__(self, generate_wave, frequency, amplitude=1, sub_modulator=None, name="AmplitudeModulator"):
        self.generate_wave = generate_wave
        self.frequency = frequency
        self.amplitude = amplitude
        self.sub_modulator = sub_modulator
        self.name = name
        if sub_modulator:
            sub_modulator.assign_wave(self)

    def apply_modulation(self, wave, duration, sample_rate, base_frequency, base_amplitude):
        if debug_modulator:
            print(self.name, " -> ", self.wave.name if self.wave.name else "unassigned")
        if graph_debug_modulator:
            visualizer.visualize_wave_with_app(wave, self.wave.name, sample_rate)
        
        # Generate this modulator's wave
        mod_wave = self.generate_wave(self.frequency, duration, sample_rate)

        if graph_debug_modulator:
            visualizer.visualize_wave_with_app(mod_wave, f"Mod for {self.name} -> {self.wave.name}", sample_rate)
        
        # If there's a sub-modulator, apply its modulation recursively
        if self.sub_modulator:
            mod_wave = self.sub_modulator.apply_modulation(mod_wave, duration, sample_rate, base_frequency) * self.amplitude
        else:
            mod_wave *= self.amplitude
        return wave * np.abs(1 - mod_wave)

class LowPassFilterModulator(Modulator):
    def __init__(self, cutoff, order=5, name="LowPassFilterModulator"):
        self.cutoff = cutoff
        self.order = order
        self.name = name

    def apply_modulation(self, wave, duration, sample_rate, base_frequency, base_amplitude):
        return self.butter_lowpass_filter(wave, self.cutoff, sample_rate, self.order)

    def butter_lowpass_filter(self, data, cutoff, sample_rate, order):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return lfilter(b, a, data)


class HighPassFilterModulator(Modulator):
    def __init__(self, cutoff, order=5, name="HighPassFilterModulator"):
        self.cutoff = cutoff
        self.order = order
        self.name = name

    def apply_modulation(self, wave, duration, sample_rate, base_frequency, base_amplitude):
        return self.butter_highpass_filter(wave, self.cutoff, sample_rate, self.order)

    def butter_highpass_filter(self, data, cutoff, sample_rate, order):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return lfilter(b, a, data)
    
class RelativeLowPassFilterModulator(Modulator):
    def __init__(self, cutoff, order=5, name="RelativeLowPassFilterModulator"):
        self.cutoff = cutoff
        self.order = order
        self.name = name

    def apply_modulation(self, wave, duration, sample_rate, base_frequency, base_amplitude):
        # Effective cutoff: the note's frequency plus the modulator's cutoff offset
        effective_cutoff = base_frequency + self.cutoff
        if effective_cutoff < 0:
            return wave
        nyquist = 0.5 * sample_rate
        normal_cutoff = effective_cutoff / nyquist
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        return lfilter(b, a, wave)


class RelativeHighPassFilterModulator(Modulator):
    def __init__(self, cutoff, order=5, name=""):
        self.cutoff = cutoff
        self.order = order
        self.name = name or "RelativeHighPassFilterModulator"

    def apply_modulation(self, wave, duration, sample_rate, base_frequency, base_amplitude):
        # Effective cutoff: the note's frequency plus the modulator's cutoff offset
        effective_cutoff = base_frequency + self.cutoff
        nyquist = 0.5 * sample_rate
        normal_cutoff = effective_cutoff / nyquist
        b, a = butter(self.order, normal_cutoff, btype='high', analog=False)
        return lfilter(b, a, wave)

class EnvelopeModulator(Modulator):
    def __init__(self, attack: float, decay: float, sustain_level: float, release: float, name="EnvelopeModulator"):
        self.attack = attack
        self.decay = decay
        self.sustain_level = sustain_level
        self.release = release
        self.name = name

    def apply_modulation(self, wave, duration, sample_rate, base_frequency, base_amplitude):
        # Ensure that the release part of the ADSR envelope will be finished
        required_duration = self.attack + self.decay + self.release
        if duration < required_duration:
            wave = self.wave.generate_wave(base_frequency, required_duration, sample_rate, base_amplitude)

        base_samples = int(duration * sample_rate)
        # Calculate the number of samples for each phase.
        attack_samples = int(self.attack * sample_rate)
        decay_samples = int(self.decay * sample_rate)
        release_samples = int(self.release * sample_rate)

        # Determine required envelope samples (with no sustain).
        required_samples = attack_samples + decay_samples + release_samples

        # If there's time remaining, it will be used for sustain.
        sustain_samples = base_samples - required_samples if base_samples > required_samples else 0

        attack_env = np.linspace(0, 1, attack_samples, endpoint=False) if attack_samples > 0 else np.array([])
        decay_env = np.linspace(1, self.sustain_level, decay_samples, endpoint=False) if decay_samples > 0 else np.array([])
        sustain_env = np.full(sustain_samples, self.sustain_level) if sustain_samples > 0 else np.array([])
        release_env = np.linspace(self.sustain_level, 0, release_samples, endpoint=True) if release_samples > 0 else np.array([])

        envelope = np.concatenate([attack_env, decay_env, sustain_env, release_env])

        # Clamp envelope values to [0, 0.98]
        # This is to prevent integer overflow when multiplying with the wave.
        for i in range(len(envelope)):
            envelope[i] = min(0.98, max(0, envelope[i]))

        if len(envelope) > len(wave):
            envelope = envelope[:len(wave)]
        elif len(envelope) < len(wave):
            extra_samples = len(wave) - len(envelope)
            extra = np.linspace(envelope[-1], 0, extra_samples, endpoint=True)
            envelope = np.concatenate([envelope, extra])
        if graph_debug_modulator:
            visualizer.visualize_wave_with_app(envelope, f"Envelope for sound", sample_rate)
        if graph_debug_modulator:
            visualizer.visualize_wave_with_app(wave * envelope, "sound", sample_rate)
        return wave * envelope

class ChainModulator(Modulator):
    def __init__(self, modulators, name="ChainModulator"):
        self.modulators = modulators  # A list of modulators to chain
        self.name = name

    def assign_wave(self, wave):
        # Assign the wave to each modulator in the chain
        for mod in self.modulators:
            mod.assign_wave(wave)

    def apply_modulation(self, wave, duration, sample_rate, base_frequency, base_amplitude):
        # Apply each modulator in sequence to the wave
        modulated_wave = wave
        for mod in self.modulators:
            modulated_wave = mod.apply_modulation(modulated_wave, duration, sample_rate, base_frequency, base_amplitude)
        return modulated_wave

class HarmonicsAdder(Modulator):
    def __init__(self, harmonics, falloff, name="HarmonicFrequencyModulator"):
        self.harmonics = harmonics
        self.falloff = falloff
        self.name = name
    
    def apply_modulation(self, wave, duration, sample_rate, base_frequency, base_amplitude):
        # Geerate the wave as the series of the harmonic sequence where frequency = (1/n)base_frequency
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        harmonic_wave = np.zeros_like(t)
        total_weight = 0.0
        for n in range(1, self.harmonics + 1):
            freq = base_frequency * n
            weight = 1 / n**(1/(freq * self.falloff))
            harmonic_wave += weight * self.wave.generate_wave(freq, duration, sample_rate)
            total_weight += weight
        # Normalize by the sum of weights (harmonic series)
        if total_weight > 0:
            harmonic_wave /= total_weight
        # Scale by base amplitude
        return harmonic_wave * base_amplitude

class Wave:
    # generate_wave must take in a frequency and duration, and output a wave
    def __init__(self, generate_wave, modulator=None, name=""):
        self.generate_wave = generate_wave
        self.modulator = modulator  # This is an optional modulator
        self.name = name    
        if modulator and modulator.assign_wave:
            modulator.assign_wave(self)

    def apply_modulation(self, wave, duration, sample_rate, base_frequency, base_amplitude):
        if self.modulator:
            return self.modulator.apply_modulation(wave, duration, sample_rate, base_frequency, base_amplitude)
        return wave

class Sound:
    # waves is a dict of shape {Wave : weight}, where weight is a function that takes
    # a frequency and outputs the scale.
    def __init__(self, waves, modulator=None):
        self.waves = waves
        self.modulator = modulator  # This is an optional modulator
        if modulator and modulator.assign_wave:
            modulator.assign_wave(self)
    def get_total_weight(self, norm_freq):
        total_weight = 0
        for wave in self.waves:
            total_weight += self.waves[wave](norm_freq)
        return total_weight
    def generate_wave(self, frequency: float, duration: float, sample_rate=44100, amplitude=0.5):
        norm_freq = frequency / 440
        total_wave = np.zeros(int(sample_rate * duration))

        # Take the weighted average of all waves in the sound
        for wave in self.waves:

            weight = self.waves[wave](norm_freq)
            wave_data = wave.generate_wave(frequency, duration, sample_rate)
            
            wave_data = wave.apply_modulation(wave_data, duration, sample_rate, frequency, amplitude)

            total_wave += wave_data * weight
        
        # Apply the sound modulator if it exists
        if self.modulator:
            total_wave = self.modulator.apply_modulation(total_wave, duration, sample_rate, frequency, amplitude)

        # Normalize to ensure it doesn't get blown out
        total_wave /= total_wave.max()

        # Multiply by the amplitude
        total_wave *= amplitude

        wave_int16 = np.int16(total_wave * 32767)  # Scale to max int16 value
        return wave_int16

# --- Sample Sounds --- #

fm_ramp_modulator = FrequencyModulator(generate_ramp, frequency=1, amplitude=0.005, name="fm_ramp_modulator")
downramp_modulator = AmplitudeModulator(generate_downramp, amplitude=1, frequency=1, name="downramp_modulator")
modulator_sine = AmplitudeModulator(generate_sin, frequency=2, name="sin_modulator")
fm_sin_modulator = FrequencyModulator(generate_sin, amplitude=1, frequency=2, name="fm_sin_modulator")
fm_sin_modulator_squared = FrequencyModulator(generate_sin, amplitude=0.00005, frequency=1, sub_modulator=fm_sin_modulator, name="fm_sin_modulator_squared")
wavvy_sin_modulator = AmplitudeModulator(generate_sin, amplitude=1, frequency=10, sub_modulator=modulator_sine, name="sin_modulator_squared")
less_wavvy_sin_modulator = AmplitudeModulator(generate_sin, amplitude=0.5, frequency=3, sub_modulator=modulator_sine, name="sin_modulator_squared_small")
harmonic_modulator = HarmonicsAdder(10, 1.5, name="harmonic_modulator")

vibrato = FrequencyModulator(generate_sin, amplitude=0.00028, frequency=1, sub_modulator=fm_ramp_modulator, name="vibrato")

low_cutoff = RelativeLowPassFilterModulator(50, order=2, name="low_cutoff")
high_cutoff = RelativeLowPassFilterModulator(-50, order=2, name="low_cutoff")

fm_square_wave = Wave(generate_square, modulator=fm_sin_modulator_squared, name="fm_square_wave")
vibrato_square_wave = Wave(generate_square, modulator=vibrato, name="vibrato_square_wave")

modulated_sin_wave = Wave(generate_sin, modulator=modulator_sine, name="modulated_sin_wave")
wavvy_modulated_sin_wave = Wave(generate_sin, modulator=wavvy_sin_modulator, name="wavvy_modulated_sin_wave")
wavvy_modulated_square_wave = Wave(generate_square, modulator=less_wavvy_sin_modulator, name="wavvy_modulated_square_wave")

ramp_noise_wave = Wave(generate_noise, modulator=downramp_modulator, name="ramp_noise_wave")

band_cutoff_sawtooth_wave = Wave(generate_sawtooth, modulator=ChainModulator([high_cutoff, low_cutoff]), name="low_cutoff_sawtooth_wave")
band_noise_wave = Wave(generate_noise, modulator=ChainModulator([low_cutoff, high_cutoff]), name="band_noise_wave")

base_sin_wave = Wave(generate_sin, name="sin_wave")
base_square_wave = Wave(generate_square, name="square_wave")
base_sawtooth_wave = Wave(generate_sawtooth, name="sawtooth_wave")
base_noise_wave = Wave(generate_noise, name="noise_wave")

harmonic_sin_wave = Wave(generate_sin, name="harmonic_sin_wave", modulator=harmonic_modulator)
harmonic_vibrato_square_wave = Wave(generate_square, modulator=ChainModulator([vibrato, harmonic_modulator], "harmonic_vibrato_square_mod"), name="vibrato_square_wave")
harmonic_vibrato_trapezoid_wave = Wave(generate_trapezoid, modulator=ChainModulator([vibrato, harmonic_modulator], "harmonic_vibrato_square_mod"), name="vibrato_square_wave")
harmonic_vibrato_smooth_trapezoid_wave = Wave(generate_smooth_trapezoid, modulator=ChainModulator([vibrato, harmonic_modulator], "harmonic_vibrato_square_mod"), name="vibrato_square_wave")

# fm_sin = Sound({
#     fm_sin_wave : (lambda freq : 1),
# })

# graph_debug_modulator = True

ringing_sin = Sound({
    base_sin_wave : (lambda freq : 1),
    vibrato_square_wave : (lambda freq : 1),
}, modulator=ChainModulator([EnvelopeModulator(0.1, 0.1, 0.5, 0.5, name="ringing_sin")]))

bassy_sin = Sound({
    base_sin_wave : (lambda freq : freq / 2),
    base_sin_wave : (lambda freq : 10),
    # wavvy_modulated_sin_wave : (lambda freq : 1 * (freq / 2)),
    base_square_wave : (lambda freq : 0.8),
    band_cutoff_sawtooth_wave : (lambda freq : (0.3 / freq)),
    band_noise_wave : (lambda freq : 0.025 * (2 / freq)),
})

string_sound = Sound({
    harmonic_sin_wave : (lambda freq : 1),
    harmonic_vibrato_smooth_trapezoid_wave : (lambda freq : 1),
}, modulator=ChainModulator([EnvelopeModulator(0.05, 1, 0.75, 0.05, name="string_sound")]))

snare = Sound({
    ramp_noise_wave : (lambda freq : 1)
})