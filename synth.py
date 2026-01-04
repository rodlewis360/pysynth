import numpy as np
from scipy.io.wavfile import write
import mido
from sound_library import *

debug = False



def generate_wave(note : Note, duration, sound : Sound) -> np.ndarray:
    return sound.generate_wave(note.frequency, duration, note.sample_rate, note.amplitude)

def midi_to_frequency(midi_note : int) -> float:
    # Convert MIDI note number to frequency (A4 = 440Hz = MIDI note 69)
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def calculate_duration(ticks, bpm, tpq):
    # Calculate note duration in seconds using BPM and tpq
    return (ticks / tpq) * (60.0 / bpm)

def calculate_amplitude(velocity):
    return velocity / 100

def generate_audio_from_midi(midi_file_path, sound : Sound, output_file='generated_audio.wav', sample_rate=44100):
    midi_file = mido.MidiFile(midi_file_path)

    # Extract BPM and TPQ from the MIDI file
    bpm = 160  # Default BPM if none found
    tpq = midi_file.ticks_per_beat  # TPQ (ticks per quarter note) from the MIDI file
    
    # Iterate through MIDI messages to find the BPM (tempo change)
    for msg in midi_file.tracks[1]:
        if msg.type == 'set_tempo':
            bpm = mido.tempo2bpm(msg.tempo)
            break

    # Count the number of samples needed in the output
    num_samples = 0
    for msg in midi_file.tracks[1]:
        num_samples = num_samples + (calculate_duration(msg.time, bpm, tpq) * sample_rate)

    # Generate empty lists of the audio data in the file
    num_samples = int(num_samples) + 1
    audio_data = [0 for _ in range(0, num_samples)]
    voice_count = [0 for _ in range(0, num_samples)]

    current_time = 0  # Keep track of the current time (in number of samples) in the MIDI sequence

    # Dict of notes and starting times
    running_notes : dict[Note, int] = {}

    # Only accepts 1 track for now (and probably will never accept 2)
    # track 0 is metadata
    for msg in midi_file.tracks[1]:
        current_time = current_time + (calculate_duration(msg.time, bpm, tpq) * sample_rate)
        if debug:
            print(msg)
            print("jump", msg.time)
            print("jumpsecs", calculate_duration(msg.time, bpm, tpq))
            print("jumpsamps", calculate_duration(msg.time, bpm, tpq) * sample_rate)
            print("time", current_time)

        if msg.type == 'set_tempo':
            bpm = mido.tempo2bpm(msg.tempo)
            break
        
        if msg.type == 'note_on':  # When a note is played
            frequency = midi_to_frequency(msg.note)  # Convert MIDI note to frequency
            amplitude = calculate_amplitude(msg.velocity)
            note = Note(frequency, amplitude, sample_rate)
            running_notes[note] = current_time
        
        if msg.type == 'note_off':
            frequency = midi_to_frequency(msg.note)  # Convert MIDI note to frequency
            for note in list(running_notes.keys()):
                if note.frequency == frequency:
                    # Calculate the note start sample
                    start_sample = int(running_notes[note])
                    
                    # Generate wave using the note's duration
                    duration = (current_time - running_notes[note]) / sample_rate
                    wave = generate_wave(note, duration, sound)
                    wave_length = len(wave)
                    end_sample = start_sample + wave_length

                    # If audio_data or voice_count isn't long enough, extend them.
                    if end_sample > len(audio_data):
                        extension = end_sample - len(audio_data)
                        audio_data.extend([0] * extension)
                        voice_count.extend([0] * extension)
                    
                    # Retrieve the existing slice of audio data and voice count.
                    data_slice = np.array(audio_data[start_sample:end_sample])
                    voice_count_slice = np.array(voice_count[start_sample:end_sample])
                    
                    # Merge the old data with the new wave based on the voice count.
                    # Here we use np.average with weights: previous voice count and 1 for the new voice.
                    merged = np.average(
                        np.array([data_slice, wave]),
                        axis=0,
                        weights=[voice_count_slice, np.ones(len(voice_count_slice))]
                    )

                    # Update the audio_data with the merged data.
                    audio_data[start_sample:end_sample] = merged.tolist()

                    # Update the voice count for this segment.
                    for i in range(start_sample, end_sample):
                        voice_count[i] += 1

                    running_notes.pop(note)
                    break
        if debug:
            print()

    # In case the wave is too short, resize it to be correct
    print(max(audio_data), "amplitude")
    print(len(audio_data), "samples")
    print(len(audio_data) / sample_rate, "seconds")
    output_data = np.array(audio_data, dtype=np.int16)
    write(output_file, sample_rate, output_data)  # Save to a WAV file
    print(f"Audio saved as '{output_file}'")

# Example usage: generate audio from a MIDI file
midi_file_path = 'test3.mid'  # Replace with the path to your MIDI file
generate_audio_from_midi(midi_file_path, string_sound, 'samples/harmonic.wav')