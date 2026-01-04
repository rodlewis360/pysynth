import mido as md

def displayMIDI(filename : str):
    MIDIfile = md.MidiFile(filename)
    i = 0
    while i < len(MIDIfile.tracks):
        print("track", i)
        track = MIDIfile.tracks[i]
        for msg in track:
            print(msg)
        i += 1

if __name__ == "__main__":
    filename = input("MIDI file: ")
    displayMIDI(filename)