import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("midis/*.mid"):
        print("Parsing %s" % file)

        midi = converter.parse(file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

        # TODO: Write this to file so it's faster
    return notes

notes = get_notes()
print(notes)
n_vocab = len(set(notes))
pitchnames = sorted(set(item for item in notes))
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

encoded_notes = [note_to_int[n] for n in notes]
print(encoded_notes)
# treat the above as a "sequence of characters"

# now do things to this @ani


#idea:
# 1. load all midi songs
# 2. concat all midi songs into one gigantic song
# 3. treat each note as a "character"