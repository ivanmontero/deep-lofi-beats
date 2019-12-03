import torch
import numpy as np
from music21 import instrument, note, stream, chord
from midinet import MidiNet
from midinet import get_notes
from midinet import generate_beats
import os

# TODO:
# 1. load model from checkpoint
# 2. eval final model code
# 3. add create midi code and output to output folders

FEATURE_SIZE = 2048
WEIGHTS_PATH = 'checkpoints/032.pt' #TODO: add in correct weight file for reconstruction
# SEQUENCE_LENGTH = 200
TEMPERATURE = 3

def load_model_from_checkpoint(checkpoint, note_to_int, device):
    model = MidiNet(len(note_to_int), FEATURE_SIZE)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device)

def eval_final_model(model, note_to_int, int_to_note, device, rando=True):
    seed_words = 'A3 G#3 B4 F#6 E-5 B4 G#3 G#3 F#6 E-5 F#6 B4 B4 G#3 F#6 G#3 B6 E-5 E-5 G#3 F#6 E-5 B4 E-5 F#6 G#3 B4 B4 E-5 G#3 E-5 G3 G#3 B4 E-5 F#6 E-5 F#6 E-5 B4 G#3 G3 G#3 F#6 B4 E-5 E-5 B4 F#6 E-5 B4 F#6 E-5 B4 B6 E-5 E-5 B4 E-5 F#6 E-5 G#3 F#6 F#6 E-5 G#3 B4 E-5 B4 F#6 F#6 E-5'.split(
        " ")
    sequence_length = 1024
    sample_beats = generate_beats(model, device, seed_words, sequence_length, note_to_int, int_to_note, 'sample', TEMPERATURE, rando=rando)
    print('generated with sample\t', sample_beats)
    beam_beats = generate_beats(model, device, seed_words, sequence_length, note_to_int, int_to_note, 'beam', TEMPERATURE, rando=rando)
    print('generated with beam\t', beam_beats)
    return sample_beats, beam_beats

def create_midi(prediction_output, sampling_strategy):
    """ convert the output from the prediction to notes and create a midi file from the notes """
    offset = 0
    output_notes = []
    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    if sampling_strategy == 'beam':
        midi_stream.write('midi', fp='beam_test_output.mid')    
    elif sampling_strategy == 'sample':
        midi_stream.write('midi', fp='sample_test_output.mid')
    else:
        print("Error! Invalid sampling strategy")

def main():
    if not os.path.exists('notes.txt'):
        print("Reading midi files")
        notes = [n for s in get_notes() for n in s]
        with open("notes.txt", "w+") as f:
            f.write(" ".join(notes))
    else:
        print("Restoring notes")
        with open('notes.txt', 'r') as f:
            notes = f.read().split(" ")
    
    n_vocab = len(set(notes))
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    int_to_note = {val: key for key, val in note_to_int.items()}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: {}'.format(device))
    
    print('Loading model from checkpoint...')
    model = load_model_from_checkpoint(WEIGHTS_PATH, note_to_int, device)
    
    print('Evaluating model...')
    sample_beats, beam_beats = eval_final_model(model, note_to_int, int_to_note, device)
    
    print('Saving results...')
    create_midi(sample_beats, 'sample')
    create_midi(beam_beats, 'beam')
  
if __name__ == "__main__":
    main()
