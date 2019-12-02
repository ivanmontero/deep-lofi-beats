import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing
import os
import tqdm
import re # regex
import numpy as np
import matplotlib.pyplot as plt

SEQUENCE_LENGTH = 100
BATCH_SIZE = 64
FEATURE_SIZE = 512
TEST_BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0005
USE_CUDA = True
PRINT_INTERVAL = 10
LOG_PATH = 'logs/log.pkl'
TEMPERATURE = 3
BEAM_WIDTH = 10

def write_log(filename, data):
    """Pickles and writes data to a file

    Args:
        filename(str): File name
        data(pickleable object): Data to save
    """

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    pickle.dump(data, open(filename, 'wb'))

def read_log(filename, default_value=None):
    """Reads pickled data or returns the default value if none found

    Args:
        filename(str): File name
        default_value(anything): Value to return if no file is found
    Returns:
        unpickled file
    """

    if os.path.exists(filename):
        return pickle.load(open(filename, 'rb'))
    return default_value

def plot(x_values, y_values, title, xlabel, ylabel):
    """Plots a line graph

    Args:
        x_values(list or np.array): x values for the line
        y_values(list or np.array): y values for the line
        title(str): Title for the plot
        xlabel(str): Label for the x axis
        ylabel(str): label for the y axis
    """

    plt.figure(figsize=(20, 10))
    plt.plot(x_values, y_values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"logs/{title}.png")

def restore_latest(net, folder):
    """Restores the most recent weights in a folder

    Args:
        net(torch.nn.module): The net to restore
        folder(str): The folder path
    Returns:
        int: Attempts to parse the epoch from the state and returns it if possible. Otherwise returns 0.
    """

    checkpoints = sorted(glob.glob(folder + '/*.pt'), key=os.path.getmtime)
    start_it = 0
    if len(checkpoints) > 0:
        restore(net, checkpoints[-1])
        try:
            start_it = int(re.findall(r'\d+', checkpoints[-1])[-1])
        except:
            pass
    return start_it

def restore(net, save_file):
    """Restores the weights from a saved file

    This does more than the simple Pytorch restore. It checks that the names
    of variables match, and if they don't doesn't throw a fit. It is similar
    to how Caffe acts. This is especially useful if you decide to change your
    network architecture but don't want to retrain from scratch.

    Args:
        net(torch.nn.Module): The net to restore
        save_file(str): The file path
    """

    net_state_dict = net.state_dict()
    restore_state_dict = torch.load(save_file)

    restored_var_names = set()

    print('Restoring:')
    for var_name in restore_state_dict.keys():
        if var_name in net_state_dict:
            var_size = net_state_dict[var_name].size()
            restore_size = restore_state_dict[var_name].size()
            if var_size != restore_size:
                print('Shape mismatch for var', var_name, 'expected', var_size, 'got', restore_size)
            else:
                if isinstance(net_state_dict[var_name], torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    net_state_dict[var_name] = restore_state_dict[var_name].data
                try:
                    net_state_dict[var_name].copy_(restore_state_dict[var_name])
                    print(str(var_name) + ' -> \t' + str(var_size) + ' = ' + str(int(np.prod(var_size) * 4 / 10**6)) + 'MB')
                    restored_var_names.add(var_name)
                except Exception as ex:
                    print('While copying the parameter named {}, whose dimensions in the model are'
                          ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                              var_name, var_size, restore_size))
                    raise ex

    ignored_var_names = sorted(list(set(restore_state_dict.keys()) - restored_var_names))
    unset_var_names = sorted(list(set(net_state_dict.keys()) - restored_var_names))
    print('')
    if len(ignored_var_names) == 0:
        print('Restored all variables')
    else:
        print('Did not restore:\n\t' + '\n\t'.join(ignored_var_names))
    if len(unset_var_names) == 0:
        print('No new variables')
    else:
        print('Initialized but did not modify:\n\t' + '\n\t'.join(unset_var_names))

    print('Restored %s' % save_file)

def save(net, file_name, num_to_keep=1):
    """Saves the net to file, creating folder paths if necessary.

    Args:
        net(torch.nn.module): The network to save
        file_name(str): the path to save the file.
        num_to_keep(int): Specifies how many previous saved states to keep once this one has been saved.
            Defaults to 1. Specifying < 0 will not remove any previous saves.
    """

    folder = os.path.dirname(file_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(net.state_dict(), file_name)
    extension = os.path.splitext(file_name)[1]
    checkpoints = sorted(glob.glob(folder + '/*' + extension), key=os.path.getmtime)
    print('Saved %s\n' % file_name)
    if num_to_keep > 0:
        for ff in checkpoints[:-num_to_keep]:
            os.remove(ff)

def get_notes_from_file(file):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

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
    return notes

def get_notes():
    # TODO: Write this to file so it's faster
    return multiprocessing.Pool(os.cpu_count()).map(get_notes_from_file, glob.glob("midis/*.mid"))

import math

class MidiDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, note_to_int, int_to_note, sequence_length, batch_size):
        super(MidiDataset, self).__init__()

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.note_to_int = note_to_int
        self.int_to_note = int_to_note

        # with open(data_file, 'rb') as data_pkl:
        #     dataset = pickle.load(data_pkl)
        # self.tokens = dataset["tokens"]
        self.tokens = tokens

        # Make multiple of match size
        self.tokens = self.tokens[:-(len(self.tokens) % self.batch_size)]

        # Fix last batch
        self.elements_per_in_last_batch = (len(self.tokens) % (self.batch_size * self.sequence_length)) // self.batch_size
        self.normal_sequences = (len(self.tokens) // (self.sequence_length*self.batch_size))*self.batch_size

    def __len__(self):
        if self.elements_per_in_last_batch == 1:
            return self.normal_sequences
        else:
            return self.normal_sequences + self.batch_size
        
    def __getitem__(self, idx):
        if idx < self.normal_sequences:
            data, labels = torch.tensor(self.tokens[idx*self.sequence_length: (idx+1)*self.sequence_length]).long(), torch.tensor(self.tokens[idx*self.sequence_length+1: (idx+1)*self.sequence_length+1]).long()
        else:
            last_batch_idx = idx - self.normal_sequences
            start = self.normal_sequences*self.sequence_length + last_batch_idx*self.elements_per_in_last_batch
            # print(f"{start} {start+(self.elements_per_in_last_batch-1)}")
            data, labels =  torch.tensor(
                self.tokens[start:start+(self.elements_per_in_last_batch-1)]).long(), torch.tensor(
                    self.tokens[start+1:start+self.elements_per_in_last_batch]).long()
        return data, labels

    def to_notes(self, indices):
        return [self.int_to_note[i] for i in indices]

class MidiNet(nn.Module):
    def __init__(self, vocab_size, feature_size):
        super(MidiNet, self).__init__()
        self.vocab_size = vocab_size
        self.feature_size = feature_size
        self.encoder = nn.Embedding(self.vocab_size, self.feature_size)
        self.lstm = nn.LSTM(self.feature_size, self.feature_size, num_layers=2, batch_first=True)
        self.decoder = nn.Linear(self.feature_size, self.vocab_size)
        
        # Can comment out below two lines.
        self.decoder.weight = self.encoder.weight
        self.decoder.bias.data.zero_()
        
        self.best_accuracy = -1
    
    def forward(self, x, hidden_state=None):
        encoded = self.encoder(x)
        out, hidden_state = self.lstm(encoded, hidden_state)
        decoded = self.decoder(out)

        return decoded, hidden_state

    def inference(self, x, hidden_state=None, temperature=1):
        x = x.view(-1, 1)
        x, hidden_state = self.forward(x, hidden_state)
        x = x.view(1, -1)
        x = x / max(temperature, 1e-20)
        x = F.softmax(x, dim=1)
        return x, hidden_state

    def loss(self, prediction, label, reduction='mean'):
        loss_val = F.cross_entropy(prediction.view(-1, self.vocab_size), label.view(-1), reduction=reduction)
        return loss_val

    def save_model(self, file_path, num_to_keep=1):
        save(self, file_path, num_to_keep)

    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if accuracy > self.best_accuracy:
            self.save_model(file_path, num_to_keep)
            self.best_accuracy = accuracy
 
    def load_model(self, file_path):
        restore(self, file_path)


def max_sampling_strategy(sequence_length, model, output, hidden):
    outputs = []
    for ii in range(sequence_length):
        topv, topi = torch.topk(output, 1)
        outputs.append(topi.item())
        output, hidden = model.inference(topi, hidden, TEMPERATURE)
    return outputs
    
def sample_sampling_strategy(sequence_length, model, output, hidden):
    outputs = []
    for ii in range(sequence_length):
        s = torch.multinomial(output, 1)
        outputs.append(s.item())
        output, hidden = model.inference(s, hidden, TEMPERATURE)
    return outputs

def beam_sampling_strategy(sequence_length, beam_width, model, output, hidden):
    beams = [([], output, hidden, 0)]
    for ii in range(sequence_length):
        new_beams = []
        for b in beams:
            samples = torch.multinomial(b[1], beam_width, replacement=True)
            for sample in samples[0]:
                score = torch.log(b[1][0][sample])
                new_beams.append((b[0] + [sample], b[1], b[2], b[3] + score))
        new_beams = list(reversed(sorted(new_beams, key=lambda x: x[3])))[:beam_width]

        for i in range(len(new_beams)):
            b = new_beams[i]
            output, hidden = model.inference(b[0][-1], b[2], TEMPERATURE)
            new_beams[i] = (b[0], output, hidden, b[3])
        beams = new_beams
    return beams[0][0]


def generate_beats(model, device, seed_notes, sequence_length, note_to_int, int_to_note, sampling_strategy='max', beam_width=BEAM_WIDTH):
    model.eval()

    with torch.no_grad():
        seed_notes_arr = torch.LongTensor([note_to_int[n] for n in seed_notes])

        # Computes the initial hidden state from the prompt (seed words).
        hidden = None
        for ind in seed_notes_arr:
            data = ind.to(device)
            output, hidden = model.inference(data, hidden)
        
        if sampling_strategy == 'max':
            outputs = max_sampling_strategy(sequence_length, model, output, hidden)

        elif sampling_strategy == 'sample':
            outputs = sample_sampling_strategy(sequence_length, model, output, hidden)

        elif sampling_strategy == 'beam':
            outputs = beam_sampling_strategy(sequence_length, beam_width, model, output, hidden)

        return [int_to_note[i.cpu().item() if torch.is_tensor(i) else i] for i in outputs]

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(model, device, optimizer, train_loader, lr, epoch, log_interval):
    model.train()
    losses = []
    hidden = None
    for batch_idx, (data, label) in enumerate(tqdm.tqdm(train_loader)):
        data, label = data.to(device), label.to(device)
        # Separates the hidden state across batches. 
        # Otherwise the backward would try to go all the way to the beginning every time.
        if hidden is not None:
            hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        output, hidden = model(data)
        pred = output.max(-1)[1]
        loss = model.loss(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        hidden = None
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output, hidden = model(data, hidden)
            test_loss += model.loss(output, label, reduction='mean').item()
            pred = output.max(-1)[1]
            correct_mask = pred.eq(label.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct
            # Comment this out to avoid printing test results
            if batch_idx % 10 == 0:
                print('Input\t%s\nGT\t%s\npred\t%s\n\n' % (
                    " ".join(test_loader.dataset.to_notes(data[0].cpu().numpy())),
                    " ".join(test_loader.dataset.to_notes(label[0].cpu().numpy())),
                    " ".join(test_loader.dataset.to_notes(pred[0].cpu().numpy()))))

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / (len(test_loader.dataset) * test_loader.dataset.sequence_length)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset) * test_loader.dataset.sequence_length,
        100. * correct / (len(test_loader.dataset) * test_loader.dataset.sequence_length)))
    return test_loss, test_accuracy

if __name__ == "__main__":
    notes = [n for s in get_notes() for n in s]
    n_vocab = len(set(notes))
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    int_to_note = {val: key for key, val in note_to_int.items()}

    encoded_notes = [note_to_int[n] for n in notes]
    # treat the above as a "sequence of characters"

    data_train = MidiDataset(encoded_notes, note_to_int, int_to_note, SEQUENCE_LENGTH, BATCH_SIZE)
    data_test = MidiDataset(encoded_notes, note_to_int, int_to_note, SEQUENCE_LENGTH, TEST_BATCH_SIZE)

    use_cuda = USE_CUDA and torch.cuda.is_available()

    # device = torch.device("cpu")
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    import multiprocessing
    num_workers = multiprocessing.cpu_count()
    print('num workers:', num_workers)

    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE,
                                               shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE,
                                              shuffle=False, **kwargs)

    model = MidiNet(len(note_to_int), FEATURE_SIZE).to(device)

    # Adam is an optimizer like SGD but a bit fancier. It tends to work faster and better than SGD.
    # We will talk more about different optimization methods in class.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    start_epoch = restore_latest(model, 'checkpoints')

    train_losses, test_losses, test_accuracies = read_log(LOG_PATH, ([], [], []))
    test_loss, test_accuracy = test(model, device, test_loader)

    test_losses.append((start_epoch, test_loss))
    test_accuracies.append((start_epoch, test_accuracy))
    
    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            lr = LEARNING_RATE * np.power(0.25, (int(epoch / 6)))
            train_loss = train(model, device, optimizer, train_loader, lr, epoch, PRINT_INTERVAL)
            test_loss, test_accuracy = test(model, device, test_loader)
            train_losses.append((epoch, train_loss))
            test_losses.append((epoch, test_loss))
            test_accuracies.append((epoch, test_accuracy))
            write_log(LOG_PATH, (train_losses, test_losses, test_accuracies))
            model.save_best_model(test_accuracy, 'checkpoints/%03d.pt' % epoch)
            # seed_words = 'Harry Potter, Voldemort, and Dumbledore walk into a bar. '
            seed_words = 'A3 G#3 B4 F#6 E-5 B4 G#3 G#3 F#6 E-5 F#6 B4 B4 G#3 F#6 G#3 B6 E-5 E-5 G#3 F#6 E-5 B4 E-5 F#6 G#3 B4 B4 E-5 G#3 E-5 G3 G#3 B4 E-5 F#6 E-5 F#6 E-5 B4 G#3 G3 G#3 F#6 B4 E-5 E-5 B4 F#6 E-5 B4 F#6 E-5 B4 B6 E-5 E-5 B4 E-5 F#6 E-5 G#3 F#6 F#6 E-5 G#3 B4 E-5 B4 F#6 F#6 E-5'.split(" ")
            generated_sentence = generate_beats(model, device, seed_words, 200, note_to_int, int_to_note, 'max')
            print('generated max\t\t', " ".join(generated_sentence))
            for ii in range(10):
                generated_sentence = generate_beats(model, device, seed_words, 200, note_to_int, int_to_note, 'sample')
                print('generated sample\t', " ".join(generated_sentence))
            generated_sentence = generate_beats(model, device, seed_words, 200, note_to_int, int_to_note, 'beam')
            print('generated beam\t\t', " ".join(generated_sentence))
            print('')

    except KeyboardInterrupt as ke:
        print('Interrupted')
    except:
        import traceback
        traceback.print_exc()
    finally:
        print('Saving final model')
        model.save_model('checkpoints/%03d.pt' % epoch, 0)
        ep, val = zip(*train_losses)
        plot(ep, val, 'Train loss', 'Epoch', 'Error')
        ep, val = zip(*test_losses)
        plot(ep, val, 'Test loss', 'Epoch', 'Error')
        ep, val = zip(*test_accuracies)
        plot(ep, val, 'Test accuracy', 'Epoch', 'Error')





# now do things to this @ani


#idea:
# 1. load all midi songs
# 2. concat all midi songs into one gigantic song
# 3. treat each note as a "character"
