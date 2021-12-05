import librosa
import numpy as np
import os
from torch.utils.data import IterableDataset
import matplotlib.pyplot as plt
base_directory = "../music/paired/"
import pickle
import threading
import random
# input(os.getcwd())
# files = os.listdir(base_directory)
# print(files)

# have a class that is responsible for each of the 250 samples
# this sample will take in a number, and it will handle the rest
#come sample time, we will just call the sample function
class Song:
    def __init__(self, base_directory, song):
        self.instruments = ["distortion", "harp", "harpsichord", "piano", "timpani"]
        self.base_instrument = "piano"
        self.song = song

        self.spectrogram_list = {}
        print("\tbase")
        self.base, self.fs = self.read_audio_spectum(base_directory + song + "_" + self.base_instrument + ".wav")

        self.length = self.base.shape[1]
        self.features = self.base.shape[0]
        thread_list = list()

        for instrument in self.instruments:
            print(f"\t{instrument}")
            S, fs = self.read_audio_spectum(base_directory + self.song + "_" + instrument + ".wav")
            assert fs == self.fs, "sample rate problem!"
            self.spectrogram_list[instrument] = S
    
    def inverse_instrument_index(self, instrument):
        for i, curr_instrument in enumerate(self.instruments):
            if instrument == curr_instrument:
                return i
        raise Exception("invalid instrument!")

    def sample(self, instrument = None):
        #returns the sample along with a one-hot vector
        one_hot = np.zeros(len(self.instruments))
        if instrument is None:
            instrument_index = np.random.randint(0, len(self.instruments))
        else:
            instrument_index = self.inverse_instrument_index(instrument)
        one_hot[instrument_index] = 1

        start_location = np.random.randint(self.length - self.features)
        # input(self.base.shape)
        # input(self.base[:, start_location : start_location + self.features].shape)
        return np.expand_dims(self.base[:, start_location : start_location + self.features], axis = 0), \
               np.expand_dims(self.spectrogram_list[self.instruments[instrument_index]][:, start_location : start_location + self.features], axis = 0), \
               one_hot


    def read_audio_spectum(self, filename):
        N_FFT = 1024
        x, fs = librosa.load(filename)  # Duration=58.05 so as to make sizes convenient
        S = librosa.stft(x, N_FFT)
        p = np.angle(S)
        S = np.log1p(np.abs(S))
        # input(S.shape)
        #nfft / 2 + 1, by
        # S = np.pad(S, ((0, 0),(0, 513 - 431)), mode = "constant", constant_values = 0)
        return S, fs


class SampleLibrary(IterableDataset):
    def __init__(self, base_directory, mode = "same_song"):
        self.songsList = list()
        self.base_directory = base_directory
        self.songs = ["bouree", "libes", "op19", "symphony4", "symphony7"]
        self.test_song_name = "pathetique_2"

        thread_list = list()
        for song in self.songs:
            t = threading.Thread(target = self.grab_song, args = (song, ))
            t.start()
            thread_list.append(t)
            # self.songsList.append(Song(self.base_directory, song))
        t = threading.Thread(target = self.grab_test_song)
        t.start()

        t.join()
        for thread in thread_list:
            thread.join()


    def grab_test_song(self):
        self.test_song = Song(self.base_directory, self.test_song_name)

    def grab_song(self, song):
        print(song)
        self.songsList.append(Song(self.base_directory, song))

    def samplePair(self, test = False, test_instrument = None):
        if test:
            return self.test_song.sample(test_instrument)

        index = np.random.randint(0, len(self.songsList))
        selectedSample = self.songsList[index]
        if test_instrument is not None:
            return selectedSample.sample(test_instrument)
        return selectedSample.sample()

    #this function returns a target, base, and style audio. The target and base
    #are the same song, but the style is not. This should demonstrate how this model is
    #just as robust as a normal style transfer model (the target is only for comparison)
#     def sampleTest(self, test_instrument = None, batch_size = 8):
#         assert test_instrument is not None, "this function currently only works with test instrument"
#         x_list = list()
#         target_list = list()
#         style_list = list()
#         for i in range(batch_size):
#             x, target, _ = self.test_song.sample(test_instrument)
#             _, style, _ = self.test_song.sample(test_instrument)
#             x_list.append(x)
#             target_list.append(target)
#             style_list.append(style)
#         x_stacked = np.stack(x_list)
#         target_stacked = np.stack(target_list)
#         style_stacked = np.stack(style_list)
#         return x_stacked, target_stacked, style_stacked
    
    
    def sampleTest(self, test_instrument = None):
        assert test_instrument is not None, "this function currently only works with test instrument"
        x, target, _ = self.test_song.sample(test_instrument)
        _, style, _ = self.test_song.sample(test_instrument)
        return x, target, style
    


    def __iter__(self):
        while True:
            yield self.samplePair()