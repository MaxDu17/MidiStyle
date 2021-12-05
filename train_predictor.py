import librosa
import pickle
import numpy as np
import random
import torch
from torch import nn
import os
import soundfile as sf
import csv 
from shutil import copyfile
import threading
import matplotlib.pyplot as plt
from ConvAE import ConvAE, create_network, accuracy_1_min_mab, normalized_loss
from pipeline_whole import SampleLibrary

sampler_dataset = pickle.load(open("simple_dataset.pkl", "rb"))


def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


sampler_dataset_iterable = iter(torch.utils.data.DataLoader(sampler_dataset,
                                                            batch_size=16,
                                                            num_workers=0,
                                                            # pin_memory=True,
                                                            worker_init_fn=worker_init_fn))  # what is pin memory?

feature_maps = 8
depth = 12
pooling_freq = 1e100  # large number to disable pooling layers
strided_conv_freq = 2
strided_conv_feature_maps = 8
code_size = 5
input_dim = (1, 513, 513)

fig, (ax1, ax2) = plt.subplots(ncols=2)
plt.ion()  # needed to prevent show() from blocking


def visualize(truth, regenerated, save=False, name=None, step='no-step', visible=True):
    ax1.clear()
    ax2.clear()
    ax1.imshow(truth)
    ax2.imshow(regenerated)
    if visible:
        plt.show()
    plt.pause(1)
    if save and name is None:
        plt.savefig("test.png")
    elif save:
        plt.savefig(f"{name}_{step}.png")


CONV_ENC_BLOCK = [("conv1", feature_maps), ("relu1", None)]
CONV_ENC_LAYERS = create_network(CONV_ENC_BLOCK, depth,
                                 pooling_freq=pooling_freq,
                                 strided_conv_freq=strided_conv_freq,
                                 strided_conv_channels=strided_conv_feature_maps,
                                 )
CONV_ENC_NW = CONV_ENC_LAYERS
CONV_ENC_NW = CONV_ENC_LAYERS + [("flatten1", None), ("linear1", code_size), ("softmax1", None)]

model = ConvAE(input_dim, enc_config=CONV_ENC_NW, disable_decoder = True)

if torch.cuda.is_available():
    print("cuda available!")
    model = model.cuda()

AE_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = nn.CrossEntropyLoss()

reconstructor_thread_list = list()


def test_evaluate(model, num_tests_per_instrument, step, save=True):
    instruments = ["distortion", "harp", "harpsichord", "piano", "timpani"]
    confusion_matrix = np.zeros((5, 5), dtype = np.int8)
    
    soft_make_dir(f"evaluation_{step}")
    os.chdir(f"evaluation_{step}")
    
    
    for instrument in instruments:
        print(instrument)
        for i in range(num_tests_per_instrument):
            print(f"\t{i}")
            data, target, one_hot = sampler_dataset.samplePair(test=True, test_instrument=instrument)
            target = np.expand_dims(target, axis=0)
            target = torch.as_tensor(target, device="cuda")
            code = model.forward(target, None) #one_hot is not used here
            if torch.cuda.is_available():
                code = code.cpu().detach().numpy()
            confusion_matrix[np.argmax(code), np.argmax(one_hot)] += 1
    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    np.savetxt("confusion_test.txt", confusion_matrix)
    if save:
        print("saving model!")
        torch.save(model.state_dict(), f"classifier_params_{step}.pt")
    os.chdir("../")
    return accuracy
    
    
def valid_evaluate(model, num_tests_per_instrument, step, save=True):
    instruments = ["distortion", "harp", "harpsichord", "piano", "timpani"]
    confusion_matrix = np.zeros((5, 5), dtype = np.int8)
    
    soft_make_dir(f"evaluation_{step}")
    os.chdir(f"evaluation_{step}")
    
    
    for instrument in instruments:
        print(instrument)
        for i in range(num_tests_per_instrument):
            print(f"\t{i}")
            data, target, one_hot = sampler_dataset.samplePair(test=False, test_instrument=instrument)
            target = np.expand_dims(target, axis=0)
            target = torch.as_tensor(target, device="cuda")
            code = model.forward(target, None) #one_hot is not used here
            if torch.cuda.is_available():
                code = code.cpu().detach().numpy()
            confusion_matrix[np.argmax(code), np.argmax(one_hot)] += 1
    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    np.savetxt("confusion_valid.txt", confusion_matrix)
    if save:
        print("saving model!")
        torch.save(model.state_dict(), f"classifier_params_{step}.pt")
    os.chdir("../")
    return accuracy 



def soft_make_dir(path):
    try:
        os.mkdir(path)
    except:
        print("directory already exists!")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    experiment = "audio_classifier_smaller"
    num_training_steps = 30000
    path = f"experiments_classifier/{experiment}"
    soft_make_dir(path)
    copyfile("train_predictor.py", f"{path}/train_predictor.py")
    copyfile("ConvAE.py", f"{path}/ConvAE.py")
    copyfile("pipeline_whole.py", f"{path}/pipeline_whole.py")
    os.chdir(path)
    
    progress_writer_file = open("valid-test.csv", "w")
    progress_writer = csv.writer(progress_writer_file, delimiter = ",")

    norm_mult = 1e-7
    progress_writer.writerow(["valid", "test"])
    for i in range(num_training_steps + 1):
        x, target, one_hot = sampler_dataset_iterable.next()

        target = torch.as_tensor(target, device=device)
        one_hot = torch.as_tensor(one_hot, device=device, dtype=torch.float32)
        code = model.forward(target, None) #one_hot should not be used 
        if i % 3000 == 0:
            print("eval time!")
            test_acc = test_evaluate(model, num_tests_per_instrument=25, step=i, save=True)
            valid_acc = valid_evaluate(model, num_tests_per_instrument=25, step=i, save=True)
            progress_writer.writerow([valid_acc, test_acc])
            progress_writer_file.flush()
        encoding_loss = loss(code, one_hot)  # + norm_mult * torch.sum(torch.abs(out))
        if i % 100 == 0:
            print(i, " ", encoding_loss.cpu().detach().numpy())
        AE_optimizer.zero_grad()
        encoding_loss.backward()
        AE_optimizer.step()
    
    progress_writer_file.close()