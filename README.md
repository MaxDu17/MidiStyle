# MidiStyle
This is the release repository for MidiStyle, a parameteric audio style transfer models developed by Maximilian Du and Aditya Saligrama.

# Code
- `ConvAE` is our autoencoder implementation, adapted from https://github.com/yrevar/Easy-Convolutional-Autoencoders-PyTorch
- `styletransfer_baseline` is the Gram Matrix baseline, adapted from https://github.com/alishdipani/Neural-Style-Transfer-Audio
- `generateDataset` will read the raw WAV data and save a specially-formatted dataset file, which is used in `train_predictor`, `train_midistyle`, as well as the metrics. We separate the dataset formatting from the dataset sampling because there is significant overhead present in loading and processing each audio file. If we save it in a format that can be loaded to RAM at once, it speeds up training by many times. 
- `pipeline_whole` will read the special dataset file written by `generateDataset` and interface with a torch DataLoader to facilitate the training and testing data pipeline. 
- `train_predictor` will train an audio style classifier used in the MidiStyle model as well as a few evaluation metrics
- `train_midistyle` will train the MidiStyle model
- `generate_stats` calculates losses on a set of generated outputs and creates bar graphs 
- `run_model` runs a trained MidiStyle model consecutively to style transfer arbitrary length audio files
- `visualize_confusion` generates the confusion matrix visualization 

# Samples
Our dataset was too large to include in this repository, but we have included a few sample outputs and graphs in `run_data`. 

If interested in adapting this codebase, feel free to contact maxjdu@stanford.edu or saligrama@stanford.edu with questions. 
