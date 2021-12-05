from pipeline_whole import SampleLibrary
import pickle
base_directory = "music/"

generatedLibrary = SampleLibrary(base_directory = base_directory)

with open("simple_dataset.pkl", "wb") as f:
    pickle.dump(generatedLibrary, f, protocol=4)
#
# with open("simple_dataset.pkl", "rb") as f:
#     generatedLibrary = pickle.load(f)

# single = generatedLibrary.samplePair(test = True)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.imshow(single[0])
# plt.show()