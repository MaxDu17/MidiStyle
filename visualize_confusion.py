import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
# file = "confusion_test.txt"
# file = "confusionExisting Audio Style Transfer Baseline.txt"
file = "confusionMidiStyle Network.txt"
arr = np.loadtxt(file)
print(arr)

instruments = ["distortion", "harp", "harpsichord", "piano", "timpani"]

df_cm = pd.DataFrame(arr, instruments, instruments)
plt.figure(figsize=(8,6))
plt.title("Style Classifier on MidiStyle Outputs")
sn.set(font_scale=1) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cbar = False) # font size

plt.show()
plt.savefig("midistyle_confusion.png")