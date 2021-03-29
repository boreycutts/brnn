import matplotlib.pyplot as plt
import numpy as np
import types

fig, axs = plt.subplots(2, 2)

loss = np.load("figures/LSTM3232/Adadelta Loss vs Epochs.npy")
axs[0, 0].plot(loss, label="BRNN3232")
print("BRNN3232")
print(loss[len(loss)-1])
loss = np.load("figures/LSTM3232/Adam Loss vs Epochs.npy")
print(loss[len(loss)-1])
axs[0, 1].plot(range(1200, 1700), loss)

loss = np.load("figures/LSTM32128/Adadelta Loss vs Epochs.npy")
axs[0, 0].plot(loss, label="BRNN32128")
print("BRNN32128")
print(loss[len(loss)-1])
loss = np.load("figures/LSTM32128/Adam Loss vs Epochs.npy")
print(loss[len(loss)-1])
axs[0, 1].plot(range(1200, 1700), loss)

loss = np.load("figures/LSTM6432/Adadelta Loss vs Epochs.npy")
axs[0, 0].plot(loss, label="BRNN6432")
print("BRNN6432")
print(loss[len(loss)-1])
loss = np.load("figures/LSTM6432/Adam Loss vs Epochs.npy")
print(loss[len(loss)-1])
axs[0, 1].plot(range(1200, 1700), loss)

loss = np.load("figures/LSTM64128/Adadelta Loss vs Epochs.npy")
axs[0, 0].plot(loss, label="BRNN64128")
print("BRNN64128")
print(loss[len(loss)-1])
loss = np.load("figures/LSTM64128/Adam Loss vs Epochs.npy")
print(loss[len(loss)-1])
axs[0, 1].plot(range(1200, 1700), loss)

loss = np.load("figures/CNN3232/Adadelta Loss vs Epochs.npy")
axs[1, 0].plot(loss, label="CNN3232")
print("CNN3232")
print(loss[len(loss)-1])
loss = np.load("figures/CNN3232/Adam Loss vs Epochs.npy")
print(loss[len(loss)-1])
axs[1, 1].plot(range(500, 750), loss)

loss = np.load("figures/CNN32128/Adadelta Loss vs Epochs.npy")
axs[1, 0].plot(loss, label="CNN32128")
print("CNN32128")
print(loss[len(loss)-1])
loss = np.load("figures/CNN32128/Adam Loss vs Epochs.npy")
print(loss[len(loss)-1])
axs[1, 1].plot(range(500, 750), loss)

loss = np.load("figures/CNN6432/Adadelta Loss vs Epochs.npy")
axs[1, 0].plot(loss, label="CNN6432")
print("CNN6432")
print(loss[len(loss)-1])
loss = np.load("figures/CNN6432/Adam Loss vs Epochs.npy")
print(loss[len(loss)-1])
axs[1, 1].plot(range(500, 750), loss)

loss = np.load("figures/CNN64128/Adadelta Loss vs Epochs.npy")
axs[1, 0].plot(loss, label="CNN64128")
print("CNN64128")
print(loss[len(loss)-1])
loss = np.load("figures/CNN64128/Adam Loss vs Epochs.npy")
print(loss[len(loss)-1])
axs[1, 1].plot(range(500, 750), loss)

axs[0, 1].yaxis.tick_right()
axs[1, 1].yaxis.tick_right()

axs[0, 0].legend()
axs[1, 0].legend()

axs[0, 0].set_title("Adadelta Phase")
axs[0, 1].set_title("Adam Phase")

pad = plt.rcParams["xtick.major.size"] + plt.rcParams["xtick.major.pad"]
def bottom_offset(self, bboxes, bboxes2):
    bottom = self.axes.bbox.ymin
    self.offsetText.set(va="top", ha="left") 
    oy = bottom - pad * self.figure.dpi / 72.0
    self.offsetText.set_position((1, oy))

axs[1, 1].yaxis._update_offset_text_position = types.MethodType(bottom_offset, axs[1, 1].yaxis)

plt.subplots_adjust(wspace=0)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

fig.text(0.5, 0.04, 'Epochs', ha='center')
fig.text(0.04, 0.5, 'Adadelta Loss', va='center', rotation='vertical')
fig.text(0.97, 0.5, 'Adam Loss', va='center', rotation=270)

plt.show()