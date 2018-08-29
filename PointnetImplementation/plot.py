import matplotlib.pyplot as plt
import numpy as np

lossClass = np.loadtxt("output/scoresExtendedClassification/lossClassification.out")
accuracyClass = np.loadtxt("output/scoresExtendedClassification/accuracyClassification.out")

lossSeg = np.loadtxt("lossSegmentation.out")
accuracySeg = np.loadtxt("accuracySegmentation.out")

plt.plot(lossClass)
plt.plot(accuracyClass)