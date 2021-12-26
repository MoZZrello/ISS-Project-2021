from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import wave
import struct

def normalizeData(audio, sampleCount):
    averageValue = sum(audio) / sampleCount
    normalizedAudio = audio - averageValue
    return (normalizedAudio / max(abs(normalizedAudio)))

def getFrames(inputSignal, sampleCount):
    framesArr = []
    for i in range(sampleCount // 512 - 1):
        framesArr.append(inputSignal[i * 512: i * 512 + 1024])
    return framesArr

def DFTcoeffs(x, N):
    coefficients = []
    for k in range(0, N // 2):
        coefficients.append(sum(x * [(np.exp(-1j * 2 * np.pi / N)) ** (k * n) for n in range(N)]))
    return coefficients


def createDFT(frames, sampleRate):
    print("Calculating Discrete Fourier Transform")
    coefficients = np.array([DFTcoeffs(np.array(frames[frame]), 1024) for frame in range(len(frames))])
    magnitudes = abs(coefficients)
    frequencies = [k * sampleRate // 1024 for k in range(512)]
    print("Finished Discrete Fourier Transform")
    return magnitudes, frequencies

def createFFT(frames, sampleRate):
    magnitudes = abs(np.array([np.fft.fft(frames[k])[0: 512] for k in range(len(frames))]))
    frequencies = [k * sampleRate // 1024 for k in range(512)]
    return magnitudes, frequencies

def getDF(cosAmount, magnitudes, frequencies):
    #Priemerna magnituda frekvencii a najdeme 4 najvacsie
    magnitudesTranspose = np.transpose(magnitudes)
    magnitudesAvg = np.array([sum(magnitudesTranspose[freq]) for freq in range(len(magnitudesTranspose))])
    maxMagnitudes = np.argpartition(magnitudesAvg, -cosAmount)[-cosAmount:]

    #Z najdenych 4 najvacsich magnitud si vyberieme frekvencie
    disruptiveFreqs = []
    for i in range(cosAmount):
        disruptiveFreqs.append(frequencies[maxMagnitudes[i]])
    print("Found disruptive frequencies: " + str(disruptiveFreqs))
    return disruptiveFreqs

def findCosines(cosAmount, df, sampleCount, sampleRate):
    cosX = np.arange(0, sampleCount)
    cosY = []
    for i in range(cosAmount):
        cosY.append([np.cos(2 * np.pi * df[i] * t / sampleRate) for t in cosX])
    return cosX, cosY

def createCosines(cosAmount, cosY):
    cosYTranspose = np.transpose(cosY)
    disCos = []
    for i in range(len(cosY[0])):
        disCos.append(sum(cosYTranspose[i]) / cosAmount)
    return disCos

def plotZP(zeros, poles):
    circle1 = plt.Circle((0, 0), 1, fill=False)
    plt.gca().add_patch(circle1)
    plt.scatter(zeros.real, zeros.imag, color='g')
    plt.title("ZEROS")
    plt.axis([-1, 1, -1, 1])
    plt.show()

    circle2 = plt.Circle((0, 0), 1, fill=False)
    plt.gca().add_patch(circle2)
    plt.scatter(poles.real, poles.imag, color='b')
    plt.title("POLES")
    plt.axis([-1, 1, -1, 1])
    plt.show()
    return

def createZP2SOS(LD, LU, UD, UU, filterAmount, sampleRate):
    zeros = []
    poles = []
    gains = []
    for i in range(filterAmount):
        WP = [LD[i], UU[i]]
        WS = [LU[i], UD[i]]
        order = signal.buttord(WP, WS, 3, 40, fs = sampleRate)
        print(order)
        z, p, g = signal.butter(order[0], order[1], "bandstop", False, "zpk", sampleRate)
        zeros.append(z)
        poles.append(p)
        gains.append(g)
    plotZP(np.array(zeros[0]),np.array((poles[0])))
    print("Zeros: ")
    print(zeros[0])
    print("Poles: ")
    print(poles[0])

    SOSformat = [signal.zpk2sos(zeros[i], poles[i], gains[i]) for i in range(dfAmount)]
    return SOSformat

def plotFrequency(SOSformat, frequencies, sampleCount, sampleRate):
    fig, axes = plt.subplots(len(SOSformat))
    for i in range(len(SOSformat)):
        freqs, mags = signal.sosfreqz(SOSformat[i], sampleCount, False, sampleRate)
        axes[i].set_title("FREQUENCY RESPONSE - " + str(frequencies[i]) + " Hz")
        axes[i].plot(freqs, abs(mags))
    fig.tight_layout()
    plt.show()
    return

def plotImpulse(SOSformat, freqs, plotCount, sampleCount):
    impulseResponseInputs = np.zeros(sampleCount)
    impulseResponseInputs[0] = 1
    fig, axes = plt.subplots(plotCount)
    for i in range(len(SOSformat)):
        axes[i].set_title("Impulse response (" + str(freqs[i]) + " Hz)")
        axes[i].plot(signal.sosfilt(SOSformat[i], impulseResponseInputs))
    fig.tight_layout()
    plt.show()
    return

def fltrAudio(filteredAudio, sampleCount, sampleRate, SOSformat):
    for settings in SOSformat:
        filteredAudio = signal.sosfilt(settings, filteredAudio)

    filteredAudio = normalizeData(filteredAudio, sampleCount)
    frames = getFrames(filteredAudio, sampleCount)
    magnitudes, frequencies = createDFT(frames, sampleRate)
    print("Creating spectrogram for filtered audio")
    spectrogram = np.transpose(10 * np.log10(abs(magnitudes) ** 2))
    time = np.arange(0, timeLength, timeLength / len(frames))
    plt.pcolormesh(time, frequencies, spectrogram, shading='gouraud')
    plt.ylabel('FREQUENCY [Hz]')
    plt.xlabel('TIME [s]')
    plt.title("SPECTROGRAM")
    cbar = plt.colorbar()
    cbar.set_label('SPECTRAL POWER DENSITY [dB]', rotation=270, labelpad=15)
    plt.show()
    print("Spectrogram created")
    return filteredAudio



print("----------------------------------------------------------")
#Otvorenie audio suboru
sampleRate, audio = wavfile.read('../audio/xharma05.wav')
print("Sample rate: ", sampleRate)

#Načítaj počet sample v audiu
sampleCount = len(audio)
print("Samples count: ", sampleCount)

#Vypočítaj časovú dĺžku audia
timeLength = sampleCount / sampleRate
print("Audio length: ", timeLength, " seconds")

#Nájdi minimálnu a maximálnu hodnotu signálu
print("Minimal value: ", min(audio))
print("Maximal value: ", max(audio))
print("----------------------------------------------------------")

#Vykresli vstupné audio
plotArr = np.arange(0, timeLength, timeLength / sampleCount)
plt.plot(plotArr, audio)
plt.title("INPUT AUDIO SIGNAL")
plt.show()

#Normalizujeme data do dynamického rozsahu
normalizedAudio = normalizeData(audio, sampleCount)

#Získam rámce
frames = getFrames(normalizedAudio, sampleCount)
print("Audio contains", len(frames), "frames, all with ", len(frames[0]), "samples.")
print("----------------------------------------------------------")

#Plot one of the frames
plotArr = np.arange(0, 1024 / sampleRate, 1 / sampleRate)
prettyFrame = 18
plt.plot(plotArr, frames[prettyFrame])
plotTitle = "FRAME #" + str(prettyFrame)
plt.ylabel("AMPLITUDE")
plt.xlabel("TIME [s]")
plt.title(plotTitle)
plt.show()

#Diskretna Fourierova Transformacia - DFT
magnitudes, frequencies = createDFT(frames, sampleRate)
FFTmagnitudes, FFTfrequencies = createFFT(frames, sampleRate)
res = np.allclose(magnitudes, FFTmagnitudes)
print("Are DFT and FFT magnitudes equal within the tolerance: ", res)
plt.plot(frequencies, magnitudes[0])
plt.title("DISCRETE FOURIER TRANSFORM")
plt.ylabel("MAGNITUDE")
plt.xlabel("FREQUENCY [Hz]")
plt.show()
print("----------------------------------------------------------")


#Spektrogram
print("Creating spectrogram")
spectrogram = np.transpose(10 * np.log10(abs(magnitudes) ** 2))
time = np.arange(0, timeLength, timeLength / len(frames))
plt.pcolormesh(time, frequencies, spectrogram, shading='gouraud')
plt.ylabel('FREQUENCY [Hz]')
plt.xlabel('TIME [s]')
plt.title("SPECTROGRAM")
cbar = plt.colorbar()
cbar.set_label('SPECTRAL POWER DENSITY [dB]', rotation=270, labelpad=15)
plt.show()
print("Spectrogram created")
print("----------------------------------------------------------")

#Hľadanie rušivých frekvencií
dfAmount = 4
df = np.array(getDF(dfAmount, magnitudes, frequencies))

disCosX, disCosY = findCosines(dfAmount, df, sampleCount, sampleRate)
disCos = createCosines(dfAmount, disCosY)

#Vytvorenie spektogramu 4cos
frames = getFrames(disCos, sampleCount)
magnitudes, frequencies = createDFT(frames, sampleRate)
print("Creating spectrogram 4cos")
spectrogram = np.transpose(10 * np.log10(abs(magnitudes) ** 2))
time = np.arange(0, timeLength, timeLength / len(frames))
plt.pcolormesh(time, frequencies, spectrogram, shading='gouraud')
plt.ylabel('FREQUENCY [Hz]')
plt.xlabel('TIME [s]')
plt.title("SPECTROGRAM")
cbar = plt.colorbar()
cbar.set_label('SPECTRAL POWER DENSITY [dB]', rotation=270, labelpad=15)
plt.show()
print("Spectrogram created")
print("----------------------------------------------------------")

#Generovanie signalu 4cos
wav_file = wave.open("../audio/4cos.wav", "w")
wav_file.setparams((1, 2, sampleRate, sampleCount, "NONE", ""))
for sample in disCos:
    wav_file.writeframes(struct.pack('h', int(sample * sampleRate)))

#FILTER
#Hranice filtra
lowerUp = (df - 50)
lowerDown = (df - 20)
upperUp = (df + 20)
upperDown = (df + 50)
print("----------------------------------------------------------")

#Nulové body a póly
SOS = createZP2SOS(lowerDown, lowerUp, upperDown, upperUp, dfAmount, sampleRate)
print("----------------------------------------------------------")

#Výpočet a vykreslenie Frekvenčnej charakteristiky
print("Filters coefficients:")
[print(SOS[i]) for i in range(dfAmount)]
plotFrequency(SOS, df, sampleCount, sampleRate)
plotImpulse(SOS, df, dfAmount, 512)

filteredAudio = fltrAudio(normalizedAudio, sampleCount, sampleRate, SOS)

#Vytvorit filtrovanu zvukovu stopu
wav_file = wave.open("../audio/clean_4pasmovezadrze.wav", "w")
wav_file.setparams((1, 2, sampleRate, sampleCount, "NONE", ""))
for sample in filteredAudio:
    wav_file.writeframes(struct.pack('h', int(sample * 0x7fff)))
print("----------------------------------------------------------")