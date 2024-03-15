import cv2
import mediapipe as mp
from pydub import AudioSegment
import math
import os
import shutil
import tensorflow as tf
import tensorflow_io as tfio
import subprocess
import numpy as np

def analyse_video(file, video_model):
    preds = []
    Distances = []
    cap = cv2.VideoCapture(file)
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

    while True:
        success, img = cap.read()
        try:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            break
        image_height, image_width, ic = img.shape
        results = faceMesh.process(imgRGB)
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:

                Points = []
                for id,lm in enumerate(faceLms.landmark):
                    x, y = int(lm.x * image_width), int(lm.y * image_height)
                    Points.append([id, x, y])

                nose_distantce = ((Points[94][1] -  Points[168][1])**2 + (Points[94][2] -  Points[168][2])**2)**0.5
                lips_gorisont_left = (((Points[214][1] -  Points[61][1])**2 + (Points[214][2] -  Points[61][2])**2)**0.5) / nose_distantce
                lips_gorisont_right = (((Points[291][1] -  Points[434][1])**2 + (Points[291][2] -  Points[434][2])**2)**0.5) / nose_distantce
                nose_lips = (((Points[94][1] -  Points[0][1])**2 + (Points[94][2] -  Points[0][2])**2)**0.5) / nose_distantce
                nose_lips_left = (((Points[39][1] -  Points[206][1])**2 + (Points[39][2] -  Points[206][2])**2)**0.5) / nose_distantce
                nose_lips_right = (((Points[269][1] -  Points[426][1])**2 + (Points[269][2] -  Points[426][2])**2)**0.5) / nose_distantce
                mouth_gorisont = (((Points[308][1] -  Points[78][1])**2 + (Points[308][2] -  Points[78][2])**2)**0.5) / nose_distantce
                mouth_vertical = (((Points[13][1] -  Points[14][1])**2 + (Points[13][2] -  Points[14][2])**2)**0.5) / nose_distantce
                lips_down_left = (((Points[204][1] -  Points[91][1])**2 + (Points[204][2] -  Points[91][2])**2)**0.5) / nose_distantce
                lips_down_right = (((Points[424][1] -  Points[321][1])**2 + (Points[424][2] -  Points[321][2])**2)**0.5) / nose_distantce
                Distances.append([lips_gorisont_left,lips_gorisont_right,nose_lips,nose_lips_left,nose_lips_right,
                      mouth_gorisont,mouth_vertical,lips_down_left,lips_down_right])

                if(len(Distances) > 15):
                    f = False
                    for i in Distances[:-15]:
                        if i[6] > 0.02:
                            f = True
                            break
                    if f:
                        preds.append(video_model.predict(Distances[-1]))
    return float(list(preds).count(1)/len(preds)*100)

class SplitWavAudio():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + filename

        self.audio = AudioSegment.from_wav(self.filepath)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min
        t2 = to_min
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + split_filename, format="wav")

    def multiple_split(self, ms):
        total_sec = math.ceil(self.get_duration())
        s = 0
        for i in range(0, total_sec*1000, ms):
            split_fn = str(s) + '.wav'
            self.single_split(i, i+ms, split_fn)
            #print(str(i) + ' Done')
            #if i == total_sec*1000 - ms:
            #    print('All splited successfully')
            s+=1
        return s - 2

def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def preprocess(file_path):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram


def analyse_audio(file, numf, audio_model):
    folder = "content/audio/" + str(numf) + "/"

    if not os.path.exists("content/audio/" + str(numf)):
        os.makedirs("content/audio/" + str(numf))

    command = "ffmpeg -i " + file + " -ab 160k -ac 2 -ar 44100 -vn " + folder + "F.wav"
    subprocess.call(command, shell=True)

    split_wav = SplitWavAudio(folder, "F.wav")
    parts_n = split_wav.multiple_split(500)
    X = []
    for i in range(parts_n):
        X.append(preprocess(folder + str(i) + '.wav'))
    X = np.array(X)
    preds = audio_model.predict(X)
    shutil.rmtree(os.getcwd() + "/content/audio/" + str(numf))
    preds = [0 if x[0] >= 0.5 else 1 for x in preds]
    return float(list(preds).count(1)/len(preds)*100)