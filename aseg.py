# Audio Segmentation Using inaSpeechSegmenter   
from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter import seg2csv
import os
import sys
import subprocess as sb
import wave as wave
from tqdm import tqdm
import tempfile
import numpy as np
import csv
import argparse

# 瞬間的な音声区間の閾値
min_keep_duration = 0.4
# この閾値より長い非音声区間には余白を挿入する
padding_silence_duration = 1
# 挿入する余白の長さ
padding_time = 0.2

if padding_time * 2 > padding_silence_duration:
    padding_silence_duration = padding_time * 2

# 位置引数
parser = argparse.ArgumentParser(description='Video Segmentation Using inaSpeechSegmenter')
parser.add_argument('arg1', help='[inputfile]')
parser.add_argument('arg2', help='[outputfile]')
parser.add_argument('-c', '--csv', default=None, help='[pathname of csv]')
parser.add_argument('-d', '--dry', action='store_true')
args = parser.parse_args()
input_file_name = os.path.abspath(args.arg1)
dest_wav_name = os.path.abspath(args.arg2)

# 入力の絶対パスを解析
if os.path.isfile(input_file_name) == False:
    print("The first argument should be an existing file.")
    sys.exit()
if os.path.isfile(dest_wav_name) == True:
    print("Output file already exists.")
    sys.exit()

if args.csv == None or os.path.isfile(args.csv) == False:
    dest_csv_name = os.path.join(os.path.dirname(os.path.abspath(args.arg2)), 'segResult.csv')
else:
    dest_csv_name = os.path.abspath(args.csv)

with tempfile.TemporaryDirectory() as dname1:
    print(dname1)
    
    input_splitext = os.path.splitext(input_file_name)[1]
    if input_splitext == '.wav' or input_splitext == '.WAV':
        tmp_wav_name = input_file_name
    elif input_splitext == '.mov' or input_splitext == '.MOV' or input_splitext == '.mp4' or input_splitext == '.MP4' or input_splitext == '.webm' or input_splitext == '.WEBM':
        tmp_wav_name = os.path.join(dname1, 'tmp.wav')
        # 動画から音声ファイルを分離
        print("Separating audio files from video...")
        command = "ffmpeg -i '"+input_file_name+"' -loglevel quiet -vn '"+tmp_wav_name+"'"
        sb.call(command, shell=True)
    else:
        print("The extension of the first argument must be one of \'.wav\', \'.WAV\', \'.mp4\', \'.MP4\', \'.webm\', \'.WEBM\', \'.mov\' or \'.MOV\'.")
        sys.exit()
    
    if os.path.isfile(dest_csv_name) == True:
        print(f'Found {os.path.basename(dest_csv_name)}')
    else:
        seg = Segmenter(vad_engine='smn', detect_gender=False)
        # 区間検出実行
        print("Interval detection in progress...")
        segmentation = seg(tmp_wav_name)
        print("End of interval detection")
        seg2csv(segmentation, dest_csv_name)# 区間ごとのラベル,開始時間,終了時間をcsv形式で保存
        del segmentation
    
    if args.dry == True:
        sys.exit()
    
    label = np.loadtxt(dest_csv_name, delimiter='\t', dtype=str, skiprows=1, usecols=[0])
    segs = np.loadtxt(dest_csv_name, delimiter='\t', dtype=np.float32, skiprows=1, usecols=[1, 2])# float16 では小数部分が切り捨てられてしまう
    
    # 断片の数を確認
    if len(label[:]) == 1:
        print("The number of fragments is one.")
        sys.exit()
    
    # 分離した音声ファイルをwaveモジュールで読み込む
    with wave.open(tmp_wav_name) as wav:
        samplewidth = wav.getsampwidth()
        nchannels = wav.getnchannels()
        framerate = wav.getframerate()
        nframes = wav.getnframes()
        framerate_nchannels = framerate * nchannels
        len_data = nframes * nchannels
        if samplewidth == 2:
            data = np.frombuffer(wav.readframes(nframes), dtype='int16').copy()
        elif samplewidth == 4:
            data = np.frombuffer(wav.readframes(nframes), dtype='int32').copy()
        else:
            # https://qiita.com/Dsuke-K/items/2ad4945a81644db1e9ff
            print("Sample width : ", samplewidth)
            sys.exit()
    
    if abs(nframes/framerate -  segs[-1, -1]) > 0.02:
        print(f'vseg: This csv file \"{os.path.basename(dest_csv_name)}\" is incompatible.')
    
    # speech noEnergy noise music
    speech = segs[label == 'speech']
    
    # 瞬間的に音がなっている箇所を除く
    duration = speech[:, 1] - speech[:, 0]
    bool_list = duration < min_keep_duration
    speech = np.delete(speech, bool_list, 0)
    duration = np.delete(duration, bool_list, 0)
    del bool_list
    
    # speechではない長い区間を検出し，無音データで埋める
    bool_list = np.zeros(len_data, dtype='bool')
    padding_time_ = round(padding_time * framerate_nchannels)
    print("Detecting a long section that is not a speech...")
    for i in tqdm(range(len(speech) - 1)):
        if (speech[i + 1][0] - speech[i][1] > padding_silence_duration):
            point_1 = round(speech[i][1] * framerate_nchannels)
            point_2 = point_1 + padding_time_
            point_4 = round(speech[i + 1][0] * framerate_nchannels)
            point_3 = point_4 - padding_time_
            data[point_1:point_2] = 0
            bool_list[point_2:point_3] = 1
            data[point_3:point_4] = 0
    
    data = np.delete(data, bool_list)
    del bool_list
    
    # 加工した音声データを書き出す
    with wave.open(dest_wav_name, 'w') as wav:
        wav.setsampwidth(samplewidth)
        wav.setframerate(framerate)
        wav.setnchannels(nchannels)
        wav.writeframes(data)
    