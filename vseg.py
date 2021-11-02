# Video Segmentation Using inaSpeechSegmenter
# Original: https://gist.github.com/tam17aki/11eb1566a2d48b382607d23dddb98891
        # https://qiita.com/Nahuel/items/aba4eaabd686a1d89c37
        # https://nantekottai.com/2020/06/14/video-cut-silence/
        # https://tam5917.hatenablog.com/entry/2020/01/25/132113 # バグがあるけど参考になる

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
parser.add_argument('-i', '--insert', default=None, help='[pathname of insert wav file]')
args = parser.parse_args()
input_video_name = os.path.abspath(args.arg1)
dest_mov_name = os.path.abspath(args.arg2)

if args.csv == None or os.path.isfile(args.csv) == False:
    dest_csv_name = os.path.join(os.path.dirname(os.path.abspath(args.arg2)), 'segResult.csv')
else:
    dest_csv_name = os.path.abspath(args.csv)

# 入力の絶対パスを解析
if os.path.isfile(input_video_name) == False:
    print("The first argument should be an existing file.")
    sys.exit()
if os.path.isfile(dest_mov_name) == True:
    print("Output file already exists.")
    sys.exit()

with tempfile.TemporaryDirectory() as dname1:
    print(dname1)
    tmp_wav_name = os.path.join(dname1, 'tmp.wav')
    tmp_mov_name = os.path.join(dname1, 'tmp.mov')
    
    if os.path.isfile(dest_csv_name) == True:
        print(f'Found {os.path.basename(dest_csv_name)}')
    else:
        # 動画から音声ファイルを分離
        print("Separating audio files from video...")
        command = "ffmpeg -i '"+input_video_name+"' -loglevel quiet -vn '"+tmp_wav_name+"'"
        sb.call(command, shell=True)
        # 区間検出実行
        seg = Segmenter(vad_engine='smn', detect_gender=False)
        print("Interval detection in progress...")
        segmentation = seg(tmp_wav_name)
        print("End of interval detection")
        seg2csv(segmentation, dest_csv_name)# 区間ごとのラベル,開始時間,終了時間をcsv形式で保存
        del segmentation
    
    label = np.loadtxt(dest_csv_name, delimiter='\t', dtype=str, skiprows=1, usecols=[0])
    segs = np.loadtxt(dest_csv_name, delimiter='\t', dtype=np.float32, skiprows=1, usecols=[1, 2])# float16 では小数部分が切り捨てられてしまう
    
    # 断片の数を確認
    if len(label[:]) == 1:
        print("The number of fragments is one.")
        sys.exit()
    
    if args.insert == None:
        insert_wav_name = tmp_wav_name
    elif os.path.isfile(args.insert) == False:
        print("The specified insert wav file does not exist.")
        insert_wav_name = tmp_wav_name
    else:
        insert_wav_name = os.path.abspath(args.insert)
    
    if os.path.isfile(insert_wav_name) == False:
        # 動画から音声ファイルを分離
        print("Separating audio files from video...")
        command = "ffmpeg -i '"+input_video_name+"' -loglevel quiet -vn '"+insert_wav_name+"'"
        sb.call(command, shell=True)
    
    # （分離した）音声ファイルをwaveモジュールで読み込む
    with wave.open(insert_wav_name) as wav:
        samplewidth = wav.getsampwidth()
        nchannels = wav.getnchannels()
        framerate = wav.getframerate()
        nframes = wav.getnframes()
        framerate_nchannels = framerate * nchannels
        if samplewidth == 2:
            data = np.frombuffer(wav.readframes(nframes), dtype='int16').copy()
        elif samplewidth == 4:
            data = np.frombuffer(wav.readframes(nframes), dtype='int32').copy()
        else:
            # https://qiita.com/Dsuke-K/items/2ad4945a81644db1e9ff
            print("vseg: Sample width is ", samplewidth)
            sys.exit()
    
    if abs(nframes/framerate -  segs[-1, -1]) > 0.02:
        if insert_wav_name == tmp_wav_name:
            print(f'vseg: This csv file \"{os.path.basename(dest_csv_name)}\" is incompatible.')
        else:
            print(f'vseg: There is a {abs(nframes/framerate -  segs[-1, -1])} second difference between the extracted information and the insert wav file \"{insert_wav_name}\"')
        sys.exit()
    
    # speech noEnergy noise music
    speech = segs[label == 'speech']
    
    # 瞬間的に音がなっている箇所を除く
    duration = speech[:, 1] - speech[:, 0]
    bool_list = duration < min_keep_duration
    speech = np.delete(speech, bool_list, 0)
    duration = np.delete(duration, bool_list, 0)
    del bool_list
    
    # speechではない長い区間を検出し，無音データで埋める
    print("Detecting a long section that is not a speech...")
    for i in tqdm(range(len(speech) - 1)):
        if (speech[i + 1][0] - speech[i][1] > padding_silence_duration):
            data[round(speech[i][1] * framerate_nchannels):round(speech[i + 1][0] * framerate_nchannels)] = 0
            # 余白を挿入する
            speech[i][1] += padding_time
            speech[i + 1][0] -= padding_time
    
    # 加工した音声データを書き出す
    with wave.open(tmp_wav_name, 'w') as wav:
        wav.setsampwidth(samplewidth)
        wav.setframerate(framerate)
        wav.setnchannels(nchannels)
        wav.writeframes(data)
    
    # 動画の音声を差し替える
    print("Replacing audio...")
    command = "ffmpeg -i '"+input_video_name+"' -loglevel info -i '"+tmp_wav_name+"' -c:v copy -c:a aac -strict experimental -map 0:v -map 1:a '"+tmp_mov_name+"'"
    sb.call(command, shell=True)
    print("Replacement completed.")
    
    # 動画を切り出す
    speech = speech.astype('str')
    duration = duration.astype('str')
    num_speech = len(speech) # 断片の個数
    num_digits = len(str(num_speech)) # 断片の個数の桁数
    with tempfile.TemporaryDirectory() as dname2:
        print("Cutting out the video...")
        for i in tqdm(range(num_speech)):
            command = "ffmpeg -ss "+speech[i][0]+" -i '"+tmp_mov_name+"' -loglevel quiet -t "+duration[i]+" '"+dname2+"/"+str(i).zfill(num_digits)+".mov'"
            sb.call(command, shell=True)
        
        # 動画を繋げる
        print("Merging videos...")
        target_list = os.path.join(dname1, 'target_list.txt')
        # 遅いが正確でファイルサイズはやや小さい
        #command = "(for f in \""+dname2+"\"/*.mov; do echo file \\'$f\\'; done)>'"+target_list+"'; ffmpeg -loglevel info -safe 0 -f concat -i '"+target_list+"' '"+dest_mov_name+"'"
        # 速いが不正確でファイルサイズはやや大きい
        command = "(for f in \""+dname2+"\"/*.mov; do echo file \\'$f\\'; done)>'"+target_list+"'; ffmpeg -loglevel quiet -safe 0 -f concat -i '"+target_list+"' -c copy '"+dest_mov_name+"'"
        
        sb.call(command, shell=True)