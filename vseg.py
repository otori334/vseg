# Video Segmentation Using inaSpeechSegmenter
# Original: https://gist.github.com/tam17aki/11eb1566a2d48b382607d23dddb98891
        # https://qiita.com/Nahuel/items/aba4eaabd686a1d89c37
        # https://nantekottai.com/2020/06/14/video-cut-silence/
        # https://tam5917.hatenablog.com/entry/2020/01/25/132113

# How To Use
# python3 vseg.py [inputfile] [outputfile]

from inaSpeechSegmenter import Segmenter
#from inaSpeechSegmenter import seg2csv
import os
import sys
import subprocess as sb
import wave as wave
from tqdm import tqdm
import tempfile
import numpy as np

# 瞬間的な音声区間の閾値
min_keep_duration = 0.4
# この閾値より長い非音声区間には余白を挿入する
padding_silence_duration = 1
# 挿入する余白の長さ
padding_time = 0.2

if padding_time * 2 > padding_silence_duration:
    padding_silence_duration = padding_time * 2

# コマンドラインの数を確認
if len(sys.argv) != 3:
    print("python3 audioSeg.py [inputfile] [outputfile]")
    sys.exit()

# 入力の絶対パスを指定
input_video_name = os.path.abspath(sys.argv[1])

# 出力のmovファイルを指定
dest_mov_name = os.path.abspath(sys.argv[2])

with tempfile.TemporaryDirectory() as dname1:
    tmp_wav_name = os.path.join(dname1, 'tmp.wav')
    tmp_mov_name = os.path.join(dname1, 'tmp.mov')
    
    # 動画から音声ファイルを分離
    print("Separating audio files from video...")
    command = "ffmpeg -i '"+input_video_name+"' -loglevel quiet -vn '"+tmp_wav_name+"'"
    sb.call(command, shell=True)
    
    seg = Segmenter(vad_engine='smn', detect_gender=False)
    # 区間検出実行
    print("Interval detection in progress...")
    segmentation = seg(tmp_wav_name)
    print("End of interval detection")
    
    """
    # 断片の数を確認
    if len(segmentation) == 1:
        print("The number of fragments is one.")
        sys.exit()
    """
    
    # 音声区間
    speech = np.array([row for row in segmentation if 'speech' in row])[:, 1:3].astype(np.float16)
    duration = speech[:, 1] - speech[:, 0]
    # 瞬間的に音がなっている箇所を除く
    bool_list = duration < min_keep_duration
    speech = np.delete(speech, bool_list, 0)
    duration = np.delete(duration, bool_list, 0)
    del bool_list
    
    # 区間ごとのラベル,開始時間,終了時間をcsv形式で保存
    #seg2csv(segmentation, 'segResult.csv')
    del segmentation

    # 分離した音声ファイルをwaveモジュールで読み込む
    with wave.open(tmp_wav_name) as wav:
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
            print("Sample width : ", samplewidth)
            sys.exit()
    
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
        