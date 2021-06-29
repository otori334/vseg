# Audio Segmentation Using inaSpeechSegmenter

# How To Use
# python3 aseg.py [inputfile] [outputfile]

from inaSpeechSegmenter import Segmenter
#from inaSpeechSegmenter import seg2csv
import os
import sys
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
input_wav_name = os.path.abspath(sys.argv[1])

# 出力のmovファイルを指定
dest_wav_name = os.path.abspath(sys.argv[2])

with tempfile.TemporaryDirectory() as dname1:
    seg = Segmenter(vad_engine='smn', detect_gender=False)
    # 区間検出実行
    print("Interval detection in progress...")
    segmentation = seg(input_wav_name)
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
    with wave.open(input_wav_name) as wav:
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
    