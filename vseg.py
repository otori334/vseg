# Video Segmentation Using inaSpeechSegmenter
# Original: https://gist.github.com/tam17aki/11eb1566a2d48b382607d23dddb98891
        # https://qiita.com/Nahuel/items/aba4eaabd686a1d89c37
        # https://nantekottai.com/2020/06/14/video-cut-silence/
        # https://tam5917.hatenablog.com/entry/2020/01/25/132113 # バグがあるけど参考になる

from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter import seg2csv
import os
import sys
import shutil
import subprocess as sb
import soundfile as sf
from tqdm import tqdm
import tempfile
import numpy as np
# import csv
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
parser.add_argument('-f', '--ffmpeg', default="", help='[Arguments to add to ffmpeg]')
parser.add_argument('-w', '--which_ffmpeg', default="", help='[which ffmpeg]')
args = parser.parse_args()
input_video_name = os.path.abspath(args.arg1)
dest_mov_name = os.path.abspath(args.arg2)

if args.csv is None or not os.path.isfile(args.csv):
    dest_csv_name = os.path.join(os.path.dirname(os.path.abspath(args.arg2)), 'segResult.csv')
else:
    dest_csv_name = os.path.abspath(args.csv)
# 入力の絶対パスを解析
if not os.path.isfile(input_video_name):
    print("The first argument should be an existing file.")
    sys.exit()
if os.path.isfile(dest_mov_name):
    print("Output file already exists.")
    sys.exit()

if os.path.isfile(args.which_ffmpeg):
    ffmpeg = args.which_ffmpeg
elif os.path.isfile('/usr/local/bin/ffmpeg'):
    ffmpeg = '/usr/local/bin/ffmpeg'
elif os.path.isfile('/usr/bin/ffmpeg'):
    ffmpeg = '/usr/bin/ffmpeg'
elif shutil.which('ffmpeg') is None:
    raise (Exception("""ffmpeg program not found"""))
else:
    ffmpeg = 'ffmpeg'

with tempfile.TemporaryDirectory() as dname1:
    print(dname1)
    tmp_wav_name = os.path.join(dname1, 'tmp.wav')
    tmp_mov_name = os.path.join(dname1, 'tmp.mov')

    if os.path.isfile(dest_csv_name):
        print(f'Found {os.path.basename(dest_csv_name)}')
    else:
        # 動画から音声ファイルを標準 PCM WAV 形式で分離
        print("Separating audio files from video...", file=sys.stderr)
        command = (
            f"{ffmpeg} -y -i '{input_video_name}' -ar 16000 -ac 1 -acodec pcm_s16le "
            f"-loglevel quiet -vn '{tmp_wav_name}'"
        )
        sb.call(command, shell=True)

        # 区間検出実行
        seg = Segmenter(vad_engine='smn', detect_gender=False, ffmpeg=ffmpeg)
        print("Interval detection in progress...", file=sys.stderr)
        segmentation = seg(tmp_wav_name)
        print("End of interval detection", file=sys.stderr)
        seg2csv(segmentation, dest_csv_name)  # 区間ごとのラベル,開始時間,終了時間をcsv形式で保存
        del segmentation

    label = np.loadtxt(dest_csv_name, delimiter='\t', dtype=str, skiprows=1, usecols=[0])
    segs = np.loadtxt(dest_csv_name, delimiter='\t', dtype=np.float32, skiprows=1, usecols=[1, 2])  # float16 では小数部分が切り捨てられてしまう

    # 断片の数を確認
    if len(label[:]) == 1:
        print("The number of fragments is one.")
        sys.exit()

    if args.insert is None:
        insert_wav_name = tmp_wav_name
    elif not os.path.isfile(args.insert):
        print("The specified insert wav file does not exist.")
        insert_wav_name = tmp_wav_name
    else:
        insert_wav_name = os.path.abspath(args.insert)

    if not os.path.isfile(insert_wav_name):
        # 動画から音声ファイルを標準 PCM WAV 形式で分離
        print("Separating audio files from video...", file=sys.stderr)
        command = (
            f"{ffmpeg} -y -i '{input_video_name}' -ar 16000 -ac 1 -acodec pcm_s16le "
            f"-loglevel quiet -vn '{insert_wav_name}'"
        )
        sb.call(command, shell=True)

    # （分離した）音声ファイルを soundfile モジュールで読み込む
    with sf.SoundFile(insert_wav_name, 'r') as wav:
        nchannels = wav.channels
        framerate = wav.samplerate
        nframes = wav.frames
        framerate_nchannels = framerate * nchannels
        len_data = nframes * nchannels

        # サンプル幅に応じてデータ型を確認
        if wav.subtype == 'PCM_16':
            dtype = 'int16'
        elif wav.subtype == 'PCM_32':
            dtype = 'int32'
        else:
            print(f'vseg: Sample width subtype: {wav.subtype} is not supported.')
            sys.exit()

        data = wav.read(dtype=dtype).copy()
        if nchannels > 1:
            data = data.flatten()
        else:
            data = data.copy()

    # セグメントの最終終了時間と音声ファイルの長さを比較
    last_segment_end = segs[-1, -1]
    audio_length = len_data / framerate_nchannels
    if abs(audio_length - last_segment_end) > 0.035:
        if insert_wav_name == tmp_wav_name:
            print(f'vseg: This csv file \"{os.path.basename(dest_csv_name)}\" is incompatible.')
        else:
            difference = abs(audio_length - last_segment_end)
            print(f'vseg: There is a {difference} second difference between the extracted information and the insert wav file \"{insert_wav_name}\"')
        sys.exit()

    # speech noEnergy noise music
    speech = segs[label == 'speech']

    # 瞬間的に音声がなっている箇所を除く
    duration = speech[:, 1] - speech[:, 0]
    bool_list = duration < min_keep_duration
    speech = np.delete(speech, bool_list, 0)
    duration = np.delete(duration, bool_list, 0)
    del bool_list

    # speechではない長い区間を検出し，無音データで埋める
    print("Detecting a long section that is not a speech...", file=sys.stderr)
    for i in tqdm(range(len(speech) - 1)):
        if (speech[i + 1][0] - speech[i][1] > padding_silence_duration):
            # 無音データで埋める
            start_fill = round(speech[i][1] * framerate_nchannels)
            end_fill = round(speech[i + 1][0] * framerate_nchannels)
            data[start_fill:end_fill] = 0

            # 余白を挿入する
            speech[i][1] += padding_time
            speech[i + 1][0] -= padding_time

    # 加工した音声データを書き出す（PCM 16-bit または PCM 32-bit WAV）
    subtype_out = 'PCM_16' if dtype == 'int16' else 'PCM_32'
    with sf.SoundFile(tmp_wav_name, 'w', samplerate=framerate, channels=nchannels, subtype=subtype_out) as wav_out:
        if nchannels > 1:
            # Reshape data to (nframes, nchannels)
            data_to_write = data.reshape(-1, nchannels)
        else:
            data_to_write = data.copy()
        wav_out.write(data_to_write)

    # 動画の音声を差し替える
    print("Replacing audio...", file=sys.stderr)
    command = (
        f"{ffmpeg} -i '{input_video_name}' -loglevel error -hide_banner -stats "
        f"-i '{tmp_wav_name}' -c:v copy -c:a aac -strict experimental "
        f"-map 0:v -map 1:a '{tmp_mov_name}'"
    )
    sb.call(command, shell=True)
    print("Replacement completed.", file=sys.stderr)

    # 動画を切り出す
    speech = speech.astype('str')
    duration = duration.astype('str')
    num_speech = len(speech)  # 断片の個数
    num_digits = len(str(num_speech))  # 断片の個数の桁数
    with tempfile.TemporaryDirectory() as dname2:
        print("Cutting out the video...", file=sys.stderr)
        for i in tqdm(range(num_speech)):
            # FFmpeg コマンドをリスト形式で定義
            command = (
                f"{ffmpeg} -ss {speech[i][0]} -i '{tmp_mov_name}' {args.ffmpeg} "
                f"-loglevel quiet -t {duration[i]} '{dname2}/{str(i).zfill(num_digits)}.mov'"
            )
            sb.call(command, shell=True)

        # 動画を繋げる
        print("Merging videos...", file=sys.stderr)
        target_list = os.path.join(dname1, 'target_list.txt')
        command = (
            f"(for f in \"{dname2}\"/*.mov; do echo file \\'$f\\'; done)>'{target_list}'; "
            f"{ffmpeg} -loglevel error -hide_banner -stats -safe 0 -f concat -i '{target_list}' "
            f"'{dest_mov_name}'"
        )
        sb.call(command, shell=True)
