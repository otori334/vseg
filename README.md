# これは何

- 動画を短くするスクリプト
- 深層学習でノイズ・無音・音楽を検知，カットする
- [動画の無音部分を自動でカットする](https://nantekottai.com/2020/06/14/video-cut-silence/) と [CNNをベースとした音声区間分割 inaSpeechSegmenter を使ってみた](https://qiita.com/Nahuel/items/aba4eaabd686a1d89c37) を混ぜた
- 環境：macOS 11.5beta, Python 3.9.4, TensorFlow 2.5.0rc3, NumPy 1.19.2, inaSpeechSegmenter 0.6.7, FFmpeg 4.4

## 環境構築

- [pip3_list.txt](pip3_list.txt)
  - 現状使ってないパッケージも入っている
- 参考
  - [最近のUnicodeDecodeError](https://qiita.com/ousttrue/items/527a9c3045f710806aa9)
  - [AttributeError: module 'keras.utils.generic_utils' has no attribute 'populate_dict_with_module_objects'](https://qiita.com/hiro716/items/f3a1c9d926d345f514cf)
  - [Kerasが使えません](https://teratail.com/questions/341681?sip=n0070000_019)

## 今後

- 飛ばすところを速くする
- 環境音検知を自分でやりたい

## 蛇足

- 着想を得た [VSSR](https://nantekottai.com/2020/07/11/vssr/) にかなりお世話になった
  - しかし動作が不安定でよく落ち，トランスコードするとノイズが混じることがあった
  - フレームレートを落としてデータ量を減らす下ごしらえをしてもVSSRを通すとフレームレートが増える（フレームレートが固定されている）
  - 自分で改善出来ないしバッチ処理しづらかったので似たものを作った
- 動画の断片を見た感じかなり性能がいい
  - よくわからんが
- 全てが正常に動作する組み合わせを探し出すのに半日かかった
  - keras-nightlyを2.5.0.dev2021032900にしたのが効いたと思う
