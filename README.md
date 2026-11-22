# 上海 画面認識ヘルパー

[Windows インストーラーをダウンロード](https://github.com/Sakaban0/MahjongHelper/raw/refs/heads/main/downloads/MahjongHelper-Setup.exe) · [ZIP版をダウンロード](https://github.com/Sakaban0/MahjongHelper/raw/refs/heads/main/downloads/MahjongHelper-Windows.zip)

EXE版は **MahjongHelper-Setup.exe** を実行し、スタートメニューの **Mahjong Helper** から起動します。アンインストールは Windows の「インストールされているアプリ」から行えます。個人設定、デバッグ画像、初回に作成される仮想環境はインストール先に残ります。ZIP版はフォルダごと展開して **Start-Desktop.vbs** をダブルクリックします。通常の起動にはどちらもコンソールを表示しません。初めて使うPCには Windows 用 Python 3.12（64ビット）と Python ランチャーを用意してください。初回起動時に必要なライブラリをインストールします。

| 場所 | 内容 |
| --- | --- |
| app | アプリ本体、操作ガイド、サンプル、認識素材 |
| debug | 保存した盤面画像と認識結果。日時別に並びます |
| json | 使用中の設定と登録した牌の見本。設定を引き継ぐときはこのフォルダごとコピーします |

[操作ガイド](app/docs/GUIDE.md) · [検証記録](app/docs/VALIDATION.md)

登録済みの牌見本161件を同梱しています。