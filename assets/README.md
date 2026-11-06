# 同梱しているモデル

## silero_vad_v6.onnx

無音を取り除くための VAD(音声区間検出)のモデル。`src/vad.rs` がバイナリに埋め込んで使う。

- 入手元: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 1.2.1 の `faster_whisper/assets/silero_vad_v6.onnx`(PyPI の wheel)
- SHA-256: `4cbf549b8326f60f80f2536d9eefeb450a9abe83365a098031c89719f1be17d2`
- 元のモデル: [Silero VAD](https://github.com/snakers4/silero-vad) v6。faster-whisper が、窓をまとめて 1 回で推論できる形(入力 `[窓の数, 576]`)に書き出したもの。
- ライセンス: どちらも MIT License。著作権表示と許諾の表示は次のとおり。

```text
Copyright (c) 2020-present Silero Team
Copyright (c) 2023 SYSTRAN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
