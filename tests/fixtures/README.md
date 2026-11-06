# テスト用のデータ

- `sine-stereo.opus`: 1.5 秒・440Hz のステレオ Ogg Opus。bot のテストと同じもの。
  `ffmpeg -f lavfi -i "sine=frequency=440:duration=1.5" -ac 2 -c:a libopus -b:a 24k -frame_duration 20 -map_metadata -1 -fflags +bitexact -flags:a +bitexact sine-stereo.opus`
- `features-reference.json`: faster-whisper 1.2.1 の `FeatureExtractor(feature_size=128)` で求めた log-Mel 特徴量の参照値。
  入力は 0.5 秒の 16kHz 信号 `0.3 * sin(2π・440t) + 0.1 * sin(2π・3000t)`(float32)。
  `src/features.rs` の移植が faster-whisper と同じ値を出すかを確かめるのに使う。
- `vad-reference.json`: faster-whisper 1.2.1 の `vad.py` で求めた、VAD の参照値。`src/vad.rs` の移植が
  faster-whisper と同じ結果を出すかを確かめるのに使う。
  - `model_probs`: `SileroVADModel`(同梱の `silero_vad_v6.onnx` を onnxruntime 1.31.1 で実行)が出した、
    窓ごとの発話の確率。入力は 1.5 秒の 16kHz 信号
    `0.3 * sin(2π・220t) * (0.5 + 0.5 * sin(2π・4t)) + 0.05 * sin(2π・1800t)`(float64 で計算して float32 にしたもの)と、
    そのあとの 0.5 秒の無音。
  - `speech_timestamps`: 窓ごとの確率を決まった乱数で作り、`get_speech_timestamps` が返した発話の区間。
    確率はモデルの代わりに直接渡した(`get_vad_model` を差し替えた)。
  - `timestamp_map`: `SpeechTimestampsMap.get_original_time` が返した時刻。
