# テスト用のデータ

- `sine-stereo.opus`: 1.5 秒・440Hz のステレオ Ogg Opus。bot のテストと同じもの。
  `ffmpeg -f lavfi -i "sine=frequency=440:duration=1.5" -ac 2 -c:a libopus -b:a 24k -frame_duration 20 -map_metadata -1 -fflags +bitexact -flags:a +bitexact sine-stereo.opus`
- `features-reference.json`: faster-whisper 1.2.1 の `FeatureExtractor(feature_size=128)` で求めた log-Mel 特徴量の参照値。
  入力は 0.5 秒の 16kHz 信号 `0.3 * sin(2π・440t) + 0.1 * sin(2π・3000t)`(float32)。
  `src/features.rs` の移植が faster-whisper と同じ値を出すかを確かめるのに使う。
