#!/bin/zsh
# Fix torchcodec audio/video decoding on Apple Silicon macOS.
# Root cause: (1) Homebrew ffmpeg is v8 (libavcodec 62) but torchcodec only
# ships cores for ffmpeg 4-7; (2) the torchcodec core dylibs reference
# @rpath/libav* yet carry NO LC_RPATH, and macOS SIP strips DYLD_LIBRARY_PATH
# for the (Xcode) python. Fix: install ffmpeg@7 and bake rpaths (ffmpeg@7 +
# torch/lib + @loader_path) into the torchcodec *7.* libs, then ad-hoc re-sign.
set -e
HOMEBREW_NO_AUTO_UPDATE=1 brew list ffmpeg@7 >/dev/null 2>&1 || brew install ffmpeg@7
FF=/opt/homebrew/opt/ffmpeg@7/lib
TC=$(python3 -c "import torchcodec, os; print(os.path.dirname(torchcodec.__file__))")
TORCHLIB=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
cd "$TC"
for f in libtorchcodec_*7.*; do
  for rp in "$FF" "$TORCHLIB" "@loader_path"; do
    install_name_tool -add_rpath "$rp" "$f" 2>/dev/null || true
  done
  codesign --force --sign - "$f" 2>/dev/null || true
  echo "patched $f"
done
python3 -c "import torchcodec; from torchcodec.decoders import AudioDecoder, VideoDecoder; print('torchcodec OK', torchcodec.__version__)"
