# Progressive COOL-CHIC
This repository contains the code to reproduce the paper **Progressive COOL-CHIC: Efficient Decoding for Dual-Resolution Images**

## Abstract
In this work, we propose Progressive Cool-Chic (PCC), a scalable overfitted neural image codec that can decode an image at two different resolutions from a single bitstream. Experiments show that our method reduces the necessary bitrate to encode two representations of the same image by up to 31.54\% in terms of BD-rate compared to encoding both representations independently using Cool-Chic while also decreasing the necessary decoding time. The bitstream is structured in a way that the low-resolution image can already be decoded, when only a part of the bitstream has been received.

## COOL-CHIC
Progressive COOL-CHIC is an extension of COOL-CHIC, an overfitted image and video codec by Orange. More details are available on the [COOL-CHIC Github page](https://orange-opensource.github.io/Cool-Chic/).

## Setup
```bash
# Clone this repository
git clone https://github.com/mbenjak/Progressive-CC.git
cd Progressive-CC

# Install Progressive COOL-CHIC
CXX=g++ pip install -e .
```

## Usage
Unlike COOL-CHIC, Progressive COOL-CHIC can only encode images, not video. To encode an image run:
```bash
python coolchic/encode.py -i={input_image}.png -o={bitstream}.cool --workdir={work_dir} --enc_cfg=cfg/enc/intra/slow_100k.cfg --dec_cfg_residue=cfg/dec/intra_residue/hop.cfg --lmbda=0.001 --separate_arm
```

To decode the image in high resolution run:
```bash
python coolchic/decode.py -i={bitstream}.cool  -o={decoded_image}.ppm --verbosity=1
```

To decode the image in low resolution run:
```bash
python coolchic/decode.py -i={bitstream}.cool  -o={decoded_image}.ppm --verbosity=1 --decode_lowres
```

## Citing this work
If you use this code in your work, we ask you to please cite our work:

```latex
@inproceedings{progressive_cool_chic,
  title={Progressive COOL-CHIC: Efficient Decoding for Dual-Resolution Images},
  author={Benjak, Martin and Chen, Yi-Hsin and Peng Wen-Hsiao and Ostermann, Jörn},
  booktitle={IEEE Visual Communications and Image Processing (VCIP)},
  year={2025}
}
```
