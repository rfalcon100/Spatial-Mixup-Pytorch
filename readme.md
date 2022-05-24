# Spatial mixup: Directional loudness modification as data augmentation for sound event localization and detection
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyTorch implementation and demo of:
Spatial mixup: Directional loudness modification as data augmentation for sound event localization and detection [[arxiv](https://arxiv.org/abs/2110.06126)]


## Summary 

This repo contains a PyTorch implementation of the Spatial Mixup [[arxiv](https://arxiv.org/abs/2110.06126)]] data 
augmentation technique for spatial audio.  Spatial mixup: Directional loudness modification as data augmentation for sound event localization and detection.
We include a Jupyter notebook that illustrates the process as well a small demo
that trains a small Sound Event Localization and Detection network, using the DCASE 2021 task 3 dataset.




## Reference

**Spatial mixup: Directional loudness modification as data augmentation for sound event localization and detection**, ICASSP 2022 [[arxiv](https://arxiv.org/abs/2110.06126)]]

-- Ricardo Falcon-Perez, Kazuki Shimada, Yuichiro Koyama, Shusuke Takahashi, Yuki Mitsufuji


**TL;DR**

- Spatial mixup modifies the directional loudness of the input signals.
- This is a soft spatial filter, than can be used as data augmentation for SELD tasks.
- The transform should not be too extreme, as the sound scene will be too different.


## Requirements
We use [[Spaudiopy](https://github.com/chris-hld/spaudiopy) ] to compute the spherical harmonics and other spatial audio operations.

```
conda create -n YOUR_ENV_NAME python=3.7
conda activate YOUR_ENV_NAME
pip install spaudiopy
```

Or you can create a conda environment from the provided file:
```
conda env create --file environment.yaml
```

## How to use it
Refer to the file  `spatial_mixup.py` which inlcudes the core processing.

Then the notebook `demo_spatial_mixup.ipnyb`shows how to use it, and a few exmaples.


## Citation
```
@misc{falconperez2021spatial,
      title={Spatial mixup: Directional loudness modification as data augmentation for sound event localization and detection}, 
      author={Ricardo Falcon-Perez and Kazuki Shimada and Yuichiro Koyama and Shusuke Takahashi and Yuki Mitsufuji},
      year={2021},
      eprint={2110.06126},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>





<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Thanks to [Chris Hold](https://github.com/chris-hld) for his spatial audio library and his comments and recomendations on how to use it.


