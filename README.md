# FLAME Video fitting
Contains code for fitting flame on videos. The code borrows heavily from [IMavatar](https://github.com/zhengyuf/IMavatar), "IMAvatar: Implicit Morphable Head Avatars from Videos" by Zheng et al.. The codebase depends on several 
packages for estimating facial landmarks, foreground masks, and flame parameters. More specifically:
* [DECA](https://github.com/yfeng95/DECA): "DECA: Detailed Expression Capture and Animation" by Yao et al.
* [face-alignment](https://pypi.org/project/face-alignment/): "FAN-Face: a Simple Orthogonal Improvement to Deep Face Recognition", by Bulat et al.
* [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting): "Robust high-resolution video matting with temporal guidance" by Lin et al.
* [face-detection-fdlite](https://github.com/patlevin/face-detection-tflite): Which is more or less a [mediapipe](https://github.com/google-ai-edge/mediapipe) feature subset.
* [FLAME](https://flame.is.tue.mpg.de/): "Learning a model of facial shape and expression from 4D scans" by Li et al.

So there are many dependencies, so brace yourselves.
Please check ``requirements.txt`` for the needed dependencies (I never ``pip install -r requirements.txt``, but instead try to manually install packages, by checking the document). 
You additionally are going to need to download the ``mobilenetv3`` checkpoint from [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting).

The entry point for optimizing one video is ``fit_video.py``. It takes one argument, ``--conf`` which is the path to a ``.yaml`` file that with the following structure:
```
input_video: /path/to/video/file
output_dir: /path/to/output/directory
rvm_chkp: /path/robust/video/matting/checkpoint
```