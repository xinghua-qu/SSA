# Speech Synthesising based Attack for Automatic Speech Recognition
- Our paper has been accepted by KDD-2022.
- Our web-page containing generated audio demos is at [SSA_Demo_page](https://sites.google.com/view/ssa-asr/home).
- Our paper has been higlighted/reported by several Chinese social media, such as [PaperWeekly]( https://mp.weixin.qq.com/s/qzesgFUIar3DXun0nkuq7Q), 语音之家，深科技，火山引擎.

## Abstract
Adversarial examples in automatic speech recognition (ASR) are naturally sounded by humans \textit{yet} capable of fooling well trained ASR models to transcribe incorrectly. Existing audio adversarial examples are typically constructed by adding constrained perturbations on benign audio inputs. Such attacks are therefore generated with an audio dependent assumption. For the first time, we propose the Speech Synthesising based Attack (SSA), a novel threat model that constructs audio adversarial examples entirely from scratch, i.e., without depending on any existing audio to fool cutting-edge ASR models. To this end, we introduce a conditional variational auto-encoder (CVAE) as the speech synthesiser. Meanwhile, an adaptive sign gradient descent algorithm is proposed to solve the adversarial audio synthesis task. Experiments on three datasets (i.e., Audio Mnist, Common Voice, and Librispeech) show that our method could synthesise naturally sounded audio adversarial examples to mislead the start-of-the-art ASR models. 

## Build the environment
```bash env_build.sh```
## Check the model is loaded properly
```python3 cvae_attack_model_prepare.py```
## Run the speech synthesis attack with the command below.
```python3 cvae_attack_mnist.py```
