# GPT-Voices

inspired by ChatWaifu
https://github.com/cjyaddone/ChatWaifu

modified codes from MoeGoe
https://github.com/CjangCjengh/MoeGoe

used pre-trained models speakers from TTS models (by CjangCjengh)
https://github.com/CjangCjengh/TTSModels


1. install anaconda.
2. change directory to where the project is saved. example:
  `cd E:\Projects\GPTVoices\'
3. create a virtual environment using anaconda.
  `conda create --name GPTVoices python=3.10`
4. activate vitual environment.
  `conda activate GPTVoices`
5. install required libraries with pip.
  `pip install -r requirements.txt`
6. download pre-trained models files into /model folder.
  link: https://drive.google.com/drive/folders/1AGAx5bjKz9wwaWDCOrhg8YgoHnXpC1aU?usp=share_link
7. run the code.
  `python GPTVoices_v0.4.1.py`
  You need **OpenAI API key** to run the code.


# Differences from existing codes:
- can ask ChatGPT in Korean, and ChatGPT will answer in Japanese, which will be translated automatically into Korean. (uses library `googletrans` to translate the answer.)
- no Selenium module is used. the program is only window on your desktop.


# 한국어
1. anaconda 설치.
2. anaconda prompt 실행 후 저장된 폴더로 이동. 예시:
  `cd E:\Projects\GPTVoices\'
3. anaconda 가상 환경 생성.
  `conda create --name GPTVoices python=3.10`
4. 가상 환경 구동.
  `conda activate GPTVoices`
5. pip 명령어 이용해 필요 라이브러리 설치.
  `pip install -r requirements.txt`
6. /model 폴더에 pre-trained 모델 다운로드. (덮어쓰기 가능)
  link: https://drive.google.com/drive/folders/1AGAx5bjKz9wwaWDCOrhg8YgoHnXpC1aU?usp=share_link
7. 코드 실행.
  `python GPTVoices_v0.4.1.py`
  정상적으로 작동시키기 위해서 **OpenAI API key** 필요.
