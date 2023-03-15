# GPT-Voices

inspired by ChatWaifu
https://github.com/cjyaddone/ChatWaifu

modified codes from MoeGoe
https://github.com/CjangCjengh/MoeGoe

used pre-trained models speakers from TTS models (by CjangCjengh)
https://github.com/CjangCjengh/TTSModels


1. install anaconda.
2. change directory to where the project is saved.
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
