#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import time
import json
import base64 as b64
import tkinter as tk
import tkinter.ttk as ttk
import openai
from winsound import PlaySound
from googletrans import Translator


### inspired by ChatWaifu ###
### https://github.com/cjyaddone/ChatWaifu

### modified codes from MoeGoe ###
### https://github.com/CjangCjengh/MoeGoe
import re
import sys
import logging
from scipy.io.wavfile import write
from torch import no_grad,LongTensor

### from MoeGoe py files
                
import utils
import commons
from models import SynthesizerTrn
from text import text_to_sequence,_clean_text

logging.getLogger('numba').setLevel(logging.WARNING)

def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text,hps.symbols,[])
    else:
        text_norm = text_to_sequence(text,hps.symbols,hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm,0)
    text_norm = LongTensor(text_norm)
    return text_norm

def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text

def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text

def generateSound(text):
    if '--escape' in sys.argv: escape = True
    else: escape = False

    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
    use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False
    emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2+1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        emotion_embedding=emotion_embedding,
        **hps_ms.model
    )
    _ = net_g_ms.eval()
    utils.load_checkpoint(model,net_g_ms)

    if n_symbols != 0:
        if not emotion_embedding:
            if True:
                choice = 't'
                text = text
                if text == '[ADVANCED]': text = ''

                length_scale,text = get_label_value(text,'LENGTH',1,'length scale')
                noise_scale,text = get_label_value(text,'NOISE',0.667,'noise scale')
                noise_scale_w,text = get_label_value(text,'NOISEW',0.8,'deviation of noise')
                cleaned,text = get_label(text,'CLEANED')

                stn_tst = get_text(text,hps_ms,cleaned=cleaned)

                speaker_id = spkrID
                out_path = 'output.wav'

                with no_grad():
                    x_tst = stn_tst.unsqueeze(0)
                    x_tst_lengths = LongTensor([stn_tst.size(0)])
                    sid = LongTensor([speaker_id])
                    audio = net_g_ms.infer(
                        x_tst,x_tst_lengths,sid=sid,noise_scale=noise_scale,noise_scale_w=noise_scale_w,length_scale=length_scale
                    )[0][0,0].data.cpu().float().numpy()
            
                write(out_path,hps_ms.data.sampling_rate,audio)

        else:
            import os
            import librosa
            import numpy as np
            import audonnx
            from torch import FloatTensor

            w2v2_folder = input('Path of a w2v2 dimensional emotion model: ')
            w2v2_model = audonnx.load(os.path.dirname(w2v2_folder))

            if True:
                choice = 't'
                text = text
                if text == '[ADVANCED]': text = ''

                length_scale,text = get_label_value(text,'LENGTH',1,'length scale')
                noise_scale,text = get_label_value(text,'NOISE',0.667,'noise scale')
                noise_scale_w,text = get_label_value(text,'NOISEW',0.8,'deviation of noise')
                cleaned,text = get_label(text,'CLEANED')

                stn_tst = get_text(text,hps_ms,cleaned=cleaned)

                speaker_id = spkrID

                emotion_reference = input('Path of an emotion reference: ')
                if emotion_reference.endswith('.npy'):
                    emotion = np.load(emotion_reference)
                    emotion = FloatTensor(emotion).unsqueeze(0)
                else:
                    audio16000,sampling_rate = librosa.load(emotion_reference,sr=16000,mono=True)
                    emotion = w2v2_model(audio16000,sampling_rate)['hidden_states']
                    emotion_reference = re.sub(r'\..*$', '', emotion_reference)
                    np.save(emotion_reference,emotion.squeeze(0))
                    emotion = FloatTensor(emotion)

                    out_path = 'output.wav'

                    with no_grad():
                        x_tst = stn_tst.unsqueeze(0)
                        x_tst_lengths = LongTensor([stn_tst.size(0)])
                        sid = LongTensor([speaker_id])
                        audio = net_g_ms.infer(
                            x_tst,x_tst_lengths,sid=sid,noise_scale=noise_scale,noise_scale_w=noise_scale_w,length_scale=length_scale,emotion_embedding=emotion
                            )[0][0, 0].data.cpu().float().numpy()
                
                write(out_path,hps_ms.data.sampling_rate,audio)

###


### build a GUI window ###
class GPTVoices():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('GPT Voices v0.4.0')
        scrW,scrH = self.root.winfo_screenwidth(),self.root.winfo_screenheight()
        winW,winH = int(scrW*(2/3)),int((scrH-150)*(3/4))
        self.root.geometry('%sx%s+50+25' % (winW,winH))  # set default window size following the screen size
        self.root.resizable(0,0)
        try:    self.root.option_add('*Font','나눔고딕 11')
        except: self.root.option_add('*Font','맑은고딕 11')
        self.root.option_add('*background','black')  # set default backgrounds of all widgets black
        self.root.option_add('*foreground','white')  # set default letter color of all widgets white

        ### menubar
        def newChat(): pass
                
        def saveAPIKey(): 
            self.apiWin = tk.Toplevel(self.root)
            self.apiWin.title('OpenAI API Key 저장하기')
            self.apiWin.geometry('300x100')
            self.apiWin.resizable(0,0)

            self.apiLabel = tk.Label(self.apiWin,text='OpenAI API key 입력 (입력 후 엔터):')
            self.apiEntry = tk.Entry(self.apiWin,width=3)
            self.apiEntry.bind('<Return>',saveKeyEvent)
            self.apiBttn = tk.Button(self.apiWin,text='입력',command=saveKey)
            self.apiLabel.pack(fill='x',padx=10,pady=2)
            self.apiEntry.pack(fill='x',padx=10,pady=2)
            self.apiBttn.pack(fill='x',padx=10,pady=2)

        def saveKey():
            key = self.apiEntry.get()
            with open('./data/key.txt','w',encoding='utf-8') as f:
                encKey = b64.b64encode(key.encode()).decode()
                f.write(encKey)
            
            openai.api_key = key
            global keyLoaded
            keyLoaded = True
            
            self.apiWin.quit()
            self.apiWin.destroy()

        def saveKeyEvent(event):
            self.saveKey()

        """
        def setVoice():
            self.voiWin = tk.Toplevel(self.root)
            self.voiWin.title('목소리 설정 저장하기')
            self.voiWin.geometry('300x%s' % (winH-100))
            self.voiWin.resizable(0,0)

            self.shwFrame = tk.Frame(self.voiWin,background='white')
            self.inpFrame = tk.Frame(self.voiWin,background='white')
            self.shwFrame.pack(side='top',fill='both',expand=1)
            self.inpFrame.pack(side='bottom',fill='x')

            self.shwBox = tk.Text(self.shwFrame,width=3,wrap='char',spacing1=3,spacing2=3,spacing3=3)
            self.shwBox.config(state='disabled')
            self.inpBox = tk.Entry(self.inpFrame,width=3)
            self.shwBox.pack(fill='both',expand=1,padx=1,pady=1)
            self.inpBox.pack(fill='x',padx=1,pady=1)

            aSpkr = self.loadSpeakers('./model/')
            bSpkr = self.loadSpeakers('./model/beta/')

            def smallUpdateText(self,text):
                self.shwBox.config(state='normal')
                self.shwBox.insert('end',text+'\n')
                self.shwBox.see('end')
                self.shwBox.config(state='disabled')

            global spkrNum
            spkrNum = 0
            for key in aSpkr['perModel'].keys():
                spkrList = []
                for i in aSpkr['perModel'][key].keys():
                    spkr = ('[%s]' % (spkrNum+i)).ljust(6) + (aSpkr['perModel'][key][spkrNum+i]).lust(8,'　')
                    spkrList.append(spkr)
                spkrList.append('--------'.ljust(22))
                smallUpdateText('')
                smallUpdateText('\n'.join(spkrList))
            smallUpdateText(('[99]').ljust(6)+'베타 (주의: 목소리 데이터 로딩 속도 느림)')
                
            def setSpeakerEvent(event):
                inpt = int(self.inpBox.get(0,'end'))
                
                self.inpBox.delete(0,'end')
                
                global speakerSelected

                if inpt == 99:
                    global listIndx
                    listIindx = 0

                    def smallShowBetaList(self,num):
                        smallUpdateText('')
                        spkrList = []
                        for key in list(bSpkr['perID'].keys())[listIndx*100:(listIndx+1)*100]:
                            spkr = ('[%s]' % key).ljust(8) + bSpkr['perID'][key]
                            spkrList.append(spkr)
                        smallUpdateText('\n'.join(spkrList))
                        if listIndx == 0:
                            smallUpdateText(('[9999]').ljust(8) + '다음 페이지')
                        elif (listIndx+1)*100 > len(bSpkr['perID'].keys()):
                            smallUpdateText(('[9998]').ljust(8) + '이전 페이지')
                        else:
                            smallUpdateText(('[9998]').ljust(8) + '이전 페이지')
                            smallUpdateText(('[9999]').ljust(8) + '다음 페이지')
                    
                    smallShowBetaList(self,listIndx)

                    def smallSelectBetaSpeakerEvent(event):
                        inpt = int(self.inptBox.get(0,'end'))
                        
                        global listIndx
                        if inpt == 9998:
                            listIndx -= 1
                            smallShowBetaList(self,listIndx)
                        elif inpt == 9999:
                            listIndx += 1
                            smallShowBetaList(self,listIndx)
                        else:
                            global indx
                            indx = inpt+100
                            smallUpdateText
            """

        def close(): 
            self.root.quit()
            self.root.destroy()

        self.menubar = tk.Menu(self.root)
        self.filemenu = tk.Menu(self.menubar,tearoff=0)
        self.filemenu.add_command(label='New chat',command=newChat)
        self.filemenu.add_command(label='Save API key',command=saveAPIKey)
        self.filemenu.add_command(label='Set voice',command=setVoice)
        self.filemenu.add_separator()
        self.filemenu.add_command(label='Close',command=close)
        self.menubar.add_cascade(label='File',menu=self.filemenu)

        self.root.config(menu=self.menubar)

        ### basic frames
        self.gptFrame = tk.Frame(self.root,background='white')
        self.usrFrame = tk.Frame(self.root,background='white')
        self.gptFrame.pack(side='top',fill='both',expand=1)
        self.usrFrame.pack(side='bottom',fill='x')
        
        ### text widget shows the input and results
        self.gptBox = tk.Text(self.gptFrame,width=3,wrap='char',spacing1=3,spacing2=3,spacing3=3)
        self.gptBox.config(state='disabled')  # lock Text widget from user
        self.gptBox.tag_configure('red',foreground='red')  # show 'red' tagged text in red
        self.gptBox.tag_configure('blue',foreground='dodgerblue') # show 'yellow' tagged text in yellow
        self.gptBox.tag_configure('yellow',foreground='yellow') # show 'yellow' tagged text in yellow
        self.gptBox.pack(fill='both',expand=1,padx=1,pady=1)

        ### text widget for user inputs
        self.usrBox = tk.Text(self.usrFrame,width=3,wrap='char',height=3,spacing1=1,spacing2=3,spacing3=1)
        self.usrBox.config(insertwidth=10,insertbackground='white')  # set cursor of the widget bold white
        self.usrBox.pack(fill='x',padx=1,pady=1)

    def updateText(self,text):
        self.gptBox.config(state='normal')
        self.gptBox.insert('end','\n%s' % text)
        self.gptBox.see('end')
        self.gptBox.config(state='disabled')
        self.root.update()
    
    def updateTextR(self,text):
        self.gptBox.config(state='normal')
        self.gptBox.insert('end',('\n%s' % text),'red')
        self.gptBox.see('end')
        self.gptBox.config(state='disabled')
        self.root.update()

    def updateTextB(self,text):
        self.gptBox.config(state='normal')
        self.gptBox.insert('end',('\n%s' % text),'blue')
        self.gptBox.see('end')
        self.gptBox.config(state='disabled')
        self.root.update()

    def updateTextY(self,text):
        self.gptBox.config(state='normal')
        self.gptBox.insert('end',('\n%s' % text),'yellow')
        self.gptBox.see('end')
        self.gptBox.config(state='disabled')
        self.root.update()

    def checkKey(self):
        global keyLoaded
        keyLoaded = False
        self.updateText('   *** OpenAI API Key 확인 중...')
        time.sleep(0.1)
        try:
            with open('./data/key.txt','r',encoding='utf-8') as f:
                encd = f.readlines()[0]
                apiKey = b64.b64decode(encd).decode()
                openai.api_key = apiKey
                keyLoaded = True
                self.updateText('')
                self.updateText('   *** OpenAI API Key 확인 완료.')
                time.sleep(0.1)
                self.updateText('')
                self.updateText('   *** ChatGPT 준비 중...')
                time.sleep(0.1)
        except:
            self.inputKey()

    def inputKey(self):
        self.updateText('')
        self.updateText('   *** OpenAI API Key 입력 (입력 후 엔터):')
        self.updateText('       (File > Save API Key 메뉴를 통해 Key 저장 가능)')

        def inputKeyEvent(event):
            input = self.usrBox.get('1.0','end-1c')
            if input[0]=='\n': apiKey = input[1:]
            else: apiKey = input

            self.usrBox.delete('1.0','end')
            self.usrBox.see('1.0')

            openai.api_key = apiKey
            global keyLoaded
            keyLoaded = True

            self.updateText('')
            self.updateText('   *** ChatGPT 준비 중...')
            time.sleep(0.1)
        
        self.usrBox.bind('<Return>',inputKeyEvent)


    ### load list of speakers from TTS models (by CjangCjengh) ###
    ### https://github.com/CjangCjengh/TTSModels
    def loadSpeakers(self,path):
        fList = os.listdir(path)
        mList = []
        for file in fList:
            if file.endswith('.json'): mList.append(file)
        
        spkrPModl = {}
        spkrIDs = {}
        indx = 0
        for modl in mList:
            with open(os.path.join(path,modl),'r') as m:
                jsonData = json.load(m)
                spkrs = {}
                for spkr in jsonData['speakers']:
                    spkrs[indx] = spkr
                    spkrIDs[indx] = spkr
                    indx += 1
                spkrPModl[modl[:-5].split('_')[1]] = spkrs
        return {'perModel':spkrPModl,'perID':spkrIDs}
        
    def selectSpeaker(self):
        global speakerSelected
        speakerSelected = False

        aSpkr = self.loadSpeakers('./model/')
        bSpkr = self.loadSpeakers('./model/beta/')

        self.updateText('\n목소리 선택 (ID 입력 후 엔터):')

        ### show list of speakers
        global spkrNum
        spkrNum = 0
        n = 4
        for key in aSpkr['perModel'].keys():
            for i in range(len(aSpkr['perModel'][key].keys())//n+1):
                spkrList = []
                if i < len(aSpkr['perModel'][key].keys())//n:
                    spkrLine = []
                    for j in range(n):
                        spkrElem = ('[%s]' % (spkrNum+n*i+j)).ljust(6) + (aSpkr['perModel'][key][spkrNum+n*i+j]).ljust(8,'　')
                        spkrLine.append(spkrElem)
                    spkrList.append(''.join(spkrLine))
                elif i == len(aSpkr['perModel'][key].keys())//n:
                    spkrLine = []
                    for j in range(n):
                        try:
                            spkrElem = ('[%s]' % (spkrNum+n*i+j)).ljust(6) + (aSpkr['perModel'][key][spkrNum+n*i+j]).ljust(8,'　')
                            spkrLine.append(spkrElem)
                        except: pass
                    spkrList.append(''.join(spkrLine))
                    spkrList.append('--------'.center(88))
                self.updateText('\n'.join(spkrList))
            spkrNum += len(aSpkr['perModel'][key].keys())
        self.updateText(('[99]'.ljust(6))+'베타 (주의: 목소리 데이터 로딩 속도 느림)')

        def selectSpeakerEvent(event):
            inpt = self.usrBox.get('1.0','end-1c')

            if inpt[0] == '\n': inpt = int(inpt[1:])
            else: inpt = int(inpt)

            self.usrBox.delete('1.0','end')
            self.usrBox.see('1.0')

            global speakerSelected
            
            if inpt == 99:
                speakerSelected = False
                global listIndx
                listIndx = 0

                def showBetaList(self,num):
                    self.updateText('')
                    spkrList = []
                    for key in list(bSpkr['perID'].keys())[listIndx*100:(listIndx+1)*100]:
                        spkrLine = ('[%s]' % key).ljust(8) + bSpkr['perID'][key]
                        spkrList.append(spkrLine)
                    self.updateText('\n'.join(spkrList))
                    if listIndx == 0: 
                        self.updateText(('[9999]').ljust(8) + '다음 페이지')
                    elif (listIndx+1)*100 > len(bSpkr['perID'].keys()):
                        self.updateText(('[9998]').ljust(8) + '이전 페이지')
                    else:
                        self.updateText(('[9998]').ljust(8) + '이전 페이지')
                        self.updateText(('[9999]').ljust(8) + '다음 페이지')
                
                showBetaList(self,listIndx)
                
                def selectBetaSpeakerEvent(event):
                    inpt = self.usrBox.get('1.0','end-1c')
                    if inpt[0] == '\n': inpt = int(inpt[1:])
                    else: inpt = int(inpt)

                    global listIndx
                    if inpt == 9998:
                        listIndx -= 1
                        showBetaList(self,listIndx)
                    elif inpt == 9999:
                        listIndx += 1
                        showBetaList(self,listIndx)
                    else:
                        global indx
                        indx = inpt+100
                        self.updateText('')
                        self.updateText(('[%s]' % (inpt)).ljust(8) + ('%s 선택' % bSpkr['perID'][inpt]))

                        global speakerSelected
                        speakerSelected = True

                    self.usrBox.delete('1.0','end')
                    self.usrBox.see('1.0')

                self.usrBox.bind('<Return>',selectBetaSpeakerEvent)

            elif inpt <= spkrNum:
                global indx
                indx = inpt
                
                self.updateText('')
                self.updateText(('[%s]' % inpt).ljust(6) + ('%s 선택' % aSpkr['perID'][inpt]))
                                
                speakerSelected = True
            
            else:
                self.updateText('')
                self.updateTextR('유효하지 않은 ID')

        self.usrBox.bind('<Return>',selectSpeakerEvent)

    def setSpeakerDir(self,indx):
        num = 0
        aSpkr = self.loadSpeakers('./model/')
        bSpkr = self.loadSpeakers('./model/beta/')
        
        global model
        global config
        global spkrID
        if indx <= spkrNum:
            for key in aSpkr['perModel'].keys():
                if (indx >= num) and (indx < num+len(aSpkr['perModel'][key].keys())):
                    model = './model/model_%s.pth' % key
                    config = './model/config_%s.json' % key
                    spkrID = indx - num
                    break
                else: num += len(aSpkr['perModel'][key].keys())
        elif indx > spkrNum:
            model = './model/beta/model_beta.pth'
            config = './model/beta/config_beta.json'
            spkrID = indx - 100

    def inputPrompt(self):
        global chatLog
        global tokn
        chatLog = [{'role':'system','content':'You are a very helpful assistant talking in Japanese.'}]
        tokn = 0

        self.updateText('')
        self.updateTextY('ChatGPT:')
        self.updateText('何を手伝ましょうか。\n무엇을 도와드릴까요?\n')

        if indx <= spkrNum:
            generateSound('何を手伝ましょうか。')
            PlaySound('./output.wav',flags=0)
        elif indx > spkrNum:
            generateSound('[JA]何を手伝ましょうか。[JA]')
            PlaySound('./output.wav',flags=0)

        def inputPromptEvent(event):
            inpt = self.usrBox.get('1.0','end-1c')
            if inpt[0] == '\n': prom = inpt[1:]
            else: prom = inpt

            self.usrBox.delete('1.0','end')
            self.updateText('')
            self.updateTextB('사용자:')
            self.updateText(prom)

            def generateResponse(self,prompt,chatLog=None,token=0):
                msgs = []
                if token > 4096*0.9:
                    for log in chatLog[1:]: msgs.append(log)
                else:
                    for log in chatLog: msgs.append(log)
                msgs.append({'role':'user','content':prompt+' 日本語で答える'})
                resp = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',messages=msgs,max_tokens=1024
                )

                return resp
            
            global tokn
            resp = generateResponse(self,prom,chatLog,tokn)
            role = resp.choices[0]['message']['role']
            ansr = resp.choices[0]['message']['content']
            trns = Translator().translate(ansr,dest='ko').text
            self.updateText('')
            self.updateTextY('ChatGPT:')
            self.updateText(ansr+'\n'+trns)

            chatLog.append({'role':'user','content':prom})
            chatLog.append({'role':role,'content':ansr})
            tokn += resp.usage['prompt_tokens']

            if indx <= spkrNum:
                generateSound(ansr)
                PlaySound('./output.wav',flags=0)
            elif indx > spkrNum:
                generateSound('[JA]%s[JA]' % ansr)
                PlaySound('./output.wav',flags=0)

        self.usrBox.bind('<Return>',inputPromptEvent)


if __name__=='__main__':
    win = GPTVoices()

    keyLoaded = False
    speakerSelected = False

    win.checkKey()
    while True:
        if keyLoaded:
            win.selectSpeaker()
            break
        else:
            win.root.update()
    while True:
        if speakerSelected:
            win.setSpeakerDir(indx)
            win.inputPrompt()
            break
        else: win.root.update()
    
    win.root.mainloop()

