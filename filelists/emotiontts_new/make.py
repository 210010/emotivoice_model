import os
import numpy as np
import sys
workspacepath = "example"
sys.path.insert(0, workspacepath)
from text import _clean_text

baseemopath = "/data/emotiontts_new/04.Emotion"
basemainpath = "/data/emotiontts_new/01.Main"

testoutpath = os.path.join(workspacepath, "filelists/emotiontts_new/emotts_new_test.txt")
validoutpath = os.path.join(workspacepath, "filelists/emotiontts_new/emotts_new_valid.txt")
trainoutpath = os.path.join(workspacepath, "filelists/emotiontts_new/emotts_new_train.txt")

def writefile(fname, wavs, scripts):
    with open(fname, "w") as f:
        for wavpath, script in zip(wavs, scripts):
            f.write(wavpath+"|"+script+"\n")

def shuffle(wavs, scripts):
    shuffled_idx = [idx for idx in range(len(wavs))]
    np.random.shuffle(shuffled_idx)
    shuffled_wavs = np.take(wavs, shuffled_idx)
    shuffled_scripts = np.take(scripts, shuffled_idx)
    return shuffled_wavs, shuffled_scripts

def validcheck(wavs, scripts):
    final_wavs = []
    final_scripts = []
    for i in range(len(wavs)):
        try:
            _clean_text(scripts[i], ['korean_cleaners'])
            final_wavs.append(wavs[i])
            final_scripts.append(scripts[i])
        except Exception as e:
            print("Skipped no: {}, text: {}, Error: {}".format(i, scripts[i], e))
    return np.array(final_wavs), np.array(final_scripts)


def read_main_wavs_and_scripts(basepath):
    wavs = []
    scripts = []
    wav_basepath = os.path.join(basepath, "wav_22k")
    script_basepath = os.path.join(basepath, "script")

    wavs = [os.path.join(wav_basepath, wav) for wav in sorted(os.listdir(wav_basepath))]
    scripts = [open(os.path.join(script_basepath, wav),"r",encoding='utf-8-sig').read().strip() for wav in sorted(os.listdir(script_basepath)) if ".txt" in wav]

    wavs = np.array(wavs)
    scripts = np.array(scripts)
    print(wavs.shape, scripts.shape)
    return wavs, scripts

def read_emo_wavs_and_scripts(basepath):
    wavs = []
    scripts = []
    emodirs = os.listdir(basepath)
    for emodir in sorted(emodirs):
        wav_basepath = os.path.join(basepath, emodir, "wav_22k")
        script_basepath = os.path.join(basepath, emodir, "script")

        emo_wavs = [os.path.join(wav_basepath, wav) for wav in sorted(os.listdir(wav_basepath))]
        emo_scripts = [open(os.path.join(script_basepath, wav),"r",encoding='utf-8-sig').read().strip() for wav in sorted(os.listdir(script_basepath)) if ".txt" in wav]

        wavs.append(emo_wavs)
        scripts.append(emo_scripts)
    wavs = np.array(wavs)
    scripts = np.array(scripts)
    print(wavs.shape, scripts.shape)
    return wavs, scripts

def split_emo_wavs_and_scripts(wavs, scripts):
    # splits (train:valid:test = 8:1:1)
    testwavs = np.array([])
    testscripts = np.array([])
    validwavs = np.array([])
    validscripts = np.array([])
    trainwavs = np.array([])
    trainscripts = np.array([])
    for i, (emo_wavs, emo_scripts) in enumerate(zip(wavs, scripts)):
        testidx_from = i*40
        testidx_until = (i+1)*40
        testwavs = np.hstack((testwavs, wavs[i][testidx_from:testidx_until]))
        testscripts = np.hstack((testscripts, scripts[i][testidx_from:testidx_until]))

        valididx_from = (i+1)*40
        valididx_until = (i+2)*40
        if valididx_from>=400:
            valididx_from = 0
            valididx_until = 40
        validwavs = np.hstack((validwavs, wavs[i][valididx_from:valididx_until]))
        validscripts = np.hstack((validscripts, scripts[i][valididx_from:valididx_until]))

        train_selected = []
        for idx in range(400):
            if (testidx_from<=idx and idx<testidx_until) or (valididx_from<=idx and idx<valididx_until):
                continue
            train_selected.append(idx)
        trainwavs = np.hstack((trainwavs, np.take(wavs[i], train_selected)))
        trainscripts = np.hstack((trainscripts, np.take(scripts[i], train_selected)))

    return (trainwavs, trainscripts), (validwavs, validscripts), (testwavs, testscripts)

def split_main_wavs_and_scripts(wavs, scripts):
    # splits (train:valid:test = 8:1:1)
    wavs = np.array(wavs)
    scripts = np.array(scripts)

    testwavs = wavs[:1300]
    testscripts = scripts[:1300]

    validwavs = wavs[1300:2*1300]
    validscripts = scripts[1300:2*1300]

    trainwavs = wavs[2*1300:]
    trainscripts = scripts[2*1300:]

    return (trainwavs, trainscripts), (validwavs, validscripts), (testwavs, testscripts)


if __name__=="__main__":
    # main
    mainwavs, mainscripts = read_main_wavs_and_scripts(basemainpath)
    (maintrainwavs, maintrainscripts), (mainvalidwavs, mainvalidscripts), \
        (maintestwavs, maintestscripts) = split_main_wavs_and_scripts(mainwavs, mainscripts)

    # emotion
    emowavs, emoscripts = read_emo_wavs_and_scripts(baseemopath)
    (emotrainwavs, emotrainscripts), (emovalidwavs, emovalidscripts), \
        (emotestwavs, emotestscripts) = split_emo_wavs_and_scripts(emowavs, emoscripts)

    # merge
    trainwavs = np.hstack((maintrainwavs, emotrainwavs))
    trainscripts = np.hstack((maintrainscripts, emotrainscripts))
    validwavs = np.hstack((mainvalidwavs, emovalidwavs))
    validscripts = np.hstack((mainvalidscripts, emovalidscripts))
    testwavs = np.hstack((maintestwavs, emotestwavs))
    testscripts = np.hstack((maintestscripts, emotestscripts))

    # shuffle
    trainwavs, trainscripts = shuffle(trainwavs, trainscripts)
    validwavs, validscripts = shuffle(validwavs, validscripts)
    testwavs, testscripts = shuffle(testwavs, testscripts)

    # valid check
    trainwavs, trainscripts = validcheck(trainwavs, trainscripts)
    validwavs, validscripts = validcheck(validwavs, validscripts)
    testwavs, testscripts = validcheck(testwavs, testscripts)

    # write
    writefile(testoutpath, testwavs, testscripts)
    writefile(validoutpath, validwavs, validscripts)
    writefile(trainoutpath, trainwavs, trainscripts)