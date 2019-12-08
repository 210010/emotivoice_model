import argparse
from tornado.httpclient import HTTPClient
from tornado.escape import url_escape

if __name__=="__main__":
    import time
    start = time.time()

    # CLI setup
    default_text = "이것은 감정을 담아 말하는 음성합성기 입니다."
    default_ref = "/home/tts_team/ai_workspace/data/emotiontts_new/04.Emotion/ema/wav_22k/ema00350.wav"
    default_out = "output.wav"

    # arguments parser settup
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ref-wav', default="")
    group.add_argument('--predef-style', default="")
    parser.add_argument('--text', default="", required=True)
    parser.add_argument('--out', default=default_out)
    args = parser.parse_args()

    http_client = HTTPClient()
    ref_wav = url_escape(args.ref_wav)
    predef_style = url_escape(args.predef_style)
    text = url_escape(args.text)
    out = url_escape(args.out)
    url = "http://localhost:8888?ref-wav={}&predef-style={}&text={}&out={}".format(ref_wav, predef_style, text, out)
    response = http_client.fetch(url)

    print("time :", time.time() - start)
