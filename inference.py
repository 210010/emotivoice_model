import argparse
from tornado.httpclient import HTTPClient

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
    url = "http://localhost:8888?ref-wav={}&predef-style={}&text={}&out={}".format(args.ref_wav, args.predef_style, args.text, args.out)
    response = http_client.fetch(url)

    print("time :", time.time() - start)
