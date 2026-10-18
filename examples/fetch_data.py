"""Download two small public datasets and write 300-row labelled samples (seed 0) to examples/data/.

AG News (4-class topics), the SMS Spam Collection (binary) and SST-2 movie-review sentiment (binary). Nothing is redistributed in this repo:
run this script to fetch them from their public mirrors.
"""
import io
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "data"
N = 300


def _get(url: str) -> bytes:
    return urllib.request.urlopen(url, timeout=60).read()


def agnews() -> pd.DataFrame:
    raw = _get("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv")
    d = pd.read_csv(io.BytesIO(raw), header=None, names=["label", "title", "desc"])
    names = {1: "world", 2: "sports", 3: "business", 4: "sci_tech"}
    return pd.DataFrame({"topic": d["label"].map(names), "text": d["title"].str.strip() + ". " + d["desc"].str.strip()})


def spam() -> pd.DataFrame:
    raw = _get("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv")
    d = pd.read_csv(io.BytesIO(raw), sep="\t", header=None, names=["label", "text"])
    return d


def sentiment() -> pd.DataFrame:
    raw = _get("https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/test.tsv")
    d = pd.read_csv(io.BytesIO(raw), sep="	", header=None, names=["text", "label"])
    return pd.DataFrame({"label": d["label"].map({0: "negative", 1: "positive"}), "text": d["text"].str.strip()})


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, df in (("agnews", agnews()), ("spam", spam()), ("sentiment", sentiment())):
        sample = df.sample(n=N, random_state=0).reset_index(drop=True)
        sample.insert(0, "id", [f"{name[:2]}{i}" for i in range(len(sample))])
        sample.to_csv(OUT / f"{name}.csv", index=False)
        print(name, len(sample), sample.iloc[:, 1].value_counts().to_dict())
