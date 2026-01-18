import pandas as pd
from typing import Tuple, List, Dict, Any
import io
import os

IN_CSV=r"CS404-Artificial-Intelligence-and-Language-Processing\Assignment2\games_sample.csv"
OUT_CSV=r"CS404-Artificial-Intelligence-and-Language-Processing\Assignment2\games_sample_v2.csv"

##The note and tags keyword
VIOLENCE_KEYS=["blood", "gore", "violence", "violent", "horror"]
ADULT_KEYS=["adult", "nudity", "sexual"]
ROMANCE_KEYS  = ["romance", "dating", "love"]

FALLBACKS = ["gb18030", "gbk", "cp1252", "latin1"]
last = None

##try to use utf8, if not, fallback to other encoder,and then fallback to utf8

def read_csv_mixed_utf8(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        raw = f.read()

    repaired_text = repair_bytes_mixed_utf8(raw)

    return pd.read_csv(io.StringIO(repaired_text))


def decode_best(chunk: bytes) -> str | None:
    for enc in FALLBACKS:
        try:
            return chunk.decode(enc, errors="strict")
        except UnicodeDecodeError:
            continue
    return None

def repair_bytes_mixed_utf8(data: bytes) -> str:

    out_parts: list[str] = []
    i = 0
    n = len(data)

    while i < n:
        try:

            s = data[i:].decode("utf-8", errors="strict")
            out_parts.append(s)
            break
        except UnicodeDecodeError as e:
            bad_start = i + e.start

            if bad_start > i:
                out_parts.append(data[i:bad_start].decode("utf-8", errors="strict"))

            repaired = None
            repaired_len = 0

            for L in range(1, 9):
                if bad_start + L > n:
                    break
                chunk = data[bad_start:bad_start + L]
                candidate = decode_best(chunk)
                if candidate is not None:
                    repaired = candidate
                    repaired_len = L

            if repaired is None:

                repaired = data[bad_start:bad_start+1].decode("latin1")
                repaired_len = 1

            out_parts.append(repaired)
            i = bad_start + repaired_len

    return "".join(out_parts)

##Normalize date
def normalize_date(x):
    '''
    x:csv time input
    output:mysql time format
    '''
    if pd.isna(x) or str(x).strip()=="":
        return pd.NA
    dt = pd.to_datetime(str(x).strip(), errors="coerce")
    if pd.isna(dt):
        return pd.NA
    return dt.strftime("%Y-%m-%d")

##clean estimated owners
def clean_estimated_owners(x):
    '''
    the number of people who play the game is not 0-0
    '''
    if pd.isna(x):
        return pd.NA
    s=str(x).strip()
    if not s:
        return pd.NA
    if s.replace(" ", "") in ("0-0", "0–0", "0—0"):
        return pd.NA
    return s

##unifiy the bool format
def clean_bool(x):
    '''
    clean the attribute value is TRUE or FALSE, such as windows ,linux,mac
    '''
    if pd.isna(x) or str(x).strip()=="":
        return pd.NA
    s=str(x).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return "TRUE"
    if s in ("false", "0", "no", "n"):
        return "FALSE"
    return pd.NA

##convert a int value to int and some 0 value to na,because some 0 value may exists, but is fake, so convert to na
def to_int_or_na(x, zero_to_na=False):
    '''
    x: the attribute value in csv
    zero_to_na: which attribute the value can not be 0
    '''
    if pd.isna(x) or str(x).strip() == "":
        return pd.NA
    try:
        v = int(float(str(x).strip()))
        if zero_to_na and v == 0:
            return pd.NA
        return v
    except:
        return pd.NA
    

##unify float format
def to_float_or_na(x):
    if pd.isna(x) or str(x).strip() == "":
        return pd.NA
    s = str(x).strip().replace("$", "").replace("%", "")
    try:
        return float(s)
    except:
        return pd.NA

##required age according to the gener and notes to estimate

def infer_age(orig_age, notes, tags):

    age = to_int_or_na(orig_age, zero_to_na=True)  # 关键：zero_to_na=True
    if not pd.isna(age):
        return max(int(age), 3)

    text = f"{'' if pd.isna(notes) else str(notes)} {'' if pd.isna(tags) else str(tags)}".lower()

    def hit(keys):
        return any(str(k).lower() in text for k in keys)

    if hit(ADULT_KEYS) or hit(VIOLENCE_KEYS):
        return 17
    if hit(ROMANCE_KEYS):
        return 10
    return 3



def main():
    ##utf-8-sig encoding provides secure reading way
    # df = pd.read_csv(IN_CSV, encoding="utf-8-sig")

    df = read_csv_mixed_utf8(IN_CSV)


    df.columns = [c.strip() for c in df.columns]

    ##Release date
    if "Release date" in df.columns:
        df["Release date"] = df["Release date"].apply(normalize_date)

    ##Estimated owners
    if "Estimated owners" in df.columns:
        df["Estimated owners"] = df["Estimated owners"].apply(clean_estimated_owners)

    ##Required age
    if "Required age" in df.columns:
        df["Required age"] = df.apply(lambda r: infer_age(r.get("Required age"), r.get("Notes"), r.get("Tags")), axis=1)

    ##Price,Discount,DLC count
    for c in ["Price", "Discount"]:
        if c in df.columns:
            df[c] = df[c].apply(to_float_or_na) if c == "Price" else df[c].apply(lambda x: to_int_or_na(x, zero_to_na=False))
    if "DLC count" in df.columns:
        df["DLC count"] = df["DLC count"].apply(lambda x: to_int_or_na(x, zero_to_na=False))

    ##bool
    for c in ["Windows", "Mac", "Linux"]:
        if c in df.columns:
            df[c] = df[c].apply(clean_bool)

    ##Metacritic score: 0 → NA
    if "Metacritic score" in df.columns:
        df["Metacritic score"] = df["Metacritic score"].apply(lambda x: to_int_or_na(x, zero_to_na=True))

    ##Achievements,ecommendations: 0 → NA
    if "Achievements" in df.columns:
        df["Achievements"] = df["Achievements"].apply(lambda x: to_int_or_na(x, zero_to_na=True))
    if "Recommendations" in df.columns:
        df["Recommendations"] = df["Recommendations"].apply(lambda x: to_int_or_na(x, zero_to_na=True))

    ##Positive,Negative(keep 0)
    for c in ["Positive", "Negative"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: to_int_or_na(x, zero_to_na=False))

    ##output csv
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Cleaned CSV saved: {OUT_CSV}")

if __name__ == "__main__":
    main()