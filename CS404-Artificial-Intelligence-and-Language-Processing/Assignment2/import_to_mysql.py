import os
import pandas as pd
import pymysql
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = os.getenv("CSV_PATH", r"CS404-Artificial-Intelligence-and-Language-Processing\Assignment2\games_sample.csv")

DB_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("MYSQL_PORT", "3306"))
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASS = os.getenv("MYSQL_PASSWORD", "")
DB_NAME = os.getenv("MYSQL_DB", "steam_games")


def main():

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    ## csv attribute name to database column name
    col_map = {
        "AppID": "app_id",
        "Name": "game_name",
        "Release date": "release_date",
        "Estimated owners": "estimated_owners",
        "Required age": "required_age",
        "Price": "price",
        "Discount": "discount",
        "DLC count": "dlc_count",
        "About the game": "about_the_game",
        "Supported languages": "supported_languages",
        "Full audio languages": "full_audio_languages",
        "Reviews": "reviews",
        "Header image": "header_image",
        "Website": "website",
        "Support url": "support_url",
        "Support email": "support_email",
        "Windows": "support_window",   
        "Mac": "support_mac",
        "Linux": "support_linux",
        "Metacritic score": "metacritic_score",
        "Metacritic url": "metacritic_url",
        "Positive": "positive",
        "Negative": "negative",
        "Achievements": "achievements",
        "Recommendations": "recommendations",
        "Notes": "notes",
        "Developers": "developers",
        "Publishers": "publishers",
        "Categories": "categories",
        "Genres": "genres",
        "Tags": "tags",
        "Screenshots": "screenshots",
    }

    csv_cols = set(df.columns)
    expected_csv_cols = set(col_map.keys())

    missing_in_csv = sorted(list(expected_csv_cols - csv_cols))
    if missing_in_csv:
        print("These columns are in col_map but NOT found in CSV:")
        for c in missing_in_csv:
            print("  -", c)

    src_cols = [c for c in col_map.keys() if c in df.columns]
    db_cols = [col_map[c] for c in src_cols]

    out = df[src_cols].rename(columns={c: col_map[c] for c in src_cols})
    out = out.astype(object)
    out = out.where(~out.isna(), None)

    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        charset="utf8mb4",
        autocommit=False
    )

    try:
        with conn.cursor() as cur:
            cur.execute("SET NAMES utf8mb4;")

            cur.execute("TRUNCATE TABLE games;")

            placeholders = ",".join(["%s"] * len(db_cols))
            sql = f"INSERT INTO games ({','.join(db_cols)}) VALUES ({placeholders})"

            batch = 1000
            vals = out.values.tolist()

            for i in range(0, len(vals), batch):
                cur.executemany(sql, vals[i:i+batch])
                # print(f"Inserted {min(i+batch, len(vals))}/{len(vals)}")

        conn.commit()
        print("Imported rows:", len(out))

    except Exception as e:
        conn.rollback()
        print("Import failed:", e)
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()
