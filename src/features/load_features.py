import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def load_features():
    sql = open("sql/features_query.sql", encoding="utf-8").read()
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


if __name__ == "__main__":
    df = load_features()
    print(df.shape)
    print(df["failed"].mean())
    print(df.groupby("repo")["failed"].mean())