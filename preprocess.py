import pandas as pd

def load_data():
    df = pd.read_csv("data/results.csv")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
