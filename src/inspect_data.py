import pandas as pd

def main():
    csv_path = "data/fer2013.csv"
    df = pd.read_csv(csv_path)
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nUsage value counts:")
    print(df["Usage"].value_counts())
    print("\nFirst 3 rows:")
    print(df.head(3))

if __name__ == "__main__":
    main()
