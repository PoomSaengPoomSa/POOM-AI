import urllib.request
import pandas as pd
import io

def main():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10"
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(html))
            print("--- FRED DFII10 data ---")
            print(df.tail(10))
            print(f"Total rows: {len(df)}")
    except Exception as e:
        print(f"Error fetching from FRED: {e}")

if __name__ == '__main__':
    main()
