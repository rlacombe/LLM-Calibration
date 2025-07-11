import pandas as pd
import requests
from googlesearch import search
import time

def get_iarc_data():
    """
    Downloads and prepares the IARC carcinogen classification data.

    Returns:
        pandas.DataFrame: A DataFrame containing the list of agents.
    """
    url = "https://monographs.iarc.who.int/list-of-classifications/#:~:text=Copy-,CSV,-Excel"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        from io import StringIO
        # Find the start of the actual data
        content = response.text
        start_index = content.find("Agent,Group,Volume,Year,Additional information")
        csv_content = content[start_index:]
        
        df = pd.read_csv(StringIO(csv_content))
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return None

def count_google_hits(query):
    """
    Performs a Google search and returns the approximate number of results.

    Args:
        query (str): The search query.

    Returns:
        int: The approximate number of search results.
    """
    try:
        # The search generator is consumed to get an approximate count.
        # This is not a perfect method for getting an exact count, but it's a proxy.
        return sum(1 for _ in search(query, stop=20)) # Checking first 20 results for a hit
    except Exception as e:
        print(f"An error occurred during search: {e}")
        return 0

def main():
    """
    Main function to download IARC data and count Google hits for each compound.
    Writes results to a CSV file line by line.
    """
    iarc_df = get_iarc_data()

    if iarc_df is not None:
        # Create output file with headers
        output_file = "iarc_google_counts.csv"
        with open(output_file, 'w') as f:
            # Write headers including all IARC columns plus our new column
            headers = list(iarc_df.columns) + ['google_hits']
            f.write(','.join(headers) + '\n')
            
            # Process each row
            for index, row in iarc_df.iterrows():
                compound_name = row['Agent']
                search_query = f'"{compound_name}" carcinogenic'
                
                print(f"Searching for: {search_query}")
                hits = count_google_hits(search_query)
                print(f"-> Hits: {hits}")
                
                # Write the row with all original data plus the hit count
                row_data = [str(row[col]) for col in iarc_df.columns] + [str(hits)]
                f.write(','.join(row_data) + '\n')
                
                # To be respectful of Google's terms of service, add a delay
                time.sleep(0.5)
        
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()