import csv
import sys

def convert_tsv_to_csv(tsv_file, csv_file):
    with open(tsv_file, 'r', encoding='utf-8') as tsv_in, \
         open(csv_file, 'w', encoding='utf-8', newline='') as csv_out:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')
        csv_writer = csv.writer(csv_out, quoting=csv.QUOTE_ALL)  # Quote all fields
        for row in tsv_reader:
            csv_writer.writerow(row)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_csv.py input.tsv output.csv")
        sys.exit(1)
    
    convert_tsv_to_csv(sys.argv[1], sys.argv[2]) 