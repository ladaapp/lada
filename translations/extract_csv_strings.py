# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import csv
import sys

def extract_csv_strings(csv_file, pot_file, column_header):
    with open(csv_file, mode='r', newline='', encoding='utf-8') as csvfile, \
            open(pot_file, mode='w', encoding='utf-8') as potfile:
        reader = csv.DictReader(csvfile, delimiter='|')
        for row in reader:
            description = row[column_header]
            description = description.replace('"', '\\"') # Escape quotes

            potfile.write(f'#: {csv_file}:{reader.line_num}\n')
            potfile.write(f'msgid "{description}"\n')
            potfile.write('msgstr ""\n\n')

    print(f"{reader.line_num - 1} Strings extracted from {csv_file} and written to {pot_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_csv_strings.py <input_csv_file> <output_pot_file> <column header>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_pot = sys.argv[2]
    column_header = sys.argv[3]
    extract_csv_strings(input_csv, output_pot, column_header)
