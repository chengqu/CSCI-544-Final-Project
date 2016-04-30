
"""
parse the loss function values in log file into a txt file for visualization

Example usage:
    python parse_loss.py title_author.pbs.o14844248 title_author_loss.txt    

"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("input",help="input log file")
parser.add_argument("output",help="output file")

args = parser.parse_args()


f = open(args.input)
content = f.read()
f.close()

content = content.split()

with open(args.output,'w') as f:
    for i in range(len(content)):
        if content[i] == 'cost,':
            f.write(content[i-1])
            f.write('\n')



