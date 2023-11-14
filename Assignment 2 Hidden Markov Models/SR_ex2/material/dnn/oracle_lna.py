#!/usr/bin/env python3

import struct
import sys
import math


def oracle_lna(phones_file, label_file, lna_file):
    phones = [l.strip() for l in open(phones_file, encoding='utf-8')]

    yes_prob = math.log(0.95)
    no_prob = math.log(0.05 / (len(phones) - 1))

    with open(lna_file, 'wb') as f:
        f.write(struct.pack(">I", len(phones)))
        f.write(struct.pack("B", 2))
        for label in open(label_file, encoding='utf-8'):
            i = phones.index(label.strip())
            for j in range(len(phones)):
                if i == j:
                    f.write(struct.pack(">H", int(-1820.0 * yes_prob + .5)))
                else:
                    f.write(struct.pack("BB", 255,255))

if __name__ == "__main__":
    # If we would be writing real programs, we would be doing real error handling
    oracle_lna(sys.argv[1], sys.argv[2], sys.argv[3])
