#!/usr/bin/env python2

from rectool import RecognizerToolbox


def main():
    rt = RecognizerToolbox()
    rt.decode_batch()


if __name__ == "__main__":
    # If we would be writing real programs, we would be doing real error handling
    main()
