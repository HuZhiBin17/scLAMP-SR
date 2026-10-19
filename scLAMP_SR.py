#!/usr/bin/env python
"""Train the paper scLAMP-SR model."""

from sclamp_sr.model import main, parse_args

if __name__ == "__main__":
    main(parse_args())
