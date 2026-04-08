#!/usr/bin/env python3
"""Run all dataset collection scripts."""

import collect_t2t
import collect_ti2t
import collect_t2i
import collect_ti2i


def main():
    print("=" * 60)
    print("Collecting ALL datasets")
    print("=" * 60)

    collect_t2t.main()
    print()
    collect_ti2t.main()
    print()
    collect_t2i.main()
    print()
    collect_ti2i.main()

    print()
    print("=" * 60)
    print("ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
