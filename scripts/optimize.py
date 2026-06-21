import sys

from waveguide_opt.cli import main

if __name__ == "__main__":
    main(["batch", *sys.argv[1:]])
