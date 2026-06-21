import sys

from waveguide_opt.cli import main

if __name__ == "__main__":
    main(["generate", *sys.argv[1:]])
