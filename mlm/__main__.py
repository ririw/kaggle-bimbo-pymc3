import logging
import coloredlogs

from mlm import main

coloredlogs.install(level=logging.INFO)

if __name__ == '__main__':
    main()

