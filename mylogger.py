import logging
import os
import sys

def get_logger(name,filepath,level=logging.DEBUG,format='[%(levelname)s]:%(name)s, %(asctime)s, %(message)s',mode='a'):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # tx = logging.StreamHandler(sys.stdout)
    # tx.setFormatter(logging.Formatter(format))
    # tx.setLevel(level)
    # logger.addHandler(tx)
    if filepath:
        if os.path.exists(os.path.dirname(filepath)):
            fh = logging.FileHandler(filename=filepath,mode=mode)
            fh.setFormatter(logging.Formatter(format))
            fh.setLevel(level)
            logger.addHandler(fh)
        else:
            logger.warning("path of logfile {} is not a valid file, considered as no logfile.".format(os.path.abspath(filepath)))
    return logger
        
