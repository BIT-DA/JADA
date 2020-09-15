import time
import logging
from logging import handlers


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # log level

    def __init__(self, logroot, filename, level='info', when='D', fmt='%(message)s'):
        filename = logroot + time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + filename + '.log'
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # set log format
        self.logger.setLevel(self.level_relations.get(level))  # set log level
        sh = logging.StreamHandler() 
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when,
                                               encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh) 
        self.logger.addHandler(th)


# sample
# if __name__ == '__main__':
#     log = Logger('../save/all.log', level='debug')
#     i = 10
#     while i:
#         log.logger.debug('acc%.4f', i/100)
#         i -= 1
