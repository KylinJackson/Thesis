import logging
import time


class Logger:
    def __init__(self, filename):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler('log/{}'.format(filename))
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter('%(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def set_log(self, message, t=None, **kwargs):
        if t is None:
            t = time.localtime()
        t = time.strftime('%Y-%m-%d %H:%M:%S', t)
        message = t + ' - ' + message

        if len(kwargs) != 0:
            for k in kwargs:
                message2 = str(kwargs[k]).replace('\n', '\n\t')
                message += '\n\t{} - {}'.format(k, message2)

        self.logger.info(message)


if __name__ == '__main__':
    logger = Logger('test2.log')
    logger.set_log('info', city='Beijing', age=20)
