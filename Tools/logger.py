"""A simple wrapper for python logging module.
"""
import os
import inspect
import logging
from .utils import ensure

GLOBAL_LOGGER_NAME = None

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)


#The background is set with 40 plus the number of the color, and the foreground with 30

#These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

def formatter_message(message, use_color = True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message

COLORS = {
    'WARNING': YELLOW,
    'INFO': CYAN,
    'DEBUG': MAGENTA,
    'CRITICAL': RED,
    'ERROR': RED
}

class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)

class QueueHandler(logging.Handler):
    """Custom logging handler for queueing the logs into given queue
    """
    def __init__(self, q):
        self.queue = q
        super(QueueHandler, self).__init__()

    def emit(self, record):
        log_entry = self.format(record)
        self.queue.put(log_entry)

def get_level_from_env():
    """Get logging level from environment variable LOGLEVEL if set.

    Returns
    -------
    logging level
        logging level
    """
    env_level = os.environ.get("LOGLEVEL")
    level = None

    if env_level == "DEBUG":
        level = logging.DEBUG
    elif env_level == "INFO":
        level = logging.INFO
    elif env_level == "WARNING":
        level = logging.WARNING
    elif env_level == "ERROR":
        level = logging.ERROR
    elif env_level == "CRITICAL":
        level = logging.CRITICAL

    return level

def setup_global_logger(name=None, level=None, logpath=None):
    """Initialize global logger

    Parameters
    ----------
    name : str, optional
        Name of the logger
    level : logging.level, optional
        Log level, by default logging.DEBUG
    logpath : str, optional
        Name of the log file, by default "diary.log"
    """
    global GLOBAL_LOGGER_NAME
    GLOBAL_LOGGER_NAME = name

    if level is None:
        level = get_level_from_env()
    if level is None:
        level = logging.DEBUG

    logger = logging.getLogger(GLOBAL_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    fmt = '[%(asctime)s] %(levelname)-7s (%(_module)s-%(_lineno)d): %(message)s'
    formatter = logging.Formatter(fmt=fmt)

    if logpath is not None:
        ensure(os.path.dirname(logpath))
        fhandler = logging.FileHandler(logpath)
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)

    colored_formatter = ColoredFormatter(formatter_message(fmt, False), False)
    # Add specific handlers
    shandler = logging.StreamHandler()
    shandler.setFormatter(colored_formatter)
    shandler.setLevel(level)
    logger.addHandler(shandler)

def common_meta(**kwargs):
    """Common meta setter
    """
    caller = os.path.basename(inspect.stack()[2][1])
    caller = caller[:caller.rfind(".")]
    line_no = inspect.stack()[2][2]
    kwargs['extra'] = {'_module': caller, '_lineno': line_no}
    return logging.getLogger(GLOBAL_LOGGER_NAME), kwargs

def debug(msg, *args, **kwargs):
    """Wrapper for logging.debug function
    """
    logger, kwargs = common_meta(**kwargs)
    logger.debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    """Wrapper for logging.info function
    """
    logger, kwargs = common_meta(**kwargs)
    logger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    """Wrapper for logging.warning function
    """
    logger, kwargs = common_meta(**kwargs)
    logger.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    """Wrapper for logging.error function
    """
    logger, kwargs = common_meta(**kwargs)
    logger.error(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    """Wrapper for logging.critical function
    """
    logger, kwargs = common_meta(**kwargs)
    logger.critical(msg, *args, **kwargs)

def attach_handler(handler):
    """Attach a handler to the global logger

    Parameters
    ----------
    handler : logging.Handler
        Handler to be added
    """
    logger = logging.getLogger(GLOBAL_LOGGER_NAME)

    # Use the default formatter
    handler.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(handler)