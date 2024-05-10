import colorlog
import logging


def _get_console_handler(logger_level: int) -> colorlog.StreamHandler:
    # Create a console handler using colorlog
    console_handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(fg_black)s%(bg_white)s%(name)s %(asctime)s%(reset)-4s"
        " %(log_color)s[%(levelname)s]: %(log_color)s%(message)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red, bg_white",
        },
        style="%",
    )
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logger_level)
    return console_handler


def get_logger(logger_name: str, logger_level: int = logging.DEBUG) -> logging.Logger:
    # Configure logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)
    logger.propagate = False

    # Check if logger already has handlers
    if not logger.handlers:
        ch = _get_console_handler(logger_level)
        logger.addHandler(ch)

    return logger


logger = get_logger(__file__, logger_level=logging.INFO)
