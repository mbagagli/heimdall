"""
This module provides customized logging facilities with color-coded output depending on the log level.

The color-coded logs help in distinguishing between different levels of logs at a glance. This module
supports logging to both the console and to a file, with the same color formatting applied to the console output.

Classes:
    CustomFormatter: A custom formatter for logging which applies different color codes to different logging levels.

Functions:
    init_logger(lgname, lvl='info', log_file=None): Initializes and returns a logger with a custom formatter.
"""

import logging

# ===============================================================
# ===============================================================
# ===============================================================
#               SEETING  UP  COLOR-LOGS

# Define new custom level STATE_LEVEL between INFO (20) and WARNING (30)
STATE_LEVEL = 25
logging.addLevelName(STATE_LEVEL, "STATE")

FMT = "[{levelname:^9}] {name} - {funcName}:{lineno} --> {message}"
FORMATS = {
    logging.DEBUG:     f"\33[35m{FMT}\33[0m",  # DEBUG in purple
    logging.INFO:      FMT,                    # INFO without color
    STATE_LEVEL:       f"\33[36m{FMT}\33[0m",  # STATE in cyan
    logging.WARNING:   f"\33[33m{FMT}\33[0m",  # WARNING in yellow
    logging.ERROR:     f"\33[31m{FMT}\33[0m",  # ERROR in red
    logging.CRITICAL:  f"\33[1m\33[31m{FMT}\33[0m"  # CRITICAL in bold red
}


def state(self, message, *args, **kws):
    """
    Log 'message' with severity 'STATE'.
    """
    if self.isEnabledFor(STATE_LEVEL):
        self._log(STATE_LEVEL, message, args, **kws)


logging.Logger.state = state


class CustomFormatter(logging.Formatter):
    """
    Custom formatter that applies color coding to log messages based on their severity.

    This formatter changes the color of log messages in the terminal depending on the log level,
    using ANSI escape sequences to set the colors. The color mappings are:
    - DEBUG: Cyan
    - INFO: No color
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Bold Red

    Methods:
        format(record): Formats a log record using the predefined log format and color scheme.
    """
    def format(self, record):
        """
        Format the specified record as text.

        Args:
            record (logging.LogRecord): A log record which will have its message formatted according to its severity.

        Returns:
            str: A formatted string with appropriate color coding for the console.
        """
        log_fmt = FORMATS[record.levelno]
        formatter = logging.Formatter(log_fmt, style="{")  # needed for custom styling
        return formatter.format(record)


def init_logger(lgname, lvl="info", log_file=None):
    """
    Initializes a logger with a custom formatter for both console and file outputs.

    This function sets up a logger to output color-coded log messages to the console and optionally to a file.
    The logging level can be specified, and a file path can be provided if file logging is required.

    Args:
        lgname (str): The name of the logger to initialize.
        lvl (str, optional): The logging level as a string (debug, info, warning, error, critical). Defaults to "info".
        log_file (str, optional): The path to the file where logs should be written. If None, logs will not be written to a file. Defaults to None.

    Returns:
        logging.Logger: The configured logger object with a custom formatter and appropriate handlers set.
    """
    map_level = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "state": STATE_LEVEL,  # Adding the custom STATE level
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    logger = logging.getLogger(lgname)
    logger.setLevel(map_level[lvl.lower()])

    # Remove any existing handlers
    logger.handlers.clear()

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

    # Setup file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(CustomFormatter())
        logger.addHandler(file_handler)

    return logger


def __testing__():
    logger = init_logger("my_logger", lvl="debug")
    logger.debug("This is a DEBUG message (purple).")
    logger.info("This is an INFO message.")
    logger.state("This is a STATE message (cyan).")
    logger.warning("This is a WARNING message. (yellow)")
    logger.error("This is an ERROR message. (red)")
    logger.critical("This is a CRITICAL message. (bold-red)")


if __name__ == "__main__":
    __testing__()
