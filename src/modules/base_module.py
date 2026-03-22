import logging


class BaseModule:
    """
    Base class for all modules.
    This class provides a common interface and logging functionality for all modules.
    """

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.logger = logging.getLogger(module_name)
