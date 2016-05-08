class BaseFitnessError(Exception):
    pass


class UnsupportedSpace(BaseFitnessError):
    """Raised when an operation is not supported for the Space"""
    def __init__(self, message):
        super(UnsupportedSpace, self).__init__(message)
