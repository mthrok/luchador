from __future__ import absolute_import


class CopyMixin(object):
    """Provide copy method which creates a copy of initialed instance"""
    ###########################################################################
    # Argument checkers
    def _store_args(self, args):
        """
        Args:
            args (dict): Arguments passed to __init__ of the subclass instance.
            This will be used to recreate a subclass instance.
        """
        self._validate_args(args)
        self.args = args

    def _validate_args(self, args):
        """Validate arguments"""
        pass

    ###########################################################################
    def copy(self):
        """Create and initialize new instance with the same argument"""
        return type(self)(**self.args)

    ###########################################################################
    # Configuration equality
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.args == other.args

    def __ne__(self, other):
        return not self.__eq__(other)

    ###########################################################################
    # IO
    def export(self):
        args = {}
        for key, value in self.args.items():
            args[key] = value
        return {
            'name': self.__class__.__name__,
            'args': args
        }
