from __future__ import absolute_import


class StoreMixin(object):
    def _store_args(self, args):
        """Store initializer arguments to `args` attribute
        Args:
            args (dict): Arguments passed to __init__.
            This will be used to recreate a subclass instance.
        """
        self._validate_args(args)
        self.args = args

    def _validate_args(self, args):
        """Validate arguments
        Args:
            args (dict): Arguments passed to __init__.
        """
        pass

    ###########################################################################
    def __repr__(self):
        return "{{'name': '{}', 'args': {}}}".format(
            self.__class__.__name__, self.args)

    ###########################################################################
    # Shallow equality
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.args == other.args
        return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    ###########################################################################
    def serialize(self):
        """Serialize object configuration (constructor arguments)"""
        args = {}
        for key, val in self.args.items():
            args[key] = val.serialize() if hasattr(val, 'serialize') else val
        return {
            'name': self.__class__.__name__,
            'args': args
        }
