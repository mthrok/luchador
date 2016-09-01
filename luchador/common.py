from __future__ import absolute_import


def get_subclasses(Class):
    """Get the list of all subclasses

    This function is intended to use in get_* functions such as
    get_agent, get_env, get_layer, get_optimizer, get_initializer ...
    for flexible access.
    """
    ret = []
    for SubClass in Class.__subclasses__():
        # As we adopt subclassing-base-interface approach to realize backend
        # switch, we prefer to return subclasses in the order similar to
        # python's inheritence resolution.
        # So the descendant classes comes before their anscestors.
        ret.extend(get_subclasses(SubClass))
        ret.append(SubClass)
    return ret
