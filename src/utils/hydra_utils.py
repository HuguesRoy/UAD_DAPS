# hydra utils function to use function as _target_


def convert_to_tuple(args):
    try:
        if args is None or isinstance(args, tuple):
            return args
        return tuple(args)
    except TypeError:
        return args
