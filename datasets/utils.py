from typing import Union


def transform_label(label: Union[int, str]) -> int:
    """Transforms the 1-indexed label into a 0-indexed label.

    The input may be either an int or a string. In the case of the latter, it is assumed
    that the input is a string representation of an int.

    Args:
        label:
            The 1-indexed label.

    Returns:
        The 0-indexed label.
    """
    return int(label) - 1
