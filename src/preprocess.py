from collections.abc import MutableMapping
from typing import Any, Dict

def _flatten_dict_gen(d: MutableMapping, parent_key: str, sep: str):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep).items()
        elif isinstance(v, list) or isinstance(v, list):
            #  For lists we transform them into strings with a join
            yield new_key, "#".join(map(str, v))
        else:
            yield new_key, v

def flatten_dict(d: MutableMapping,
                 parent_key: str = '',
                 sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a dictionary using recursion (via an auxiliary funciton).
    The list/tuples values are flattened as a string.
    Parameters
    ----------
    d : MutableMapping
        Dictionary (or, more generally something that is a MutableMapping) to flatten.
        It might be nested, thus the function will traverse it to flatten it.
    parent_key : str
        Key of the parent dictionary in order to append to the path of keys.
    sep : str
        Separator to use in order to represent nested structures.
    Returns
    -------
    Dict[str, Any]
        The flattened dict where each nested dictionary is expressed as a path with
        the `sep`.
    >>> flatten_dict({'a': {'b': 1, 'c': 2}, 'd': {'e': {'f': 3}}})
    {'a.b': 1, 'a.c': 2, 'd.e.f': 3}
    >>> flatten_dict({'a': {'b': [1, 2]}})
    {'a.b': '1#2'}
    """
    return dict(_flatten_dict_gen(d, parent_key, sep))
