from .hub_entries import HubEntries


def get_hub_entries(path, default_file="hubconfig"):
    """
    get enty point of a hub folder
    :param path: path to hub folder
    :param default_file: default entry file, default="hubconfig"
    :return: HubEntries
    """
    return HubEntries(path, default_file)
