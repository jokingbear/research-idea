from .hub_entries import HubEntries


def get_hub_entries(path, default_file="hubconfig"):
    return HubEntries(path, default_file)
