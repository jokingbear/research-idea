import pkgutil

arr = [*pkgutil.iter_modules(['plasma/utils.py'])]

pkgutil.get_importer()