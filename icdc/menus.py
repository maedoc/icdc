"""
Implements a menu bar that is built dynamically from other modules in the package
and the actions described there in. 

Bugs aside, this file should only require modification to add a module-menu.

"""

from .qt import QtCore, QtGui
from .core import Action
from . import file, edit, view, detect, classify, help

MODULE_MENUS = [file, edit, view, detect, classify, "separator", help]

# TODO
_mb = []


def update_recent_files():
    mb = _mb[0]
    for act in mb._recent_file_actions:
        mb.file_menu.removeAction(act)
    fs = file.read_recent_files(QtCore.QSettings("INS", "IC/DC"))[::-1]
    for i, f in enumerate(fs):
        act = file.RecentFileAction(i, f, parent=mb.file_menu)
        mb.file_menu.addAction(act)
        mb._recent_file_actions.append(act)
    if fs:
        act = file.ClearRecentFiles(parent=mb.file_menu)
        mb.file_menu.addAction(act)
        mb._recent_file_actions.append(act)


def setup_menubar(mb, parent=None):
    _mb.append(mb)
    for mod in MODULE_MENUS:
        if mod == "separator":
            mb.addSeparator()
        else:
            title = getattr(mod, "title", "&" + mod.__name__.split(".")[-1].title())
            menu = mb.addMenu(title)
            for action in mod.actions:
                if action == "separator":
                    menu.addSeparator()
                else:
                    act = action(parent=menu)
                    menu.addAction(act)
                    # disable actions that don't implement their own run
                    if action.run == Action.run:
                        act.setEnabled(False)

            if mod == file:
                mb.file_menu = menu
                mb._recent_file_actions = []
                update_recent_files()
                menu.aboutToShow.connect(update_recent_files)
