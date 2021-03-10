"""
ICDC

Usage:

    TODO docopt style options. 

Default behavior, start ui

"""

import sys

# import docopt

use_ui = True

if use_ui:
    from .qt import QtCore, QtGui

    app = QtGui.QApplication(sys.argv)
    try:
        from .main import Main

        main = Main()
        main.show()
    except Exception as exc:
        import traceback

        t, v, tb = sys.exc_info()

        class err(QtGui.QTextEdit):
            def sizeHint(self):
                return QtCore.QSize(640, 300)

        box = err()
        box.setHtml("<pre>%s\n%r</pre>" % ("\n".join(traceback.format_tb(tb)), exc))
        box.show()
    sys.exit(app.exec_())
