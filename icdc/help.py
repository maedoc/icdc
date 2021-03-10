from .qt import QtCore, QtGui
from .core import Action


# TODO this should capture log to file user can send as attachment
class ReportBug(Action):
    _text = "Report &bug"


class LogView(QtGui.QWidget):
    def __init__(self, lines, parent=None):
        QtGui.QWidget.__init__(self, parent=parent)
        self.lay = QtGui.QVBoxLayout()
        self.setLayout(self.lay)

        self.setWindowTitle("Log")

        self.textedit = QtGui.QTextEdit()
        print(lines)
        self.textedit.setHtml("<pre>%s</pre>" % ("\n".join(lines),))
        self.lay.addWidget(self.textedit)

        self.lay_buttons = QtGui.QHBoxLayout()
        self.lay.addLayout(self.lay_buttons)

        self.pb_copy = QtGui.QPushButton("&Copy log to clipboard")
        self.pb_copy.clicked.connect(self.copy_msg)
        self.lay_buttons.addWidget(self.pb_copy)

        self.pb_close = QtGui.QPushButton("&OK")
        self.pb_close.clicked.connect(self.close)
        self.lay_buttons.addWidget(self.pb_close)

    def copy_msg(self):
        qc = QtGui.QClipboard()
        qc.setText(self.msg)

    def sizeHint(self):
        return QtCore.QSize(400, 400)


class ShowLog(Action):
    _text = "Show &log"

    def run(self):
        self.view = LogView(self.main.log)
        self.view.show()


class ShowConsole(Action):
    _text = "Show &console"

    def run(self):
        from pyqtgraph.console import ConsoleWidget

        ns = {}
        ns.update(globals())
        ns.update(locals())
        self.con = ConsoleWidget(namespace=ns)
        self.con.show()


actions = [ShowLog, ShowConsole]
