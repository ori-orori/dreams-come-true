# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:57:38 2023

@author: jellyho
"""

from RCServer import RCServer
import sys
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QMessageBox, QSlider
from PyQt5.QtCore import Qt

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Heuristic')
        self.resize(320, 240)
        self.setupUi()
        self.server = RCServer(carA_ip='192.168.0.15', port=1234)
        self.show()
        
    def setupUi(self):
        self.text = QLabel('stop', self)
        self.text.setGeometry(120, 120, 200, 100)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.move(30, 30)
        self.slider.setRange(0, 100)
        self.slider.setSingleStep(1)
        self.slider.setTickPosition(1)
        
        # self.text.move(120, 120)
        self.text.setFont(QFont('Arial', 50))
                
        
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_W:
            self.server.HeuristicAct('A', 'W', int(self.slider.value())/100.0)
            self.text.setText(f'front:{self.slider.value()}')
        elif e.key() == Qt.Key_S:
            self.server.HeuristicAct('A', 'S', int(self.slider.value())/100.0)
            self.text.setText('back')
        elif e.key() == Qt.Key_D:
            self.server.HeuristicAct('A', 'D', int(self.slider.value())/100.0)
            self.text.setText('right')
        elif e.key() == Qt.Key_A:
            self.server.HeuristicAct('A', 'A', int(self.slider.value())/100.0)
            self.text.setText('left')
        else:
            self.server.HeuristicAct('A', '?', int(self.slider.value())/100.0)
            self.text.setText('stop')
            
    def closeEvent(self, event):
        quit_msg = "Want to exit?"
        reply = QMessageBox.question(self, 'Message', quit_msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.server.sock.close()
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Main()
    sys.exit(app.exec_())