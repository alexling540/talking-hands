# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import sys
from PyQt5.QtWidgets import QApplication
from window import UI_Window

app = QApplication([])
window = UI_Window()
window.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sys.exit(app.exec_())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
