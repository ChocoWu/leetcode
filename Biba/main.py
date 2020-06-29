#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu


from Biba.Biba import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QMainWindow, QApplication
import sys


class Main(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.user_permission_dict = {r"C:/Users/wushe/Desktop/A.txt": 'A',
                                     r"C:/Users/wushe/Desktop/B.txt": 'B',
                                     r"C:/Users/wushe/Desktop/C.txt": 'C',
                                     r"C:/Users/wushe/Desktop/D.txt": 'D'}
        self.setupUi(self)
        self.choose.clicked.connect(self.openFile)
        self.readButton.clicked.connect(self.read)
        self.writeButton.clicked.connect(self.write)

    def read(self):
        print('write', self.filepath.text())
        fgrade = self.user_permission_dict[self.filepath.text()]
        ugrade = self.user_permission.text()
        print(fgrade)
        print(ugrade)
        if fgrade > ugrade:
            QMessageBox.warning(self, "warning", "当前用户等级太高，不能读取", QMessageBox.Yes, QMessageBox.Yes)
            self.showContext.close()
        else:
            with open(self.filepath.text(), 'r') as f:
                lines = ' '.join(f.readlines())
                self.showContext.setText(lines)
            # QMessageBox.information(self, "提示框", "读取成功", QMessageBox.Yes, QMessageBox.Yes)

    def write(self):
        # dict = self.getGrade()
        print('write', self.filepath.text())
        fgrade = self.user_permission_dict[self.filepath.text()]
        ugrade = self.user_permission.text()
        print(fgrade)
        print(ugrade)
        if fgrade < ugrade:
            QMessageBox.warning(self, "warning", "当前用户等级太低，不能写入", QMessageBox.Yes, QMessageBox.Yes)
            # print("当前用户等级太低，不能写入")
        else:
            with open(self.filepath.text(), 'a') as f:
                lines = self.writeContext.toPlainText()
                f.write(lines)
            QMessageBox.information(self, "提示框", "写入成功", QMessageBox.Yes, QMessageBox.Yes)

        # with open(self.filepath.text(), 'a') as f:
        #     lines = self.writeContext.toPlainText()
        #     f.write(lines)

    def openFile(self):
        get_filename_path, ok = QFileDialog.getOpenFileName(self, "选取单个文件", "C:/",
                                                            "All Files (*);;Text Files (*.txt)")
        print(str(get_filename_path))
        if ok:
            self.filepath.setText(str(get_filename_path))
        else:
            QMessageBox.warning(self, '警告提示框', '打开文件错误', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = Main()
    myWin.show()
    sys.exit(app.exec_())
    # with open('./C.txt', 'a') as f:
    #     f.writelines('we are family')
