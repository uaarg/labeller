import sys
import os
import json
from PyQt6 import QtGui
from PyQt6.QtWidgets import (QApplication, QGraphicsScene, QGraphicsView,
                             QGraphicsRectItem, QGraphicsPixmapItem,
                             QVBoxLayout, QWidget, QPushButton, QFileDialog)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


class DraggableRectItem(QGraphicsRectItem):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)

    def itemChange(self, change, value):
        if change == QGraphicsRectItem.GraphicsItemChange.ItemPositionChange:
            self.adjust_annotations()
        return super().itemChange(change, value)

    def adjust_annotations(self):
        # TODO
        pass


class App(QWidget):

    def __init__(self):
        super().__init__()

        self.image_path = None
        self.annotations_path = None
        self.init_ui()

    def init_ui(self):
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing,
                                True)  # Enable Antialiasing

        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)

        self.rect_items = []

        layout = QVBoxLayout(self)
        layout.addWidget(self.view)

        self.add_box_button = QPushButton("Add Bounding Box", self)
        self.add_box_button.clicked.connect(self.add_bounding_box)
        layout.addWidget(self.add_box_button)

        self.load_image_button = QPushButton("Load Image", self)
        self.load_image_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_image_button)

        self.save_annotations_button = QPushButton("Save Annotations", self)
        self.save_annotations_button.clicked.connect(self.save_annotations)
        layout.addWidget(self.save_annotations_button)

        self.setLayout(layout)

    def load_image(self):
        file_dialog = QFileDialog()
        file_dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            file_path = file_dialog.selectedFiles()[0]
            self.load_annotations(file_path)
            self.set_image(file_path)

    def set_image(self, file_path: str):
        self.image_path = file_path
        self.annotations_path = self.get_annotations_path(file_path)
        pixmap = QPixmap(file_path)
        self.image_item.setPixmap(pixmap)

    def get_annotations_path(self, image_path: str) -> str:
        base_path, ext = os.path.splitext(image_path)
        return base_path + '.json'

    def load_annotations(self, image_path: str):
        annotations_path = self.get_annotations_path(image_path)
        if os.path.exists(annotations_path):
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)

            for annotation in annotations:
                rect_item = DraggableRectItem(annotation['x'], annotation['y'], annotation['width'], annotation['height'])
                rect_item.setPen(QtGui.QColor().red())
                self.scene.addItem(rect_item)
                self.rect_items.append(rect_item)

    def save_annotations(self):
        if self.image_path is not None:
            annotations = []
            for rect_item in self.rect_items:
                annotations.append({
                    'x': rect_item.x(),
                    'y': rect_item.y(),
                    'width': rect_item.rect().width(),
                    'height': rect_item.rect().height(),
                })

            assert self.annotations_path is not None
            with open(self.annotations_path, 'w') as f:
                json.dump(annotations, f)

    def add_bounding_box(self):
        if self.image_path is not None:
            rect_item = DraggableRectItem(
                0, 0, 100, 100)  # Initial size, you can adjust as needed
            rect_item.setPos(50,
                             50)  # Initial position, you can adjust as needed
            rect_item.setPen(QtGui.QColor().red())
            self.scene.addItem(rect_item)
            self.rect_items.append(rect_item)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected_boxes()

    def delete_selected_boxes(self):
        for rect_item in self.scene.selectedItems():
            if isinstance(rect_item, DraggableRectItem):
                self.scene.removeItem(rect_item)
                self.rect_items.remove(rect_item)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    window.setGeometry(100, 100, 800, 600)
    window.setWindowTitle('Bounding Box Annotation Tool')
    window.show()
    sys.exit(app.exec())
