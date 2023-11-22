import sys
import os
import json

from typing import Optional

from PyQt6 import QtGui
from PyQt6.QtWidgets import (QApplication, QGestureEvent, QGraphicsScene, QGraphicsTextItem,
                             QGraphicsView, QGraphicsRectItem,
                             QGraphicsPixmapItem, QPinchGesture, QVBoxLayout, QWidget,
                             QPushButton, QFileDialog, QToolBar)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QEvent, QObject, QPointF, QRectF, Qt


class DraggableRectItem(QGraphicsRectItem):

    def __init__(self, x, y, width, height, title=""):
        super().__init__(0, 0, width, height)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(
            QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        self.setPen(QtGui.QColor().red())

        self.title = title
        self.title_item = QGraphicsTextItem(self.title, self)
        self.title_item.setDefaultTextColor(QtGui.QColor().fromRgb(
            255, 255, 255))
        self.adjust_annotations()

        self.setPos(x, y)
        print(self.x(), self.y())

    def itemChange(self, change, value):
        if change == QGraphicsRectItem.GraphicsItemChange.ItemPositionChange:
            self.adjust_annotations()

        return super().itemChange(change, value)

    def adjust_annotations(self):
        self.title_item.setPos(self.rect().topLeft() + QPointF(5, 5))

    def mouseMoveEvent(self, event):
        if event:
            if self.isSelected() and event.buttons(
            ) == Qt.MouseButton.RightButton:
                self.resize(event.pos() - self.rect().topLeft())

        super().mouseMoveEvent(event)

    def resize(self, size):
        self.prepareGeometryChange()
        self.setRect(QRectF(self.rect().topLeft(), size))


class App(QWidget):

    def __init__(self):
        super().__init__()

        self.image_path = None
        self.annotations_path = None
        self.rect_items = []

        layout = QVBoxLayout(self)

        self.tool_bar = QToolBar()
        layout.addWidget(self.tool_bar)

        self.tool_bar.addAction(QtGui.QIcon.fromTheme("zoom-in"), "Zoom In",
                                self.zoom_in)
        self.tool_bar.addAction(QtGui.QIcon.fromTheme("zoom-out"), "Zoom Out",
                                self.zoom_out)

        self.scene = QGraphicsScene(self)
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        layout.addWidget(self.view)

        viewport = self.view.viewport()
        assert viewport is not None
        viewport.grabGesture(Qt.GestureType.PinchGesture)
        viewport.installEventFilter(self)

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
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.jpeg)")
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
                rect_item = DraggableRectItem(annotation['x'], annotation['y'],
                                              annotation['width'],
                                              annotation['height'],
                                              annotation['title'])
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
                    'title': rect_item.title,
                })

            assert self.annotations_path is not None
            with open(self.annotations_path, 'w') as f:
                json.dump(annotations, f)

    def add_bounding_box(self):
        if self.image_path is not None:
            rect_item = DraggableRectItem(0, 0, 100, 100, title="orange")
            rect_item.setPos(50, 50)
            self.scene.addItem(rect_item)
            self.rect_items.append(rect_item)

    def keyPressEvent(self, event: Optional[QtGui.QKeyEvent]):
        if event and event.key() == Qt.Key.Key_Delete:
            self.delete_selected_boxes()

    def delete_selected_boxes(self):
        for rect_item in self.scene.selectedItems():
            if isinstance(rect_item, DraggableRectItem):
                self.scene.removeItem(rect_item)
                self.rect_items.remove(rect_item)

    def zoom_in(self):
        self.view.scale(1.1, 1.1)

    def zoom_out(self):
        self.view.scale(0.9, 0.9)

    def pinch_trigger(self, gesture):
        # Adjust the scale factor based on the pinch gesture
        zoom_factor = gesture.scaleFactor()
        self.view.setTransform(self.view.transform().scale(zoom_factor, zoom_factor))

    def eventFilter(self, source: Optional[QObject], event: Optional[QEvent]) -> bool:
        if event and event.type() == QEvent.Type.Gesture:
            assert isinstance(event, QGestureEvent)
            gesture_event = event
            for gesture in gesture_event.gestures():
                if gesture.state() == Qt.GestureState.GestureUpdated and isinstance(gesture, QPinchGesture):
                    self.pinch_trigger(gesture)

        return super().eventFilter(source, event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    window.setGeometry(100, 100, 800, 600)
    window.setWindowTitle('Bounding Box Annotation Tool')
    window.show()
    sys.exit(app.exec())
