import sys
import os
import json
import zipfile
import glob
import numpy as np

from typing import Optional, Tuple, List

from PyQt6 import QtGui
from PyQt6.QtWidgets import (QApplication, QErrorMessage, QGestureEvent,
                             QGraphicsScene, QGraphicsTextItem, QGraphicsView,
                             QGraphicsRectItem, QGraphicsPixmapItem, QLabel,
                             QMessageBox, QPinchGesture, QVBoxLayout, QWidget,
                             QFileDialog, QToolBar)
from PyQt6.QtGui import QKeyEvent, QPixmap
from PyQt6.QtCore import QEvent, QObject, QPointF, QRectF, Qt
import PIL
from PIL import Image


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

        self.error_dialog = QErrorMessage()
        self.message_dialog = QMessageBox()

        self.installEventFilter(self)

        layout = QVBoxLayout(self)

        self.tool_bar = QToolBar()
        layout.addWidget(self.tool_bar)

        self.tool_bar.addAction(QtGui.QIcon.fromTheme("zoom-in"), "Zoom In",
                                self.zoom_in)
        self.tool_bar.addAction(QtGui.QIcon.fromTheme("zoom-out"), "Zoom Out",
                                self.zoom_out)

        self.tool_bar.addSeparator()

        self.tool_bar.addAction(QtGui.QIcon.fromTheme("select"),
                                "Add Bounding Box", self.add_bounding_box)
        self.tool_bar.addAction(QtGui.QIcon.fromTheme("save"),
                                "Save Annotations", self.save_annotations)

        self.tool_bar.addSeparator()

        self.tool_bar.addAction(QtGui.QIcon.fromTheme("previous"),
                                "Previous Image", self.prev_image)
        self.tool_bar.addAction(QtGui.QIcon.fromTheme("next"), "Next Image",
                                self.next_image)

        self.tool_bar.addSeparator()

        self.tool_bar.addAction(QtGui.QIcon.fromTheme("image"), "Load Bundle",
                                self.load_bundle)
        self.tool_bar.addAction(QtGui.QIcon.fromTheme("image"),
                                "Complete Bundle", self.complete_bundle)

        self.image_progress_text = QLabel()
        self.update_image_progress_label()
        self.tool_bar.addWidget(self.image_progress_text)

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

        self.setLayout(layout)

    def update_image_progress_label(self):
        """
        Update the status message displayed in the image progress label.
        """
        if self.image_path is None:
            self.image_progress_text.setText("No Image Open")
            return

        curr_image_details = self.parse_curr_image()
        assert curr_image_details is not None
        _, name, ext = curr_image_details
        curr = int(name)

        # TODO: We should cache this
        images = glob.glob(
            os.path.join(os.path.dirname(self.image_path), "*.jpeg"))
        no_images = len(images)
        start_image = 2**32
        for image in images:
            image_name, _ = os.path.splitext(os.path.basename(image))
            no = int(image_name)

            if no <= start_image:
                start_image = no

        self.image_progress_text.setText(
            f"Image {curr - start_image + 1}/{no_images}")

    def load_bundle(self):
        """
        Open the load bundle dialog.
        """
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Bundle (*.zip)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            file_path = file_dialog.selectedFiles()[0]

            dirname = os.path.dirname(file_path)
            with zipfile.ZipFile(file_path) as zip:
                images: List[str] = zip.namelist()
                image_nos: List[int] = [2**32 for _ in images]
                for i, path in enumerate(images):
                    if path[-1] == '/':
                        continue  # skip directories

                    file = os.path.basename(path)
                    name, _ext = os.path.splitext(file)
                    image_nos[i] = int(name)
                first_image = images[np.argmin(image_nos)]

                zip.extractall(path=dirname)

            image_path = os.path.join(dirname, first_image)
            self.set_image(image_path)
            self.load_annotations(image_path)

    def complete_bundle(self):
        """
        Complete the currently loaded bundle.
        """
        if self.image_path is None:
            return

        self.save_annotations()

        bundle_path = os.path.dirname(self.image_path)
        bundle_name = os.path.basename(bundle_path)
        completed_bundle = bundle_path + "-complete.zip"

        images_and_labels = []
        images = glob.glob(os.path.join(bundle_path, "*.jpeg"))
        for image_path in images:
            file = os.path.basename(image_path)
            name, _ext = os.path.splitext(file)

            label_path = os.path.join(bundle_path, name + ".label")
            if os.path.exists(label_path):
                images_and_labels.append((label_path, image_path, name))
            else:
                # Maybe the image here is invalid? (In many of our bundles,
                # images were corrupted as we were saturating the pi's SD all
                # the way until shutdown. This meant that not all image file
                # contents could properly be flushed to disk.)
                try:
                    Image.open(image_path)

                    # If we got here, the image is valid, and the label should
                    # be created before continuing.
                    self.error_dialog.showMessage(
                        "WARNING: Missing label at %r" % label_path)
                    return
                except PIL.UnidentifiedImageError:
                    # The image is invalid. Don't care about it missing a
                    # .label file. Don't add it to the complete bundle either.
                    pass

        with zipfile.ZipFile(completed_bundle, "w") as zip:
            for label_path, image_path, name in images_and_labels:
                zip.write(image_path, bundle_name + "/" + name + ".jpeg")
                zip.write(label_path, bundle_name + "/" + name + ".label")

        self.message_dialog.setText("Bundle Completed")
        self.message_dialog.setDetailedText(
            "You can safely close the application, or begin labelling a new bundle"
        )
        self.message_dialog.show()

    def parse_curr_image(self) -> Optional[Tuple[str, int, str]]:
        """
        If there is an image currently open, return that image's dirname, name,
        extension as a tuple.

        Otherwise return None.
        """
        if self.image_path is None:
            return None

        dirname = os.path.dirname(self.image_path)
        basename = os.path.basename(self.image_path)
        name, ext = os.path.splitext(basename)

        try:
            idx = int(name)
        except ValueError:
            return None

        return dirname + "/", idx, ext

    def try_load_image(self, path: str) -> bool:
        """
        Returns True iff a new image is loaded.
        """
        if os.path.exists(path):
            self.set_image(path)
            self.load_annotations(path)

            return True

        return False

    def next_image(self):
        """
        Try to load the next image in the current folder.

        ie. If we are on image bundle/0.jpeg, we will try to load bundle/1.jpeg
        """
        path = self.parse_curr_image()
        if not path:
            return

        self.save_annotations()

        dirname, idx, ext = path
        newpath = dirname + str(idx + 1) + ext
        self.try_load_image(newpath)

    def prev_image(self):
        """
        Try to load the previous image in the current folder.

        ie. If we are on image bundle/1.jpeg, we will try to load bundle/0.jpeg
        """
        path = self.parse_curr_image()
        if not path:
            return

        self.save_annotations()

        dirname, idx, ext = path
        newpath = dirname + str(idx - 1) + ext
        self.try_load_image(newpath)

    def set_image(self, file_path: str):
        """
        Update the image using a given file path.
        """
        self.image_path = file_path
        self.annotations_path = self.get_annotations_path(file_path)
        pixmap = QPixmap(file_path)
        self.image_item.setPixmap(pixmap)

        self.update_image_progress_label()

    def get_annotations_path(self, image_path: str) -> str:
        """
        Extract the annotation JSON file path from an image path.
        """
        base_path, ext = os.path.splitext(image_path)
        return base_path + '.label'

    def load_annotations(self, image_path: str):
        """
        Try to load annotations for a given image.
        """
        annotations_path = self.get_annotations_path(image_path)
        if os.path.exists(annotations_path):
            for old_rect_item in self.rect_items:
                self.scene.removeItem(old_rect_item)
            self.rect_items.clear()

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
        """
        Save any annotations for the current image.
        """
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
        """
        Add a bounding box over the current image.
        """
        if self.image_path is not None:
            rect_item = DraggableRectItem(0, 0, 100, 100, title="orange")
            rect_item.setPos(50, 50)
            self.scene.addItem(rect_item)
            self.rect_items.append(rect_item)

    def keyPressEvent(self, event: Optional[QtGui.QKeyEvent]):
        """
        Dispatch various key press shortcuts.
        """
        if not event:
            return

        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected_boxes()

    def delete_selected_boxes(self):
        """
        Remove all selected annotation boxes.
        """
        for rect_item in self.scene.selectedItems():
            if isinstance(rect_item, DraggableRectItem):
                self.scene.removeItem(rect_item)
                self.rect_items.remove(rect_item)

    def zoom_in(self):
        """
        Zoom in by an increment of 10%.
        """
        self.view.scale(1.1, 1.1)

    def zoom_out(self):
        """
        Zoom out by an increment of 10%.
        """
        self.view.scale(0.9, 0.9)

    def pinch_trigger(self, gesture):
        """
        Adjust the scale factor based on the pinch gesture
        """
        zoom_factor = gesture.scaleFactor()
        self.view.setTransform(self.view.transform().scale(
            zoom_factor, zoom_factor))

    def eventFilter(self, source: Optional[QObject],
                    event: Optional[QEvent]) -> bool:
        """
        Manage gesture events. Used for, eg. zooming in the image.
        """
        if event and event.type() == QEvent.Type.Gesture:
            assert isinstance(event, QGestureEvent)
            for gesture in event.gestures():
                if gesture.state(
                ) == Qt.GestureState.GestureUpdated and isinstance(
                        gesture, QPinchGesture):
                    self.pinch_trigger(gesture)

        elif event and event.type() == QEvent.Type.ShortcutOverride:
            assert isinstance(event, QKeyEvent)

            # Some keys, ie. the arrow keys, are not sent to keyPressEvent. So
            # we have to handle them in an event filter.
            if event.key() == Qt.Key.Key_Left:
                self.prev_image()
            elif event.key() == Qt.Key.Key_Right:
                self.next_image()

        return super().eventFilter(source, event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    window.setGeometry(100, 100, 800, 600)
    window.setWindowTitle('Bounding Box Annotation Tool')
    window.show()
    sys.exit(app.exec())
