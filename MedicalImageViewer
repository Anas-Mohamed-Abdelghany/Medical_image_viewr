import sys
from functools import partial
import cv2
import numpy as np
import SimpleITK as sitk
import vtk
from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QIcon
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QGridLayout,
    QWidget, QSlider, QFrame, QAction, QSizePolicy
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util import numpy_support


class ImageLabel(QLabel):
    def __init__(self, viewer, index, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.viewer = viewer
        self.index = index
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.last_point = None
        self.zoom_factor = 1.0
        self.pan_start = None
        self.dragging_point = False
        self.marked_point = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Convert click position to image coordinates
            pos = self.convert_screen_to_image_coords(event.pos())
            if pos:
                self.last_point = pos
                self.dragging_point = True
                self.viewer.mark_point(self.index, pos)
        elif event.button() == Qt.RightButton:
            self.last_point = event.pos()
        elif event.button() == Qt.MiddleButton:
            self.pan_start = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging_point and event.buttons() & Qt.LeftButton:
            pos = self.convert_screen_to_image_coords(event.pos())
            if pos:
                self.viewer.mark_point(self.index, pos)
        elif event.buttons() & Qt.RightButton:
            dx = event.x() - self.last_point.x()
            dy = event.y() - self.last_point.y()
            self.viewer.adjust_brightness_contrast(self.index, dx, dy)
            self.last_point = event.pos()
        elif event.buttons() & Qt.MiddleButton and self.pan_start:
            delta = event.pos() - self.pan_start
            self.viewer.pan_image(self.index, delta)
            self.pan_start = event.pos()
        
        # Update crosshair even when not dragging
        pos = self.convert_screen_to_image_coords(event.pos())
        if pos:
            self.viewer.update_crosshair(self.index, pos)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging_point = False
        elif event.button() == Qt.MiddleButton:
            self.pan_start = None

    def convert_screen_to_image_coords(self, pos):
        """Convert screen coordinates to image coordinates with bounds checking"""
        if not self.pixmap():
            return None

        # Get the dimensions
        label_width = self.width()
        label_height = self.height()
        pixmap_width = self.pixmap().width()
        pixmap_height = self.pixmap().height()

        # Calculate ratios
        x_ratio = pixmap_width / label_width
        y_ratio = pixmap_height / label_height

        # Convert to image coordinates
        x = int(pos.x() * x_ratio)
        y = int(pos.y() * y_ratio)

        # Get image dimensions based on view
        if not hasattr(self.viewer, 'image_array'):
            return None
        
        depth, height, width = self.viewer.image_array.shape

        # Clamp coordinates based on view
        if self.index == 0:  # Axial
            x = min(max(0, x), width - 1)
            y = min(max(0, y), height - 1)
        elif self.index == 1:  # Sagittal
            x = min(max(0, x), depth - 1)
            y = min(max(0, y), height - 1)
        else:  # Coronal
            x = min(max(0, x), width - 1)
            y = min(max(0, y), depth - 1)

        return QPoint(x, y)

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ITK-SNAP-like 3D Medical Imaging Viewer")
        self.setGeometry(100, 100, 1600, 900)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QGridLayout(self.main_widget)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(10, 10, 10, 10)

        self.image_labels = [ImageLabel(self, i) for i in range(3)]
        self.sliders_h = [QSlider(Qt.Horizontal) for _ in range(3)]
        self.sliders_v = [QSlider(Qt.Vertical) for _ in range(3)]
        self.images = [None] * 3
        self.current_slice = [(0, 0), (0, 0), (0, 0)]
        self.image_size = (350, 350)

        self.notification_label = QLabel("")
        self.notification_label.setStyleSheet("color: green; font-size: 14px;")
        self.notification_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.notification_label, 0, 0, 1, 3, alignment=Qt.AlignCenter)

        # Initialize brightness and contrast sliders
        self.brightness_sliders = [QSlider(Qt.Horizontal) for _ in range(3)]
        self.contrast_sliders = [QSlider(Qt.Horizontal) for _ in range(3)]

        for i in range(3):
            self.create_view_section(i)

        self.create_menus()

        # 3D Volume Rendering
        self.vtk_widget = QVTKRenderWindowInteractor(self.main_widget)
        self.layout.addWidget(self.vtk_widget, 1, 2, 3, 1)
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # Intensity windowing
        self.window_level_dialog = None

        # Drawing attributes
        self.drawing_color = (255, 0, 0)  # Red
        self.drawing_thickness = 20  # Increased thickness for better visibility
        # Cache for VTK image to avoid redundant conversions
        self.vtk_image_cached = None

        # Initialize coronal slice variable
        self.coronalSlice = 0

    def create_view_section(self, index):
        section_widget = QFrame()
        section_widget.setFrameShape(QFrame.StyledPanel)
        section_widget.setFrameShadow(QFrame.Raised)

        section_layout = QGridLayout(section_widget)
        section_layout.setSpacing(5)

        image_frame = QFrame()
        image_frame.setFixedSize(*self.image_size)
        image_frame.setStyleSheet("background-color: black; border: 1px solid #444444;")
        image_layout = QGridLayout(image_frame)
        image_layout.setSpacing(0)

        slider_v = self.sliders_v[index]
        slider_v.setMinimum(0)
        slider_v.setMaximum(0)
        slider_v.setValue(0)
        slider_v.setTickPosition(QSlider.TicksRight)
        slider_v.setTickInterval(1)
        slider_v.valueChanged.connect(partial(self.update_image_slice, index))
        image_layout.addWidget(slider_v, 0, 0, 3, 1)

        slider_h = self.sliders_h[index]
        slider_h.setMinimum(0)
        slider_h.setMaximum(0)
        slider_h.setValue(0)
        slider_h.setTickPosition(QSlider.TicksBelow)
        slider_h.setTickInterval(1)
        slider_h.valueChanged.connect(partial(self.update_image_slice, index))
        image_layout.addWidget(slider_h, 2, 1, 1, 3)

        image_label = self.image_labels[index]
        image_label.setFixedSize(*self.image_size)
        image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        image_layout.addWidget(image_label, 0, 1, 2, 3)

        # Move brightness and contrast sliders under the image label
        brightness_slider = self.brightness_sliders[index]
        brightness_slider.setMinimum(-100)
        brightness_slider.setMaximum(100)
        brightness_slider.setValue(0)
        brightness_slider.setTickPosition(QSlider.TicksBelow)
        brightness_slider.setTickInterval(10)
        brightness_slider.valueChanged.connect(partial(self.update_image_slice, index))
        image_layout.addWidget(brightness_slider, 3, 1, 1, 3)

        contrast_slider = self.contrast_sliders[index]
        contrast_slider.setMinimum(-100)
        contrast_slider.setMaximum(100)
        contrast_slider.setValue(0)
        contrast_slider.setTickPosition(QSlider.TicksBelow)
        contrast_slider.setTickInterval(10)
        contrast_slider.valueChanged.connect(partial(self.update_image_slice, index))
        image_layout.addWidget(contrast_slider, 4, 1, 1, 3)

        # Add zoom slider below brightness and contrast sliders
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)  # 10%
        self.zoom_slider.setMaximum(200)  # 200%
        self.zoom_slider.setValue(100)  # Default at 100%
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(10)
        self.zoom_slider.valueChanged.connect(lambda value: self.updateZoom(index, value))
        image_layout.addWidget(self.zoom_slider, 5, 1, 1, 3)

        section_layout.addWidget(image_frame, 0, 0, 1, 3)

        title_label = QLabel(self.get_section_title(index))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 12px; color: white;")
        section_layout.addWidget(title_label, 2, 0, 1, 3)

        section_layout.setRowStretch(3, 1)

        section_widget.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 5px;
            }
            QSlider::groove:horizontal {
                background: #4a4a4a;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #8f8f8f;
                width: 18px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 9px;
            }
            QSlider::groove:vertical {
                background: #4a4a4a;
                width: 8px;
                border-radius: 4px;
            }
            QSlider::handle:vertical {
                background: #8f8f8f;
                height: 18px;
                margin-left: -5px;
                margin-right: -5px;
                border-radius: 9px;
            }
        """)

        row = 1 + (index // 2) * 2
        col = index % 2
        self.layout.addWidget(section_widget, row, col)

    def get_section_title(self, index):
        titles = ["Axial", "Sagittal", "Coronal"]
        return titles[index] if index < 3 else "Unknown"

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Image Files (*.nii *.nii.gz *.mha *.mhd *.dcm *.jpg *.jpeg *.png *.bmp *.tiff *.tif)"
        )
        if file_name:
            try:
                image = sitk.ReadImage(file_name)
                self.image_array = sitk.GetArrayFromImage(image)

                print(f"Image loaded: shape {self.image_array.shape}")

                if self.image_array.size > 0:
                    for label in self.image_labels:
                        label.clear()

                    self.notification_label.setText("Image loaded successfully.")

                    self.update_image_slices()
                    self.render_3d_volume()
                else:
                    self.notification_label.setText("Error: Image data is invalid or could not be loaded.")
                    print("Error: Image data is invalid or could not be loaded.")
            except Exception as e:
                self.notification_label.setText("Error: Failed to load image.")
                print(f"Error loading image: {e}")

    def update_image_slices(self):
        if not hasattr(self, 'image_array'):
            return

        depth, height, width = self.image_array.shape

        for i in range(3):
            if i == 0:  # Axial
                self.sliders_v[i].setMaximum(depth - 1)
                self.sliders_h[i].setMaximum(width - 1)
                mid_slice_v, mid_slice_h = depth // 2, width // 2
            elif i == 1:  # Sagittal
                self.sliders_v[i].setMaximum(height - 1)
                self.sliders_h[i].setMaximum(depth - 1)
                mid_slice_v, mid_slice_h = height // 2, depth // 2
            else:  # Coronal
                self.sliders_v[i].setMaximum(height - 1)
                self.sliders_h[i].setMaximum(width - 1)
                mid_slice_v, mid_slice_h = height // 2, width // 2

            self.sliders_v[i].blockSignals(True)
            self.sliders_h[i].blockSignals(True)

            self.sliders_v[i].setValue(mid_slice_v)
            self.sliders_h[i].setValue(mid_slice_h)
            self.current_slice[i] = (mid_slice_v, mid_slice_h)

            self.sliders_v[i].blockSignals(False)
            self.sliders_h[i].blockSignals(False)

            self.update_image_slice(i)

        # Set the coronal slider range
        self.sliders_h[2].setMinimum(0)
        self.sliders_h[2].setMaximum(depth - 1)  # Assuming coronal slices are along the z-axis

    def update_image_slice(self, index):
        if not hasattr(self, 'image_array'):
            return

        slice_index_v = self.sliders_v[index].value()
        slice_index_h = self.sliders_h[index].value()
        self.current_slice[index] = (slice_index_v, slice_index_h)

        # Ensure slice indices are within bounds
        if index == 0:  # Axial
            slice_img = self.image_array[slice_index_v, :, :]
        elif index == 1:  # Sagittal
            slice_img = self.image_array[:, slice_index_v, :]
        else:  # Coronal
            slice_img = self.image_array[:, :, slice_index_h]

        # Normalize the image using NumPy's efficient operations
        min_val, max_val = slice_img.min(), slice_img.max()
        if max_val > min_val:
            normalized_img = ((slice_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized_img = np.zeros_like(slice_img, dtype=np.uint8)

        # Adjust brightness and contrast
        brightness = self.brightness_sliders[index].value()
        contrast = self.contrast_sliders[index].value()
        normalized_img = cv2.convertScaleAbs(normalized_img, alpha=1 + contrast / 100, beta=brightness)

        # Convert to RGB for overlay
        color_img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2RGB)

        # Apply zoom
        zoom_factor = self.image_labels[index].zoom_factor
        if zoom_factor != 1.0:
            height, width = color_img.shape[:2]
            new_size = (int(width * zoom_factor), int(height * zoom_factor))
            color_img = cv2.resize(color_img, new_size, interpolation=cv2.INTER_LINEAR)

        # Draw crosshairs
        self.draw_crosshairs(color_img, index)

        # Convert to QImage and set pixmap
        q_img = QImage(color_img.data, color_img.shape[1], color_img.shape[0],
                      3 * color_img.shape[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_labels[index].setPixmap(pixmap.scaled(
            self.image_size[0], self.image_size[1],
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        # Apply zoom factor to the displayed pixmap
        self.apply_zoom_to_label(index)

    def apply_zoom_to_label(self, index):
        label = self.image_labels[index]
        pixmap = label.pixmap()
        if pixmap is not None:
            new_width = int(pixmap.width() * label.zoom_factor)
            new_height = int(pixmap.height() * label.zoom_factor)
            scaled_pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio)
            label.setPixmap(scaled_pixmap)

    def draw_crosshairs(self, image, index):
        """Draw crosshairs and point marker on the image"""
        if not hasattr(self, 'image_array'):
            return

        depth, height, width = self.image_array.shape
        
        # Draw green crosshairs
        color = (0, 255, 0)  # Green
        thickness = 1
        
        # Get current slice positions
        if index == 0:  # Axial
            x = self.current_slice[2][1]  # Sagittal position
            y = self.current_slice[2][0]  # Coronal position
        elif index == 1:  # Sagittal
            x = self.current_slice[0][0]  # Axial position
            y = self.current_slice[2][0]  # Coronal position
        else:  # Coronal
            x = self.current_slice[2][1]  # Sagittal position
            y = self.current_slice[0][0]  # Axial position

        # Draw crosshairs
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            cv2.line(image, (0, y), (image.shape[1], y), color, thickness)
            cv2.line(image, (x, 0), (x, image.shape[0]), color, thickness)
            
            # Draw red point at intersection
            point_color = (255, 0, 0)  # Red
            point_size = 3
            cv2.circle(image, (x, y), point_size, point_color, -1)  # -1 fills the circle

    def update_crosshair(self, index, pos):
        label = self.image_labels[index]
        pixmap = label.pixmap()
        if pixmap:
            label_width = label.width()
            label_height = label.height()
            pixmap_width = pixmap.width()
            pixmap_height = pixmap.height()

            x_ratio = pixmap_width / label_width
            y_ratio = pixmap_height / label_height

            x = int(pos.x() * x_ratio)
            y = int(pos.y() * y_ratio)

            # Clamp x and y to prevent out-of-bounds access
            if index == 0:  # Axial
                max_slice_h = self.sliders_h[0].maximum()
                max_slice_v = self.sliders_v[0].maximum()
                x = min(x, max_slice_h)
                y = min(y, max_slice_v)
                # y corresponds to Sagittal (V slider index 1)
                # x corresponds to Coronal (H slider index 2)
                self.current_slice[1] = (y, self.current_slice[1][1])  # Sagittal: Axial index
                self.current_slice[2] = (self.current_slice[2][0], x)  # Coronal: Axial index
            elif index == 1:  # Sagittal
                max_slice_h = self.sliders_h[1].maximum()
                max_slice_v = self.sliders_v[1].maximum()
                x = min(x, max_slice_h)
                y = min(y, max_slice_v)
                # y corresponds to Axial (V slider index 0)
                # x corresponds to Coronal (H slider index 2)
                self.current_slice[0] = (y, self.current_slice[0][1])  # Axial: Sagittal index
                self.current_slice[2] = (self.current_slice[2][0], x)  # Coronal: Sagittal index
            else:  # Coronal
                max_slice_h = self.sliders_h[2].maximum()
                max_slice_v = self.sliders_v[2].maximum()
                x = min(x, max_slice_h)
                y = min(y, max_slice_v)
                # y corresponds to Axial (V slider index 0)
                # x corresponds to Sagittal (V slider index 1)  # Corrected logic for Coronal view
                self.current_slice[0] = (y, self.current_slice[0][1])  # Axial: Coronal index
                self.current_slice[1] = (x, self.current_slice[1][1])  # Sagittal: Coronal index

            # Update sliders for the other two views
            self.sync_sliders(index)

            # Update images for all views
            for i in range(3):
                self.update_image_slice(i)

    def sync_sliders(self, clicked_index):
        """
        Synchronize the sliders based on the current_slice after a click.
        """
        if not hasattr(self, 'image_array'):
            return

        for i in range(3):
            if i == clicked_index:
                continue  # Skip the clicked view

            slice_v, slice_h = self.current_slice[i]
            self.sliders_v[i].blockSignals(True)
            self.sliders_h[i].blockSignals(True)

            # Set slider values based on the updated current_slice coordinates
            self.sliders_v[i].setValue(slice_v)
            self.sliders_h[i].setValue(slice_h)

            self.sliders_v[i].blockSignals(False)
            self.sliders_h[i].blockSignals(False)

    def mark_point(self, index, pos):
        """Update point marker position with proper bounds checking"""
        if not hasattr(self, 'image_array'):
            return

        depth, height, width = self.image_array.shape
        
        # Get current positions
        axial_z = self.sliders_v[0].value()
        sagittal_x = self.sliders_v[1].value()
        coronal_y = self.sliders_v[2].value()

        # Update positions based on view
        if index == 0:  # Axial
            sagittal_x = min(max(0, pos.x()), width - 1)
            coronal_y = min(max(0, pos.y()), height - 1)
        elif index == 1:  # Sagittal
            axial_z = min(max(0, pos.x()), depth - 1)
            coronal_y = min(max(0, pos.y()), height - 1)
        else:  # Coronal
            sagittal_x = min(max(0, pos.x()), width - 1)
            axial_z = min(max(0, pos.y()), depth - 1)

        # Update current slices with bounded values
        self.current_slice[0] = (axial_z, sagittal_x)  # Axial (z, x)
        self.current_slice[1] = (coronal_y, axial_z)   # Sagittal (y, z)
        self.current_slice[2] = (coronal_y, sagittal_x) # Coronal (y, x)

        # Update sliders
        self.sync_sliders(index)

        # Update all views
        for i in range(3):
            self.update_image_slice(i)

    def render_3d_volume(self):
        if not hasattr(self, 'image_array') or self.image_array is None:
            return

        if self.vtk_image_cached is None:
            # Create VTK image data only once and cache it
            vtk_image = vtk.vtkImageData()
            vtk_image.SetDimensions(self.image_array.shape[::-1])
            vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

            # Normalize and flip the image array
            min_val, max_val = self.image_array.min(), self.image_array.max()
            if max_val > min_val:
                normalized_array = ((self.image_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized_array = np.zeros_like(self.image_array, dtype=np.uint8)

            flipped_array = np.flip(normalized_array, axis=0)
            vtk_array = numpy_support.numpy_to_vtk(flipped_array.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            vtk_image.GetPointData().SetScalars(vtk_array)

            self.vtk_image_cached = vtk_image
        else:
            # Update existing VTK image data if segmentation mask changes
            vtk_array = numpy_support.numpy_to_vtk(
                np.flip(self.image_array, axis=0).ravel(),
                deep=True,
                array_type=vtk.VTK_UNSIGNED_CHAR
            )
            self.vtk_image_cached.GetPointData().SetScalars(vtk_array)

        # Clear existing actors
        self.renderer.RemoveAllViewProps()

        # Create volume mapper and property
        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputData(self.vtk_image_cached)

        volume_property = vtk.vtkVolumeProperty()
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()

        # Create and set color transfer function
        color_transfer_function = vtk.vtkColorTransferFunction()
        color_transfer_function.AddRGBPoint(0, 0.0, 0.0, 0.0)
        color_transfer_function.AddRGBPoint(128, 0.5, 0.5, 0.5)
        color_transfer_function.AddRGBPoint(255, 1.0, 1.0, 1.0)
        volume_property.SetColor(color_transfer_function)

        # Create and set opacity transfer function
        opacity_transfer_function = vtk.vtkPiecewiseFunction()
        opacity_transfer_function.AddPoint(0, 0.0)
        opacity_transfer_function.AddPoint(128, 0.5)
        opacity_transfer_function.AddPoint(255, 1.0)
        volume_property.SetScalarOpacity(opacity_transfer_function)

        # Create volume
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        # Add volume to renderer
        self.renderer.AddVolume(volume)

        # Reset camera and render
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def create_menus(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        open_action = QAction('Open', self)
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)

    def updateZoom(self, index, value):
        self.image_labels[index].zoom_factor = value / 100  # Convert slider value to a zoom factor
        self.update_image_slice(index)  # Redraw the view with the new zoom level

    def adjust_brightness_contrast(self, index, dx, dy):
        brightness_slider = self.brightness_sliders[index]
        contrast_slider = self.contrast_sliders[index]

        brightness_slider.setValue(brightness_slider.value() + dx)
        contrast_slider.setValue(contrast_slider.value() + dy)

    def pan_image(self, index, delta):
        # Placeholder for pan image logic
        pass

    def updateCoronalView(self, value):
        self.coronalSlice = value
        self.renderCoronalView()  # Make sure this method re-renders the view

    def renderCoronalView(self):
        coronal_image = self.image_array[:, :, self.coronalSlice]  # Assuming coronal slices are along the z-axis
        # Update the view with the extracted slice
        self.displayCoronalImage(coronal_image)

    def displayCoronalImage(self, image):
        # Convert the coronal image to a format suitable for display
        min_val, max_val = image.min(), image.max()
        if max_val > min_val:
            normalized_img = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized_img = np.zeros_like(image, dtype=np.uint8)

        # Convert to RGB for overlay
        color_img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2RGB)

        # Convert to QImage and set pixmap
        q_img = QImage(color_img.data, color_img.shape[1], color_img.shape[0],
                      3 * color_img.shape[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_labels[2].setPixmap(pixmap.scaled(
            self.image_size[0], self.image_size[1],
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
