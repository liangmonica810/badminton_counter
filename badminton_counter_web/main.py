import os
import sys
import subprocess
import re
import time
import bisect
from typing import List, Optional, Tuple
from PyQt5 import QtCore, QtGui, QtWidgets


class InferenceWorker(QtCore.QThread):
    """背景推論工作執行緒"""
    finished = QtCore.pyqtSignal(str, str)  # 成功時發送 (images_dir, labels_dir)
    error = QtCore.pyqtSignal(str)  # 失敗時發送錯誤訊息
    output = QtCore.pyqtSignal(str)  # 發送輸出訊息
    progress = QtCore.pyqtSignal(int, int)  # 發送進度 (current, total)
    
    def __init__(self, video_path: str, output_root: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.output_root = output_root
    
    def run(self):
        """在背景執行推論"""
        try:
            images_dir, labels_dir = self._run_yolo_inference(self.video_path, self.output_root)
            self.finished.emit(images_dir, labels_dir)
        except Exception as e:
            self.error.emit(str(e))
    
    def _run_yolo_inference(self, video_path: str, output_root: str):
        """使用外部 yolo_predict_new.py 腳本執行 YOLOv8 推論。"""
        
        # 準備輸出目錄
        images_dir, labels_dir = self._prepare_output_dirs(output_root)
        
        # 取得推論腳本路徑
        project_root = os.path.dirname(__file__)
        script_path = os.path.join(project_root, "inference_code", "yolo_predict_new.py")
        if not os.path.exists(script_path):
            raise RuntimeError(f"找不到推論腳本：{script_path}")
        
        # 直接執行腳本，透過命令列參數傳遞影片路徑和輸出目錄
        # 設定環境變數確保使用 UTF-8 編碼（解決 Windows 編碼問題）
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # 使用 Popen 即時讀取輸出並打印到 console
        process = subprocess.Popen(
            [sys.executable, script_path, video_path, output_root],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=os.path.join(project_root, "inference_code"),
            env=env,
            bufsize=1,
            universal_newlines=True
        )
        
        # 即時讀取並打印輸出
        output_lines = []
        for line in process.stdout:
            if line:
                line = line.strip()
                print(line)  # 打印到 console
                output_lines.append(line)
                # 發送輸出訊息
                self.output.emit(line)
                # 解析 tqdm 進度條並發送進度
                progress_info = self._parse_tqdm_progress(line)
                if progress_info:
                    current, total = progress_info
                    self.progress.emit(current, total)
        
        process.wait()
        
        if process.returncode != 0:
            error_msg = '\n'.join(output_lines[-10:])  # 取最後 10 行作為錯誤訊息
            raise RuntimeError(f"推論腳本執行失敗（返回碼: {process.returncode}）：\n{error_msg}")
        
        # 檢查輸出是否正確產生
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            raise RuntimeError("推論完成但未找到輸出目錄")
        
        return images_dir, labels_dir
    
    def _parse_tqdm_progress(self, line: str) -> Optional[tuple]:
        """
        解析 tqdm 進度條輸出，返回 (current, total)
        例如: "Processing video: 50%|████████  | 50/100 [00:10<00:10, 5.00it/s]"
        返回: (50, 100)
        """
        # tqdm 格式: "desc: percentage|bar| current/total [time<time, speed]"
        # 匹配模式: 數字/數字 (例如: 50/100)
        match = re.search(r'(\d+)/(\d+)', line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            return (current, total)
        return None
    
    def _prepare_output_dirs(self, output_root: str):
        """準備輸出目錄結構。"""
        os.makedirs(output_root, exist_ok=True)
        images_dir = os.path.join(output_root, "images")
        labels_dir = os.path.join(output_root, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        return images_dir, labels_dir


class BoundingBox:
    """單一框標註容器（以影像座標記錄）。

    屬性:
        class_name: 類別名稱
        rect: Qt 影像座標中的矩形區域 (QRectF)
    """
    # Rect 是「rectangle（矩形）」的縮寫。Qt 依慣例用 QRect 表示整數矩形、QRectF 表示浮點矩形，F 代表 floating（浮點）。
    def __init__(self, cls_name: str, rect: QtCore.QRectF):
        self.class_name = cls_name
        self.rect = QtCore.QRectF(rect)

    # to_yolo 是「to YOLO format（轉成 YOLO 格式）」的縮寫。
    def to_yolo(self, img_w: int, img_h: int) -> str:
        """輸出 YOLO 格式 (cx, cy, w, h 皆為 0~1 正規化)。"""
        cx = (self.rect.left() + self.rect.right()) / 2.0 / img_w #  計算中心 x（cx）：(left+right)/2，再除以影像寬 img_w 正規化到 0~1
        cy = (self.rect.top() + self.rect.bottom()) / 2.0 / img_h
        w = self.rect.width() / img_w #實際像素寬度除以影像寬 img_w → 正規化
        h = self.rect.height() / img_h
        return f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    # 將 YOLO 正規化字串還原為影像座標 QRectF（只處理幾何，不含類別）
    @staticmethod
    def from_yolo(norm_str: str, img_w: int, img_h: int) -> QtCore.QRectF:
        """由 YOLO 格式 (cx cy w h) 產生影像座標矩形。"""
        cx, cy, w, h = map(float, norm_str.split())
        rect_w = w * img_w
        rect_h = h * img_h
        left = cx * img_w - rect_w / 2.0
        top = cy * img_h - rect_h / 2.0
        return QtCore.QRectF(left, top, rect_w, rect_h)
    
    # BoundingBox 負責「單一框的幾何資料」與「YOLO幾何格式互轉」，互動與檔案處理由其他類別負責。

class ImageCanvas(QtWidgets.QLabel):
    """顯示影像與互動式框選的畫布元件。"""
    boxAdded = QtCore.pyqtSignal()
    boxDeleted = QtCore.pyqtSignal()
    selectionChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        """初始化畫布與狀態變數。"""
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setBackgroundRole(QtGui.QPalette.Base)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)
        
        # 設置畫布樣式
        self.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f5f9fd, stop:1 #e9f3fb);
                border: 2px solid #cfe3f3;
                border-radius: 12px;
            }
        """)

        self._image: Optional[QtGui.QImage] = None
        self._pixmap: Optional[QtGui.QPixmap] = None

        self.boxes: List[BoundingBox] = []
        self.active_box_index: int = -1
        self.drawing: bool = False
        self.start_pos: Optional[QtCore.QPointF] = None
        self.current_rect: Optional[QtCore.QRectF] = None
        self.current_class: str = "object"

        # 縮放與平移狀態
        self._zoom_level: float = 1.0        # 縮放倍率 (1.0 = 原始大小)
        self._pan_offset = QtCore.QPointF(0, 0)  # 平移偏移量
        self._panning: bool = False          # 是否正在平移
        self._pan_start: Optional[QtCore.QPoint] = None  # 平移起始點
        self._last_middle_click_time: float = 0  # 上次中鍵點擊時間（用於雙擊檢測）

        # 縮放參數常數
        self.ZOOM_MIN = 0.5   # 最小縮放 50%
        self.ZOOM_MAX = 10.0  # 最大縮放 1000%
        self.ZOOM_STEP = 1.15 # 每次縮放倍率

        # 調整大小狀態
        self._resizing: bool = False  # 是否正在調整大小
        self._resize_handle: Optional[str] = None  # 當前拖曳的手柄 (tl/t/tr/l/r/bl/b/br)
        self._resize_start_rect: Optional[QtCore.QRectF] = None  # 調整前的原始矩形
        self._resize_start_pos: Optional[QtCore.QPointF] = None  # 調整開始時的滑鼠位置
        self.HANDLE_SIZE = 8  # 手柄大小（像素）

        # 移動框框狀態
        self._moving: bool = False  # 是否正在移動框框
        self._move_start_pos: Optional[QtCore.QPointF] = None  # 移動開始時的滑鼠位置（影像座標）
        self._move_start_rect: Optional[QtCore.QRectF] = None  # 移動前的原始矩形

    def set_image(self, image_path: str):
        """載入影像並重置當前標註狀態。"""
        image = QtGui.QImage(image_path)
        if image.isNull():
            return
        self._image = image
        self._pixmap = QtGui.QPixmap.fromImage(image)
        self.boxes.clear()
        self.active_box_index = -1
        # 重置縮放與平移
        self._zoom_level = 1.0
        self._pan_offset = QtCore.QPointF(0, 0)
        self.update()

    def set_current_class(self, class_name: str):
        """設定目前繪製框的類別名稱。"""
        self.current_class = class_name

    def sizeHint(self) -> QtCore.QSize:
        """預設視窗大小建議。"""
        return QtCore.QSize(800, 600)

    def paintEvent(self, event: QtGui.QPaintEvent):
        """繪製影像、既有框及目前正在繪製的框。"""
        super().paintEvent(event)
        if self._pixmap is None:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # 將影像依比例置中縮放顯示
        target_rect = self._target_rect()
        # 使用 QRectF 版本的 drawPixmap，需提供 sourceRect
        source_rect = QtCore.QRectF(self._pixmap.rect())
        painter.drawPixmap(target_rect, self._pixmap, source_rect)

        # 繪製已存在的框（選中框以橘色表示）
        for i, box in enumerate(self.boxes):
            is_active = (i == self.active_box_index)
            color = LabelingWindow.BOX_COLOR_NORMAL if not is_active else LabelingWindow.BOX_COLOR_ACTIVE
            pen = QtGui.QPen(color, LabelingWindow.PEN_WIDTH)
            painter.setPen(pen)
            widget_rect = self._image_to_widget(box.rect)
            painter.drawRect(widget_rect)

            # 在框左上方顯示類別文字背景與文字
            label = box.class_name
            if label:
                text_rect = QtCore.QRectF(widget_rect.left(), widget_rect.top() - LabelingWindow.LABEL_TEXT_HEIGHT, 
                                         LabelingWindow.LABEL_TEXT_WIDTH, LabelingWindow.LABEL_TEXT_HEIGHT)
                painter.fillRect(text_rect, LabelingWindow.LABEL_BG_COLOR)
                painter.setPen(QtCore.Qt.white)
                painter.drawText(text_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, label)

            # 繪製選中框的調整手柄
            if is_active:
                handles = self._get_resize_handles(widget_rect)
                painter.setPen(QtGui.QPen(QtCore.Qt.white, 1))
                painter.setBrush(QtGui.QBrush(LabelingWindow.BOX_COLOR_ACTIVE))
                for handle_rect in handles.values():
                    painter.drawRect(handle_rect)

        # 繪製正在拖曳中的臨時矩形（虛線）
        if self.drawing and self.current_rect is not None:
            painter.setPen(QtGui.QPen(LabelingWindow.BOX_COLOR_DRAWING, LabelingWindow.PEN_WIDTH, QtCore.Qt.DashLine))
            painter.drawRect(self._image_to_widget(self.current_rect))

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        """滑鼠按下：統一左鍵操作（選取/調整/移動/繪製）、Ctrl+左鍵平移、中鍵重置。"""
        self.setFocus(QtCore.Qt.MouseFocusReason)
        if self._image is None:
            return
        if event.button() == QtCore.Qt.LeftButton:
            # 檢查是否按住 Ctrl 鍵 → 平移模式
            if event.modifiers() & QtCore.Qt.ControlModifier:
                self._panning = True
                self._pan_start = event.pos()
                return
            
            # 1. 檢查是否點擊在選中框的調整手柄上 → 調整大小
            handle = self._hit_test_handle(event.pos())
            if handle:
                self._resizing = True
                self._resize_handle = handle
                self._resize_start_rect = QtCore.QRectF(self.boxes[self.active_box_index].rect)
                self._resize_start_pos = self._widget_to_image(event.pos())
                return
            
            # 2. 檢查是否點擊在某個框框內部
            clicked_index = self._hit_test(event.pos())
            if clicked_index != -1:
                # 點擊在框框上
                if clicked_index == self.active_box_index:
                    # 點擊在已選中的框框內部 → 移動模式
                    self._moving = True
                    self._move_start_pos = self._widget_to_image(event.pos())
                    self._move_start_rect = QtCore.QRectF(self.boxes[self.active_box_index].rect)
                else:
                    # 點擊在其他框框上 → 選取該框
                    self.active_box_index = clicked_index
                    self.selectionChanged.emit(self.active_box_index)
                    self.update()
                return
            
            # 3. 點擊在空白區域 → 取消選取並開始繪製新框
            self.active_box_index = -1
            self.selectionChanged.emit(self.active_box_index)
            img_pt = self._widget_to_image(event.pos())
            self.drawing = True
            self.start_pos = img_pt
            self.current_rect = QtCore.QRectF(img_pt, img_pt)
            self.update()
            
        elif event.button() == QtCore.Qt.RightButton:
            # 右鍵：可用於未來的右鍵選單功能
            pass
            
        elif event.button() == QtCore.Qt.MiddleButton:
            # 中鍵：雙擊重置縮放
            current_time = time.time()
            if current_time - self._last_middle_click_time < 0.3:
                self._reset_zoom()
            self._last_middle_click_time = current_time

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        """滑鼠移動：平移、調整大小、移動框框、或繪製新框。"""
        if self._image is None:
            return
        
        # 處理平移模式（Ctrl+左鍵拖曳）
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_offset.setX(self._pan_offset.x() + delta.x())
            self._pan_offset.setY(self._pan_offset.y() + delta.y())
            self._pan_start = event.pos()
            self.update()
            return
        
        # 處理調整大小模式
        if self._resizing and self._resize_handle and self._resize_start_rect and self._resize_start_pos:
            current_img_pos = self._widget_to_image(event.pos())
            delta = current_img_pos - self._resize_start_pos
            new_rect = self._apply_resize(self._resize_handle, self._resize_start_rect, delta)
            self.boxes[self.active_box_index].rect = new_rect
            self.update()
            return
        
        # 處理移動框框模式
        if self._moving and self._move_start_pos and self._move_start_rect:
            current_img_pos = self._widget_to_image(event.pos())
            delta = current_img_pos - self._move_start_pos
            new_rect = self._move_start_rect.translated(delta.x(), delta.y())
            # 確保框框不會移出影像範圍
            new_rect = self._clamp_rect_to_image(new_rect)
            self.boxes[self.active_box_index].rect = new_rect
            self.update()
            return
        
        # 處理繪製模式
        if self.drawing and self.start_pos is not None:
            img_pt = self._widget_to_image(event.pos())
            rect = QtCore.QRectF(self.start_pos, img_pt).normalized()
            rect = self._clamped_to_image(rect)
            self.current_rect = rect
            self.update()
            return
        
        # 閒置狀態：更新滑鼠游標
        self._update_cursor(event.pos())

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        """滑鼠放開：結束各種操作模式，保持選取狀態。"""
        if self._image is None:
            return
        if event.button() == QtCore.Qt.LeftButton:
            # 結束平移模式
            if self._panning:
                self._panning = False
                self._pan_start = None
            # 結束調整大小模式（保持選取）
            elif self._resizing:
                self._resizing = False
                self._resize_handle = None
                self._resize_start_rect = None
                self._resize_start_pos = None
                self.boxAdded.emit()  # 通知框已修改
                self.update()
            # 結束移動框框模式（保持選取）
            elif self._moving:
                self._moving = False
                self._move_start_pos = None
                self._move_start_rect = None
                self.boxAdded.emit()  # 通知框已修改
                self.update()
            # 結束繪製模式
            elif self.drawing and self.current_rect is not None:
                if (self.current_rect.width() > LabelingWindow.MIN_BOX_SIZE and 
                    self.current_rect.height() > LabelingWindow.MIN_BOX_SIZE):
                    self.boxes.append(BoundingBox(self.current_class, self.current_rect))
                    self.active_box_index = len(self.boxes) - 1
                    self.selectionChanged.emit(self.active_box_index)
                    self.boxAdded.emit()
                self.drawing = False
                self.start_pos = None
                self.current_rect = None
                self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """鍵盤事件：Delete 刪除目前選中框。"""
        key = event.key()
        if key == QtCore.Qt.Key_Delete and 0 <= self.active_box_index < len(self.boxes):
            del self.boxes[self.active_box_index]
            self.active_box_index = min(self.active_box_index, len(self.boxes) - 1)
            self.boxDeleted.emit()  # 通知主視窗框已被刪除
            self.selectionChanged.emit(self.active_box_index)
            self.update()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        """處理滾輪事件：Ctrl+滾輪進行縮放（以滑鼠位置為中心）"""
        if self._image is None:
            return
        
        # 檢查是否按住 Ctrl 鍵
        if event.modifiers() & QtCore.Qt.ControlModifier:
            # 取得滑鼠位置（作為縮放中心）
            mouse_pos = event.pos()
            
            # 計算縮放前滑鼠對應的影像座標
            img_pos_before = self._widget_to_image(mouse_pos)
            
            # 根據滾輪方向調整縮放倍率
            if event.angleDelta().y() > 0:
                new_zoom = self._zoom_level * self.ZOOM_STEP
            else:
                new_zoom = self._zoom_level / self.ZOOM_STEP
            
            # 限制縮放範圍
            old_zoom = self._zoom_level
            self._zoom_level = max(self.ZOOM_MIN, min(self.ZOOM_MAX, new_zoom))
            
            # 如果縮放倍率實際有變化，調整平移偏移使滑鼠位置指向的影像點保持不變
            if self._zoom_level != old_zoom:
                # 重新計算縮放後的影像座標
                img_pos_after = self._widget_to_image(mouse_pos)
                # 計算偏移補償
                delta = img_pos_after - img_pos_before
                scale = self._get_current_scale()
                self._pan_offset.setX(self._pan_offset.x() + delta.x() * scale)
                self._pan_offset.setY(self._pan_offset.y() + delta.y() * scale)
            
            self.update()
            event.accept()
        else:
            super().wheelEvent(event)

    def _reset_zoom(self):
        """重置縮放與平移至初始狀態"""
        self._zoom_level = 1.0
        self._pan_offset = QtCore.QPointF(0, 0)
        self.update()

    def _get_current_scale(self) -> float:
        """取得當前影像到視窗的縮放比例（不含 zoom_level）"""
        if self._pixmap is None:
            return 1.0
        pix_size = self._pixmap.size()
        widget_size = self.size()
        scaled = pix_size.scaled(widget_size, QtCore.Qt.KeepAspectRatio)
        return scaled.width() / pix_size.width()

    def _target_rect(self) -> QtCore.QRectF:
        """計算影像縮放後在元件中的顯示區域（含縮放與平移）。"""
        assert self._pixmap is not None
        pix_size = self._pixmap.size()
        widget_size = self.size()
        # 先按比例縮放
        scaled = pix_size.scaled(widget_size, QtCore.Qt.KeepAspectRatio)
        # 套用縮放倍率
        scaled_w = scaled.width() * self._zoom_level
        scaled_h = scaled.height() * self._zoom_level
        # 計算置中位置，並套用平移偏移
        x = (widget_size.width() - scaled_w) / 2 + self._pan_offset.x()
        y = (widget_size.height() - scaled_h) / 2 + self._pan_offset.y()
        return QtCore.QRectF(x, y, scaled_w, scaled_h)

    def _widget_to_image(self, pt: QtCore.QPoint) -> QtCore.QPointF:
        """將視窗座標轉為影像座標，並夾在顯示區域內。"""
        if self._image is None:
            return QtCore.QPointF()
        target = self._target_rect()
        # 使用浮點數座標進行計算
        px, py = float(pt.x()), float(pt.y())
        # 夾在顯示區域內
        px = min(max(px, target.left()), target.right())
        py = min(max(py, target.top()), target.bottom())
        # 計算相對位置並轉換為影像座標
        rel_x = (px - target.left()) / target.width() if target.width() > 0 else 0
        rel_y = (py - target.top()) / target.height() if target.height() > 0 else 0
        img_x = rel_x * self._image.width()
        img_y = rel_y * self._image.height()
        return QtCore.QPointF(img_x, img_y)

    def _image_to_widget(self, rect: QtCore.QRectF) -> QtCore.QRectF:
        """將影像座標矩形換算到視窗座標矩形。"""
        target = self._target_rect()
        if self._image is None or target.width() == 0 or target.height() == 0:
            return QtCore.QRectF()
        sx = target.width() / self._image.width()
        sy = target.height() / self._image.height()
        left = target.left() + rect.left() * sx
        top = target.top() + rect.top() * sy
        w = rect.width() * sx
        h = rect.height() * sy
        return QtCore.QRectF(left, top, w, h)

    def _clamped_to_image(self, rect: QtCore.QRectF) -> QtCore.QRectF:
        """將矩形限制在影像尺寸邊界內。"""
        if self._image is None:
            return rect
        left = max(0.0, min(rect.left(), self._image.width() - 1.0))
        top = max(0.0, min(rect.top(), self._image.height() - 1.0))
        right = max(0.0, min(rect.right(), float(self._image.width())))
        bottom = max(0.0, min(rect.bottom(), float(self._image.height())))
        return QtCore.QRectF(QtCore.QPointF(left, top), QtCore.QPointF(right, bottom)).normalized()

    def _clamp_rect_to_image(self, rect: QtCore.QRectF) -> QtCore.QRectF:
        """將矩形移動到影像範圍內（保持大小不變）"""
        if self._image is None:
            return rect
        img_w, img_h = float(self._image.width()), float(self._image.height())
        # 計算需要的偏移量
        dx, dy = 0.0, 0.0
        if rect.left() < 0:
            dx = -rect.left()
        elif rect.right() > img_w:
            dx = img_w - rect.right()
        if rect.top() < 0:
            dy = -rect.top()
        elif rect.bottom() > img_h:
            dy = img_h - rect.bottom()
        return rect.translated(dx, dy)

    def _update_cursor(self, widget_pt: QtCore.QPoint):
        """根據滑鼠位置更新游標形狀"""
        # 優先檢查是否在調整手柄上
        handle = self._hit_test_handle(widget_pt)
        if handle:
            self.setCursor(self._get_cursor_for_handle(handle))
            return
        # 檢查是否在選中的框框內部（顯示移動游標）
        if 0 <= self.active_box_index < len(self.boxes):
            widget_rect = self._image_to_widget(self.boxes[self.active_box_index].rect)
            if widget_rect.contains(QtCore.QPointF(widget_pt)):
                self.setCursor(QtCore.Qt.SizeAllCursor)  # 移動游標
                return
        # 檢查是否在其他框框上（顯示手指游標）
        if self._hit_test(widget_pt) != -1:
            self.setCursor(QtCore.Qt.PointingHandCursor)
            return
        # 預設：十字游標（繪製）
        self.setCursor(QtCore.Qt.CrossCursor)

    def _hit_test(self, widget_pt: QtCore.QPoint) -> int:
        """回傳包含此視窗座標點的第一個框索引；若無則 -1。"""
        for i in reversed(range(len(self.boxes))):
            if self._image_to_widget(self.boxes[i].rect).contains(widget_pt):
                return i
        return -1

    def _get_resize_handles(self, widget_rect: QtCore.QRectF) -> dict:
        """取得框框的8個調整手柄位置（視窗座標）"""
        hs = self.HANDLE_SIZE / 2  # 手柄半徑
        l, t, r, b = widget_rect.left(), widget_rect.top(), widget_rect.right(), widget_rect.bottom()
        cx, cy = widget_rect.center().x(), widget_rect.center().y()
        return {
            'tl': QtCore.QRectF(l - hs, t - hs, self.HANDLE_SIZE, self.HANDLE_SIZE),  # 左上
            't':  QtCore.QRectF(cx - hs, t - hs, self.HANDLE_SIZE, self.HANDLE_SIZE),  # 上中
            'tr': QtCore.QRectF(r - hs, t - hs, self.HANDLE_SIZE, self.HANDLE_SIZE),  # 右上
            'l':  QtCore.QRectF(l - hs, cy - hs, self.HANDLE_SIZE, self.HANDLE_SIZE),  # 左中
            'r':  QtCore.QRectF(r - hs, cy - hs, self.HANDLE_SIZE, self.HANDLE_SIZE),  # 右中
            'bl': QtCore.QRectF(l - hs, b - hs, self.HANDLE_SIZE, self.HANDLE_SIZE),  # 左下
            'b':  QtCore.QRectF(cx - hs, b - hs, self.HANDLE_SIZE, self.HANDLE_SIZE),  # 下中
            'br': QtCore.QRectF(r - hs, b - hs, self.HANDLE_SIZE, self.HANDLE_SIZE),  # 右下
        }

    def _hit_test_handle(self, widget_pt: QtCore.QPoint) -> Optional[str]:
        """檢測滑鼠是否在選中框的調整手柄上，返回手柄名稱或 None"""
        if not (0 <= self.active_box_index < len(self.boxes)):
            return None
        widget_rect = self._image_to_widget(self.boxes[self.active_box_index].rect)
        handles = self._get_resize_handles(widget_rect)
        for name, handle_rect in handles.items():
            if handle_rect.contains(QtCore.QPointF(widget_pt)):
                return name
        return None

    def _get_cursor_for_handle(self, handle: Optional[str]) -> QtCore.Qt.CursorShape:
        """根據手柄名稱返回對應的游標形狀"""
        cursor_map = {
            'tl': QtCore.Qt.SizeFDiagCursor,  # ↘
            'br': QtCore.Qt.SizeFDiagCursor,  # ↘
            'tr': QtCore.Qt.SizeBDiagCursor,  # ↙
            'bl': QtCore.Qt.SizeBDiagCursor,  # ↙
            't':  QtCore.Qt.SizeVerCursor,    # ↕
            'b':  QtCore.Qt.SizeVerCursor,    # ↕
            'l':  QtCore.Qt.SizeHorCursor,    # ↔
            'r':  QtCore.Qt.SizeHorCursor,    # ↔
        }
        return cursor_map.get(handle, QtCore.Qt.ArrowCursor)

    def _apply_resize(self, handle: str, start_rect: QtCore.QRectF, delta: QtCore.QPointF) -> QtCore.QRectF:
        """根據手柄類型和移動量計算新的矩形（影像座標）"""
        l, t, r, b = start_rect.left(), start_rect.top(), start_rect.right(), start_rect.bottom()
        dx, dy = delta.x(), delta.y()
        
        # 根據手柄調整對應的邊
        if 'l' in handle:
            l += dx
        if 'r' in handle:
            r += dx
        if 't' in handle:
            t += dy
        if 'b' in handle:
            b += dy
        
        # 確保最小尺寸
        min_size = LabelingWindow.MIN_BOX_SIZE
        if r - l < min_size:
            if 'l' in handle:
                l = r - min_size
            else:
                r = l + min_size
        if b - t < min_size:
            if 't' in handle:
                t = b - min_size
            else:
                b = t + min_size
        
        new_rect = QtCore.QRectF(QtCore.QPointF(l, t), QtCore.QPointF(r, b))
        return self._clamped_to_image(new_rect)


class HitCountWorker(QtCore.QThread):
    """在背景用虛擬感應線計算整支影片的擊球幀。"""
    hitsReady = QtCore.pyqtSignal(int, list)
    hitsFailed = QtCore.pyqtSignal(int, str)

    def __init__(self, token: int, video_path: str, region: Tuple[float, float, float, float], parent=None):
        super().__init__(parent)
        self.token = token
        self.video_path = video_path
        self.region = region

    def run(self):
        try:
            from count_hits import detect_tripwire
            crossings, _ = detect_tripwire(self.video_path, self.region)
            self.hitsReady.emit(self.token, [(int(frame), d) for frame, d in crossings])
        except Exception as exc:
            self.hitsFailed.emit(self.token, str(exc))


class CameraWorker(QtCore.QThread):
    """在背景讀取攝影機：即時顯示畫面、同步錄成 mp4，並邊錄邊計算擊球次數。"""
    started_ok = QtCore.pyqtSignal(float, int, int, str)  # (fps, 寬, 高, 錄影檔路徑)
    frameReady = QtCore.pyqtSignal(QtGui.QImage, int)  # (畫面, 第幾幀)
    hitsUpdated = QtCore.pyqtSignal(list)  # 目前所有過網 [(幀, 方向)]
    failed = QtCore.pyqtSignal(str)
    stopped = QtCore.pyqtSignal(str, int, list)  # (錄影檔路徑, 總幀數, 過網 [(幀, 方向)])

    FPS_PROBE_FRAMES = 20  # 開始錄影前先量測攝影機實際幀率用的幀數

    def __init__(self, camera_index: int, out_path: str, region: Tuple[float, float, float, float], parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.out_path = out_path
        self.region = region
        self._running = True

    def stop(self):
        """要求停止錄影（執行緒會在讀完目前這一幀後結束）。"""
        self._running = False

    def run(self):
        import cv2
        from count_hits import TripwireCounter

        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.failed.emit(f"無法開啟攝影機 {self.camera_index}。\n請確認攝影機已連接、沒有被其他程式佔用，並已允許本程式使用攝影機。")
            return

        writer = None
        frame_idx = 0
        counter = None
        try:
            # 先讀幾幀量測實際幀率，讓錄下來的影片播放速度正確
            probe = []
            t0 = time.monotonic()
            while self._running and len(probe) < self.FPS_PROBE_FRAMES:
                ok, frame = cap.read()
                if not ok:
                    break
                probe.append(frame)
            if not probe:
                if self._running:
                    self.failed.emit("攝影機沒有傳回畫面，請確認攝影機是否正常。")
                return
            elapsed = time.monotonic() - t0
            measured = (len(probe) - 1) / elapsed if elapsed > 0 and len(probe) > 1 else 0
            reported = cap.get(cv2.CAP_PROP_FPS)
            fps = measured if measured > 1 else (reported if reported and reported > 1 else 30.0)
            fps = round(min(fps, 120.0), 2)

            height, width = probe[0].shape[:2]
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            writer = cv2.VideoWriter(self.out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                writer.release()
                writer = None
                self.failed.emit(f"無法建立錄影檔：\n{self.out_path}")
                return
            counter = TripwireCounter(fps, width, height, self.region)
            self.started_ok.emit(fps, width, height, self.out_path)

            def handle(frame):
                nonlocal frame_idx
                writer.write(frame)
                if counter.update(frame):
                    self.hitsUpdated.emit(list(counter.crossings))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QtGui.QImage(rgb.data, width, height, 3 * width, QtGui.QImage.Format_RGB888).copy()
                self.frameReady.emit(image, frame_idx)
                frame_idx += 1

            for frame in probe:
                handle(frame)
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    self.failed.emit("攝影機畫面中斷，已停止錄影。")
                    break
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                handle(frame)
        except Exception as exc:
            self.failed.emit(f"錄影發生錯誤：{exc}")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            hits = []
            if counter is not None:
                counter.flush()
                hits = list(counter.crossings)
            if writer is not None:
                self.stopped.emit(self.out_path, frame_idx, hits)


class VideoPane(QtWidgets.QWidget):
    """播放整支影片，不把影片拆成單張影像。"""
    frameChanged = QtCore.pyqtSignal(int)
    playbackChanged = QtCore.pyqtSignal(bool)
    stopLiveRequested = QtCore.pyqtSignal()  # 即時錄影中按下「停止錄影」或空白鍵

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.path: Optional[str] = None
        self.fps = 30.0
        self.total = 0
        self.frame_idx = 0
        self._frame_image: Optional[QtGui.QImage] = None
        self._slider_held = False
        self.live = False  # 是否正在即時錄影

        self.screen = QtWidgets.QLabel("請按「開始錄影」或開啟影片")
        self.screen.setAlignment(QtCore.Qt.AlignCenter)
        self.screen.setMinimumSize(640, 360)
        self.screen.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.screen.setStyleSheet("""
            QLabel {
                background: #0b1c2c;
                color: #d6e8f8;
                border: 2px solid #cfe3f3;
                border-radius: 12px;
                font-size: 16px;
            }
        """)

        self.play_btn = QtWidgets.QPushButton("播放")
        self.play_btn.setFixedWidth(96)
        self.play_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.play_btn.clicked.connect(self.toggle_play)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_value)

        self.time_label = QtWidgets.QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("color: #0a2742; background: transparent; border: none; font-size: 13px;")

        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.time_label)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self.screen, 1)
        layout.addLayout(controls)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        space = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self, self.toggle_play)
        space.setContext(QtCore.Qt.WidgetWithChildrenShortcut)

    def load(self, path: str) -> bool:
        """開啟影片並停在第一幀。"""
        import cv2
        self.close_video()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        self.cap = cap
        self.path = path
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 30.0
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, self.total - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        ok, frame = cap.read()
        if ok:
            self._show_bgr(frame)
        self._update_time_label()
        self.play_btn.setText("播放")
        self.frameChanged.emit(0)
        return True

    def play(self):
        """由目前位置繼續播放。播到結尾時從頭開始。"""
        if self.cap is None:
            return
        if self.total > 1 and self.frame_idx >= self.total - 1:
            self.seek(0)
        self.timer.start(max(1, int(round(1000 / self.fps))))
        self.play_btn.setText("暫停")
        self.playbackChanged.emit(True)

    def pause(self):
        """暫停播放。"""
        self.timer.stop()
        self.play_btn.setText("播放")
        self.playbackChanged.emit(False)

    def toggle_play(self):
        """切換播放與暫停。即時錄影時則是要求停止錄影。"""
        if self.live:
            self.stopLiveRequested.emit()
            return
        if self.timer.isActive():
            self.pause()
        else:
            self.play()

    def seek(self, index: int):
        """跳到指定幀。"""
        if self.cap is None or self.total <= 0:
            return
        index = max(0, min(int(index), self.total - 1))
        self.cap.set(cv2_pos_frames(), index)
        ok, frame = self.cap.read()
        if not ok:
            return
        self.frame_idx = index
        self._show_bgr(frame)
        self._sync_slider()
        self._update_time_label()
        self.frameChanged.emit(self.frame_idx)

    def close_video(self):
        """停止播放並釋放影片。"""
        self.timer.stop()
        self.playbackChanged.emit(False)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.path = None
        self.total = 0
        self.frame_idx = 0
        self._frame_image = None
        self.screen.clear()
        self.screen.setText("請按「開始錄影」或開啟影片")
        self.play_btn.setText("播放")
        self.time_label.setText("00:00 / 00:00")
        self.slider.blockSignals(True)
        self.slider.setRange(0, 0)
        self.slider.setValue(0)
        self.slider.blockSignals(False)

    # ── 即時錄影模式 ──
    def begin_live(self):
        """切換成即時錄影模式：釋放原本的影片、停用進度條。"""
        self.close_video()
        self.live = True
        self.screen.setText("攝影機啟動中...")
        self.slider.setEnabled(False)
        self.play_btn.setText("停止錄影")
        self.time_label.setText("● REC 00:00")

    def show_live_frame(self, image: QtGui.QImage, frame_idx: int, fps: float):
        """顯示攝影機傳來的一幀，並更新錄影時間。"""
        if not self.live:
            return
        self.fps = fps
        self.frame_idx = frame_idx
        self._frame_image = image
        self._apply_pixmap()
        self.time_label.setText(f"● REC {self._format_time(frame_idx / fps if fps else 0)}")
        self.frameChanged.emit(frame_idx)

    def end_live(self):
        """離開即時錄影模式，恢復一般播放器。"""
        if not self.live:
            return
        self.live = False
        self.slider.setEnabled(True)
        self.play_btn.setText("播放")
        self.time_label.setText("00:00 / 00:00")
        if self._frame_image is None:
            self.screen.setText("請按「開始錄影」或開啟影片")

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self._apply_pixmap()

    def _on_tick(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.pause()
            return
        self.frame_idx = min(self.frame_idx + 1, max(0, self.total - 1))
        self._show_bgr(frame)
        self._sync_slider()
        self._update_time_label()
        self.frameChanged.emit(self.frame_idx)
        if self.total > 0 and self.frame_idx >= self.total - 1:
            self.pause()

    def _on_slider_pressed(self):
        self._slider_held = True

    def _on_slider_released(self):
        self._slider_held = False
        self.seek(self.slider.value())

    def _on_slider_value(self, value: int):
        if self._slider_held:
            return
        if value != self.frame_idx:
            self.seek(value)

    def _sync_slider(self):
        self.slider.blockSignals(True)
        self.slider.setValue(self.frame_idx)
        self.slider.blockSignals(False)

    def _update_time_label(self):
        current = self.frame_idx / self.fps if self.fps else 0
        total = self.total / self.fps if self.fps else 0
        self.time_label.setText(f"{self._format_time(current)} / {self._format_time(total)}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _show_bgr(self, frame):
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QtGui.QImage(rgb.data, width, height, channels * width, QtGui.QImage.Format_RGB888)
        self._frame_image = image.copy()
        self._apply_pixmap()

    def _apply_pixmap(self):
        if self._frame_image is None or self._frame_image.isNull():
            return
        target = self.screen.size()
        if target.width() < 2 or target.height() < 2:
            target = QtCore.QSize(960, 540)
        pixmap = QtGui.QPixmap.fromImage(self._frame_image)
        self.screen.setPixmap(pixmap.scaled(target, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


def cv2_pos_frames() -> int:
    """取得 OpenCV 的幀位置常數，避免播放器在未載入影片時就依賴 cv2。"""
    import cv2
    return cv2.CAP_PROP_POS_FRAMES


class LabelingWindow(QtWidgets.QMainWindow):
    """主視窗：含畫布、側邊標註清單與工具列/快捷鍵。"""
    # 常數定義
    SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    DEFAULT_CLASSES = ["object", "person", "ball"]
    INFERENCE_SCRIPT_DIR = "inference_code"
    OUTPUT_DIR = "output"
    CLASSES_FILE = "classes.txt"
    MIN_BOX_SIZE = 3  # 最小框尺寸
    COUNTDOWN_MS = 3 * 60 * 1000  # 倒數計時預設 3 分鐘
    # 網柱上方感應區（畫面寬高比例 x0, x1, y0, y1），與 count_hits.py 預設相同
    TRIPWIRE_REGION = (0.40, 0.60, 0.23, 0.45)
    
    # UI 顏色常數
    BOX_COLOR_NORMAL = QtGui.QColor(0, 255, 0)  # 綠色
    BOX_COLOR_ACTIVE = QtGui.QColor(255, 165, 0)  # 橘色
    BOX_COLOR_DRAWING = QtGui.QColor(0, 200, 255)  # 青色
    LABEL_BG_COLOR = QtGui.QColor(0, 0, 0, 140)  # 半透明黑色
    FRAME_LABEL_COLOR = QtGui.QColor(0, 0, 0, 140)
    
    # UI 尺寸常數
    LABEL_TEXT_WIDTH = 120
    LABEL_TEXT_HEIGHT = 18
    PEN_WIDTH = 2
    
    # 主題配色
    THEME = {
        "base_bg": "#e9f3fb",
        "text": "#0a2742",
        "sidebar_bg": "#e1eef8",
        "sidebar_border": "#cfe3f3",
        "combo_bg": "#ffffff",
        "combo_border": "#c7dbef",
        "combo_hover": "#a9c9e8",
        "combo_focus": "#1572d3",
        "btn_bg": "#1572d3",
        "btn_hover": "#2c84de",
        "btn_pressed": "#1262b6",
        "btn_disabled": "#bcd5ef",
        "list_bg": "#ffffff",
        "list_border": "#ddd",
        "list_hover": "#f0f7ff",
        "list_selected": "#e3f2ff",
        "list_selected_text": "#1976d2",
        "frame_bg": "#f0f8ff",
        "frame_border": "#b8d4f0",
        "canvas_bg": "#f5f9fd",
        "toolbar_bg": "#e1eef8",
        "accent": "#1572d3",
        # 右側資訊卡片
        "card_bg": "#175e81",
        "card_border": "#0b2a3c",
        "card_text": "#ffffff",
    }
    # 失誤判斷：球過網後超過幾秒沒有下一次過網，就算失誤一次
    MISTAKE_GAP_SEC = 3.0
    # 是否在工具列顯示舊的影像標註功能（開啟資料夾、儲存、清除…）；改成 True 即可恢復
    SHOW_LABELING_TOOLS = False
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("影像標註工具 (Simple Labeling)")
        self.resize(1100, 750)

        # 套用主題樣式
        self._apply_theme()

        self.canvas = ImageCanvas()
        self.canvas.set_current_class("ball")
        self.video_pane = VideoPane()
        self.viewer = QtWidgets.QStackedWidget()
        self.viewer.addWidget(self.video_pane)
        self.viewer.addWidget(self.canvas)
        self.viewer.setCurrentWidget(self.video_pane)
        # 中央版面（左：畫面，右：資訊卡片）在 _setup_sidebar 建立

        self.image_folder: Optional[str] = None
        self.labels_folder: Optional[str] = None  # 標註資料夾路徑
        self.image_paths: List[str] = []
        self.current_index: int = -1
        self.inference_msg_box: Optional[QtWidgets.QDialog] = None  # 推論訊息框
        # 緩存專案根目錄路徑，避免重複計算
        self._project_root = os.path.dirname(__file__)
        # 追蹤已修改的標註：{影像路徑: [BoundingBox列表]}
        self.modified_annotations: dict[str, List[BoundingBox]] = {}
        # 當前影像路徑（用於追蹤修改）
        self._current_image_path: Optional[str] = None
        # 目前載入影片的擊球幀，與對應的計算工作
        self._hit_frames: List[int] = []
        self._crossings: List[Tuple[int, str]] = []  # 過網 (幀, 方向)，用來分左右失誤
        self._hits_ready = False
        self._hit_token = 0
        self._hit_worker: Optional[HitCountWorker] = None
        # 即時錄影
        self._camera_worker: Optional[CameraWorker] = None
        self._live_fps = 30.0
        self._countdown_left_ms = self.COUNTDOWN_MS
        self._countdown_mark: Optional[float] = None
        self._countdown_timer = QtCore.QTimer(self)
        self._countdown_timer.setInterval(100)
        self._countdown_timer.timeout.connect(self._on_countdown_tick)

        # 右側 Dock：類別輸入/選單與框列表
        self._setup_sidebar()

        # 工具列
        self._setup_toolbar()

    def _apply_theme(self):
        """套用現代化主題樣式"""
        p = self.THEME
        stylesheet = f"""
        /* 主視窗背景 */
        QMainWindow {{
            background: {p['base_bg']};
        }}
        
        /* 中央畫布區域 */
        QLabel {{
            background: {p['canvas_bg']};
            border: 1px solid {p['sidebar_border']};
            border-radius: 8px;
        }}
        
        /* Dock 側邊欄 */
        QDockWidget {{
            font-size: 14px;
            font-weight: 600;
            color: {p['text']};
        }}
        QDockWidget::title {{
            background: {p['sidebar_bg']};
            padding: 10px 12px;
            border-bottom: 1px solid {p['sidebar_border']};
            font-size: 16px;
        }}
        QDockWidget > QWidget {{
            background: {p['sidebar_bg']};
            border: none;
        }}
        
        /* ComboBox 下拉選單 */
        QComboBox {{
            background: {p['combo_bg']};
            color: {p['text']};
            border: 2px solid {p['combo_border']};
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 14px;
            min-height: 20px;
        }}
        QComboBox::drop-down {{
            width: 28px;
            border: none;
            border-left: 1px solid {p['combo_border']};
            border-top-right-radius: 8px;
            border-bottom-right-radius: 8px;
        }}
        QComboBox::down-arrow {{
            width: 12px;
            height: 12px;
        }}
        QComboBox:hover {{
            border: 2px solid {p['combo_hover']};
        }}
        QComboBox:focus {{
            border: 2px solid {p['combo_focus']};
        }}
        QComboBox QAbstractItemView {{
            background: {p['combo_bg']};
            color: {p['text']};
            selection-background-color: {p['list_selected']};
            selection-color: {p['list_selected_text']};
            border: 1px solid {p['combo_border']};
            border-radius: 4px;
            padding: 4px;
        }}
        
        /* 按鈕樣式 */
        QPushButton {{
            background: {p['btn_bg']};
            color: #ffffff;
            border: none;
            border-radius: 8px;
            padding: 12px 18px;
            font-size: 14px;
            font-weight: 600;
            min-height: 20px;
        }}
        QPushButton:hover {{
            background: {p['btn_hover']};
        }}
        QPushButton:pressed {{
            background: {p['btn_pressed']};
        }}
        QPushButton:disabled {{
            background: {p['btn_disabled']};
            color: #f5f7fb;
        }}
        
        /* 列表樣式 */
        QListWidget {{
            background: {p['list_bg']};
            border: 2px solid {p['list_border']};
            border-radius: 8px;
            font-size: 13px;
            padding: 4px;
        }}
        QListWidget::item {{
            padding: 10px 12px;
            border-bottom: 1px solid #f0f0f0;
            border-radius: 4px;
            margin: 2px;
        }}
        QListWidget::item:hover {{
            background: {p['list_hover']};
        }}
        QListWidget::item:selected {{
            background: {p['list_selected']};
            color: {p['list_selected_text']};
            font-weight: 600;
        }}
        
        /* 工具列 */
        QToolBar {{
            background: {p['toolbar_bg']};
            border: none;
            border-bottom: 1px solid {p['sidebar_border']};
            padding: 8px 12px;
            spacing: 8px;
        }}
        QToolBar::separator {{
            width: 2px;
            background: {p['sidebar_border']};
            margin: 6px 8px;
        }}
        QToolButton {{
            background: transparent;
            color: {p['text']};
            border: 2px solid transparent;
            border-radius: 8px;
            padding: 8px 14px;
            font-size: 13px;
            font-weight: 500;
        }}
        QToolButton:hover {{
            background: {p['combo_bg']};
            border: 2px solid {p['combo_border']};
        }}
        QToolButton:pressed {{
            background: {p['list_selected']};
            border: 2px solid {p['accent']};
        }}
        
        /* 捲軸 */
        QScrollBar:vertical {{
            background: {p['base_bg']};
            width: 12px;
            margin: 0;
            border-radius: 6px;
        }}
        QScrollBar::handle:vertical {{
            background: {p['combo_border']};
            border-radius: 6px;
            min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {p['combo_hover']};
        }}
        QScrollBar::add-line, QScrollBar::sub-line {{
            height: 0;
        }}
        
        /* 訊息框 */
        QMessageBox {{
            background: {p['combo_bg']};
        }}
        QMessageBox QLabel {{
            color: {p['text']};
            background: transparent;
            border: none;
        }}
        """
        self.setStyleSheet(stylesheet)
    
    def _make_card(self, title: str, value: str, value_pt: int, sub: Optional[str] = None):
        """建立右側資訊卡片（標題 + 大字數值 + 可選的小字說明）。"""
        p = self.THEME
        card = QtWidgets.QFrame()
        card.setObjectName("infoCard")
        card.setStyleSheet(f"""
            QFrame#infoCard {{
                background: {p['card_bg']};
                border: 2px solid {p['card_border']};
                border-radius: 10px;
            }}
            QLabel {{ background: transparent; border: none; color: {p['card_text']}; }}
        """)
        card.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(2)
        layout.addStretch(1)

        title_label = QtWidgets.QLabel(title)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QtGui.QFont()
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        if title:
            layout.addWidget(title_label)

        value_label = QtWidgets.QLabel(value)
        value_label.setAlignment(QtCore.Qt.AlignCenter)
        value_label.setWordWrap(True)
        value_font = QtGui.QFont()
        value_font.setBold(True)
        value_font.setPointSize(value_pt)
        value_label.setFont(value_font)
        layout.addWidget(value_label)

        sub_label = None
        if sub is not None:
            sub_label = QtWidgets.QLabel(sub)
            sub_label.setAlignment(QtCore.Qt.AlignCenter)
            sub_font = QtGui.QFont()
            sub_font.setPointSize(13)
            sub_label.setFont(sub_font)
            layout.addWidget(sub_label)
        layout.addStretch(1)
        return card, value_label, sub_label

    def _setup_sidebar(self):
        """版面：左邊即時攝影機畫面，右邊由上而下三張卡片（擊球次數、倒數計時、失誤次數）。"""
        # 擊球次數
        hit_card, self.frame_label, self.hit_total_label = self._make_card("", "Hit 到 0 次球", 22, "全片共 0 次")
        # 倒數計時（預設 3 分鐘，跟著錄影 / 播放）
        countdown_card, self.countdown_label, _ = self._make_card("倒數計時", "03:00", 30)
        # 失誤次數分左右兩格：球過網後太久沒有下一次過網，算「沒把球打回去的那一方」失誤一次
        left_miss_card, self.left_mistake_label, _ = self._make_card("左方失誤", "0 次", 22)
        right_miss_card, self.right_mistake_label, _ = self._make_card("右方失誤", "0 次", 22)
        mistake_row = QtWidgets.QWidget()
        mistake_layout = QtWidgets.QHBoxLayout(mistake_row)
        mistake_layout.setContentsMargins(0, 0, 0, 0)
        mistake_layout.setSpacing(16)
        mistake_layout.addWidget(left_miss_card, 1)
        mistake_layout.addWidget(right_miss_card, 1)

        right = QtWidgets.QWidget()
        right.setFixedWidth(300)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(24)
        right_layout.addWidget(hit_card, 1)
        right_layout.addWidget(countdown_card, 1)
        right_layout.addWidget(mistake_row, 1)

        # 標註工具的元件仍保留（舊的標註功能會用到），但不顯示在版面上
        self.box_list = QtWidgets.QListWidget()
        self.delete_btn = QtWidgets.QPushButton("刪除選取框 (Del)")
        self.box_list.hide()
        self.delete_btn.hide()

        central = QtWidgets.QWidget()
        central_layout = QtWidgets.QHBoxLayout(central)
        central_layout.setContentsMargins(16, 16, 16, 16)
        central_layout.setSpacing(24)
        central_layout.addWidget(self.viewer, 1)
        central_layout.addWidget(right)
        self.setCentralWidget(central)

    def _setup_toolbar(self):
        """設置工具列"""
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(20, 20))
        
        # 即時錄影：選擇攝影機編號後按「開始錄影」
        cam_label = QtWidgets.QLabel(" 攝影機 ")
        tb.addWidget(cam_label)
        self.camera_spin = QtWidgets.QSpinBox()
        self.camera_spin.setRange(0, 9)
        self.camera_spin.setToolTip("攝影機編號：0 通常是內建鏡頭，外接鏡頭多為 1 或 2")
        self.camera_spin.setFocusPolicy(QtCore.Qt.ClickFocus)
        tb.addWidget(self.camera_spin)
        self.act_live = tb.addAction("● 開始錄影 (R)")
        tb.addSeparator()
        act_open_video = tb.addAction("開啟影片")

        # 連接訊號
        self.act_live.triggered.connect(self.toggle_live_recording)
        self.video_pane.stopLiveRequested.connect(self.stop_live_recording)
        act_open_video.triggered.connect(self.open_video)
        self.video_pane.frameChanged.connect(self._refresh_hit_label)
        self.video_pane.playbackChanged.connect(self._on_playback_changed)

        self.delete_btn.clicked.connect(self.delete_selected)
        self.box_list.currentRowChanged.connect(self._on_box_selected)
        self.canvas.boxAdded.connect(self._on_box_added)
        self.canvas.boxDeleted.connect(self._on_box_added)  # 刪除框時也標記為已修改
        self.canvas.selectionChanged.connect(self._on_canvas_selection)

        # 快捷鍵綁定
        QtWidgets.QShortcut(QtGui.QKeySequence("R"), self, self.toggle_live_recording)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+O"), self, self.open_video)

        if not self.SHOW_LABELING_TOOLS:
            return

        # ── 以下為舊的影像標註功能（預設隱藏）──
        tb.addSeparator()
        act_open = tb.addAction("開啟資料夾 (O)")
        tb.addSeparator()
        act_save = tb.addAction("儲存 (S)")
        act_save_as = tb.addAction("另存至...")
        tb.addSeparator()
        act_clear = tb.addAction("清除 (C)")
        act_open.triggered.connect(self.open_folder)
        act_save.triggered.connect(self.save_annotations)
        act_save_as.triggered.connect(self.save_annotations_as)
        act_clear.triggered.connect(self.clear_current_image)
        QtWidgets.QShortcut(QtGui.QKeySequence("O"), self, self.open_folder)
        for key in (QtCore.Qt.Key_A, QtCore.Qt.Key_Left):
            QtWidgets.QShortcut(QtGui.QKeySequence(key), self, self.prev_image)
        for key in (QtCore.Qt.Key_D, QtCore.Qt.Key_Right):
            QtWidgets.QShortcut(QtGui.QKeySequence(key), self, self.next_image)
        QtWidgets.QShortcut(QtGui.QKeySequence("S"), self, self.save_annotations)
        QtWidgets.QShortcut(QtGui.QKeySequence("C"), self, self.clear_current_image)

        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self, self.delete_selected)

    def _check_and_save_unsaved_changes(self, action_name: str = "關閉") -> bool:
        """檢查並處理未儲存的修改，返回是否應該繼續執行操作"""
        # 先標記當前影像為已修改（如果正在查看某張影像）
        if self._is_valid_image_index():
            self._mark_current_image_modified()
        
        # 檢查是否有未儲存的修改
        if not self.modified_annotations:
            return True
        
        # 有未儲存的修改，詢問用戶
        reply = QtWidgets.QMessageBox.question(
            self,
            "未儲存的修改",
            f"您有 {len(self.modified_annotations)} 個未儲存的標註修改。\n\n是否要儲存後再{action_name}？",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save
        )
        
        if reply == QtWidgets.QMessageBox.Save:
            self.save_annotations(auto=True)
            return True
        elif reply == QtWidgets.QMessageBox.Discard:
            return True
        else:
            return False
    
    def closeEvent(self, event):
        """視窗關閉事件：檢查是否有未儲存的修改，並提示用戶。"""
        if self._check_and_save_unsaved_changes("關閉"):
            if self._camera_worker is not None:
                # 關閉前先停止錄影，確保錄影檔正確寫完
                worker = self._camera_worker
                for sig in (worker.started_ok, worker.frameReady, worker.hitsUpdated,
                            worker.failed, worker.stopped, worker.finished):
                    try:
                        sig.disconnect()
                    except TypeError:
                        pass
                worker.stop()
                worker.wait(5000)
                self._camera_worker = None
            self.video_pane.close_video()
            event.accept()
            super().closeEvent(event)
        else:
            event.ignore()

    def _get_inference_dir(self, *paths):
        """取得 inference_code 目錄下的路徑"""
        return os.path.join(self._project_root, self.INFERENCE_SCRIPT_DIR, *paths)
    
    def _get_file_basename_without_ext(self, file_path: str) -> str:
        """取得檔案名稱（不含副檔名）"""
        return os.path.splitext(os.path.basename(file_path))[0]
    
    def _get_annotation_filename(self, img_path: str) -> str:
        """根據影像路徑取得對應的標註檔名"""
        return self._get_file_basename_without_ext(img_path) + ".txt"
    
    def _copy_boxes(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """深拷貝 BoundingBox 列表"""
        return [BoundingBox(box.class_name, QtCore.QRectF(box.rect)) for box in boxes]
    
    def _is_valid_image_index(self) -> bool:
        """檢查當前影像索引是否有效"""
        return self._current_image_path is not None and 0 <= self.current_index < len(self.image_paths)
    
    def _safe_file_operation(self, operation, default=None):
        """安全執行文件操作，捕獲 IOError、OSError 和 FileNotFoundError"""
        try:
            return operation()
        except (IOError, OSError, FileNotFoundError):
            return default
    
    def _get_classes_path(self, img_path: Optional[str] = None) -> str:
        """取得 classes.txt 檔案路徑"""
        # 優先順序：labels_folder > image_folder 父目錄 > image_folder > img_path 目錄 > 當前目錄
        if self.labels_folder:
            return os.path.join(self.labels_folder, self.CLASSES_FILE)
        if self.image_folder:
            parent = os.path.dirname(self.image_folder)
            target = parent if parent and parent != self.image_folder else self.image_folder
            return os.path.join(target, self.CLASSES_FILE)
        folder = os.path.dirname(img_path) if img_path else os.getcwd()
        return os.path.join(folder, self.CLASSES_FILE)
    
    def _read_classes_file(self, classes_path: str) -> List[str]:
        """讀取 classes.txt 檔案，返回類別列表"""
        if not os.path.exists(classes_path):
            return []
        def _read():
            with open(classes_path, "r", encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
        return self._safe_file_operation(_read, default=[])
    
    def _write_classes_file(self, classes_path: str, classes: List[str]) -> bool:
        """寫入 classes.txt 檔案，返回是否成功"""
        def _write():
            with open(classes_path, "w", encoding="utf-8") as f:
                f.write("\n".join(classes) + "\n")
            return True
        return self._safe_file_operation(_write, default=False) is not False
    
    def _load_image_safely(self, img_path: str) -> Optional[QtGui.QImage]:
        """安全載入影像，失敗時返回 None"""
        img = QtGui.QImage(img_path)
        return img if not img.isNull() else None
    
    def _create_inference_dialog(self, video_path: str):
        """創建推論訊息對話框"""
        p = self.THEME
        
        self.inference_msg_box = QtWidgets.QDialog(self)
        self.inference_msg_box.setWindowTitle("🤖 自動標註處理中")
        self.inference_msg_box.setModal(False)
        self.inference_msg_box.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.inference_msg_box.setStyleSheet(f"""
            QDialog {{
                background: {p['base_bg']};
            }}
            QLabel {{
                color: {p['text']};
                background: transparent;
                border: none;
            }}
        """)
        
        layout = QtWidgets.QVBoxLayout(self.inference_msg_box)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # 標題區塊
        title_frame = QtWidgets.QFrame()
        title_frame.setStyleSheet(f"""
            QFrame {{
                background: {p['frame_bg']};
                border: 2px solid {p['frame_border']};
                border-radius: 12px;
                padding: 12px;
            }}
        """)
        title_layout = QtWidgets.QVBoxLayout(title_frame)
        
        label = QtWidgets.QLabel("🔄 模型正在處理中...")
        label.setFont(QtGui.QFont("", 18))
        label.setStyleSheet(f"color: {p['accent']}; font-weight: 600; background: transparent; border: none;")
        label.setAlignment(QtCore.Qt.AlignCenter)
        title_layout.addWidget(label)
        layout.addWidget(title_frame)
        
        # 進度條
        self.inference_progress = QtWidgets.QProgressBar()
        self.inference_progress.setFormat("%p% (%v/%m)")
        self.inference_progress.setMinimumHeight(28)
        self.inference_progress.setStyleSheet(f"""
            QProgressBar {{
                background: {p['combo_bg']};
                border: 2px solid {p['combo_border']};
                border-radius: 8px;
                text-align: center;
                font-size: 13px;
                font-weight: 600;
                color: {p['text']};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {p['accent']}, stop:1 {p['btn_hover']});
                border-radius: 6px;
            }}
        """)
        layout.addWidget(self.inference_progress)
        
        # 輸出日誌區塊
        log_label = QtWidgets.QLabel("📋 處理日誌：")
        log_label.setStyleSheet(f"font-size: 14px; font-weight: 600; color: {p['text']}; background: transparent; border: none;")
        layout.addWidget(log_label)
        
        self.inference_text = QtWidgets.QPlainTextEdit()
        self.inference_text.setReadOnly(True)
        self.inference_text.setMaximumBlockCount(1000)
        self.inference_text.setMinimumHeight(180)
        self.inference_text.setStyleSheet(f"""
            QPlainTextEdit {{
                background: {p['combo_bg']};
                border: 2px solid {p['list_border']};
                border-radius: 8px;
                font-family: Consolas, Monaco, monospace;
                font-size: 12px;
                color: {p['text']};
                padding: 8px;
            }}
        """)
        layout.addWidget(self.inference_text)
        
        self.inference_msg_box.resize(550, 420)
        self.inference_msg_box.show()

        # 顯示初始訊息
        self.inference_text.appendPlainText(f"[系統] 已選取影片: {os.path.basename(video_path)}")
        self.inference_text.appendPlainText(f"[系統] 影片路徑: {video_path}")
        self.inference_text.appendPlainText(f"[系統] 準備開始推論...\n")
    
    def auto_label_from_video(self):
        """選擇影片，使用 YOLOv8 模型自動推論，輸出 640x480 逐幀與 YOLO 標註，並載入供人工校正。"""
        default_dir = self._get_inference_dir()
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Video", default_dir, "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if not video_path:
            return

        # 創建推論訊息框
        self._create_inference_dialog(video_path)

        # 取得影片檔名（不含副檔名）作為資料夾名稱
        video_basename = self._get_file_basename_without_ext(video_path)
        
        # 在 output 目錄下建立以影片檔名命名的資料夾
        output_base = self._get_inference_dir(self.OUTPUT_DIR)
        out_dir = os.path.join(output_base, video_basename)

        # 建立並啟動背景推論執行緒
        self.inference_worker = InferenceWorker(video_path, out_dir, self)
        self.inference_worker.finished.connect(self._on_inference_finished)
        self.inference_worker.error.connect(self._on_inference_error)
        self.inference_worker.output.connect(self._on_inference_output)
        self.inference_worker.progress.connect(self._on_inference_progress)
        self.inference_worker.start()

    def _on_inference_finished(self, images_dir: str, labels_dir: str):
        """推論完成時的回調函數"""
        # 在 labels 資料夾建立 classes.txt（供本標註器載入類別名稱）
        classes_path = os.path.join(labels_dir, self.CLASSES_FILE)
        self._write_classes_file(classes_path, ["ball"])

        # 清空已修改標註（載入新推論結果時）
        self.modified_annotations.clear()
        self._current_image_path = None

        # 載入輸出影像資料夾
        self.image_folder = images_dir
        self.labels_folder = labels_dir
        self.image_paths = self._discover_images(images_dir)
        if not self.image_paths:
            self._close_inference_msg_box()
            QtWidgets.QMessageBox.warning(self, "No Images", "自動輸出中未找到影像檔。")
            self._update_ui_after_image_change()
            return
        self.current_index = 0
        self._load_current_image()
        QtCore.QTimer.singleShot(100, self._close_inference_msg_box)

    def _close_inference_msg_box(self):
        """關閉推論訊息框的輔助方法"""
        if self.inference_msg_box:
            self.inference_msg_box.close()  # 關閉對話框
            self.inference_msg_box.deleteLater()  # 延遲刪除，確保安全
            self.inference_msg_box = None

    def _has_inference_ui(self) -> bool:
        """檢查推論 UI 是否已初始化"""
        return self.inference_msg_box is not None
    
    def _on_inference_output(self, text: str):
        """接收推論輸出並顯示在訊息框中"""
        if self._has_inference_ui():
            self.inference_text.appendPlainText(text)
            self.inference_text.verticalScrollBar().setValue(self.inference_text.verticalScrollBar().maximum())
    
    def _on_inference_progress(self, current: int, total: int):
        """更新進度條"""
        if self._has_inference_ui():
            self.inference_progress.setMaximum(total)
            self.inference_progress.setValue(current)
    
    def _on_inference_error(self, error_msg: str):
        """推論失敗時的回調函數"""
        if self._has_inference_ui():
            self.inference_text.appendPlainText(f"\n[錯誤] {error_msg}")
        # 關閉推論訊息框
        self._close_inference_msg_box()
        QtWidgets.QMessageBox.critical(self, "Auto Label Failed", f"推論失敗：\n{error_msg}")


    def _restore_dock(self):
        """將標註面板重新停靠回主視窗右側並顯示。"""
        self.dock.setFloating(False)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock)
        self.dock.show()

    def _on_box_selected(self, row: int):
        """列表選取框時反映到畫布的選中索引。"""
        self.canvas.active_box_index = row
        self.canvas.update()

    def _on_canvas_selection(self, row: int):
        """畫布改變選中框時，反映到列表選取。"""
        self.box_list.setCurrentRow(row)
    
    def _on_box_added(self):
        """當新增框時，標記當前影像為已修改並同步列表。"""
        self._mark_current_image_modified()
        self._sync_list_from_boxes()
    
    def _mark_current_image_modified(self):
        """標記當前影像為已修改，儲存當前的標註狀態。"""
        if not self._is_valid_image_index():
            return
        
        # 檢查當前標註是否與檔案中的標註一致
        if self._is_annotation_unchanged(self._current_image_path):
            # 如果標註沒有改變，就不標記為已修改
            self.modified_annotations.pop(self._current_image_path, None)
            return
        
        # 深拷貝當前的 boxes 列表
        self.modified_annotations[self._current_image_path] = self._copy_boxes(self.canvas.boxes)
    
    def _read_annotation_file(self, txt_path: str, img_w: int, img_h: int, id_to_class: dict) -> List[BoundingBox]:
        """讀取標註檔案，返回 BoundingBox 列表"""
        def _read():
            boxes = []
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    box = self._parse_annotation_line(line, img_w, img_h, id_to_class)
                    if box is not None:
                        boxes.append(box)
            return boxes
        return self._safe_file_operation(_read, default=[])
    
    def _box_to_comparison_tuple(self, box: BoundingBox) -> tuple:
        """將 BoundingBox 轉換為比較用的元組"""
        return (box.class_name, box.rect.left(), box.rect.top(), box.rect.width(), box.rect.height())
    
    def _compare_boxes(self, box1: BoundingBox, box2: BoundingBox, tolerance: float = 0.1) -> bool:
        """比較兩個框是否相同（允許小的浮點數誤差）"""
        t1 = self._box_to_comparison_tuple(box1)
        t2 = self._box_to_comparison_tuple(box2)
        if t1[0] != t2[0]:  # 類別不同
            return False
        # 比較位置和尺寸（允許誤差）
        return all(abs(a - b) <= tolerance for a, b in zip(t1[1:], t2[1:]))
    
    def _is_annotation_unchanged(self, img_path: str) -> bool:
        """檢查當前畫布上的標註是否與檔案中的標註一致"""
        if self.canvas._image is None:
            return True
        
        # 讀取檔案中的標註
        txt_path = self._get_annotation_path(img_path)
        if not os.path.exists(txt_path):
            # 如果檔案不存在，檢查畫布上是否有標註
            return len(self.canvas.boxes) == 0
        
        # 載入檔案中的標註
        img_w, img_h = self.canvas._image.width(), self.canvas._image.height()
        classes = self._load_or_build_classes_file()
        id_to_class = self._build_id_to_class_map(classes)
        
        file_boxes = self._read_annotation_file(txt_path, img_w, img_h, id_to_class)
        
        # 比較數量
        if len(self.canvas.boxes) != len(file_boxes):
            return False
        
        # 比較每個框的內容（簡化比較：只比較類別和位置）
        canvas_boxes_sorted = sorted(self.canvas.boxes, key=self._box_to_comparison_tuple)
        file_boxes_sorted = sorted(file_boxes, key=self._box_to_comparison_tuple)
        
        # 逐一比較
        return all(self._compare_boxes(c_box, f_box) for c_box, f_box in zip(canvas_boxes_sorted, file_boxes_sorted))

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """將鍵盤事件轉發給畫布（提供 Delete 操作）。"""
        self.canvas.keyPressEvent(event)

    def _detect_folder_structure(self, folder: str) -> Tuple[str, Optional[str], str]:
        """
        智能判斷資料夾結構，返回 (image_folder, labels_folder, search_folder)
        
        優先順序：
        1. 標準 output 結構（有 images 和 labels 子資料夾）
        2. 選擇的是 images 資料夾，父目錄有 labels
        3. 有 labels 子資料夾但沒有 images
        4. 預設（影像同目錄）
        """
        images_sub = os.path.join(folder, "images")
        labels_sub = os.path.join(folder, "labels")
        folder_name = os.path.basename(folder)
        parent = os.path.dirname(folder)
        parent_labels = os.path.join(parent, "labels")
        
        if os.path.exists(images_sub) and os.path.exists(labels_sub):
            return images_sub, labels_sub, images_sub
        elif folder_name == "images" and os.path.exists(parent_labels):
            return folder, parent_labels, folder
        elif os.path.exists(labels_sub):
            return folder, labels_sub, folder
        else:
            return folder, None, folder
    
    def open_folder(self):
        """選取資料夾並載入影像清單，切換至第一張。"""
        default_folder = self.image_folder or self._project_root
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Image Folder", default_folder)
        if not folder:
            return
        
        # 清空已修改標註（開啟新資料夾時）
        self.modified_annotations.clear()
        self._current_image_path = None
        
        self.image_folder, self.labels_folder, search_folder = self._detect_folder_structure(folder)
        self.image_paths = self._discover_images(search_folder)
        
        if not self.image_paths:
            QtWidgets.QMessageBox.warning(self, "No Images", "此資料夾沒有可支援的影像檔。")
            self._update_ui_after_image_change()
            return
        self.current_index = 0
        self._load_current_image()

    def _discover_images(self, folder: str) -> List[str]:
        """掃描資料夾內支援格式的影像檔並回傳排序清單。"""
        paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in self.SUPPORTED_IMAGE_EXTS
        ]
        paths.sort()
        return paths

    # ═════════════════════════════════════════════
    # 即時錄影
    # ═════════════════════════════════════════════
    def _is_recording(self) -> bool:
        return self._camera_worker is not None and self._camera_worker.isRunning()

    def toggle_live_recording(self):
        """開始或停止即時錄影。"""
        if self._is_recording():
            self.stop_live_recording()
        else:
            self.start_live_recording()

    def start_live_recording(self):
        """開啟攝影機，即時顯示並錄影，同時邊錄邊計算擊球次數。"""
        if self._is_recording():
            return
        record_dir = os.path.join(self._project_root, "recordings")
        out_path = os.path.join(record_dir, f"live_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

        # 停止原本影片的擊球計算結果，切到播放畫面
        self.viewer.setCurrentWidget(self.video_pane)
        self.video_pane.begin_live()
        self.video_pane.setFocus()
        self._hit_token += 1
        self._hit_frames, self._crossings = [], []
        self._hits_ready = True
        self.frame_label.setText("Hit 到 0 次球")
        self.hit_total_label.setText("攝影機啟動中...")
        self.set_mistake_count(0)
        self._reset_countdown()

        worker = CameraWorker(self.camera_spin.value(), out_path, self.TRIPWIRE_REGION, self)
        worker.started_ok.connect(self._on_live_started)
        worker.frameReady.connect(self._on_live_frame)
        worker.hitsUpdated.connect(self._on_live_hits)
        worker.failed.connect(self._on_live_failed)
        worker.stopped.connect(self._on_live_stopped)
        worker.finished.connect(self._on_live_thread_finished)
        self._camera_worker = worker
        self.camera_spin.setEnabled(False)
        self.act_live.setText("■ 停止錄影 (R)")
        worker.start()

    def stop_live_recording(self):
        """停止錄影；錄好的影片會自動載入，可以回放檢查。"""
        if self._camera_worker is not None:
            self._camera_worker.stop()
            self.act_live.setEnabled(False)
            self.act_live.setText("儲存錄影中...")

    def _on_live_started(self, fps: float, width: int, height: int, path: str):
        self._live_fps = fps
        self.setWindowTitle(f"影像標註工具 - 即時錄影中 ({width}x{height}, {fps:.0f} fps)")
        self.hit_total_label.setText("錄影中，邊錄邊計算")
        self._on_playback_changed(True)  # 開始倒數

    def _on_live_frame(self, image: QtGui.QImage, frame_idx: int):
        self.video_pane.show_live_frame(image, frame_idx, self._live_fps)

    def _set_crossings(self, crossings: list):
        """記錄所有過網 [(幀, 方向)]：擊球次數用幀，左右失誤用方向。"""
        self._crossings = sorted((int(f), d) for f, d in crossings)
        self._hit_frames = [f for f, _ in self._crossings]

    def _on_live_hits(self, crossings: list):
        self._set_crossings(crossings)
        self._refresh_hit_label(self.video_pane.frame_idx)

    def _on_live_failed(self, message: str):
        QtWidgets.QMessageBox.warning(self, "即時錄影", message)

    def _on_live_stopped(self, path: str, total_frames: int, crossings: list):
        """錄影結束：顯示最終次數，並把錄好的影片載入播放器。"""
        from count_hits import detect_mistakes_by_side
        self._on_playback_changed(False)
        self.video_pane.end_live()
        if total_frames <= 0 or not os.path.exists(path):
            return
        mistakes = detect_mistakes_by_side(crossings, self._live_fps, total_frames - 1, self.MISTAKE_GAP_SEC)
        left = sum(1 for _, s in mistakes if s == 'L')
        right = sum(1 for _, s in mistakes if s == 'R')
        self._load_video(path, autoplay=False, hit_frames=crossings)
        QtWidgets.QMessageBox.information(
            self, "錄影完成",
            f"共打到 {len(crossings)} 次球，失誤 {len(mistakes)} 次"
            f"（左方 {left} 次、右方 {right} 次）。\n\n"
            f"錄影檔已儲存：\n{path}\n\n可以按「播放」回放檢查。")

    def _on_live_thread_finished(self):
        """攝影機執行緒結束後恢復按鈕狀態。"""
        self._on_playback_changed(False)
        self.video_pane.end_live()
        self._camera_worker = None
        self.camera_spin.setEnabled(True)
        self.act_live.setEnabled(True)
        self.act_live.setText("● 開始錄影 (R)")
        if self.video_pane.path is None:
            self.setWindowTitle("影像標註工具 (Simple Labeling)")

    def open_video(self):
        """選擇一支影片，直接播放並在側邊顯示擊球次數。"""
        if self._is_recording():
            QtWidgets.QMessageBox.information(self, "錄影中", "請先停止錄影，再開啟其他影片。")
            return
        start_dir = os.path.dirname(self.video_pane.path) if self.video_pane.path else self._project_root
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "選擇影片", start_dir, "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if video_path:
            self._load_video(video_path)

    def _load_video(self, video_path: str, autoplay: bool = True, hit_frames: Optional[list] = None):
        """載入影片、開始播放，並在背景計算擊球次數。
        hit_frames 有值時（剛錄完的影片）直接使用錄影時算好的結果，不再重算。"""
        self.viewer.setCurrentWidget(self.video_pane)
        self.video_pane.setFocus()
        if not self.video_pane.load(video_path):
            QtWidgets.QMessageBox.warning(self, "無法開啟影片", f"無法讀取：\n{video_path}")
            return
        self.setWindowTitle(f"影像標註工具 - {os.path.basename(video_path)}")
        self._hit_token += 1
        self._hit_frames, self._crossings = [], []
        self._hits_ready = False
        self.frame_label.setText("計算擊球次數中...")
        self.hit_total_label.setText("全片共 — 次")
        self.set_mistake_count(None)
        self._reset_countdown()
        if hit_frames is not None:
            self._on_hits_ready(self._hit_token, hit_frames)
        else:
            worker = HitCountWorker(self._hit_token, video_path, self.TRIPWIRE_REGION, self)
            worker.hitsReady.connect(self._on_hits_ready)
            worker.hitsFailed.connect(self._on_hits_failed)
            self._hit_worker = worker
            worker.start()
        if autoplay:
            self.video_pane.play()

    def _on_hits_ready(self, token: int, crossings: list):
        """擊球計算完成後，依目前播放位置更新側邊次數。crossings: [(幀, 方向)]"""
        if token != self._hit_token:
            return
        self._set_crossings(crossings)
        self._hits_ready = True
        self._refresh_hit_label(self.video_pane.frame_idx)

    def _on_hits_failed(self, token: int, message: str):
        """擊球計算失敗時保留影片播放，並把原因顯示在側邊。"""
        if token != self._hit_token:
            return
        self._hit_frames, self._crossings = [], []
        self._hits_ready = False
        self.frame_label.setText("Hit 到 0 次球")
        self.hit_total_label.setText("無法計算擊球次數")
        QtWidgets.QMessageBox.warning(self, "擊球次數", f"計算失敗：\n{message}")

    def _refresh_hit_label(self, frame_idx: int):
        """側邊顯示播到目前這一刻，累計打到幾次球。"""
        if not self._hits_ready:
            return
        count = bisect.bisect_right(self._hit_frames, frame_idx)
        self.frame_label.setText(f"Hit 到 {count} 次球")
        if self.video_pane.live:
            self.hit_total_label.setText("錄影中，邊錄邊計算")
        else:
            self.hit_total_label.setText(f"全片共 {len(self._hit_frames)} 次")
        self._refresh_mistake_label(frame_idx)

    def _refresh_mistake_label(self, frame_idx: int):
        """失誤次數：球過網後超過 MISTAKE_GAP_SEC 秒都沒有下一次過網，算失誤一次。
        依最後一球的方向分成左方 / 右方失誤（L>R 之後沒回球 → 右方失誤）。"""
        from count_hits import detect_mistakes_by_side
        fps = self.video_pane.fps or 30.0
        if self.video_pane.live:
            # 過網要約 0.1 秒後才會確認，留一點緩衝避免誤判
            end_frame = frame_idx - int(0.2 * fps)
        else:
            end_frame = max(0, self.video_pane.total - 1)
        mistakes = detect_mistakes_by_side(self._crossings, fps, end_frame, self.MISTAKE_GAP_SEC)
        seen = [s for m, s in mistakes if m <= frame_idx]  # 只算播到目前這一刻的失誤
        self.set_mistake_count(seen.count('L'), seen.count('R'))

    def _reset_hit_label(self):
        """沒有載入影片時，側邊擊球次數歸零。"""
        self._hit_token += 1
        self._hit_frames, self._crossings = [], []
        self._hits_ready = False
        self.frame_label.setText("Hit 到 0 次球")
        self.hit_total_label.setText("全片共 0 次")
        self.set_mistake_count(0)

    def _update_ui_after_image_change(self):
        """統一更新影像切換後的 UI 狀態"""
        self._sync_list_from_boxes()
    
    def _load_current_image(self):
        """載入當前索引影像、讀取其標註並同步列表顯示。"""
        if not (0 <= self.current_index < len(self.image_paths)):
            return
        self.video_pane.pause()
        self.viewer.setCurrentWidget(self.canvas)
        
        # 在切換前，儲存當前影像的標註狀態（如果有修改）
        if self._current_image_path:
            self._mark_current_image_modified()
        
        path = self.image_paths[self.current_index]
        self._current_image_path = path
        self.setWindowTitle(f"Simple Labeling - {os.path.basename(path)} ({self.current_index + 1}/{len(self.image_paths)})")
        
        # 載入影像
        self.canvas.set_image(path)
        
        # 如果該影像有已修改的標註，優先使用修改後的標註
        if path in self.modified_annotations:
            self.canvas.boxes = self._copy_boxes(self.modified_annotations[path])
        else:
            self._load_annotations_for(path)
        
        self._update_ui_after_image_change()

    def _format_box_item(self, index: int, box: BoundingBox) -> str:
        """格式化框列表項目文字"""
        r = box.rect
        return f"[{index+1}] {box.class_name} - x:{int(r.left())}, y:{int(r.top())}, w:{int(r.width())}, h:{int(r.height())}"
    
    def _sync_list_from_boxes(self):
        """以畫布內的框同步右側列表內容與選取。"""
        # 保存當前選中索引
        active_idx = self.canvas.active_box_index
        
        # 暫時阻止信號，避免 clear() 觸發 currentRowChanged 導致 active_box_index 被設為 -1
        self.box_list.blockSignals(True)
        self.box_list.clear()
        for i, box in enumerate(self.canvas.boxes):
            self.box_list.addItem(self._format_box_item(i, box))
        self.box_list.setCurrentRow(active_idx)
        self.box_list.blockSignals(False)

    def _navigate_image(self, direction: int):
        """切換影像（direction: -1 為前一張，1 為下一張）"""
        if not self.image_paths:
            return
        self.current_index = (self.current_index + direction) % len(self.image_paths)
        self._load_current_image()
    
    def prev_image(self):
        """切到前一張影像。"""
        self._navigate_image(-1)

    def next_image(self):
        """切到下一張影像。"""
        self._navigate_image(1)

    def delete_selected(self):
        """刪除目前列表選取的框並同步畫布。"""
        idx = self.box_list.currentRow()
        if 0 <= idx < len(self.canvas.boxes):
            del self.canvas.boxes[idx]
            self.canvas.active_box_index = min(idx, len(self.canvas.boxes) - 1)
            self._mark_current_image_modified()
            self._sync_list_from_boxes()
            self.canvas.update()
    
    def _clear_all_data(self):
        """清除所有資料（影像路徑、標註、畫布狀態）"""
        self.image_folder = None
        self.labels_folder = None
        self.image_paths = []
        self.current_index = -1
        self._current_image_path = None
        self.modified_annotations.clear()
        self.video_pane.close_video()
        self.viewer.setCurrentWidget(self.video_pane)
        self._reset_hit_label()
        self._reset_countdown()
        self.set_mistake_count(None)
        
        # 清除畫布
        self.canvas.boxes.clear()
        self.canvas.active_box_index = -1
        self.canvas._image = None
        self.canvas._pixmap = None
    
    def clear_current_image(self):
        """清除所有影像和標籤，讓使用者可以打開新的影像或影片，不會累加。如果有未儲存的修改，會先顯示提醒。"""
        # 先標記當前影像為已修改（如果正在查看某張影像）
        if self._is_valid_image_index():
            self._mark_current_image_modified()
        
        # 檢查是否有未儲存的修改
        if self.modified_annotations:
            reply = QtWidgets.QMessageBox.question(
                self,
                "未儲存的修改",
                f"您有 {len(self.modified_annotations)} 個未儲存的標註修改。\n\n清除所有內容將會丟失未儲存的修改。\n\n是否要繼續清除？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            
            if reply == QtWidgets.QMessageBox.No:
                return
        
        # 清除所有資料
        self._clear_all_data()
        
        # 更新顯示
        self._update_ui_after_image_change()
        self.setWindowTitle("Simple Labeling (mini anylabeling)")
        self.canvas.update()

    def _collect_used_classes(self, boxes: List[BoundingBox], img_w: int, img_h: int) -> Tuple[List[str], List[str]]:
        """收集使用的類別並生成 YOLO 格式行，返回 (used_classes, yolo_lines)"""
        used_classes = []
        class_to_id = {}
        yolo_lines = []
        
        for box in boxes:
            if box.class_name not in class_to_id:
                class_to_id[box.class_name] = len(used_classes)
                used_classes.append(box.class_name)
            cls_id = class_to_id[box.class_name]
            yolo_lines.append(f"{cls_id} {box.to_yolo(img_w, img_h)}")
        
        return used_classes, yolo_lines
    
    def save_annotations(self, auto: bool = False):
        """將所有已修改的影像標註以 YOLO 文字檔與 classes.txt 儲存。

        參數:
            auto: 若為 True，不顯示完成提示。
        """
        if not self.image_paths:
            return
        
        # 先標記當前影像為已修改（如果正在查看某張影像）
        if self._is_valid_image_index():
            self._mark_current_image_modified()
        
        if not self.modified_annotations:
            if not auto:
                QtWidgets.QMessageBox.information(self, "No Changes", "沒有需要儲存的修改。")
            return
        
        # 收集所有已修改影像使用的類別
        all_used_classes = []
        saved_count = 0
        failed_paths = []
        
        # 遍歷所有已修改的影像並儲存
        for img_path, boxes in self.modified_annotations.items():
            if not os.path.exists(img_path):
                failed_paths.append(img_path)
                continue
            
            img = self._load_image_safely(img_path)
            if img is None:
                failed_paths.append(img_path)
                continue
            
            used_classes, yolo_lines = self._collect_used_classes(boxes, img.width(), img.height())
            all_used_classes.extend(used_classes)
            
            # 寫入 YOLO 標註檔
            if self._write_annotation_file(img_path, yolo_lines):
                saved_count += 1
            else:
                failed_paths.append(img_path)
        
        # 更新 classes.txt（使用所有已修改影像的類別）
        if all_used_classes:
            # 去重並保持順序
            unique_classes = list(dict.fromkeys(all_used_classes))
            classes_path = self._get_classes_path()
            self._update_classes_file(classes_path, unique_classes)
        
        # 清空已修改標註字典
        self.modified_annotations.clear()
        
        # 重新載入當前影像的標註，確保畫布上的標註和檔案中的標註一致
        # 這樣下次檢查時就不會誤判為已修改
        if self._is_valid_image_index():
            current_path = self.image_paths[self.current_index]
            if current_path == self._current_image_path:
                # 重新載入標註
                self._load_annotations_for(current_path)
                self._update_ui_after_image_change()
        
        if not auto:
            self._show_save_result_message(saved_count, failed_paths)
    
    def save_annotations_as(self):
        """選擇儲存位置，將所有標註儲存到指定資料夾。"""
        if not self.image_paths:
            QtWidgets.QMessageBox.warning(self, "No Images", "請先載入影像。")
            return
        
        # 先標記當前影像為已修改（如果正在查看某張影像）
        if self._is_valid_image_index():
            self._mark_current_image_modified()
        
        # 預設目錄
        default_dir = self.labels_folder or self.image_folder or self._project_root
        
        # 選擇儲存資料夾
        save_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "選擇標註儲存位置", default_dir)
        if not save_folder:
            return
        
        # 詢問儲存範圍
        reply = QtWidgets.QMessageBox.question(
            self,
            "儲存範圍",
            "請選擇要儲存的範圍：\n\n"
            "• 「是」：儲存所有影像的標註（包含未修改的）\n"
            "• 「否」：只儲存已修改的標註",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Yes
        )
        
        if reply == QtWidgets.QMessageBox.Cancel:
            return
        
        save_all = (reply == QtWidgets.QMessageBox.Yes)
        
        # 執行儲存
        saved_count = 0
        failed_paths = []
        all_used_classes = []
        
        # 決定要儲存的影像列表
        if save_all:
            # 儲存所有影像
            images_to_save = [(path, self._get_boxes_for_image(path)) for path in self.image_paths]
        else:
            # 只儲存已修改的
            images_to_save = list(self.modified_annotations.items())
        
        for img_path, boxes in images_to_save:
            if not boxes:  # 沒有標註的影像也要處理（建立空檔案或跳過）
                if not save_all:
                    continue
            
            img = self._load_image_safely(img_path)
            if img is None:
                failed_paths.append(img_path)
                continue
            
            used_classes, yolo_lines = self._collect_used_classes(boxes, img.width(), img.height())
            all_used_classes.extend(used_classes)
            
            # 寫入 YOLO 標註檔到指定資料夾
            txt_name = self._get_annotation_filename(img_path)
            txt_path = os.path.join(save_folder, txt_name)
            if self._write_annotation_file_to(txt_path, yolo_lines):
                saved_count += 1
            else:
                failed_paths.append(img_path)
        
        # 更新 classes.txt
        if all_used_classes:
            unique_classes = list(dict.fromkeys(all_used_classes))
            classes_path = os.path.join(save_folder, self.CLASSES_FILE)
            self._update_classes_file(classes_path, unique_classes)
        
        # 更新 labels_folder 為新的儲存位置
        self.labels_folder = save_folder
        
        # 清空已修改標註字典
        self.modified_annotations.clear()
        
        # 顯示結果
        QtWidgets.QMessageBox.information(
            self, "儲存完成",
            f"已成功儲存 {saved_count} 個標註檔到：\n{save_folder}\n\n"
            f"已更新: {self.CLASSES_FILE}\n\n"
            f"後續儲存將使用此位置。"
        )
    
    def _get_boxes_for_image(self, img_path: str) -> List[BoundingBox]:
        """取得指定影像的標註框（優先使用已修改的，否則從檔案載入）"""
        # 如果是當前顯示的影像，使用畫布上的框
        if img_path == self._current_image_path:
            return self._copy_boxes(self.canvas.boxes)
        # 如果有已修改的標註，使用修改後的
        if img_path in self.modified_annotations:
            return self.modified_annotations[img_path]
        # 否則從檔案載入
        return self._load_boxes_from_file(img_path)
    
    def _load_boxes_from_file(self, img_path: str) -> List[BoundingBox]:
        """從檔案載入標註框"""
        txt_path = self._get_annotation_path(img_path)
        if not os.path.exists(txt_path):
            return []
        
        img = self._load_image_safely(img_path)
        if img is None:
            return []
        
        classes = self._load_or_build_classes_file()
        id_to_class = self._build_id_to_class_map(classes)
        return self._read_annotation_file(txt_path, img.width(), img.height(), id_to_class)
    
    def _write_annotation_file_to(self, txt_path: str, yolo_lines: List[str]) -> bool:
        """寫入 YOLO 標註檔案到指定路徑，返回是否成功"""
        def _write():
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
            return True
        return self._safe_file_operation(_write, default=False) is not False
    
    def _show_save_result_message(self, saved_count: int, failed_paths: List[str]):
        """顯示儲存結果訊息"""
        if failed_paths:
            failed_names = [os.path.basename(p) for p in failed_paths[:5]]
            msg = f"已儲存 {saved_count} 個標註檔。\n\n失敗的檔案 ({len(failed_paths)} 個):\n" + "\n".join(failed_names)
            if len(failed_paths) > 5:
                msg += f"\n... 還有 {len(failed_paths) - 5} 個檔案"
            QtWidgets.QMessageBox.warning(self, "Save Completed with Errors", msg)
        else:
            QtWidgets.QMessageBox.information(
                self, "Saved", 
                f"已成功儲存 {saved_count} 個標註檔。\n\n已更新: {self.CLASSES_FILE}"
            )

    def _update_classes_file(self, classes_path: str, used_classes: List[str]):
        """更新 classes.txt 檔案，合併現有類別和新增的類別。"""
        existing_classes = self._read_classes_file(classes_path)
        # 合併現有類別和新增類別，保持順序且去重
        all_classes = list(dict.fromkeys(existing_classes + used_classes))
        self._write_classes_file(classes_path, all_classes)

    def _build_id_to_class_map(self, classes: List[str]) -> dict:
        """建立類別 ID 到類別名稱的映射字典"""
        return {i: c for i, c in enumerate(classes)}
    
    def _load_or_build_classes_file(self) -> List[str]:
        """讀取或建立 classes.txt。"""
        default_classes = list(self.DEFAULT_CLASSES)
        if not self.image_folder:
            return default_classes

        classes_path = self._get_classes_path()
        lines = self._read_classes_file(classes_path)
        classes = lines or default_classes
        if "ball" in classes:
            self.canvas.set_current_class("ball")
        elif classes:
            self.canvas.set_current_class(classes[0])
        return classes

    def _on_playback_changed(self, playing: bool):
        """影片播放時倒數，暫停或停止時停住。"""
        if playing and self._countdown_left_ms > 0:
            self._countdown_mark = time.monotonic()
            self._countdown_timer.start()
        else:
            self._countdown_timer.stop()
            self._countdown_mark = None

    def _on_countdown_tick(self):
        """依實際經過時間減少剩餘秒數。"""
        now = time.monotonic()
        if self._countdown_mark is not None:
            elapsed_ms = int((now - self._countdown_mark) * 1000)
            self._countdown_left_ms = max(0, self._countdown_left_ms - elapsed_ms)
        self._countdown_mark = now
        self._update_countdown_label()
        if self._countdown_left_ms <= 0:
            self._countdown_timer.stop()
            self._countdown_mark = None
            # 即時錄影時，倒數 3 分鐘結束就自動停止錄影
            if self._is_recording():
                self.stop_live_recording()

    def _reset_countdown(self):
        """倒數回到預設 3 分鐘。"""
        self._countdown_timer.stop()
        self._countdown_mark = None
        self._countdown_left_ms = self.COUNTDOWN_MS
        self._update_countdown_label()

    def _update_countdown_label(self):
        """更新側邊倒數顯示。"""
        total_seconds = self._countdown_left_ms // 1000
        self.countdown_label.setText(f"{total_seconds // 60:02d}:{total_seconds % 60:02d}")

    def set_mistake_count(self, left: Optional[int], right: Optional[int] = None):
        """更新左方 / 右方失誤次數。只傳一個值（0 或 None）時兩格一起設定；None 顯示「—」。"""
        if right is None:
            right = left
        self.left_mistake_label.setText("—" if left is None else f"{left} 次")
        self.right_mistake_label.setText("—" if right is None else f"{right} 次")

    def _get_annotation_path(self, img_path: str) -> str:
        """根據標註資料夾設置決定標註檔案路徑。"""
        if self.labels_folder:
            txt_name = self._get_annotation_filename(img_path)
            return os.path.join(self.labels_folder, txt_name)
        return os.path.splitext(img_path)[0] + ".txt"
    
    def _write_annotation_file(self, img_path: str, yolo_lines: List[str]) -> bool:
        """寫入 YOLO 標註檔案，返回是否成功"""
        txt_path = self._get_annotation_path(img_path)
        def _write():
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
            return True
        return self._safe_file_operation(_write, default=False) is not False

    def _parse_annotation_line(self, line: str, img_w: int, img_h: int, id_to_class: dict) -> Optional[BoundingBox]:
        """解析單行標註，返回 BoundingBox 或 None"""
        line = line.strip()
        if not line:
            return None
        parts = line.split()
        if len(parts) != 5:
            return None
        try:
            cls_id = int(parts[0])
            rect = BoundingBox.from_yolo(" ".join(parts[1:]), img_w, img_h)
            cls_name = id_to_class.get(cls_id, "object")
            return BoundingBox(cls_name, rect)
        except (ValueError, IndexError):
            return None
    
    def _load_annotations_for(self, img_path: str):
        """讀取單張影像對應的 YOLO 標註檔，載入至畫布框清單。"""
        self.canvas.boxes.clear()
        if self.canvas._image is None:
            return
        img_w, img_h = self.canvas._image.width(), self.canvas._image.height()

        classes = self._load_or_build_classes_file()
        id_to_class = self._build_id_to_class_map(classes)
        txt_path = self._get_annotation_path(img_path)
        
        if not os.path.exists(txt_path):
            return
        
        boxes = self._read_annotation_file(txt_path, img_w, img_h, id_to_class)
        self.canvas.boxes.extend(boxes)


def run():
    """啟動應用程式並顯示主視窗。"""
    app = QtWidgets.QApplication(sys.argv)
    win = LabelingWindow()
    win.show()
    
    # 連接應用程式退出信號（作為備用方案，如果 closeEvent 沒有被觸發）
    def on_about_to_quit():
        print("應用程式即將退出")  # 調試用
        # 檢查是否有未儲存的修改
        if hasattr(win, 'modified_annotations') and win.modified_annotations:
            # 如果 closeEvent 沒有被觸發，這裡可以作為備用處理
            # 但由於 aboutToQuit 時已經太晚了，我們主要還是依賴 closeEvent
            pass
    
    app.aboutToQuit.connect(on_about_to_quit)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()


