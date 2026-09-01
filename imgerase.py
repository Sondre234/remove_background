#!/usr/bin/env python3
"""
imgerase - erase parts of an image with the cursor, with deep zoom.

Erase paints alpha to 0; restore paints the original pixels back, so
overshooting at high zoom is cheap to fix. Saves PNG with alpha.
"""

import math
import os
import sys

import numpy as np
from PySide6.QtCore import QDir, QPoint, QPointF, QRect, QRectF, Qt, QUrl
from PySide6.QtGui import (QAction, QActionGroup, QBrush, QColor, QGuiApplication,
                           QImage, QKeySequence, QPainter, QPen, QPixmap,
                           QPolygonF)
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QLabel,
                               QMainWindow, QMessageBox, QSpinBox, QToolBar,
                               QWidget)

MIN_SCALE = 0.02
MAX_SCALE = 64.0
GRID_SCALE = 12.0        # show the pixel grid at or above this zoom
UNDO_LIMIT = 40

BACKGROUNDS = [
    ("Checker", None),
    ("Black", QColor(0, 0, 0)),
    ("White", QColor(255, 255, 255)),
    ("Magenta", QColor(255, 0, 255)),
    ("Mid grey", QColor(128, 128, 128)),
]


def qimage_pair(w, h):
    """A premultiplied-ARGB QImage plus an (h, w, 4) numpy view of its pixels."""
    qi = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    qi.fill(0)
    bpl = qi.bytesPerLine()
    flat = np.asarray(qi.bits()).view(np.uint8).reshape(h, bpl)
    arr = flat.reshape(h, bpl // 4, 4)[:, :w, :]
    if not arr.flags.writeable:
        raise RuntimeError("QImage buffer came back read-only")
    return qi, arr


def load_array(path):
    """Load any image file into an (h, w, 4) premultiplied BGRA array."""
    src = QImage(path)
    if src.isNull():
        raise ValueError("could not decode %s" % os.path.basename(path))
    src = src.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    w, h = src.width(), src.height()
    bpl = src.bytesPerLine()
    flat = np.asarray(src.constBits()).view(np.uint8).reshape(h, bpl)
    return flat.reshape(h, bpl // 4, 4)[:, :w, :].copy()


def build_kernel(radius, hardness, strength):
    """Round brush stamp as uint8 alpha, plus the centre offset within it."""
    r = max(0.5, float(radius))
    n = int(math.ceil(r - 0.5)) * 2 + 1
    c = (n - 1) / 2.0
    y, x = np.mgrid[0:n, 0:n]
    d = np.hypot(x - c, y - c) / r
    if hardness >= 0.999:
        # one image pixel of antialiasing, so edges are not stair-stepped
        a = np.clip((1.0 - d) * r + 0.5, 0.0, 1.0)
    else:
        t = np.clip((d - hardness) / (1.0 - hardness), 0.0, 1.0)
        a = 1.0 - (t * t * (3.0 - 2.0 * t))
    a = a.astype(np.float32) * float(strength)
    return (a * 255.0 + 0.5).astype(np.uint8), c


def make_checker():
    pm = QPixmap(16, 16)
    pm.fill(QColor(74, 74, 78))
    p = QPainter(pm)
    p.fillRect(0, 0, 8, 8, QColor(94, 94, 99))
    p.fillRect(8, 8, 8, 8, QColor(94, 94, 99))
    p.end()
    return pm


class Canvas(QWidget):
    def __init__(self, window):
        super().__init__()
        self.win = window
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.BlankCursor)

        self.qimg = None
        self.arr = None          # live pixels
        self.orig = None         # pristine pixels, source for restore
        self.pre = None          # pixels as of the current stroke's start
        self.mask = None         # accumulated alpha of the current stroke
        self.w = self.h = 0
        self.path = None
        self.dirty = False

        self.scale = 1.0
        self.origin = QPointF(0, 0)   # widget coords of image pixel (0, 0)

        self.brush = 32.0             # diameter, image pixels
        self.hardness = 0.9
        self.strength = 1.0
        self.mode = "erase"
        self.tool = "brush"
        self.tolerance = 32
        self._kernel = None
        self._kernel_key = None

        self.bg_index = 0
        self.checker = make_checker()
        self.show_grid = True

        self.stroke_mode = None
        self.stroke_bbox = None
        self.lasso_pts = []
        self.lasso_active = False
        self.lasso_dragging = False
        self.lasso_mode = 'erase'
        self.press_pos = None
        self.drag_dist = 0.0
        self.pts_at_press = 0
        self.last_pt = None
        self.carry = 0.0
        self.panning = False
        self.pan_anchor = None
        self.space_down = False
        self.cursor_pos = None
        self._needs_fit = False

        self.undo_stack = []
        self.redo_stack = []

    # ---------- image lifecycle ----------

    def load(self, path):
        try:
            src = load_array(path)
        except ValueError as exc:
            QMessageBox.warning(self, "imgerase", str(exc))
            return False
        h, w = src.shape[:2]
        self.qimg, self.arr = qimage_pair(w, h)
        self.arr[:] = src
        self.orig = src
        self.pre = np.empty_like(src)
        self.mask = np.zeros((h, w), np.uint8)
        self.w, self.h = w, h
        self.path = path
        self.dirty = False
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.fit()
        self._needs_fit = True
        self.win.refresh()
        return True

    def save(self, path):
        out = QImage(self.qimg).convertToFormat(QImage.Format.Format_ARGB32)
        if not out.save(path, "PNG"):
            QMessageBox.warning(self, "imgerase", "Could not write %s" % path)
            return False
        self.path = path
        self.dirty = False
        self.win.refresh()
        return True

    # ---------- coordinate transforms ----------

    def to_image(self, pt):
        return QPointF((pt.x() - self.origin.x()) / self.scale,
                       (pt.y() - self.origin.y()) / self.scale)

    def to_widget(self, pt):
        return QPointF(pt.x() * self.scale + self.origin.x(),
                       pt.y() * self.scale + self.origin.y())

    def image_rect(self):
        return QRectF(self.origin, QPointF(self.origin.x() + self.w * self.scale,
                                           self.origin.y() + self.h * self.scale))

    def fit(self):
        if self.arr is None:
            return
        self._needs_fit = False
        pad = 16
        sw = max(1, self.width() - pad * 2)
        sh = max(1, self.height() - pad * 2)
        s = min(sw / self.w, sh / self.h)
        self.scale = max(MIN_SCALE, min(MAX_SCALE, s))
        self.center()

    def center(self):
        self.origin = QPointF((self.width() - self.w * self.scale) / 2.0,
                              (self.height() - self.h * self.scale) / 2.0)
        self.update()
        self.win.refresh()

    def zoom_to(self, new_scale, anchor=None):
        if self.arr is None:
            return
        new_scale = max(MIN_SCALE, min(MAX_SCALE, new_scale))
        if abs(new_scale - self.scale) < 1e-9:
            return
        if anchor is None:
            anchor = QPointF(self.width() / 2.0, self.height() / 2.0)
        img_pt = self.to_image(anchor)
        self.scale = new_scale
        self.origin = QPointF(anchor.x() - img_pt.x() * new_scale,
                              anchor.y() - img_pt.y() * new_scale)
        self.update()
        self.win.refresh()

    # ---------- painting ----------

    def kernel(self):
        key = (round(self.brush, 2), round(self.hardness, 3), round(self.strength, 3))
        if key != self._kernel_key:
            self._kernel = build_kernel(self.brush / 2.0, self.hardness, self.strength)
            self._kernel_key = key
        return self._kernel

    def stamp(self, cx, cy):
        k, c = self.kernel()
        n = k.shape[0]
        x0 = int(math.floor(cx) - c)
        y0 = int(math.floor(cy) - c)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(self.w, x0 + n), min(self.h, y0 + n)
        if dx1 <= dx0 or dy1 <= dy0:
            return None
        sub = k[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0]
        dst = self.mask[dy0:dy1, dx0:dx1]
        np.maximum(dst, sub, out=dst)
        return [dx0, dy0, dx1, dy1]

    def composite(self, rect):
        """Rebuild `rect` from the stroke-start pixels and the accumulated mask."""
        x0, y0, x1, y1 = rect
        m = (self.mask[y0:y1, x0:x1].astype(np.float32) / 255.0)[..., None]
        pre = self.pre[y0:y1, x0:x1].astype(np.float32)
        if self.stroke_mode == "restore":
            src = self.orig[y0:y1, x0:x1].astype(np.float32)
            out = pre * (1.0 - m) + src * m
        else:
            out = pre * (1.0 - m)
        self.arr[y0:y1, x0:x1] = (out + 0.5).astype(np.uint8)

    def start_edit(self, mode):
        """Snapshot the pixels and clear the mask, ready for a brush or lasso edit."""
        self.stroke_mode = mode
        np.copyto(self.pre, self.arr)
        if self.stroke_bbox is not None:
            x0, y0, x1, y1 = self.stroke_bbox
            self.mask[y0:y1, x0:x1] = 0
        else:
            self.mask[:] = 0
        self.stroke_bbox = None

    def begin_stroke(self, img_pt, mode):
        self.start_edit(mode)
        self.last_pt = img_pt
        self.carry = 0.0
        self.apply_segment(img_pt, img_pt)

    def grow_bbox(self, seg):
        if self.stroke_bbox is None:
            self.stroke_bbox = seg[:]
        else:
            sb = self.stroke_bbox
            sb[0] = min(sb[0], seg[0]); sb[1] = min(sb[1], seg[1])
            sb[2] = max(sb[2], seg[2]); sb[3] = max(sb[3], seg[3])

    def apply_segment(self, a, b):
        spacing = max(0.35, (self.brush / 2.0) * 0.2)
        dx, dy = b.x() - a.x(), b.y() - a.y()
        dist = math.hypot(dx, dy)
        rects = []
        if dist < 1e-6:
            r = self.stamp(a.x(), a.y())
            if r:
                rects.append(r)
        else:
            t = self.carry
            while t <= dist:
                f = t / dist
                r = self.stamp(a.x() + dx * f, a.y() + dy * f)
                if r:
                    rects.append(r)
                t += spacing
            self.carry = t - dist
        if not rects:
            return
        seg = rects[0][:]
        for r in rects[1:]:
            seg[0] = min(seg[0], r[0]); seg[1] = min(seg[1], r[1])
            seg[2] = max(seg[2], r[2]); seg[3] = max(seg[3], r[3])
        self.composite(seg)
        self.grow_bbox(seg)
        wr = QRectF(self.to_widget(QPointF(seg[0], seg[1])),
                    self.to_widget(QPointF(seg[2], seg[3])))
        self.update(wr.toAlignedRect().adjusted(-2, -2, 2, 2))

    def similar_to(self, sx, sy):
        """Bool map of pixels within tolerance of the seed, channel by channel.

        The pixels are premultiplied BGRA, so an erased pixel is (0,0,0,0) and
        differs from any opaque colour by a full 255 in alpha - which is exactly
        why a brushed-out outline stops the flood.
        """
        seed = self.arr[sy, sx]
        tol = int(self.tolerance)
        similar = np.ones(self.arr.shape[:2], bool)
        for c in range(4):
            d = self.arr[..., c].astype(np.int16)
            d -= int(seed[c])
            np.abs(d, out=d)
            similar &= d <= tol
        return similar

    def flood_region(self, similar, sx, sy):
        """Connected run-fill from the seed. Scanline spans, so the Python loop
        runs once per span rather than once per pixel."""
        h, w = similar.shape
        out = np.zeros((h, w), bool)
        if not similar[sy, sx]:
            return out
        stack = [(sx, sy)]
        while stack:
            x, y = stack.pop()
            row = similar[y] & ~out[y]
            if not row[x]:
                continue
            blocked = np.flatnonzero(~row[:x])
            x0 = blocked[-1] + 1 if blocked.size else 0
            blocked = np.flatnonzero(~row[x + 1:])
            x1 = x + blocked[0] if blocked.size else w - 1
            out[y, x0:x1 + 1] = True
            for ny in (y - 1, y + 1):
                if not (0 <= ny < h):
                    continue
                seg = similar[ny, x0:x1 + 1] & ~out[ny, x0:x1 + 1]
                idx = np.flatnonzero(seg)
                if not idx.size:
                    continue
                # one seed per contiguous run in the neighbouring row
                starts = np.concatenate(([idx[0]], idx[np.flatnonzero(np.diff(idx) > 1) + 1]))
                for start in starts:
                    stack.append((x0 + int(start), ny))
        return out

    def fill_at(self, img_pt, mode):
        """Bucket fill: take everything connected to the clicked pixel."""
        sx = int(math.floor(img_pt.x()))
        sy = int(math.floor(img_pt.y()))
        if not (0 <= sx < self.w and 0 <= sy < self.h):
            return
        region = self.flood_region(self.similar_to(sx, sy), sx, sy)
        ys, xs = np.where(region)
        if not xs.size:
            return
        seg = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        self.start_edit(mode)
        sub = region[seg[1]:seg[3], seg[0]:seg[2]]
        value = int(round(255 * self.strength))
        dst = self.mask[seg[1]:seg[3], seg[0]:seg[2]]
        np.maximum(dst, np.where(sub, value, 0).astype(np.uint8), out=dst)
        self.composite(seg)
        self.grow_bbox(seg)
        self.end_stroke()
        self.update()
        self.win.statusBar().showMessage("Filled %d px" % int(region.sum()), 3000)

    def rasterize_polygon(self, pts):
        """Fill the closed outline into the stroke mask; returns its clipped bbox."""
        xs = [q.x() for q in pts]
        ys = [q.y() for q in pts]
        x0 = max(0, int(math.floor(min(xs))) - 1)
        y0 = max(0, int(math.floor(min(ys))) - 1)
        x1 = min(self.w, int(math.ceil(max(xs))) + 2)
        y1 = min(self.h, int(math.ceil(max(ys))) + 2)
        if x1 <= x0 or y1 <= y0:
            return None
        tmp, tarr = qimage_pair(x1 - x0, y1 - y0)
        pr = QPainter(tmp)
        pr.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pr.setPen(Qt.PenStyle.NoPen)
        pr.setBrush(QColor(255, 255, 255))
        pr.drawPolygon(QPolygonF([QPointF(q.x() - x0, q.y() - y0) for q in pts]))
        pr.end()
        a = tarr[..., 3]
        if self.strength < 0.999:
            a = (a.astype(np.float32) * self.strength + 0.5).astype(np.uint8)
        dst = self.mask[y0:y1, x0:x1]
        np.maximum(dst, a, out=dst)
        return [x0, y0, x1, y1]

    def commit_lasso(self):
        pts = self.lasso_pts
        mode = self.lasso_mode
        self.cancel_lasso()
        if len(pts) < 3:
            return
        self.start_edit(mode)
        seg = self.rasterize_polygon(pts)
        if seg is None:
            self.stroke_mode = None
            return
        self.composite(seg)
        self.grow_bbox(seg)
        self.end_stroke()
        self.update()

    def cancel_lasso(self):
        self.lasso_active = False
        self.lasso_dragging = False
        self.lasso_pts = []
        self.press_pos = None
        self.drag_dist = 0.0
        self.pts_at_press = 0
        self.update()

    def end_stroke(self):
        if self.stroke_mode is None:
            return
        self.stroke_mode = None
        sb = self.stroke_bbox
        if sb is None:
            return
        x0, y0, x1, y1 = sb
        before = self.pre[y0:y1, x0:x1].copy()
        after = self.arr[y0:y1, x0:x1].copy()
        if np.array_equal(before, after):
            return
        self.undo_stack.append((tuple(sb), before, after))
        if len(self.undo_stack) > UNDO_LIMIT:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.dirty = True
        self.win.refresh()

    def undo(self):
        self._step(self.undo_stack, self.redo_stack, 1)

    def redo(self):
        self._step(self.redo_stack, self.undo_stack, 2)

    def _step(self, src, dst, idx):
        if not src:
            return
        entry = src.pop()
        x0, y0, x1, y1 = entry[0]
        self.arr[y0:y1, x0:x1] = entry[idx]
        dst.append(entry)
        self.dirty = True
        self.update()
        self.win.refresh()

    # ---------- events ----------

    def resizeEvent(self, event):
        # load() runs before the compositor has sized the window, so the fit
        # it computed is against a stale size; redo it on the first real resize
        if self.arr is not None and self._needs_fit:
            self.fit()
            if self.width() > 1 and self.height() > 1:
                self._needs_fit = False
        super().resizeEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(32, 32, 36))
        if self.arr is None:
            p.setPen(QColor(150, 150, 155))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Drop an image here, or Ctrl+O to open")
            return

        target = self.image_rect()
        clipped = target.intersected(QRectF(event.rect()))
        if not clipped.isEmpty():
            p.save()
            p.setClipRect(clipped)
            name, colour = BACKGROUNDS[self.bg_index]
            if colour is None:
                p.fillRect(clipped, QBrush(self.checker))
            else:
                p.fillRect(clipped, colour)
            p.restore()

            # draw only the visible slice of the image, so deep zoom stays cheap
            src = QRectF(self.to_image(clipped.topLeft()),
                         self.to_image(clipped.bottomRight()))
            src = src.intersected(QRectF(0, 0, self.w, self.h))
            isrc = QRect(int(math.floor(src.left())), int(math.floor(src.top())),
                         0, 0)
            isrc.setRight(int(math.ceil(src.right())))
            isrc.setBottom(int(math.ceil(src.bottom())))
            isrc = isrc.intersected(QRect(0, 0, self.w, self.h))
            if not isrc.isEmpty():
                dst = QRectF(self.to_widget(QPointF(isrc.left(), isrc.top())),
                             self.to_widget(QPointF(isrc.left() + isrc.width(),
                                                    isrc.top() + isrc.height())))
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,
                                self.scale < 1.5)
                p.drawImage(dst, self.qimg, QRectF(isrc))

        if self.show_grid and self.scale >= GRID_SCALE:
            self.draw_grid(p, target.intersected(QRectF(event.rect())))

        p.setPen(QPen(QColor(120, 120, 130), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(target.adjusted(-0.5, -0.5, 0.5, 0.5))

        if self.lasso_active and self.lasso_pts:
            self.draw_lasso(p)
        self.draw_brush(p)

    def draw_lasso(self, p):
        pts = [self.to_widget(q) for q in self.lasso_pts]
        if not self.lasso_dragging and self.cursor_pos is not None:
            pts.append(QPointF(self.cursor_pos))   # rubber band to the cursor
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        accent = (QColor(120, 220, 255) if self.lasso_mode == 'restore'
                  else QColor(255, 150, 90))
        if len(pts) >= 3:
            tint = QColor(accent)
            tint.setAlpha(46)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(tint)
            p.drawPolygon(QPolygonF(pts))
        p.setBrush(Qt.BrushStyle.NoBrush)
        poly = QPolygonF(pts)
        p.setPen(QPen(QColor(0, 0, 0, 200), 3))
        p.drawPolyline(poly)
        ants = QPen(QColor(255, 255, 255), 1)
        ants.setDashPattern([4, 4])
        p.setPen(ants)
        p.drawPolyline(poly)
        if len(pts) >= 2:
            p.setPen(QPen(QColor(0, 0, 0, 160), 3))
            p.drawLine(pts[-1], pts[0])
            close = QPen(accent, 1)
            close.setDashPattern([3, 5])
            p.setPen(close)
            p.drawLine(pts[-1], pts[0])
        if not self.lasso_dragging:
            verts = [self.to_widget(v) for v in self.lasso_pts]
            p.setPen(QPen(QColor(0, 0, 0, 210), 1))
            p.setBrush(QColor(255, 255, 255))
            for q in verts[1:]:
                p.drawEllipse(q, 3.0, 3.0)
            # the start vertex is the close target, so make it unmistakable
            start = verts[0]
            p.setPen(QPen(QColor(0, 0, 0, 210), 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(start, 7.5, 7.5)
            p.setPen(QPen(accent, 2))
            p.drawEllipse(start, 7.5, 7.5)
            p.setPen(QPen(QColor(0, 0, 0, 210), 1))
            p.setBrush(accent)
            p.drawEllipse(start, 3.0, 3.0)

    def draw_grid(self, p, area):
        if area.isEmpty():
            return
        p.setPen(QPen(QColor(255, 255, 255, 36), 1))
        x = int(math.floor((area.left() - self.origin.x()) / self.scale))
        while True:
            wx = self.origin.x() + x * self.scale
            if wx > area.right():
                break
            if wx >= area.left():
                p.drawLine(QPointF(wx, area.top()), QPointF(wx, area.bottom()))
            x += 1
        y = int(math.floor((area.top() - self.origin.y()) / self.scale))
        while True:
            wy = self.origin.y() + y * self.scale
            if wy > area.bottom():
                break
            if wy >= area.top():
                p.drawLine(QPointF(area.left(), wy), QPointF(area.right(), wy))
            y += 1

    def stamp_centre(self, pos):
        """Widget position of the centre of the pixel a stamp there would land on."""
        ip = self.to_image(pos)
        return self.to_widget(QPointF(math.floor(ip.x()) + 0.5,
                                      math.floor(ip.y()) + 0.5))

    def brush_rect(self):
        if self.cursor_pos is None:
            return QRect()
        r = self.brush / 2.0 * self.scale + 4 + self.scale
        c = self.stamp_centre(self.cursor_pos)
        return QRectF(c.x() - r, c.y() - r, r * 2, r * 2).toAlignedRect()

    def draw_brush(self, p):
        if self.cursor_pos is None or self.panning or self.space_down:
            return
        if self.tool in ("lasso", "fill"):
            c = QPointF(self.cursor_pos)
            for pen, w in ((QPen(QColor(0, 0, 0, 190), 3), 7), (QPen(QColor(255, 255, 255), 1), 7)):
                p.setPen(pen)
                p.drawLine(QPointF(c.x() - w, c.y()), QPointF(c.x() + w, c.y()))
                p.drawLine(QPointF(c.x(), c.y() - w), QPointF(c.x(), c.y() + w))
            return
        r = max(2.0, self.brush / 2.0 * self.scale)
        c = self.stamp_centre(self.cursor_pos)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(0, 0, 0, 190), 3))
        p.drawEllipse(c, r, r)
        colour = QColor(120, 220, 255) if self.mode == "restore" else QColor(255, 255, 255)
        p.setPen(QPen(colour, 1))
        p.drawEllipse(c, r, r)
        p.setPen(QPen(QColor(0, 0, 0, 190), 3))
        p.drawPoint(c)
        p.setPen(QPen(colour, 1))
        p.drawPoint(c)

    def wheelEvent(self, event):
        if self.arr is None:
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            return
        mods = event.modifiers()
        if mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            self.set_brush(self.brush * (1.15 ** steps))
        else:
            self.zoom_to(self.scale * (1.25 ** steps), event.position())
        self.update(self.brush_rect().adjusted(-40, -40, 40, 40))

    def mousePressEvent(self, event):
        if self.arr is None:
            return
        self.setFocus()
        btn = event.button()
        if btn == Qt.MouseButton.MiddleButton or self.space_down:
            self.panning = True
            self.pan_anchor = event.position()
            self.update()
            return
        if btn == Qt.MouseButton.RightButton:
            if self.lasso_active:
                self.cancel_lasso()
            return
        if btn != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        if self.tool == "fill":
            self.fill_at(self.to_image(pos), self.effective_mode(event.modifiers()))
            return
        if self.tool == "lasso":
            self.press_pos = pos
            self.drag_dist = 0.0
            self.lasso_dragging = True
            if not self.lasso_active:
                self.lasso_active = True
                self.lasso_mode = self.effective_mode(event.modifiers())
                self.lasso_pts = [self.to_image(pos)]
            else:
                first = self.to_widget(self.lasso_pts[0])
                near = math.hypot(pos.x() - first.x(), pos.y() - first.y())
                if near <= 9.0 and len(self.lasso_pts) >= 3:
                    self.commit_lasso()
                    return
                self.lasso_pts.append(self.to_image(pos))
            self.pts_at_press = len(self.lasso_pts)
            self.update()
            return
        self.begin_stroke(self.to_image(pos), self.effective_mode(event.modifiers()))

    def effective_mode(self, mods):
        """Alt/Ctrl momentarily swaps erase and restore."""
        if mods & (Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.ControlModifier):
            return "restore" if self.mode == "erase" else "erase"
        return self.mode

    def mouseDoubleClickEvent(self, event):
        if self.tool == "lasso" and self.lasso_active and len(self.lasso_pts) >= 3:
            self.commit_lasso()

    def mouseMoveEvent(self, event):
        pos = event.position()
        old = self.brush_rect()
        self.cursor_pos = pos
        if self.panning and self.pan_anchor is not None:
            d = pos - self.pan_anchor
            self.origin += d
            self.pan_anchor = pos
            self.update()
            self.win.refresh()
            return
        if self.lasso_active:
            if self.lasso_dragging and self.press_pos is not None:
                d = math.hypot(pos.x() - self.press_pos.x(), pos.y() - self.press_pos.y())
                self.drag_dist = max(self.drag_dist, d)
                last = self.to_widget(self.lasso_pts[-1])
                if math.hypot(pos.x() - last.x(), pos.y() - last.y()) >= 1.5:
                    self.lasso_pts.append(self.to_image(pos))
            self.update()
            self.win.refresh()
            return
        if self.stroke_mode is not None:
            pt = self.to_image(pos)
            self.apply_segment(self.last_pt, pt)
            self.last_pt = pt
        self.update(old.united(self.brush_rect()).adjusted(-2, -2, 2, 2))
        self.win.refresh()

    def mouseReleaseEvent(self, event):
        if self.panning and event.button() in (Qt.MouseButton.MiddleButton,
                                               Qt.MouseButton.LeftButton):
            self.panning = False
            self.pan_anchor = None
            self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.lasso_active:
            self.lasso_dragging = False
            if self.drag_dist > 6.0:
                self.commit_lasso()          # a drag is a freehand outline
            else:
                del self.lasso_pts[self.pts_at_press:]   # a click places a vertex
                self.update()
            return
        self.end_stroke()

    def leaveEvent(self, event):
        old = self.brush_rect()
        self.cursor_pos = None
        self.update(old.adjusted(-2, -2, 2, 2))

    def set_brush(self, value):
        self.brush = max(1.0, min(2000.0, float(value)))
        self.win.sync_brush()
        self.update()
        self.win.refresh()

    def set_tool(self, tool):
        if tool != self.tool:
            self.cancel_lasso()
        self.tool = tool
        self.win.sync_tool()
        self.update()
        self.win.refresh()

    def set_mode(self, mode):
        self.mode = mode
        self.win.sync_mode()
        self.update()
        self.win.refresh()

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.space_down = True
            self.update()
        elif k == Qt.Key.Key_BracketLeft:
            self.set_brush(self.brush - max(1.0, self.brush * 0.15))
        elif k == Qt.Key.Key_BracketRight:
            self.set_brush(self.brush + max(1.0, self.brush * 0.15))
        elif k == Qt.Key.Key_Escape:
            self.cancel_lasso()
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.lasso_active and len(self.lasso_pts) >= 3:
                self.commit_lasso()
        elif k == Qt.Key.Key_P:
            self.set_tool("brush")
        elif k == Qt.Key.Key_L:
            self.set_tool("lasso")
        elif k == Qt.Key.Key_F:
            self.set_tool("fill")
        elif k == Qt.Key.Key_E:
            self.set_mode("erase")
        elif k == Qt.Key.Key_R:
            self.set_mode("restore")
        elif k == Qt.Key.Key_X:
            self.set_mode("restore" if self.mode == "erase" else "erase")
        elif k == Qt.Key.Key_B:
            self.bg_index = (self.bg_index + 1) % len(BACKGROUNDS)
            self.win.sync_bg()
            self.update()
        elif k == Qt.Key.Key_G:
            self.show_grid = not self.show_grid
            self.update()
        elif k == Qt.Key.Key_0:
            self.fit()
            self.update()
        elif k == Qt.Key.Key_1:
            self.zoom_to(1.0, self.cursor_pos)
        elif k in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_to(self.scale * 1.25, self.cursor_pos)
        elif k == Qt.Key.Key_Minus:
            self.zoom_to(self.scale / 1.25, self.cursor_pos)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.space_down = False
            self.update()
        else:
            super().keyReleaseEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.win.open_path(urls[0].toLocalFile())


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.canvas = Canvas(self)
        self.setCentralWidget(self.canvas)
        self.resize(1280, 860)
        self.build_toolbar()
        self.build_status()
        self.refresh()

    def build_toolbar(self):
        tb = QToolBar("Tools")
        tb.setMovable(False)
        self.addToolBar(tb)

        def act(text, shortcut, slot):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            self.addAction(a)
            tb.addAction(a)
            return a

        act("Open", "Ctrl+O", self.open_dialog)
        self.act_save = act("Save", "Ctrl+S", self.save)
        act("Save As", "Ctrl+Shift+S", self.save_as)
        tb.addSeparator()
        self.act_undo = act("Undo", "Ctrl+Z", self.canvas.undo)
        self.act_redo = act("Redo", "Ctrl+Shift+Z", self.canvas.redo)
        tb.addSeparator()

        tools = QActionGroup(self)
        self.act_brush = QAction("Brush", self, checkable=True, checked=True)
        self.act_brush.setToolTip("Brush (P) — paint over what you want gone")
        self.act_fill = QAction("Fill", self, checkable=True)
        self.act_fill.setToolTip("Fill (F) — click inside an area and everything "
                                 "connected to that pixel goes; a brushed-out "
                                 "outline stops it, like a paint bucket")
        self.act_lasso = QAction("Lasso", self, checkable=True)
        self.act_lasso.setToolTip("Lasso (L) — drag to outline freehand, or click to place "
                                  "vertices; release/double-click/Enter removes the inside, "
                                  "Esc cancels")
        for a, tool in ((self.act_brush, "brush"), (self.act_fill, "fill"),
                        (self.act_lasso, "lasso")):
            tools.addAction(a)
            tb.addAction(a)
            a.triggered.connect(lambda _=False, t=tool: self.canvas.set_tool(t))
        tb.addSeparator()

        group = QActionGroup(self)
        self.act_erase = QAction("Erase", self, checkable=True, checked=True)
        self.act_restore = QAction("Restore", self, checkable=True)
        for a, mode in ((self.act_erase, "erase"), (self.act_restore, "restore")):
            group.addAction(a)
            tb.addAction(a)
            a.triggered.connect(lambda _=False, m=mode: self.canvas.set_mode(m))
        tb.addSeparator()

        tb.addWidget(QLabel(" Size "))
        self.sp_size = QSpinBox()
        self.sp_size.setRange(1, 2000)
        self.sp_size.setValue(int(self.canvas.brush))
        self.sp_size.setSuffix(" px")
        self.sp_size.setKeyboardTracking(False)
        self.sp_size.valueChanged.connect(self.on_size)
        tb.addWidget(self.sp_size)

        tb.addWidget(QLabel("  Hardness "))
        self.sp_hard = QSpinBox()
        self.sp_hard.setRange(0, 100)
        self.sp_hard.setValue(int(self.canvas.hardness * 100))
        self.sp_hard.setSuffix(" %")
        self.sp_hard.setKeyboardTracking(False)
        self.sp_hard.valueChanged.connect(
            lambda v: setattr(self.canvas, "hardness", v / 100.0))
        tb.addWidget(self.sp_hard)

        tb.addWidget(QLabel("  Strength "))
        self.sp_flow = QSpinBox()
        self.sp_flow.setRange(1, 100)
        self.sp_flow.setValue(int(self.canvas.strength * 100))
        self.sp_flow.setSuffix(" %")
        self.sp_flow.setKeyboardTracking(False)
        self.sp_flow.valueChanged.connect(
            lambda v: setattr(self.canvas, "strength", v / 100.0))
        tb.addWidget(self.sp_flow)

        tb.addWidget(QLabel("  Fill tol "))
        self.sp_tol = QSpinBox()
        self.sp_tol.setRange(0, 255)
        self.sp_tol.setValue(self.canvas.tolerance)
        self.sp_tol.setKeyboardTracking(False)
        self.sp_tol.setToolTip("How different a pixel may be from the clicked one "
                               "and still be taken by Fill")
        self.sp_tol.valueChanged.connect(
            lambda v: setattr(self.canvas, "tolerance", v))
        tb.addWidget(self.sp_tol)

        tb.addWidget(QLabel("  Behind "))
        self.cb_bg = QComboBox()
        self.cb_bg.addItems([n for n, _ in BACKGROUNDS])
        self.cb_bg.currentIndexChanged.connect(self.on_bg)
        tb.addWidget(self.cb_bg)

        tb.addSeparator()
        act("Fit", None, self.canvas.fit)
        act("1:1", None, lambda: self.canvas.zoom_to(1.0))

    def build_status(self):
        self.lbl_zoom = QLabel()
        self.lbl_pos = QLabel()
        self.lbl_hint = QLabel("scroll zoom · MMB/space pan · P brush / F fill / L lasso · "
                               "X swap · B backdrop ")
        sb = self.statusBar()
        sb.addWidget(self.lbl_zoom)
        sb.addWidget(self.lbl_pos)
        sb.addPermanentWidget(self.lbl_hint)

    # ---------- toolbar plumbing ----------

    def on_size(self, v):
        if abs(self.canvas.brush - v) > 0.5:
            self.canvas.brush = float(v)
            self.canvas.update()

    def on_bg(self, i):
        self.canvas.bg_index = i
        self.canvas.update()

    def sync_brush(self):
        self.sp_size.blockSignals(True)
        self.sp_size.setValue(int(round(self.canvas.brush)))
        self.sp_size.blockSignals(False)

    def sync_tool(self):
        {"brush": self.act_brush, "fill": self.act_fill,
         "lasso": self.act_lasso}[self.canvas.tool].setChecked(True)

    def sync_mode(self):
        (self.act_erase if self.canvas.mode == "erase" else self.act_restore).setChecked(True)

    def sync_bg(self):
        self.cb_bg.blockSignals(True)
        self.cb_bg.setCurrentIndex(self.canvas.bg_index)
        self.cb_bg.blockSignals(False)

    def refresh(self):
        c = self.canvas
        has = c.arr is not None
        self.act_save.setEnabled(has)
        self.act_undo.setEnabled(bool(c.undo_stack))
        self.act_redo.setEnabled(bool(c.redo_stack))
        if not has:
            self.setWindowTitle("imgerase")
            self.lbl_zoom.setText("")
            self.lbl_pos.setText("")
            return
        name = os.path.basename(c.path) if c.path else "untitled"
        self.setWindowTitle("%s%s — imgerase" % (name, "*" if c.dirty else ""))
        if c.tool == "brush":
            detail = "brush %d px" % round(c.brush)
        elif c.tool == "fill":
            detail = "fill · tolerance %d" % c.tolerance
        else:
            detail = "lasso · %d pts" % len(c.lasso_pts) if c.lasso_active else "lasso"
        self.lbl_zoom.setText("  %s %s  ·  %d×%d  ·  %.0f%%  ·  %s  "
                             % (c.tool, c.mode, c.w, c.h, c.scale * 100, detail))
        if c.cursor_pos is not None:
            p = c.to_image(c.cursor_pos)
            self.lbl_pos.setText("· x %d  y %d " % (math.floor(p.x()), math.floor(p.y())))
        else:
            self.lbl_pos.setText("")

    # ---------- files ----------

    def file_dialog(self, caption, start, save):
        """Qt's own dialog, with dotfiles visible - ~/.config etc. are real targets here."""
        d = start if os.path.isdir(start) else os.path.dirname(start)
        dlg = QFileDialog(self, caption, d or os.path.expanduser("~"))
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setFilter(dlg.filter() | QDir.Filter.Hidden)
        if save:
            dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            dlg.setDefaultSuffix("png")
            dlg.setNameFilter("PNG (*.png)")
        else:
            dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
            dlg.setNameFilters([
                "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.avif)",
                "All files (*)"])
        home = os.path.expanduser("~")
        dlg.setSidebarUrls(dlg.sidebarUrls() + [
            QUrl.fromLocalFile(os.path.join(home, sub))
            for sub in (".config", "Pictures", "Downloads")
            if os.path.isdir(os.path.join(home, sub))])
        if not os.path.isdir(start):
            dlg.selectFile(os.path.basename(start))
        if dlg.exec() != QFileDialog.DialogCode.Accepted:
            return None
        picked = dlg.selectedFiles()
        return picked[0] if picked else None

    def open_dialog(self):
        start = os.path.dirname(self.canvas.path) if self.canvas.path else os.path.expanduser("~")
        path = self.file_dialog("Open image", start, save=False)
        if path:
            self.open_path(path)

    def open_path(self, path):
        if not path or not os.path.isfile(path):
            return
        if not self.confirm_discard():
            return
        self.canvas.load(path)

    def confirm_discard(self):
        if not self.canvas.dirty:
            return True
        r = QMessageBox.question(self, "imgerase", "Discard unsaved changes?",
                                 QMessageBox.StandardButton.Discard |
                                 QMessageBox.StandardButton.Cancel)
        return r == QMessageBox.StandardButton.Discard

    def save(self):
        if self.canvas.arr is None:
            return
        path = self.canvas.path
        if not path:
            return self.save_as()
        root, ext = os.path.splitext(path)
        if ext.lower() != ".png":
            # alpha needs PNG; never silently flatten into the source jpeg
            path = root + ".png"
            if os.path.exists(path):
                return self.save_as()
        self.canvas.save(path)
        self.statusBar().showMessage("Saved %s" % path, 4000)

    def save_as(self):
        if self.canvas.arr is None:
            return
        start = self.canvas.path or os.path.expanduser("~/out.png")
        start = os.path.splitext(start)[0] + ".png"
        path = self.file_dialog("Save PNG", start, save=True)
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        self.canvas.save(path)
        self.statusBar().showMessage("Saved %s" % path, 4000)

    def closeEvent(self, event):
        if self.confirm_discard():
            event.accept()
        else:
            event.ignore()


def main():
    QGuiApplication.setDesktopFileName("imgerase")
    app = QApplication(sys.argv)
    app.setApplicationName("imgerase")
    win = Window()
    win.show()
    if len(sys.argv) > 1:
        win.open_path(os.path.abspath(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
