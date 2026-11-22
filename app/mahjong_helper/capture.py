"""Swappable capture provider: only the selected HWND's client rendering.

PrintWindow is intentionally used instead of desktop screenshots so the helper
overlay and unrelated applications are not sampled. Some accelerated browsers
do not support PrintWindow; failure is surfaced rather than silently falling
back to screen capture.
"""
import ctypes as C
from ctypes import wintypes as W
from dataclasses import dataclass
from typing import Protocol
import os

import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    screen_rect: tuple[int, int, int, int]
    client_size: tuple[int, int]


class CaptureProvider(Protocol):
    def capture(self, hwnd: int, roi: list[float]) -> Frame: ...


if os.name == 'nt':
    user = C.WinDLL('user32', use_last_error=True)
    gdi = C.WinDLL('gdi32', use_last_error=True)
    user.GetDC.argtypes = [W.HWND]
    user.GetDC.restype = W.HDC
    user.ReleaseDC.argtypes = [W.HWND, W.HDC]
    user.GetForegroundWindow.restype = W.HWND
    user.GetClientRect.argtypes = [W.HWND, C.POINTER(W.RECT)]
    user.ClientToScreen.argtypes = [W.HWND, C.POINTER(W.POINT)]
    user.IsIconic.argtypes = [W.HWND]
    user.IsWindow.argtypes = [W.HWND]
    user.IsWindowVisible.argtypes = [W.HWND]
    user.PrintWindow.argtypes = [W.HWND, W.HDC, W.UINT]
    user.GetWindowTextLengthW.argtypes = [W.HWND]
    user.GetWindowTextW.argtypes = [W.HWND, W.LPWSTR, C.c_int]
    user.SetWindowPos.argtypes = [W.HWND, W.HWND, C.c_int, C.c_int, C.c_int, C.c_int, W.UINT]
    user.SetWindowDisplayAffinity.argtypes = [W.HWND, W.DWORD]
    gdi.CreateCompatibleDC.argtypes = [W.HDC]
    gdi.CreateCompatibleDC.restype = W.HDC
    gdi.CreateCompatibleBitmap.argtypes = [W.HDC, C.c_int, C.c_int]
    gdi.CreateCompatibleBitmap.restype = W.HBITMAP
    gdi.SelectObject.argtypes = [W.HDC, W.HGDIOBJ]
    gdi.SelectObject.restype = W.HGDIOBJ
    gdi.DeleteObject.argtypes = [W.HGDIOBJ]
    gdi.DeleteDC.argtypes = [W.HDC]
    gdi.GetDIBits.argtypes = [W.HDC, W.HBITMAP, W.UINT, W.UINT, W.LPVOID, W.LPVOID, W.UINT]


def enable_dpi_awareness():
    if os.name == 'nt':
        # Before QApplication; PER_MONITOR_AWARE_V2, compatible with Qt 6.
        user.SetProcessDpiAwarenessContext(C.c_void_p(-4))


def list_windows():
    if os.name != 'nt':
        return []
    found = []
    callback_type = C.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)
    def visit(hwnd, _):
        if user.IsWindowVisible(hwnd):
            length = user.GetWindowTextLengthW(hwnd)
            buf = C.create_unicode_buffer(length + 1)
            user.GetWindowTextW(hwnd, buf, length + 1)
            if buf.value:
                found.append((int(hwnd), buf.value))
        return True
    callback = callback_type(visit)
    user.EnumWindows(callback, 0)
    return found


def active(hwnd):
    return os.name == 'nt' and bool(user.IsWindow(hwnd)) and not user.IsIconic(hwnd) and user.GetForegroundWindow() == hwnd


def client_geometry(hwnd):
    rect, point = W.RECT(), W.POINT(0, 0)
    if not user.GetClientRect(hwnd, C.byref(rect)) or not user.ClientToScreen(hwnd, C.byref(point)):
        raise RuntimeError('対象ウィンドウの座標を取得できません')
    return point.x, point.y, rect.right, rect.bottom


def roi_pixels(width, height, roi):
    x, y, w, h = roi
    left, top = round(x * width), round(y * height)
    right, bottom = round((x+w) * width), round((y+h) * height)
    return left, top, max(1, right-left), max(1, bottom-top)


class BitmapInfoHeader(C.Structure):
    _fields_ = [('size', W.DWORD), ('width', W.LONG), ('height', W.LONG),
                ('planes', W.WORD), ('bits', W.WORD), ('compression', W.DWORD),
                ('image_size', W.DWORD), ('xppm', W.LONG), ('yppm', W.LONG),
                ('used', W.DWORD), ('important', W.DWORD)]


class PrintWindowCapture:
    def capture(self, hwnd, roi):
        if os.name != 'nt':
            raise RuntimeError('ライブ取得はWindows専用です')
        if not user.IsWindow(hwnd) or user.IsIconic(hwnd):
            raise RuntimeError('対象が閉じているか最小化されています')
        sx, sy, width, height = client_geometry(hwnd)
        if width < 8 or height < 8 or width * height > 50_000_000:
            raise RuntimeError('取得サイズが範囲外です')
        dc = user.GetDC(hwnd)
        memory = gdi.CreateCompatibleDC(dc)
        bitmap = gdi.CreateCompatibleBitmap(dc, width, height)
        if not dc or not memory or not bitmap:
            if bitmap: gdi.DeleteObject(bitmap)
            if memory: gdi.DeleteDC(memory)
            if dc: user.ReleaseDC(hwnd, dc)
            raise RuntimeError('画像取得用の領域を確保できません')
        previous = gdi.SelectObject(memory, bitmap)
        try:
            if not user.PrintWindow(hwnd, memory, 3):  # CLIENTONLY | RENDERFULLCONTENT
                raise RuntimeError('PrintWindowに未対応です。スクリーンショットを使用してください')
            gdi.SelectObject(memory, previous)
            previous = None
            info = BitmapInfoHeader()
            info.size, info.width, info.height, info.planes, info.bits = C.sizeof(info), width, -height, 1, 32
            pixels = np.empty((height, width, 4), dtype=np.uint8)
            if gdi.GetDIBits(memory, bitmap, 0, height, pixels.ctypes.data, C.byref(info), 0) != height:
                raise RuntimeError('画像の転送に失敗しました')
            x, y, w, h = roi_pixels(width, height, roi)
            image = pixels[y:y+h, x:x+w, :3].copy()
            if image.size == 0 or image.max() < 5 or image.std() < 1:
                raise RuntimeError('対象画像が空です。ブラウザの画像取得に未対応の可能性があります')
            return Frame(image, (sx+x, sy+y, image.shape[1], image.shape[0]), (width, height))
        finally:
            if previous: gdi.SelectObject(memory, previous)
            gdi.DeleteObject(bitmap)
            gdi.DeleteDC(memory)
            user.ReleaseDC(hwnd, dc)
