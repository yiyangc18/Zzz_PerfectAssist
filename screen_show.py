import time
import win32api
import win32gui
import win32con
import numpy as np
import ctypes

class OverlayText:
    def __init__(self):
        self.hwnd = None
        self.text = "Initializing..."
        self.color = (255, 255, 255)  # Default color: white
        self.create_overlay()

    def create_overlay(self):
        # Define window class
        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = self.wnd_proc
        wc.lpszClassName = "OverlayTextClass"
        wc.hbrBackground = win32con.COLOR_MENU
        wc.hCursor = win32api.LoadCursor(0, win32con.IDC_ARROW)
        wc.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW

        # Register the window class
        try:
            class_atom = win32gui.RegisterClass(wc)
            print("Window class registered.")
        except Exception as e:
            print(f"Failed to register window class: {e}")
            return

        # Create the window
        self.hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_TOPMOST | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT,
            wc.lpszClassName,
            "OverlayText",
            win32con.WS_POPUP,
            0, 0, 400, 100,  # Increase width and height
            0, 0, 0, None
        )

        if not self.hwnd:
            print("Failed to create window.")
            return
        print("Window created.")

        # Set the window to be transparent
        win32gui.SetLayeredWindowAttributes(self.hwnd, 0, 255, win32con.LWA_ALPHA)

        # Set window position to top-right
        screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        print(f"Screen width: {screen_width}")  # Debug information
        win32gui.SetWindowPos(self.hwnd, win32con.HWND_TOPMOST, screen_width - 400, 0, 400, 100, 0)  # Adjust position to right and increase size

        # Show the window
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
        win32gui.UpdateWindow(self.hwnd)  # Ensure WM_PAINT is triggered

    def update_text(self, new_text, text_color=(255, 255, 255)):
        if self.text != new_text or self.color != text_color:
            self.text = new_text
            self.color = text_color
            # Temporarily remove InvalidateRect and UpdateWindow to test
            win32gui.InvalidateRect(self.hwnd, None, True)
            win32gui.UpdateWindow(self.hwnd)
            # print(f"Text updated to: {self.text} with color {self.color}")  # Debug信息


    def wnd_proc(self, hwnd, message, wparam, lparam):
        # print(f"Received message: {message}")
        if message == win32con.WM_PAINT:
            hdc, paint_struct = win32gui.BeginPaint(hwnd)
            rect = win32gui.GetClientRect(hwnd)
            win32gui.SetTextColor(hdc, win32api.RGB(*self.color))  # Set the text color from self.color
            win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
            font = ctypes.windll.gdi32.CreateFontW(
                -24, 0, 0, 0, win32con.FW_BOLD, 0, 0, 0,
                win32con.ANSI_CHARSET, win32con.OUT_DEFAULT_PRECIS,
                win32con.CLIP_DEFAULT_PRECIS, win32con.DEFAULT_QUALITY,
                win32con.DEFAULT_PITCH | win32con.FF_SWISS, "Arial"
            )
            old_font = win32gui.SelectObject(hdc, font)
            win32gui.DrawText(
                hdc,
                self.text,
                -1,
                rect,
                win32con.DT_CENTER | win32con.DT_VCENTER | win32con.DT_SINGLELINE
            )
            win32gui.SelectObject(hdc, old_font)
            ctypes.windll.gdi32.DeleteObject(font)
            win32gui.EndPaint(hwnd, paint_struct)
            return 0
        elif message == win32con.WM_DESTROY:
            win32gui.PostQuitMessage(0)
            return 0
        else:
            return win32gui.DefWindowProc(hwnd, message, wparam, lparam)

if __name__ == "__main__":
    overlay = OverlayText()

    # 示例：每秒更新一次文本，并随机指定颜色
    try:
        while True:
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            overlay.update_text(f"Class: {np.random.randint(0, 10)}, FPS: {np.random.randint(20, 60)}", text_color=color)
            time.sleep(1)
    except KeyboardInterrupt:
        win32gui.DestroyWindow(overlay.hwnd)
