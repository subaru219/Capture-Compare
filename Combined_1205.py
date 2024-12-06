# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QLabel, QLineEdit, QHBoxLayout, QCheckBox
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import Qt
import tkinter as tk
from tkinter import Canvas
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import mss
import glob
from datetime import datetime  # 新增，用於時間戳記

# --- SIFT特徵比對相關方法 ---
def getMatchNum(matches, ratio):
    '''返回特徵點匹配數量和matchesMask'''
    matchesMask = [[0, 0] for i in range(len(matches))]
    matchNum = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:  # 將距離比率小於ratio的匹配點篩選出來
            matchesMask[i] = [1, 0]
            matchNum += 1
    return matchNum, matchesMask

# --- 螢幕截圖部分 ---
class ScreenCaptureApp:
    def __init__(self):
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.rect = None
        self.root = None  # 用來指向 tkinter 的 root 視窗
        self.preview_image = None  # 儲存框選範圍的截圖

    def open_select_region(self):
        # 創建 tkinter 視窗，用來進行框選
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)  # 全螢幕顯示
        self.root.attributes('-alpha', 0.3)  # 半透明
        self.root.attributes('-topmost', True)  # 保持在最上層
        self.root.config(bg="black")

        # 創建 Canvas 畫布
        self.canvas = Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 綁定滑鼠事件
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # 按下 Esc 鍵可以退出選區
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        # 開始 tkinter 視窗主迴圈
        self.root.mainloop()

    def on_button_press(self, event):
        # 設定矩形起始點
        self.start_x = event.x
        self.start_y = event.y

        # 清除先前繪製的矩形
        if self.rect:
            self.canvas.delete(self.rect)
        
        # 創建新矩形
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)

    def on_mouse_drag(self, event):
        # 動態調整矩形大小
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        # 獲取最終矩形的座標
        self.end_x = event.x
        self.end_y = event.y
        self.root.destroy()  # 關閉選區視窗

    def capture_screenshot_preview(self):
        
        # 檢查是否有選區
        if None in (self.start_x, self.start_y, self.end_x, self.end_y):
            print("請先選擇一個區域！")
            return None

        # 計算選區座標
        x1 = min(self.start_x, self.end_x)
        y1 = min(self.start_y, self.end_y)
        x2 = max(self.start_x, self.end_x)
        y2 = max(self.start_y, self.end_y)
        width = x2 - x1
        height = y2 - y1

        # 使用 mss 截取全螢幕
        with mss.mss() as sct:
            screenshot = sct.grab(sct.monitors[1])
            img = np.array(screenshot)[:, :, :3]  # 去除 alpha 通道

        # 裁剪出選定區域並儲存成預覽影像
        self.preview_image = img[y1:y1+height, x1:x1+width]
        return self.preview_image

    def capture_screenshot(self, save_path="screenshots/Screenshot/"):
        if self.preview_image is None:
            print("請先選擇一個區域！")
            return

        # 確保儲存路徑存在
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 儲存截圖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_path, f"screenshot_{timestamp}.png")
        cv2.imwrite(filename, cv2.cvtColor(self.preview_image, cv2.COLOR_RGB2BGR))
        print(f"Screenshot saved as {filename}")
        

# --- PyQt5主視窗設置 ---
class MainWindow(QWidget):
    
    def __init__(self, origin_sample_path, sample_path, output_path, query_path, screen_capture_app):
        super().__init__()
        
        self.screen_capture_app = screen_capture_app
        
        # 設定影像路徑
        self.originPath = origin_sample_path
        self.samplePath = sample_path
        self.outputPath = output_path
        self.queryPath = query_path
        
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # 圖片顯示區域
        image_layout = QHBoxLayout()
        
        # Golden 圖片區域
        self.sample_image_label = QLabel(self)
        self.sample_image_label.setAlignment(Qt.AlignCenter)
        self.sample_image_label.setStyleSheet("border: 1px solid black;")  # 加邊框
        golden_layout = QVBoxLayout()
        golden_title = QLabel("Golden", self)
        golden_title.setAlignment(Qt.AlignCenter)
        golden_title.setFont(QFont('Arial', 14, QFont.Bold))  # 設定字體
        golden_layout.addWidget(golden_title)
        golden_layout.addWidget(self.sample_image_label)
        
        # Screenshot 圖片區域
        self.query_image_label = QLabel(self)
        self.query_image_label.setAlignment(Qt.AlignCenter)
        self.query_image_label.setStyleSheet("border: 1px solid black;")  # 加邊框
        screenshot_layout = QVBoxLayout()
        screenshot_title = QLabel("Actual", self)
        screenshot_title.setAlignment(Qt.AlignCenter)
        screenshot_title.setFont(QFont('Arial', 14, QFont.Bold))  # 設定字體
        screenshot_layout.addWidget(screenshot_title)
        screenshot_layout.addWidget(self.query_image_label)
        
        # 添加至圖片佈局
        image_layout.addLayout(golden_layout)
        image_layout.addLayout(screenshot_layout)
        layout.addLayout(image_layout)
        
        # 訊息顯示區域
        self.message_label = QLabel(self)
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setFont(QFont('Arial', 12))
        self.message_label.setStyleSheet("color: red;")  # 設定文字顏色
        layout.addWidget(self.message_label)
        
        # 預覽區域
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)
        
        # IsFirstWafer Checkbox
        self.is_first_wafer_checkbox = QCheckBox('IsFirstWafer', self)
        self.is_first_wafer_checkbox.setFont(QFont('Arial', 14))
        layout.addWidget(self.is_first_wafer_checkbox)
        
        # 新增文字框和按鈕區域
        input_layout = QHBoxLayout()
        self.textbox = QLineEdit(self)
        self.textbox.setPlaceholderText("輸入型號")
        self.textbox.setFont(QFont('Arial', 12))
        confirm_button = QPushButton('確認', self)
        confirm_button.setFont(QFont('Arial', 12))
        confirm_button.clicked.connect(self.update_sample_path)  # 點擊觸發更新方法
        input_layout.addWidget(self.textbox)
        input_layout.addWidget(confirm_button)
        layout.addLayout(input_layout)
        
        # 選擇區域按鈕
        self.select_region_button = QPushButton('Select Region', self)
        self.select_region_button.setFont(QFont('Arial', 14))
        self.select_region_button.setStyleSheet('background-color: #4CAF50; color: white;')
        self.select_region_button.clicked.connect(self.select_region)
        
        # 截圖按鈕
        self.capture_screenshot_button = QPushButton('Capture Screenshot', self)
        self.capture_screenshot_button.setFont(QFont('Arial', 14))
        self.capture_screenshot_button.setStyleSheet('background-color: #4CAF50; color: white;')
        self.capture_screenshot_button.clicked.connect(self.capture_screenshot)

        # 比對圖片按鈕
        self.compare_images_button = QPushButton('Compare Images', self)
        self.compare_images_button.setFont(QFont('Arial', 14))
        self.compare_images_button.setStyleSheet('background-color: #4CAF50; color: white;')
        self.compare_images_button.clicked.connect(self.compare_images)
        
        # 布局
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_region_button)
        button_layout.addWidget(self.capture_screenshot_button)
        button_layout.addWidget(self.compare_images_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        self.setWindowTitle('Screenshot App')
        self.setGeometry(100, 100, 400, 300)
        
    def select_region(self):
        
        # 清空顯示結果
        self.preview_label.clear()
        self.sample_image_label.clear()
        self.query_image_label.clear()
    
        # 打開 tkinter 視窗進行選區
        self.screen_capture_app.open_select_region()

        # 截取選區預覽
        preview_image = self.screen_capture_app.capture_screenshot_preview()

        # 顯示選區的縮略圖
        if preview_image is not None:
            height, width, channel = preview_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(preview_image.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(200, 200, Qt.KeepAspectRatio)  # 縮放圖像以適應 QLabel 大小
            self.preview_label.setPixmap(pixmap)
            print("選取區域預覽顯示成功！")

    def capture_screenshot(self):
        
        # 根據選區範圍儲存截圖
        self.screen_capture_app.capture_screenshot()
        
        self.message_label.setText(f"Capture screenshot success !")
        
        # 從 TextBox 取得檔名
        image_name = self.textbox.text().strip()
        
        # 確認檔案名稱是否與 IsFirstWafer 狀態相關
        if self.is_first_wafer_checkbox.isChecked():
            # 檔名與 TextBox 輸入名稱相同
            filename = f"{image_name}.png"
            # 儲存的檔案路徑
            self.samplePath = os.path.join(self.originPath, filename)
        
        # 如果勾選 IsFirstWafer，將截圖另存為 Golden_Image
        if self.is_first_wafer_checkbox.isChecked():
            if self.screen_capture_app.preview_image is not None:
                if not os.path.exists(os.path.dirname(self.samplePath)):
                    os.makedirs(os.path.dirname(self.samplePath))
                cv2.imwrite(self.samplePath, cv2.cvtColor(self.screen_capture_app.preview_image, cv2.COLOR_RGB2BGR))
                self.message_label.setText(f"Golden Image saved as {self.samplePath}")
            else:
                self.message_label.setText("No preview image available. Please select a region first.")
                
    def compare_images(self):
        '''執行影像比對並顯示結果於主視窗'''
        if not os.path.exists(self.queryPath):
            print("請先選擇一個區域！")
            return
        
        self.message_label.setText(f"")
        
        # 找到 query_path 資料夾中最新的檔案
        query_images = glob.glob(os.path.join(self.queryPath, "*.png"))  # 查找所有 .png 檔案
        if not query_images:
            print("No query images found in the folder!")
            return

        # 根據檔案修改時間排序，選取最新的檔案
        latest_query_image_path = max(query_images, key=os.path.getmtime)
        print(f"Using latest query image: {latest_query_image_path}")
        
        # 從 TextBox 取得檔名
        image_name = self.textbox.text().strip()
        
        # 讀取影像
        sampleImage = cv2.imread(self.samplePath, 0)
        queryImage = cv2.imread(latest_query_image_path, 0)
        
        if sampleImage is None or queryImage is None:
            self.message_label.setText(f"型號: {image_name} 沒有建立 Golden_image! 請通知領班確認!")
            # raise ValueError("Failed to read one or both images")
            return

        # 提取特徵點和描述子
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(sampleImage, None)
        kp2, des2 = sift.detectAndCompute(queryImage, None)

        # 建立FLANN匹配器並進行匹配
        indexParams = dict(algorithm=0, trees=5)
        searchParams = dict(checks=50)
        flann = cv2.FlannBasedMatcher(indexParams, searchParams)
        matches = flann.knnMatch(des1, des2, k=2)
        
        matchNum, matchesMask = getMatchNum(matches, 0.9)  # 計算匹配程度
        matchRatio = matchNum * 100 / len(matches)

        # 繪製比對結果
        drawParams = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=0)
        comparisonImage = cv2.drawMatchesKnn(sampleImage, kp1, queryImage, kp2, matches, None, **drawParams)

        # 判斷相似度是否通過門檻
        threshold = 90.0
        status = "PASS" if matchRatio >= threshold else "FAIL"

        # 儲存比對結果圖片
        outputFilename = os.path.join(self.outputPath, f'comparison_{matchRatio:.2f}_{status}.png')
        cv2.imwrite(outputFilename, comparisonImage)

        # 更新主視窗顯示結果
        result_text = f"Matching ratio: {matchRatio:.2f}%, Result: {status}"
        
        # 設置字體大小
        font = QFont('Arial', 16)  # 設定字體為 Arial，大小為 16
        self.preview_label.setFont(font)
        self.preview_label.setText(result_text)
        print(result_text)
        
        # 顯示樣本圖像
        self.display_image(sampleImage, self.sample_image_label)

        # 顯示截圖影像
        self.display_image(queryImage, self.query_image_label)
    
    def update_sample_path(self):
        '''根據使用者輸入更新 samplePath 並載入對應影像'''
        image_name = self.textbox.text().strip()  # 獲取使用者輸入
        if not image_name:
            self.message_label.setText("請輸入型號")
            return

        # 更新 samplePath
        new_sample_path = f"screenshots/Golden_image/{image_name}.png"
        if os.path.exists(new_sample_path):
            self.samplePath = new_sample_path
            # self.message_label.setText(f"Sample Path updated to: {self.samplePath}")

            # 顯示更新的 Golden 圖片
            sample_image = cv2.imread(self.samplePath, cv2.IMREAD_GRAYSCALE)
            if sample_image is not None:
                self.display_image(sample_image, self.sample_image_label)
            else:
                print("找不到Golden，請確認檔案是否正確")
        else:
            self.message_label.setText(f"型號: {image_name} 的 Golden不存在")
            
    def display_image(self, image, label):
        '''將 OpenCV 圖片轉換為 QPixmap 並顯示於 QLabel'''
        height, width = image.shape
        bytes_per_line = width  # 灰度圖每行的位元組數
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img).scaled(200, 200, Qt.KeepAspectRatio)  # 縮放以適應 QLabel
        label.setPixmap(pixmap)

# --- 執行主程式 ---
if __name__ == '__main__':
    
    screen_capture_app = ScreenCaptureApp()
    
    app = QApplication(sys.argv)

    # 設定樣本影像路徑和輸出路徑
    origin_sample_path = 'screenshots/Golden_image/'
    sample_path = ''
    query_path = 'screenshots/Screenshot'
    output_path = 'screenshots/Compare_Result'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    main_window = MainWindow(origin_sample_path, sample_path, output_path, query_path, screen_capture_app)
    main_window.show()
    sys.exit(app.exec_())
