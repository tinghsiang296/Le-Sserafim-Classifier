# LE SSERAFIM 成員圖片分類器 (LE SSERAFIM Image Classifier)

一個使用深度學習模型與 [Streamlit](https://streamlit.io/) 建立的網頁應用程式，  
可以根據使用者上傳的照片，預測最像哪一位 **LE SSERAFIM** 成員。

> Demo 網址：  
> https://le-sserafim-classifier-c3vse6mkuswuk9fktnbakt.streamlit.app/

## 功能介紹 (Features)

- 上傳一張成員照片（或照片截圖）
- 模型自動進行前處理與特徵抽取
- 輸出預測結果：最有可能的團員名稱
- 顯示各成員的機率分佈（可選）
- 簡單易用的 Web 介面，無需安裝任何軟體即可使用（透過瀏覽器開啟）

目前支援的成員（可依實際模型調整）：

- Kim Chaewon
- Sakura
- Huh Yunjin
- Kazuha
- Hong Eunchae

## 專案背景 (Background)

本專案靈感改寫自蔡炎龍老師的 fastai 圖像分類 Demo Notebook，  
原始範例為「AI 與人類 PK 辨識 IVE 成員」，本專案則將資料集換成 **LE SSERAFIM 團員** 照片，  
並重新訓練分類模型，最後利用 Streamlit 部署成線上 Demo。

## 技術架構 (Tech Stack)

- 語言：Python 3.x
- 深度學習框架：fastai / PyTorch（依實作調整）
- 前端框架：Streamlit
- 其他：
  - Pillow / OpenCV 作為影像讀取與處理
  - requirements.txt 管理套件
  - 部署於 Streamlit Cloud


