
<h1 align="center">✨ LE SSERAFIM 團員分類器 ✨</h1>
<p align="center">一個基於深度學習的圖片分類器，幫你辨識 LE SSERAFIM 成員！</p>

---

<p align="center">
  <a href="https://le-sserafim-classifier-c3vse6mkuswuk9fktnbakt.streamlit.app/">
    <img src="https://img.shields.io/badge/💗 Live Demo-Streamlit-ff6fb0?style=for-the-badge">
  </a>
</p>

<p align="center">
  🔗 <strong>Demo URL：</strong><br>
  https://le-sserafim-classifier-c3vse6mkuswuk9fktnbakt.streamlit.app/
</p>

<p align="center">
  <img src="banner.png" width="650">
</p>

---

## 📚 目錄
- [簡介](#簡介)
- [功能特色](#功能特色)
- [技術架構](#技術架構)
- [專案結構](#專案結構)
- [安裝與執行](#安裝與執行)
- [模型訓練流程](#模型訓練流程)
- [Demo 網站](#demo-網站)

---

## 📝 簡介
這是一個使用 **深度學習（fastai + PyTorch）** 訓練的圖片分類模型，  
並透過 **Streamlit Cloud** 建立的線上 Demo，  
可辨識 LE SSERAFIM 的五位成員。

本專案改寫自蔡炎龍老師的 fastai 圖像辨識 Demo，資料集已換成 LE SSERAFIM 成員。

---

## ✨ 功能特色
- 📸 支援上傳照片並即時辨識  
- 🌐 免費線上體驗（Streamlit Cloud）  
- 🤖 使用 ResNet 預訓練模型（高準確率）  
- 📊 顯示預測機率與推論結果  
- 🎨 介面簡潔、美觀、易於使用  

---

## 🧠 技術架構

- **Python 3.x**
- **fastai / PyTorch**（模型訓練與推論）
- **Streamlit**（前端介面）
- **Pillow / OpenCV**（影像處理）
- **Streamlit Cloud** 部署

---

## 📁 專案結構

```text
.
├── app.py                    # Streamlit 主程式
├── model/
│   └── le_sserafim.pkl       # 訓練好的模型（fastai 匯出）
├── data/
│   ├── train/                # 訓練資料（依需要放）
│   └── valid/                # 驗證資料
├── requirements.txt          # Python 套件
└── README.md                 # 說明文件
