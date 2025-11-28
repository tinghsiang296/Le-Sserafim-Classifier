# -*- coding: utf-8 -*-
"""
Le Sserafim 成員辨識與 PK 遊戲
修改自: Demo03v 和AI_PK看誰比較會認IVE成員
"""

import os
from time import sleep
from fastai.vision.all import *
default_device(torch.device('cpu'))
from duckduckgo_search import DDGS

# 設定隨機種子以重現結果
set_seed(42)

# 定義 Le Sserafim 成員 (類別名稱)
MEMBERS = ['Sakura', 'Kim Chaewon', 'Huh Yunjin', 'Kazuha', 'Hong Eunchae']
# 對應的搜尋關鍵字 (加上團名以提高準確度)
SEARCH_TERMS = [f'Le Sserafim {m}' for m in MEMBERS]

# 設定資料路徑
PATH = Path('le_sserafim_images')

def search_images(term, max_images=30):
    """使用 DuckDuckGo 搜尋圖片"""
    print(f"正在搜尋: {term} ...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                # 使用 ddgs.images 搜尋
                results = list(ddgs.images(term, max_results=max_images))
                urls = [r['image'] for r in results]
                return L(urls)
        except Exception as e:
            print(f"搜尋失敗 (嘗試 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                sleep_time = (attempt + 1) * 10
                print(f"等待 {sleep_time} 秒後重試...")
                sleep(sleep_time)
            else:
                print("搜尋失敗，跳過此項目。")
                return L()

def download_dataset():
    """下載並整理資料集"""
    if not PATH.exists():
        PATH.mkdir()
    
    for i, member in enumerate(MEMBERS):
        dest = PATH/member
        dest.mkdir(exist_ok=True, parents=True)
        
        # 檢查是否已有圖片
        if len(get_image_files(dest)) > 10:
            print(f"{member} 的圖片已存在，跳過下載。")
            continue

        # 搜尋圖片網址
        urls = search_images(SEARCH_TERMS[i])
        
        if not urls:
            print(f"無法找到 {member} 的圖片。")
            continue

        # 下載圖片
        print(f"正在下載 {member} 的圖片...")
        download_images(dest, urls=urls)
        
        # 稍微暫停避免被封鎖
        print("休息 10 秒避免被封鎖...")
        sleep(10)
        
        # 調整圖片大小 (選擇性)
        resize_images(PATH/member, max_size=400, dest=PATH/member)
    
    # 清理無法開啟的圖片
    print("正在清理損壞的圖片...")
    failed = verify_images(get_image_files(PATH))
    failed.map(Path.unlink)
    print(f"已移除 {len(failed)} 張損壞圖片。")

def train_model():
    """訓練模型"""
    print("準備資料載入器 (DataLoaders)...")
    # 定義 DataBlock
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files, 
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(224), # 提升圖片解析度至 224
        batch_tfms=aug_transforms(min_scale=0.75)) # 加入資料增強
    
    dls = dblock.dataloaders(PATH, device=torch.device('cpu'), bs=16)
    
    print("開始訓練模型 (使用 ResNet34)...")
    # 使用預訓練的 resnet34 模型並進行微調 (更強大的模型)
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    # 增加訓練次數，先凍結訓練 2 次，再解凍訓練 8 次
    learn.fine_tune(10, freeze_epochs=2)
    
    print("訓練完成！")
    return learn, dls

def play_pk_game(learn, dls):
    """簡單的 PK 環節"""
    print("\n" + "="*30)
    print("  和 AI PK：看誰比較會認 Le Sserafim 成員  ")
    print("="*30 + "\n")
    
    # 從驗證集中隨機選一張圖
    valid_files = get_image_files(PATH)
    # 簡單起見，隨機選一張 (不一定是驗證集，但為了測試方便)
    test_file = valid_files[random.randint(0, len(valid_files)-1)]
    
    true_label = parent_label(test_file)
    
    print(f"系統選了一張圖片 (路徑: {test_file})")
    print("請猜猜看她是誰？")
    print("選項: " + ", ".join(MEMBERS))
    
    user_guess = input("請輸入你的答案 (英文名字): ").strip()
    
    # AI 預測
    pred, pred_idx, probs = learn.predict(test_file)
    ai_prob = probs[pred_idx]
    
    print("\n" + "-"*20)
    print(f"正確答案是: {true_label}")
    print(f"你的猜測: {user_guess}")
    print(f"AI 的猜測: {pred} (信心度: {ai_prob:.4f})")
    
    if user_guess.lower() == true_label.lower():
        print("結果: 你答對了！")
    else:
        print("結果: 你答錯了！")
        
    if pred == true_label:
        print("AI 答對了！")
    else:
        print("AI 答錯了！")
    print("-"*20)

if __name__ == '__main__':
    # 1. 下載資料
    download_dataset()
    
    # 2. 訓練模型
    learn, dls = train_model()
    
    # 3. 匯出模型 (可選)
    learn.export('le_sserafim_model.pkl')
    print("模型已儲存為 le_sserafim_model.pkl")
    
    # 4. 進入 PK 模式
    while True:
        play_pk_game(learn, dls)
        again = input("要再玩一次嗎？(y/n): ")
        if again.lower() != 'y':
            break