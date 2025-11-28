from pathlib import Path
from PIL import Image, ImageDraw
import os

MEMBERS = ['Sakura', 'Kim Chaewon', 'Huh Yunjin', 'Kazuha', 'Hong Eunchae']
PATH = Path('le_sserafim_images')
COLORS = ['pink', 'green', 'orange', 'blue', 'red']

def create_dummy_images():
    if not PATH.exists():
        PATH.mkdir()
    
    for i, member in enumerate(MEMBERS):
        dest = PATH/member
        dest.mkdir(exist_ok=True, parents=True)
        
        print(f"Generating dummy images for {member}...")
        for j in range(15): # Generate 15 images
            img = Image.new('RGB', (400, 400), color=COLORS[i])
            d = ImageDraw.Draw(img)
            d.text((10,10), f"{member} {j}", fill="white")
            img.save(dest/f"{member}_{j}.jpg")

if __name__ == '__main__':
    create_dummy_images()
    print("Dummy dataset created.")
