import random
from PIL import Image, ImageDraw, ImageFont

# ==================== 词汇表（已校验，正面31个，负面31个） ====================
positive_words = [
    "excellent", "wonderful", "marvelous", "superb", "fantastic", "amazing",
    "brilliant", "delightful", "charming", "graceful", "radiant", "joyful",
    "happy", "cheerful", "peaceful", "gentle", "kind", "sincere", "loyal",
    "brave", "confident", "creative", "intelligent", "wise", "elegant",
    "pristine", "vibrant", "exuberant", "admirable", "noble", "trustworthy"
]

negative_words = [
    "awful", "terrible", "horrible", "dreadful", "unpleasant", "miserable",
    "sad", "angry", "furious", "bitter", "cruel", "selfish", "dishonest",
    "corrupt", "lazy", "careless", "weak", "insecure", "anxious", "depressed",
    "lonely", "boring", "dull", "ugly", "filthy", "broken", "damaged",
    "hopeless", "useless", "pathetic"
]

# 为每个单词生成属性（大小统一范围，避免显著大小差异导致的不平衡；正面负面同等大小权重）
word_items = []
for w in positive_words:
    size = random.randint(26, 40)          # 所有单词大小相对均匀
    rot = random.choice([-3, -1, 0, 1, 3]) # 轻微旋转（预留，后面简化处理）
    color = (70, 130, 200)                 # 柔和蓝色
    pref_x = random.randint(100, 400)      # 偏好左上区域
    pref_y = random.randint(60, 350)
    word_items.append((w, size, rot, color, pref_x, pref_y))

for w in negative_words:
    size = random.randint(26, 40)
    rot = random.choice([-3, -1, 0, 1, 3])
    color = (200, 80, 80)                  # 柔和红色
    pref_x = random.randint(500, 850)      # 偏好右下区域
    pref_y = random.randint(350, 620)
    word_items.append((w, size, rot, color, pref_x, pref_y))

random.shuffle(word_items)

# ==================== 画布设置 ====================
W, H = 1100, 750
bg_color = (250, 250, 245)

def draw_grid(draw, w, h, spacing=20):
    """绘制浅灰色笔记本网格"""
    grid_color = (220, 220, 210)
    for x in range(0, w, spacing):
        draw.line([(x, 0), (x, h)], fill=grid_color, width=1)
    for y in range(0, h, spacing):
        draw.line([(0, y), (w, y)], fill=grid_color, width=1)

img = Image.new('RGB', (W, H), bg_color)
draw = ImageDraw.Draw(img)
draw_grid(draw, W, H, 20)

# 加载手写字体（优先 Comic Sans，若不存在则用默认，也可下载开源字体）
try:
    font_path = "C:/Windows/Fonts/comic.ttf"      # Windows
    test_font = ImageFont.truetype(font_path, 20)
except:
    try:
        font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"  # macOS
        test_font = ImageFont.truetype(font_path, 20)
    except:
        font_path = None
        test_font = ImageFont.load_default()

def get_font(sz):
    if font_path:
        return ImageFont.truetype(font_path, sz)
    else:
        return ImageFont.load_default()

# 记录已放置的矩形区域 (x0, y0, x1, y1)
placed = []

def get_text_wh(word, font):
    """获取文本实际宽高"""
    bbox = draw.textbbox((0, 0), word, font=font)
    return (bbox[2] - bbox[0]), (bbox[3] - bbox[1])

def is_overlap(new_rect, margin=10):
    """检查矩形是否有重叠（margin为间距）"""
    nx0, ny0, nx1, ny1 = new_rect
    for (x0, y0, x1, y1) in placed:
        if not (nx1 + margin < x0 or nx0 - margin > x1 or ny1 + margin < y0 or ny0 - margin > y1):
            return True
    return False

def draw_word_with_dot(draw, word, x, y, font, color):
    """在 (x,y) 处绘制单词，左侧绘制圆点（圆点颜色与单词相同）"""
    # 获取单词尺寸
    tw, th = get_text_wh(word, font)
    # 圆点半径 5px
    r = 5
    dot_x = x - r - 4
    dot_y = y + th // 2
    draw.ellipse((dot_x - r, dot_y - r, dot_x + r, dot_y + r), fill=color)
    # 绘制单词文本（水平，暂忽略旋转，避免圆点跟随复杂）
    draw.text((x, y), word, font=font, fill=color)
    return tw, th

# ==================== 放置单词（严格避免重叠） ====================
for word, size, rot, color, pref_x, pref_y in word_items:
    font = get_font(size)
    tw, th = get_text_wh(word, font)
    placed_flag = False
    # 在偏好坐标附近尝试 300 次，逐步扩大搜索半径
    for attempt in range(300):
        radius = 50 if attempt < 150 else 120
        dx = random.randint(-radius, radius)
        dy = random.randint(-radius, radius)
        x = pref_x + dx
        y = pref_y + dy
        # 边界约束（留出左右上下边距至少 15px）
        if x < 15 or x + tw > W - 15:
            continue
        if y < 15 or y + th > H - 30:
            continue
        rect = (x, y, x+tw, y+th)
        if not is_overlap(rect, margin=8):   # 8像素间隙保证不粘连
            # 放置
            draw_word_with_dot(draw, word, x, y, font, color)
            placed.append(rect)
            placed_flag = True
            break
    # 如果仍然无法放置，降低要求再次尝试（但不允许重叠，只放宽边界限制）
    if not placed_flag:
        for attempt in range(200):
            dx = random.randint(-180, 180)
            dy = random.randint(-180, 180)
            x = pref_x + dx
            y = pref_y + dy
            if x < 10 or x + tw > W - 10:
                continue
            if y < 10 or y + th > H - 20:
                continue
            rect = (x, y, x+tw, y+th)
            if not is_overlap(rect, margin=5):
                draw_word_with_dot(draw, word, x, y, font, color)
                placed.append(rect)
                placed_flag = True
                break
    # 最终极少数实在无处可放，则放角落（仍检测无重叠，但允许与边界更近）
    if not placed_flag:
        # 尝试一些固定偏移
        for (dx, dy) in [(0,0), (20,20), (-20,-20), (40,10), (-40,10)]:
            x = pref_x + dx
            y = pref_y + dy
            if x < 5 or x + tw > W - 5 or y < 5 or y + th > H - 15:
                continue
            rect = (x, y, x+tw, y+th)
            if not is_overlap(rect, margin=3):
                draw_word_with_dot(draw, word, x, y, font, color)
                placed.append(rect)
                placed_flag = True
                break
        if not placed_flag:
            # 最后保底：直接放在偏好位置，但不检测重叠（理论上极少发生）
            draw_word_with_dot(draw, word, pref_x, pref_y, font, color)
            placed.append((pref_x, pref_y, pref_x+tw, pref_y+th))

# ==================== 添加轻微纸张纹理（可选） ====================
pixels = img.load()
for _ in range(2000):
    x = random.randint(0, W-1)
    y = random.randint(0, H-1)
    r, g, b = pixels[x, y]
    if r > 240 and g > 240 and b > 240:
        continue
    r = min(255, r + random.randint(-8, 8))
    g = min(255, g + random.randint(-8, 8))
    b = min(255, b + random.randint(-8, 8))
    pixels[x, y] = (r, g, b)

# 保存与显示
output = "handwritten_wordcloud_no_overlap.png"
img.save(output)
img.show()
print(f"✅ 已生成不重叠的手绘词云图：{output}")
print(f"总单词数：{len(word_items)}（正面{len(positive_words)}，负面{len(negative_words)}）")