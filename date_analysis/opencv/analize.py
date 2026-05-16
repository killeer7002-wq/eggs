import cv2
import numpy as np
from pathlib import Path
import os


def compute_soft_gloss_index(
    gloss_score,
    yolk_mask,
    glossy_region_mask,
    matte_texture_mask,
    highlight_mask
):
    yolk_bool = yolk_mask > 0

    if np.count_nonzero(yolk_bool) == 0:
        return 0.0

    # gloss_score должен быть примерно в диапазоне 0..1
    score = gloss_score.copy().astype(np.float32)
    score = np.clip(score, 0, 1)

    # Делаем мягкий переход:
    # score < 0.45 почти не считается глянцем
    # score > 0.75 считается сильным глянцем
    gloss_strength = (score - 0.45) / (0.75 - 0.45)
    gloss_strength = np.clip(gloss_strength, 0, 1)

    # Всё, что вообще не попало в область потенциального глянца,
    # не должно давать большой вклад.
    gloss_strength[glossy_region_mask == 0] *= 0.25

    # Текстурно-матовые области сильно штрафуем.
    # Это важно для 4-го яйца справа.
    gloss_strength[matte_texture_mask > 0] *= 0.10

    # Явные блики считаем максимально глянцевыми.
    gloss_strength[highlight_mask > 0] = 1.0

    gloss_index = 100.0 * np.mean(gloss_strength[yolk_bool])

    return gloss_index

def largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    # label 0 is background
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8) * 255

def robust_norm_by_mask(x, mask_bool, p_low=20, p_high=90):
    x = x.astype(np.float32)
    vals = x[mask_bool]

    lo, hi = np.percentile(vals, [p_low, p_high])

    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def local_laplacian_energy(channel, win=9):
    channel = channel.astype(np.float32)

    lap = np.abs(cv2.Laplacian(channel, cv2.CV_32F, ksize=3))
    energy = cv2.blur(lap, (win, win))

    return energy


def detect_matte_texture(img_bgr, yolk_mask, highlight_mask=None):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    h, s, v = cv2.split(hsv)
    L, a, b = cv2.split(lab)

    yolk_bool = yolk_mask > 0

    # Немного отступаем от границы желтка,
    # чтобы не ловить границу белок/желток как "текстуру"
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inner_mask = cv2.erode(yolk_mask, inner_kernel, iterations=1)
    inner_bool = inner_mask > 0

    if np.count_nonzero(inner_bool) == 0:
        return np.zeros_like(yolk_mask)

    # Текстура именно по цвету/насыщенности.
    # Правая матовая часть хорошо ловится через S и Lab-b.
    texture_raw = (
        0.55 * local_laplacian_energy(s, win=9) +
        0.30 * local_laplacian_energy(b, win=9) +
        0.15 * local_laplacian_energy(v, win=9)
    )

    texture = robust_norm_by_mask(texture_raw, inner_bool, p_low=20, p_high=90)

    # Главный параметр.
    # Меньше => больше областей считаются матовыми.
    # Больше => строже, меньше матовой области.
    MATTE_TEXTURE_THRESHOLD = 0.45

    matte_seed = (
        inner_bool &
        (texture >= MATTE_TEXTURE_THRESHOLD)
    )

    matte_mask = matte_seed.astype(np.uint8) * 255

    # Убираем мелкий шум
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    matte_mask = cv2.morphologyEx(matte_mask, cv2.MORPH_OPEN, open_kernel)

    # Склеиваем текстурные точки в одну матовую область.
    # Для 4-го яйца это важно: справа текстура идёт полосками/пятнами.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    matte_mask = cv2.morphologyEx(matte_mask, cv2.MORPH_CLOSE, close_kernel)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    matte_mask = cv2.dilate(matte_mask, dilate_kernel, iterations=1)

    matte_mask = cv2.bitwise_and(matte_mask, yolk_mask)

    # Явные блики не должны становиться матовыми
    if highlight_mask is not None:
        highlight_dilated = cv2.dilate(
            highlight_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1
        )
        matte_mask[highlight_dilated > 0] = 0

    # Убираем очень маленькие компоненты
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(matte_mask, connectivity=8)

    cleaned = np.zeros_like(matte_mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= 80:
            cleaned[labels == label] = 255

    return cleaned

def keep_best_component(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    h, w = mask.shape
    cx0, cy0 = w / 2, h / 2

    best_label = None
    best_score = -1

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]

        if area < 300:
            continue

        cx, cy = centroids[label]

        # Штрафуем компоненты далеко от центра кадра,
        # чтобы куски желтка на ноже или по краям не выбирались вместо основного желтка.
        dist2 = (cx - cx0) ** 2 + (cy - cy0) ** 2
        center_bonus = np.exp(-dist2 / (2 * (0.4 * min(h, w)) ** 2))

        score = area * (1 + center_bonus)

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return np.zeros_like(mask)

    return ((labels == best_label).astype(np.uint8) * 255)


def get_yolk_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    h, s, v = cv2.split(hsv)
    L, a, b = cv2.split(lab)

    # Более жёсткие условия для желтка
    hsv_mask = (
        (h >= 7) & (h <= 32) &
        (s >= 95) &
        (v >= 60)
    )

    lab_mask = (
        (b >= 145) &
        (a >= 122)
    )

    mask = (hsv_mask & lab_mask).astype(np.uint8) * 255

    # Убираем мелкий мусор
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    # Слегка склеиваем близкие куски желтка
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # Оставляем лучшую компоненту
    mask = keep_best_component(mask)

    # ВАЖНО: не делаем агрессивное fill_mask_holes(mask),
    # потому что оно может захватить белок рядом с желтком.

    # Сжимаем маску, чтобы она не залезала на белок
    shrink_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, shrink_kernel, iterations=1)

    return mask


def local_std(channel, ksize=31):
    channel = channel.astype(np.float32)

    mean = cv2.blur(channel, (ksize, ksize))
    mean2 = cv2.blur(channel * channel, (ksize, ksize))

    return np.sqrt(np.maximum(mean2 - mean * mean, 0))


def get_gloss_masks(img_bgr, yolk_mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    h, s, v = cv2.split(hsv)
    L, a, b = cv2.split(lab)

    yolk_bool = yolk_mask > 0

    if np.count_nonzero(yolk_bool) == 0:
        raise ValueError("Yolk mask is empty")

    # -------------------------------
    # 1. Локальная неоднородность цвета
    # -------------------------------

    s_std = local_std(s, ksize=31)
    b_std = local_std(b, ksize=31)
    v_std = local_std(v, ksize=31)

    # Чем больше texture, тем более мутная/матовая область
    texture = (
        0.7 * (s_std / 60.0) +
        0.2 * (b_std / 25.0) +
        0.1 * (v_std / 30.0)
    )

    texture = np.clip(texture, 0, 1)

    # -------------------------------
    # 2. Расстояние до края желтка
    # -------------------------------

    dist = cv2.distanceTransform(
        (yolk_mask > 0).astype(np.uint8),
        cv2.DIST_L2,
        5
    ).astype(np.float32)

    dist_norm = dist / (dist.max() + 1e-6)

    # -------------------------------
    # 3. Цветовой критерий желтка
    # -------------------------------

    orange = (
        (h >= 7) &
        (h <= 26) &
        (s >= 160) &
        (v >= 100) &
        (b >= 160)
    )

    # -------------------------------
    # 4. Итоговый score глянца
    # -------------------------------

    S = s.astype(np.float32) / 255.0
    V = v.astype(np.float32) / 255.0
    B = (b.astype(np.float32) - 128.0) / 127.0

    gloss_score = (
        0.30 * S +
        0.15 * B +
        0.35 * (1.0 - texture) +
        0.25 * dist_norm +
        0.05 * V
    )

    gloss_score[~orange] = 0
    gloss_score[~yolk_bool] = 0

    # -------------------------------
    # 5. Порог глянца
    # -------------------------------

    GLOSS_THRESHOLD = 150

    glossy_region = (
        yolk_bool &
        orange &
        ((gloss_score * 255) >= GLOSS_THRESHOLD)
    )

    glossy_region_mask = glossy_region.astype(np.uint8) * 255

    # Только лёгкая морфология, без сильного раздувания
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    glossy_region_mask = cv2.morphologyEx(
        glossy_region_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    glossy_region_mask = cv2.bitwise_and(glossy_region_mask, yolk_mask)

    # -------------------------------
    # 6. Явные маленькие блики отдельно
    # -------------------------------

    v_float = v.astype(np.float32)
    v_blur = cv2.GaussianBlur(v_float, (0, 0), sigmaX=18, sigmaY=18)

    specular_strength = v_float - v_blur
    specular_strength[specular_strength < 0] = 0

    v_yolk = v[yolk_bool]
    spec_yolk = specular_strength[yolk_bool]

    highlight = (
        yolk_bool &
        (v >= np.percentile(v_yolk, 88)) &
        (specular_strength >= max(8, np.percentile(spec_yolk, 85)))
    )

    highlight_mask = highlight.astype(np.uint8) * 255

    # Блики тоже считаем глянцем
    glossy_region_mask = cv2.bitwise_or(glossy_region_mask, highlight_mask)
    glossy_region_mask = cv2.bitwise_and(glossy_region_mask, yolk_mask)

    matte_texture_mask = detect_matte_texture(
        img_bgr,
        yolk_mask,
        highlight_mask=highlight_mask
    )

    # ВАЖНО:
    # текстурная матовая область должна вычитаться из глянца
    glossy_region_mask = cv2.bitwise_and(
        glossy_region_mask,
        cv2.bitwise_not(matte_texture_mask)
    )

    # Но явные блики всё равно считаем глянцем
    glossy_region_mask = cv2.bitwise_or(
        glossy_region_mask,
        highlight_mask
    )

    glossy_region_mask = cv2.bitwise_and(glossy_region_mask, yolk_mask)

    matte_mask = cv2.bitwise_and(
        yolk_mask,
        cv2.bitwise_not(glossy_region_mask)
    )

    return highlight_mask, glossy_region_mask, matte_mask, gloss_score, texture, matte_texture_mask


def analyze_yolk_matte_glossy(image_path, output_prefix="result"):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    yolk_mask = get_yolk_mask(img)

    highlight_mask, glossy_region_mask, matte_mask, gloss_score, texture, matte_texture_mask = get_gloss_masks(
        img,
        yolk_mask
    )

    gloss_index = compute_soft_gloss_index(
        gloss_score=gloss_score,
        yolk_mask=yolk_mask,
        glossy_region_mask=glossy_region_mask,
        matte_texture_mask=matte_texture_mask,
        highlight_mask=highlight_mask
    )

    yolk_area = np.count_nonzero(yolk_mask)
    glossy_area = np.count_nonzero(glossy_region_mask)
    matte_area = np.count_nonzero(matte_mask)
    highlight_area = np.count_nonzero(highlight_mask)

    glossy_percent = 100.0 * glossy_area / yolk_area
    matte_percent = 100.0 * matte_area / yolk_area
    highlight_percent = 100.0 * highlight_area / yolk_area

    overlay = img.copy()

    # Синим — глянцевая область
    blue_layer = overlay.copy()
    blue_layer[glossy_region_mask > 0] = (255, 0, 0)
    overlay = cv2.addWeighted(blue_layer, 0.35, overlay, 0.65, 0)

    # Красным — явные блики
    overlay[highlight_mask > 0] = (0, 0, 255)

    # Зелёным — контур желтка
    contours, _ = cv2.findContours(
        yolk_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    p0 = Path(output_prefix)
    p1 = p0 / "yolk_mask.png"
    p2 = p0 / "glossy_region_mask.png"
    p3 = p0 / "matte_mask.png"
    p4 = p0 / "highlight_mask.png"
    p5 = p0 / "overlay.png"
    p6 = p0 / "gloss_score.png"
    p7 = p0 / "texture.png"

    os.makedirs(p0, exist_ok=True)

    cv2.imwrite(p1, yolk_mask)
    cv2.imwrite(p2, glossy_region_mask)
    cv2.imwrite(p3, matte_mask)
    cv2.imwrite(p4, highlight_mask)
    cv2.imwrite(p5, overlay)

    # Отладочные картинки
    score_vis = np.clip(gloss_score * 255, 0, 255).astype(np.uint8)
    texture_vis = np.clip(texture * 255, 0, 255).astype(np.uint8)

    cv2.imwrite(p6, score_vis)
    cv2.imwrite(p7, texture_vis)

    return {
        "matte_percent": matte_percent,
        "glossy_percent": glossy_percent,
        "highlight_percent": highlight_percent,
        "gloss_index": gloss_index,
        "yolk_area_px": int(yolk_area),
        "glossy_area_px": int(glossy_area),
        "matte_area_px": int(matte_area),
        "highlight_area_px": int(highlight_area),
    }


if __name__ == "__main__":
    name = "egg6"
    result = analyze_yolk_matte_glossy(f"{name}.jpg", output_prefix=f"{name}_result")

    print(f"Glossy area: {result['glossy_percent']:.2f}%")
    print(f"Matte area:  {result['matte_percent']:.2f}%")
    print(f"Highlights:  {result['highlight_percent']:.2f}%")
    print(f"Gloss index: {result['gloss_index']:.2f}")