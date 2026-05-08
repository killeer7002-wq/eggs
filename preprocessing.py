import numpy as np
import pandas as pd

from scipy.signal import medfilt, find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve

from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator

def shift_array_with_nan(y, shift):
    """
    shift > 0: двигает спектр вправо по индексам
    shift < 0: двигает спектр влево по индексам
    """
    y = np.asarray(y, dtype=float)
    result = np.full_like(y, np.nan, dtype=float)

    if shift > 0:
        result[shift:] = y[:-shift]
    elif shift < 0:
        result[:shift] = y[-shift:]
    else:
        result[:] = y

    return result

def find_peak_index_in_interval(x, y, left=950, right=1050):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (x >= left) & (x <= right)

    if not np.any(mask):
        raise ValueError("На заданном интервале нет точек")

    indices = np.flatnonzero(mask)
    local_max = np.argmax(y[mask])

    return indices[local_max]

def shift_peak(
        data: pd.DataFrame,
        x_col="Raman_shift",
        y_col="CCD",
        left=950,
        right=1050,
        target=1003
):
    x = np.asarray(data[x_col], dtype=float)
    y = np.asarray(data[y_col], dtype=float)

    target_idx = np.argmin(np.abs(x - target))

    peak_idx = find_peak_index_in_interval(
        x,
        y,
        left=left,
        right=right,
    )

    shift = target_idx - peak_idx

    y_shifted = shift_array_with_nan(y, shift)

    return y_shifted

def find_two_peaks_in_interval(
    data: pd.DataFrame,
    x_col="Raman_shift",
    y_col="CCD",
    left=950,
    right=1100,
    prominence=None,
    distance=None,
):
    df_part = data[(data[x_col] >= left) & (data[x_col] <= right)]

    if len(df_part) == 0:
        raise ValueError("На заданном интервале нет точек")

    x = df_part[x_col].to_numpy()
    y = df_part[y_col].to_numpy()

    peaks_local, properties = find_peaks(
        y,
        prominence=prominence,
        distance=distance,
    )

    if len(peaks_local) < 2:
        peaks_local = np.array([peaks_local[0], peaks_local[0]])

    # выбираем два самых высоких пика
    top2_local = peaks_local[np.argsort(y[peaks_local])[-2:]]

    # сортируем слева направо
    top2_local = top2_local[np.argsort(x[top2_local])]

    # индексы исходного DataFrame
    peak_indices = df_part.index[top2_local]

    return peak_indices

def shift_two_peak(
        data: pd.DataFrame,
        x_col="Raman_shift",
        y_col="CCD",
        left=950,
        right=1050,
        target=1003
):
    x = np.asarray(data[x_col], dtype=float)
    y = np.asarray(data[y_col], dtype=float)

    target_idx = np.argmin(np.abs(x - target))

    peak_indices = find_two_peaks_in_interval(
        data,
        x_col=x_col,
        y_col=y_col,
        left=left,
        right=right,
    )

    peaks = data.loc[peak_indices]

    mean = (peaks[x_col].iloc[1] + peaks[x_col].iloc[0]) / 2

    half_width = (peaks[x_col].iloc[1] - peaks[x_col].iloc[0]) / 2 + 8

    peak_idx = np.argmin(np.abs(x - mean))

    shift = target_idx - peak_idx

    y_shifted = shift_array_with_nan(y, shift)

    return y_shifted, half_width

def remove_spikes(y, kernel_size=5, threshold=8):
    y = np.asarray(y, dtype=float)
    med = medfilt(y, kernel_size=kernel_size)
    diff = y - med
    mad = np.median(np.abs(diff - np.median(diff)))
    if mad == 0:
      return y.copy()

    mask = np.abs(diff) > threshold * 1.5 * mad
    y_clean = y.copy()
    y_clean[mask] = med[mask]
    return y_clean

def expand_mask(mask, radius=3):
    """
    Расширяет найденные плохие точки влево и вправо,
    чтобы удалить не только сам минимум провала, но и его края.
    """
    mask = np.asarray(mask, dtype=bool)
    expanded = mask.copy()

    bad_indices = np.flatnonzero(mask)

    for i in bad_indices:
        left = max(0, i - radius)
        right = min(len(mask), i + radius + 1)
        expanded[left:right] = True

    return expanded


def robust_sigma_mad(r):
    """
    Робастная оценка масштаба шума через MAD.
    """
    r = np.asarray(r, dtype=float)
    med = np.median(r)
    mad = np.median(np.abs(r - med))

    if mad == 0:
        return np.std(r)

    return 1.4826 * mad


def remove_negative_artifacts(
    x,
    y,
    window_length=41,
    polyorder=3,
    threshold=4.0,
    expand=4,
    niter=3,
):
    """
    Удаляет отрицательные артефакты прибора.

    x             : Raman shift
    y             : intensity
    window_length : окно Savitzky-Golay для грубой гладкой аппроксимации
    polyorder     : порядок полинома Savitzky-Golay
    threshold     : насколько сильно точка должна отклониться вниз
    expand        : насколько расширять плохую область вокруг найденных точек
    niter         : число итераций поиска/замены

    return:
        y_clean   : спектр с заменёнными отрицательными артефактами
        bad       : маска плохих точек
        approx    : последняя гладкая аппроксимация
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    y_clean = y.copy()
    bad_total = np.zeros_like(y, dtype=bool)

    # window_length должен быть нечётным
    if window_length % 2 == 0:
        window_length += 1

    if window_length >= len(y):
        window_length = len(y) - 1 if len(y) % 2 == 0 else len(y)

    for _ in range(niter):
        # 1. Гладкая аппроксимация текущего очищенного спектра
        approx = savgol_filter(
            y_clean,
            window_length=window_length,
            polyorder=polyorder,
        )

        # 2. Остатки: насколько реальный спектр отличается от аппроксимации
        residual = y_clean - approx

        sigma = robust_sigma_mad(residual)

        # 3. Ищем только сильные провалы вниз
        bad = residual < -threshold * sigma

        # 4. Расширяем плохие области
        bad = expand_mask(bad, radius=expand)

        # 5. Добавляем к общей маске плохих точек
        bad_total |= bad

        good = ~bad_total

        if np.sum(good) < 2:
            break

        # 6. Интерполируем плохие точки по хорошим
        interpolator = PchipInterpolator(x[good], y_clean[good])
        y_clean[bad_total] = interpolator(x[bad_total])

    # Финальная аппроксимация для визуализации
    approx = savgol_filter(
        y_clean,
        window_length=window_length,
        polyorder=polyorder,
    )

    return y_clean, bad_total, approx

def baseline_asls(y, lam=1e7, p=0.001, niter=20):
    y = np.asarray(y, dtype=float)
    n = len(y)

    D = sparse.diags(
        [1.0, -2.0, 1.0],
        [0, 1, 2],
        shape=(n - 2, n),
        format="csc"
    )

    DTD = (D.T @ D).tocsc()

    w = np.ones(n)

    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(n, n), format="csc")
        Z = W + lam * DTD

        z = spsolve(Z, w * y)

        w = p * (y > z) + (1 - p) * (y <= z)

    return z

def correct_baseline(y, lam=1e7, p=0.001):
    baseline = baseline_asls(y, lam=lam, p=p)
    corrected = y - baseline
    return corrected, baseline

def normalize_to_peak(x, y, center=1003, half_width=8, mode="area"):
    mask = (x >= center - half_width) & (x <= center + half_width)

    if mode == "height":
        factor = np.max(y[mask])
    elif mode == "area":
        factor = np.trapezoid(np.maximum(y[mask], 0), x[mask])
    else:
        raise ValueError("mode must be 'height' or 'area'")

    if factor == 0:
        return y
    return y / factor

def area_peak(x, y, center=1003, half_width=8):
    mask = (x >= center - half_width) & (x <= center + half_width)
    return np.trapezoid(np.maximum(y[mask], 0), x[mask])

def normalize_by_unchanged_parts(
    data,
    x_col,
    y_col,
    mask=None,
    mode="vector",
    new_col=None,
):
    """
    Нормирует спектр по неизменяемым частям.

    data    : pd.DataFrame
    x_col   : колонка Raman shift
    y_col   : колонка интенсивности
    mask    : True там, где спектр может изменяться;
              эти точки НЕ используются для расчёта нормировки
    mode    : "vector", "area", "mean", "max"
    new_col : если None, перезаписывает y_col;
              иначе создаёт новую колонку

    return:
        data_norm, factor
    """

    data_norm = data.copy()

    x = data_norm[x_col].to_numpy(dtype=float)
    y = data_norm[y_col].to_numpy(dtype=float)

    if mask is None:
        mask = np.zeros_like(x)
    mask = np.asarray(mask, dtype=bool)

    if len(mask) != len(data_norm):
        raise ValueError("mask должен иметь ту же длину, что и data")

    unchanged = ~mask

    # убираем NaN/inf
    good = unchanged & np.isfinite(x) & np.isfinite(y)

    if np.sum(good) == 0:
        raise ValueError("Нет точек для нормировки: вся область замаскирована или содержит NaN")

    x_good = x[good]
    y_good = y[good]

    if mode == "vector":
        factor = np.linalg.norm(y_good)

    elif mode == "area":
        # Для baseline-corrected Raman лучше брать abs,
        # потому что после вычитания baseline могут быть отрицательные значения.
        factor = np.trapezoid(np.abs(y_good), x_good)

    elif mode == "mean":
        factor = np.mean(y_good)

    elif mode == "max":
        factor = np.max(np.abs(y_good))

    else:
        raise ValueError("mode должен быть 'vector', 'area', 'mean' или 'max'")

    if factor == 0:
        raise ValueError("Нормировочный коэффициент равен нулю")

    y_norm = y / factor

    if new_col is None:
        data_norm[y_col] = y_norm
    else:
        data_norm[new_col] = y_norm

    return data_norm, factor
