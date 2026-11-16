"""
計算影片前 N 秒內的擊球次數（打一下算一次）

流程：
    1. 擷取影片前 N 秒（預設 180 秒 = 3 分鐘），並縮小解析度存成 mp4
    2. 呼叫 predict.py 用 TrackNetV3 預測每一幀的球座標
    3. 從球的軌跡偵測「突然轉向」的時間點，每一次轉向算一次擊球

用法：
    python count_hits.py --video_file IMG_0319.mp4
    # 已經有預測好的 csv，只想重算擊球次數（調參數用）：
    python count_hits.py --video_file IMG_0319.mp4 --skip_predict
"""
import os
import sys
import bisect
import time
import argparse
import subprocess

import cv2
import numpy as np
import pandas as pd


def trim_video(video_file, out_file, duration, max_width):
    """ 擷取影片前 duration 秒，寬度大於 max_width 時等比例縮小。回傳 fps。"""
    cap = cv2.VideoCapture(video_file)
    assert cap.isOpened(), f'無法開啟影片: {video_file}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / w)
    out_w, out_h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    n_frames = min(total, int(round(duration * fps)))
    print(f'原始影片: {w}x{h}, {fps:.2f} fps, {total} 幀 ({total / fps:.1f} 秒)')
    print(f'擷取前 {n_frames} 幀 ({n_frames / fps:.1f} 秒)，輸出 {out_w}x{out_h}')

    writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()
    cap.release()
    return fps


def detect_hits(df, fps, max_gap_sec=0.2, win_sec=0.07, min_angle=50, min_speed=0.004, min_interval_sec=0.35, img_diag=1.0,
                img_w=None, hit_zone=None):
    """ 由球軌跡偵測擊球時間點。

        擊球的特徵是球的運動方向「瞬間」大幅改變；重力造成的拋物線頂點是緩慢轉向，
        且頂點附近速度很低，所以用「前後速度向量夾角」加上「速度門檻」來區分。

        Args:
            df (pd.DataFrame): predict.py 輸出的 csv，欄位 Frame, Visibility, X, Y
            fps (float): 影片幀率
            max_gap_sec (float): 球消失不超過這個秒數時用內插補點
            win_sec (float): 計算前後平均速度的時間窗
            min_angle (float): 前後速度夾角大於此角度(度)才算轉向
            min_speed (float): 擊球後速度門檻（每秒移動的畫面對角線比例 / fps）
            min_interval_sec (float): 兩次擊球最短間隔
            img_diag (float): 畫面對角線長度(像素)，用來把速度正規化
            img_w (int, optional): 畫面寬度(像素)，搭配 hit_zone 使用
            hit_zone (Tuple[float], optional): (左邊界, 右邊界) 畫面寬度比例。
                只有 X < 左邊界 或 X > 右邊界 的轉向才算擊球，用來排除畫面中間背景場地的球

        Returns:
            hit_frames (List[int]): 擊球發生的幀
    """
    x = np.array(df['X'], dtype=float)
    y = np.array(df['Y'], dtype=float)
    vis = df['Visibility'].to_numpy() == 1
    x[~vis], y[~vis] = np.nan, np.nan

    # 短暫遺失的幀用線性內插補起來，長時間遺失（死球、出畫面）保持 NaN
    max_gap = max(1, int(round(max_gap_sec * fps)))
    x = pd.Series(x).interpolate(limit=max_gap, limit_area='inside').to_numpy()
    y = pd.Series(y).interpolate(limit=max_gap, limit_area='inside').to_numpy()

    # 中值濾波去除單幀跳點
    x = pd.Series(x).rolling(3, center=True, min_periods=1).median().to_numpy()
    y = pd.Series(y).rolling(3, center=True, min_periods=1).median().to_numpy()

    w = max(2, int(round(win_sec * fps)))
    n = len(x)
    scores = np.zeros(n)
    for t in range(w, n - w):
        # 前一段與後一段的位移向量
        v_in = np.array([x[t] - x[t - w], y[t] - y[t - w]])
        v_out = np.array([x[t + w] - x[t], y[t + w] - y[t]])
        if np.isnan(v_in).any() or np.isnan(v_out).any():
            continue
        s_in, s_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
        speed_th = min_speed * img_diag * w
        if s_out < speed_th or s_in < speed_th * 0.5:
            continue
        cos = np.dot(v_in, v_out) / (s_in * s_out)
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        if hit_zone is not None and img_w * hit_zone[0] <= x[t] <= img_w * hit_zone[1]:
            continue
        if angle >= min_angle:
            scores[t] = angle

    # 非極大值抑制：最短間隔內只保留轉角最大的那一幀
    min_interval = int(round(min_interval_sec * fps))
    hit_frames = []
    for t in np.argsort(-scores):
        if scores[t] <= 0:
            break
        if all(abs(t - h) >= min_interval for h in hit_frames):
            hit_frames.append(int(t))
    return sorted(hit_frames)


def detect_net_crossings(df, fps, img_w, side=0.3, max_gap_sec=0.2, max_travel_sec=1.2, max_step=0.06, min_interval_sec=0.4):
    """ 計算球過網的次數（每過網一次 = 一拍）。

        相機架在網柱旁時，網子在畫面上是中間一條直線，球過網就是從畫面一側飛到另一側。
        只計算「從左側區域連續飛到右側區域（或反向）」的軌跡，背景場地的球在畫面上
        移動範圍小、不會橫跨整個畫面，因此會被排除。

        Args:
            df (pd.DataFrame): predict.py 輸出的 csv，欄位 Frame, Visibility, X, Y
            fps (float): 影片幀率
            img_w (int): 畫面寬度(像素)
            side (float): 左右區域的寬度比例，球要從 X < side*W 飛到 X > (1-side)*W 才算過網
            max_gap_sec (float): 球消失不超過這個秒數時用內插補點
            max_travel_sec (float): 從一側飛到另一側的最長時間
            max_step (float): 單幀最大水平位移（畫面寬度比例），超過視為追蹤跳到別的物體
            min_interval_sec (float): 兩次過網最短間隔

        Returns:
            crossings (List[Tuple[int, str]]): (過網的幀, 方向 'L>R' 或 'R>L')
    """
    x = np.array(df['X'], dtype=float)
    x[np.array(df['Visibility']) != 1] = np.nan
    max_gap = max(1, int(round(max_gap_sec * fps)))
    x = pd.Series(x).interpolate(limit=max_gap, limit_area='inside')
    x = x.rolling(3, center=True, min_periods=1).median().to_numpy()

    left, right = img_w * side, img_w * (1 - side)
    n, max_travel = len(x), int(max_travel_sec * fps)
    crossings = []
    t = 0
    while t < n:
        # 起點必須在左側或右側區域
        if np.isnan(x[t]) or left <= x[t] <= right:
            t += 1
            continue
        from_left = x[t] < left
        arrive = None
        for k in range(t + 1, min(n, t + max_travel)):
            # 軌跡斷掉或跳太遠就放棄這段
            if np.isnan(x[k]) or abs(x[k] - x[k - 1]) > img_w * max_step:
                break
            if (from_left and x[k] > right) or (not from_left and x[k] < left):
                arrive = k
                break
            # 還在起點那一側，把起點往後更新
            if (from_left and x[k] < left) or (not from_left and x[k] > right):
                t = k
        if arrive is None:
            t += 1
            continue
        # 以最接近畫面中線（網子）的那一幀當作過網時間
        mid = t + int(np.argmin(np.abs(x[t:arrive + 1] - img_w / 2)))
        if not crossings or mid - crossings[-1][0] >= min_interval_sec * fps:
            crossings.append((mid, 'L>R' if from_left else 'R>L'))
        t = arrive
    return crossings


def get_tripwire_box(img_w, img_h, region):
    """ 將比例座標 (x0, x1, y0, y1) 轉成像素座標。"""
    x0, x1, y0, y1 = region
    return int(x0 * img_w), int(x1 * img_w), int(y0 * img_h), int(y1 * img_h)


def detect_tripwire(video_file, region, diff_th=30, energy_th=0.0024, merge_sec=0.1, min_interval_sec=0.3):
    """ 虛擬感應線：計算球飛越網子上方的次數（每過網一次 = 一拍）。

        相機架在網柱旁時，前景的球每次過網都會快速穿過網柱正上方的區域。
        這塊區域沒有球員，背景場地在畫面下方，所以只要偵測這塊區域內「有東西快速移動」
        就代表球過網；撿球、走位時這裡沒有動態，自然不會被計算。

        Args:
            video_file (str): 影片路徑
            region (Tuple[float]): 感應區 (x0, x1, y0, y1)，畫面寬高的比例
            diff_th (int): 相鄰兩幀灰階差異大於此值的像素視為移動
            energy_th (float): 移動像素佔感應區面積的比例大於此值，視為有球通過
            merge_sec (float): 間隔小於此秒數的動態合併為同一次過網
            min_interval_sec (float): 兩次過網最短間隔

        Returns:
            crossings (List[Tuple[int, str]]): (過網的幀, 方向 'L>R' 或 'R>L')
            energy (np.ndarray): 每一幀感應區的移動像素數
    """
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, x1, y0, y1 = get_tripwire_box(w, h, region)
    min_pixels = energy_th * (x1 - x0) * (y1 - y0)

    ok, prev = cap.read()
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
    energy, cx = [0], [np.nan]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        moving = cv2.absdiff(gray, prev) > diff_th
        ys, xs = np.nonzero(moving)
        energy.append(len(xs))
        cx.append(xs.mean() if len(xs) > 15 else np.nan)
        prev = gray
    cap.release()
    energy, cx = np.array(energy), np.array(cx)

    # 把連續有動態的幀合併成一次過網事件
    events = []
    for f in np.nonzero(energy > min_pixels)[0]:
        if events and f - events[-1][-1] <= merge_sec * fps:
            events[-1].append(f)
        else:
            events.append([f])

    crossings = []
    for ev in events:
        # 動態重心由左往右移 = 球從左飛到右
        xs = cx[ev[0]:ev[-1] + 1]
        xs = xs[~np.isnan(xs)]
        direction = '?' if len(xs) < 2 else ('L>R' if xs[-1] > xs[0] else 'R>L')
        peak = ev[int(np.argmax(energy[ev]))]
        if crossings and peak - crossings[-1][0] < min_interval_sec * fps:
            continue
        crossings.append((int(peak), direction))
    return crossings, energy


def detect_mistakes(hit_frames, fps, end_frame, gap_sec=3.0):
    """ 失誤判斷：球過網後，超過 gap_sec 秒都沒有下一次過網，代表球沒有被打回來（沒過網、出界或沒接到），算一次失誤。

        Args:
            hit_frames (List[int]): 每次過網的幀（由小到大）
            fps (float): 影片幀率
            end_frame (int): 目前看到的最後一幀（錄影中為目前幀，影片則為最後一幀）
            gap_sec (float): 同一回合兩次過網之間允許的最長間隔（秒）

        Returns:
            mistakes (List[int]): 每次失誤被確認的幀（= 最後一次過網的幀 + gap_sec 秒）
    """
    gap = gap_sec * (fps if fps and fps > 1 else 30.0)
    hits = sorted(hit_frames)
    mistakes = []
    for i, h in enumerate(hits):
        next_hit = hits[i + 1] if i + 1 < len(hits) else None
        if next_hit is not None:
            if next_hit - h > gap:
                mistakes.append(int(h + gap))
        elif end_frame - h >= gap:
            mistakes.append(int(h + gap))
    return mistakes


def detect_mistakes_by_side(crossings, fps, end_frame, gap_sec=3.0):
    """ 失誤判斷（分左右）：規則與 detect_mistakes 相同，另外依「最後一次過網的方向」判斷是哪一方失誤。

        最後一球 L>R（左方打到右方）之後沒有下一次過網 → 右方沒把球打回來 → 右方失誤；
        最後一球 R>L → 左方失誤。方向無法判斷（'?'）時，用前一球方向的反方向推算。
        （左、右指的是攝影機畫面的左邊與右邊）

        Args:
            crossings (List[Tuple[int, str]]): (過網的幀, 方向 'L>R' / 'R>L' / '?')，由小到大
            fps, end_frame, gap_sec: 同 detect_mistakes

        Returns:
            mistakes (List[Tuple[int, str]]): (失誤被確認的幀, 'L' 或 'R' 或 '?')
    """
    crossings = sorted(crossings)
    frames = [f for f, _ in crossings]
    by_frame = {}
    prev_dir = None
    for f, d in crossings:
        if d not in ('L>R', 'R>L') and prev_dir is not None:
            d = 'R>L' if prev_dir == 'L>R' else 'L>R'
        by_frame[f] = d
        if d in ('L>R', 'R>L'):
            prev_dir = d
    gap = gap_sec * (fps if fps and fps > 1 else 30.0)
    result = []
    for m in detect_mistakes(frames, fps, end_frame, gap_sec):
        last_hit = int(round(m - gap))
        # 找出觸發這次失誤的那一球（m = 該球的幀 + gap，取整前後可能差 1）
        idx = bisect.bisect_right(frames, last_hit + 1) - 1
        d = by_frame.get(frames[idx]) if idx >= 0 else None
        side = 'R' if d == 'L>R' else 'L' if d == 'R>L' else '?'
        result.append((m, side))
    return result


class TripwireCounter:
    """ 即時版虛擬感應線：一幀一幀餵入攝影機畫面，邊錄邊計算過網次數。

        判斷規則與 detect_tripwire 相同，只是改成逐幀處理，不需要等整支影片讀完。
        一次過網事件要等感應區安靜超過 merge_sec 才能確定，所以計數會比畫面晚約 0.1 秒。

        用法：
            counter = TripwireCounter(fps, frame_w, frame_h, region)
            for frame in 攝影機畫面:
                new_crossings = counter.update(frame)   # 這一幀新確定的過網 [(幀, 方向)]
            counter.flush()                             # 錄影結束時結算最後一次事件
    """

    def __init__(self, fps, img_w, img_h, region, diff_th=30, energy_th=0.0024, merge_sec=0.1, min_interval_sec=0.3):
        self.fps = fps if fps and fps > 1 else 30.0
        self.x0, self.x1, self.y0, self.y1 = get_tripwire_box(img_w, img_h, region)
        self.diff_th = diff_th
        self.min_pixels = energy_th * (self.x1 - self.x0) * (self.y1 - self.y0)
        self.merge_frames = merge_sec * self.fps
        self.min_interval_frames = min_interval_sec * self.fps
        self.frame_idx = -1
        self.prev = None
        self.crossings = []      # 已確定的過網 [(幀, 方向)]
        self._event = []         # 目前進行中的事件：[(幀, 移動像素數)]
        self._event_cx = []      # 事件期間每一幀的動態重心 x

    def update(self, frame):
        """ 餵入一幀 BGR 畫面，回傳這一幀新確定的過網清單。"""
        self.frame_idx += 1
        f = self.frame_idx
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[self.y0:self.y1, self.x0:self.x1]
        if self.prev is None or self.prev.shape != gray.shape:
            self.prev = gray
            return []
        moving = cv2.absdiff(gray, self.prev) > self.diff_th
        self.prev = gray
        ys, xs = np.nonzero(moving)
        energy = len(xs)
        cx = xs.mean() if len(xs) > 15 else np.nan

        new = []
        # 感應區安靜太久，前一個事件結束
        if self._event and f - self._event[-1][0] > self.merge_frames:
            new += self._finish_event()
        if self._event:
            self._event_cx.append(cx)
        if energy > self.min_pixels:
            if not self._event:
                self._event_cx = [cx]
            self._event.append((f, energy))
        return new

    def flush(self):
        """ 錄影結束時，結算最後一個還沒結束的事件。"""
        return self._finish_event() if self._event else []

    def _finish_event(self):
        frames = [fr for fr, _ in self._event]
        energies = [en for _, en in self._event]
        span = frames[-1] - frames[0] + 1
        xs = np.array(self._event_cx[:span], dtype=float)
        xs = xs[~np.isnan(xs)]
        direction = '?' if len(xs) < 2 else ('L>R' if xs[-1] > xs[0] else 'R>L')
        peak = frames[int(np.argmax(energies))]
        self._event, self._event_cx = [], []
        if self.crossings and peak - self.crossings[-1][0] < self.min_interval_frames:
            return []
        self.crossings.append((int(peak), direction))
        return [self.crossings[-1]]


def write_tripwire_video(video_file, hit_frames, fps, out_file, region, diff_th=30):
    """ 輸出標註影片：黃框為感應區，球通過時標紅色軌跡並累計次數。"""
    cap = cv2.VideoCapture(video_file)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, x1, y0, y1 = get_tripwire_box(w, h, region)
    writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    hit_set, count, flash = set(hit_frames), 0, 0
    prev, f = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if f in hit_set:
            count += 1
            flash = int(fps * 0.3)
        if flash > 0:
            if prev is not None:
                moving = cv2.absdiff(gray, prev)[y0:y1, x0:x1] > diff_th
                frame[y0:y1, x0:x1][moving] = (0, 0, 255)
            cv2.putText(frame, 'HIT', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            flash -= 1
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255) if flash > 0 else (0, 255, 255), 2)
        cv2.putText(frame, f'Hits: {count}  t={f / fps:.1f}s', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        writer.write(frame)
        prev = gray
        f += 1
    writer.release()
    cap.release()


def write_hit_video(video_file, df, hit_frames, fps, out_file):
    """ 輸出標註擊球時間點與累計次數的影片，方便人工核對。"""
    cap = cv2.VideoCapture(video_file)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    hit_set, count, flash = set(hit_frames), 0, 0
    pos = {int(r.Frame): (int(r.X), int(r.Y)) for r in df.itertuples() if r.Visibility == 1}
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f in hit_set:
            count += 1
            flash = int(fps * 0.3)
        if f in pos:
            cv2.circle(frame, pos[f], 6, (0, 0, 255) if flash > 0 else (0, 255, 255), -1)
        if flash > 0:
            cv2.putText(frame, 'HIT', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            flash -= 1
        cv2.putText(frame, f'Hits: {count}  t={f / fps:.1f}s', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        writer.write(frame)
        f += 1
    writer.release()
    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 直接在 VS Code 按 ▶ 執行時沒有參數，會用這裡的預設影片
    parser.add_argument('--video_file', type=str, default='IMG_0319.mp4', help='原始影片路徑')
    parser.add_argument('--duration', type=float, default=180, help='只計算前幾秒（預設 180 秒 = 3 分鐘）')
    parser.add_argument('--tracknet_file', type=str, default='ckpts/TrackNet_best.pt')
    parser.add_argument('--inpaintnet_file', type=str, default='ckpts/InpaintNet_best.pt')
    parser.add_argument('--save_dir', type=str, default='hit_result')
    parser.add_argument('--max_width', type=int, default=1280, help='擷取影片的最大寬度，降低記憶體用量')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--skip_predict', action='store_true', default=False, help='跳過擷取與預測，直接用已存在的檔案計算')
    parser.add_argument('--mode', type=str, default='tripwire', choices=['tripwire', 'net', 'hit'],
                        help='tripwire: 網柱上方虛擬感應線，計算球過網次數（預設，不需 TrackNet）；'
                             'net: 用 TrackNet 軌跡計算過網次數；hit: 用 TrackNet 軌跡偵測球的轉向')
    parser.add_argument('--tripwire', type=lambda s: [float(v) for v in s.split(',')], default=[0.40, 0.60, 0.23, 0.45],
                        help='[tripwire] 感應區 "x0,x1,y0,y1"（畫面寬高比例），預設為網柱正上方')
    parser.add_argument('--net_side', type=float, default=0.3, help='[net] 球要從畫面左/右側多少比例範圍內飛到另一側才算過網')
    parser.add_argument('--net_max_step', type=float, default=0.06, help='[net] 單幀最大水平位移（畫面寬度比例）')
    parser.add_argument('--min_angle', type=float, default=50, help='判定為擊球的最小轉向角度(度)')
    parser.add_argument('--min_interval', type=float, default=0.5, help='兩次擊球最短間隔(秒)')
    parser.add_argument('--hit_zone', type=lambda s: [float(v) for v in s.split(',')], default=[0.3, 0.7],
                        help='擊球區：畫面寬度比例 "左,右"，只計算左右兩側的擊球；設 "0,0" 表示不限制')
    parser.add_argument('--no_video', action='store_true', default=False, help='不輸出標註影片')
    args = parser.parse_args()

    # 相對路徑一律以本程式所在資料夾為基準，不受終端機目前位置影響
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    os.makedirs(args.save_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(args.video_file))[0]
    clip_name = f'{video_name}_{int(args.duration)}s'
    clip_file = os.path.join(args.save_dir, f'{clip_name}.mp4')
    csv_file = os.path.join(args.save_dir, f'{clip_name}_ball.csv')

    # Step 1: 擷取影片前 N 秒
    if not args.skip_predict or not os.path.exists(clip_file):
        trim_video(args.video_file, clip_file, args.duration, args.max_width)

    cap = cv2.VideoCapture(clip_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    img_diag = float(np.hypot(img_w, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    hit_csv = os.path.join(args.save_dir, f'{clip_name}_hits.csv')
    out_video = os.path.join(args.save_dir, f'{clip_name}_hits.mp4')

    if args.mode == 'tripwire':
        # Step 2: 虛擬感應線計算過網次數（不需要 TrackNet）
        crossings, _ = detect_tripwire(clip_file, args.tripwire, min_interval_sec=0.3)
        hit_frames = [f for f, _ in crossings]
        hit_df = pd.DataFrame({'Hit': range(1, len(hit_frames) + 1),
                               'Frame': hit_frames,
                               'Time_sec': [round(f / fps, 2) for f in hit_frames],
                               'Direction': [d for _, d in crossings]})
        try:
            hit_df.to_csv(hit_csv, index=False)
        except PermissionError:
            # csv 正被 Excel 等程式開啟時無法覆寫，改存成另一個檔名
            print(f'注意：{hit_csv} 正被其他程式開啟（例如 Excel），改存成新檔名')
            hit_csv = hit_csv.replace('.csv', f'_{time.strftime("%H%M%S")}.csv')
            hit_df.to_csv(hit_csv, index=False)

        print('=' * 40)
        print(f' 影片: {args.video_file}')
        print(f' 計算範圍: 前 {n_frames / fps:.1f} 秒')
        print(f' 計算方式: 球過網次數（網柱上方虛擬感應線）')
        print(f' 擊球次數: {len(hit_frames)} 次')
        for a in range(0, int(np.ceil(n_frames / fps)), 30):
            n = sum(a <= f / fps < a + 30 for f in hit_frames)
            print(f'   {a:>3}-{a + 30:<3} 秒: {n:>3} 拍')
        print('=' * 40)
        print(f'擊球時間點已存到 {hit_csv}')
        if not args.no_video:
            write_tripwire_video(clip_file, hit_frames, fps, out_video, args.tripwire)
            print(f'標註影片已存到 {out_video}（請播放核對計數是否正確）')
        sys.exit(0)

    # Step 2: 用 TrackNetV3 預測球的位置
    if not args.skip_predict:
        # 優先用專案內的 .venv（torch 可正常使用 GPU）；anaconda 的 torch 會有 OpenMP 衝突
        venv_python = os.path.join(script_dir, '.venv', 'Scripts', 'python.exe')
        python_exe = venv_python if os.path.exists(venv_python) else sys.executable
        cmd = [python_exe, 'predict.py',
               '--video_file', clip_file,
               '--tracknet_file', args.tracknet_file,
               '--inpaintnet_file', args.inpaintnet_file,
               '--save_dir', args.save_dir,
               '--batch_size', str(args.batch_size),
               '--large_video']
        print('執行:', ' '.join(cmd))
        subprocess.run(cmd, check=True, cwd=script_dir)

    # Step 3: 計算擊球次數
    df = pd.read_csv(csv_file)
    if args.mode == 'net':
        crossings = detect_net_crossings(df, fps, img_w, side=args.net_side, max_step=args.net_max_step)
        hit_frames = [f for f, _ in crossings]
        directions = [d for _, d in crossings]
    else:
        hit_zone = args.hit_zone if args.hit_zone[1] > args.hit_zone[0] else None
        hit_frames = detect_hits(df, fps, min_angle=args.min_angle, min_interval_sec=args.min_interval, img_diag=img_diag,
                                 img_w=img_w, hit_zone=hit_zone)
        directions = [''] * len(hit_frames)

    hit_df = pd.DataFrame({'Hit': range(1, len(hit_frames) + 1),
                           'Frame': hit_frames,
                           'Time_sec': [round(f / fps, 2) for f in hit_frames],
                           'Direction': directions})
    hit_df.to_csv(hit_csv, index=False)

    print('=' * 40)
    print(f' 影片: {args.video_file}')
    print(f' 計算範圍: 前 {len(df) / fps:.1f} 秒')
    print(f' 計算方式: {"球過網次數" if args.mode == "net" else "球轉向偵測"}')
    print(f' 擊球次數: {len(hit_frames)} 次')
    print('=' * 40)
    print(f'擊球時間點已存到 {hit_csv}')

    if not args.no_video:
        write_hit_video(clip_file, df, hit_frames, fps, out_video)
        print(f'標註影片已存到 {out_video}（請播放核對計數是否正確）')
