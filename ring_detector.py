#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铃声识别 (ring detector)
=========================
目标：实时监听音频输入，当识别到与 reference.wav 相同的铃声时，输出标识 "1"；
     否则持续监听（不输出）。

技术方案（选型理由）
--------------------
1. 采集：sounddevice (PortAudio / Windows WASAPI)。pip 直装、无编译、延迟低。
   默认用麦克风阵列采集“外放”的声音；--loopback 可用“立体声混音”直接内录系统声音。
2. 特征：log-Mel 滤波器组 + MFCC(13) + 倒谱均值归一化(CMN)。
   CMN 能抵消“外放→麦克风”造成的频响/增益差异，这是原方案“MFCC 均值+余弦”所缺的。
3. 匹配：子序列 DTW（open-begin / open-end）。允许铃声出现在监听流的任意位置，
   对时长伸缩与轻微混响更鲁棒；原方案对整段取均值再算余弦，丢失时间信息，短铃声极易误判。
4. 输出：命中即打印 "1"，并调用可插拔钩子 on_match()（原 Arduino 串口逻辑改为预留函数）。

用法
----
  python ring_detector.py                       # 实时麦克风监听
  python ring_detector.py --list                # 列出音频设备
  python ring_detector.py --selftest            # 离线自测（正/负样本）
  python ring_detector.py --input a.wav         # 用音频文件当作输入流（调试）
  python ring_detector.py --device 12 --loopback  # 内录系统声音
"""

import argparse
import os
import random
import shutil
import sys
import tempfile
import time
import threading
import collections

import numpy as np
from scipy.signal import resample_poly, get_window
from scipy.fftpack import dct
from scipy.ndimage import uniform_filter1d

try:  # Windows 控制台默认 GBK，避免中文/特殊字符报错
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))

# ==================== 配置参数 ====================
def _pick_reference():
    """优先 reference.mp3，其次 wav/ogg/flac。"""
    for fn in ("reference.mp3", "reference.wav", "reference.ogg", "reference.flac"):
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            return p
    return os.path.join(HERE, "reference.mp3")


REFERENCE_FILE = _pick_reference()
CAPTURE_SR = 16000        # 采集采样率（参考音频为 16k；设备不支持时自动回退）
FRAME_MS = 25             # 帧长
HOP_MS = 10               # 帧移
N_FFT = 512
N_MELS = 40
PREEMPH = 0.97
STEP_SEC = 0.25           # 匹配滑动步长（秒）
COOLDOWN_SEC = 2.0        # 命中后冷却（秒，文件回放等无状态场景用）
RELEASE_SEC = 1.0         # 复位时间（秒）：连续 RELEASE_SEC 未命中则重新武装，使多次铃声分别输出一个 1
ENERGY_GATE = 0.02        # 能量门控(RMS)，低于此值视为静音不做匹配（环境实测底噪≈0.002）
DTW_THRESHOLD = 0.30       # 归一化 DTW 距离阈值（余弦尺度，0~2；误报多则降到0.25，漏报多则升到0.35）
HIT_CONFIRM = 1           # 命中确认次数：1=命中即输出；调大（2~3）可进一步抑制偶发误报
ENERGY_FLOOR_DB = 14.0    # 帧能量加权：峰值下 14dB 内线性保留权重，更激进地突显铃声高能量帧
DETREND_W = 5             # mel 轴去趋势窗：去掉频谱缓慢倾斜（扬声器/麦克风频响），保留峰值结构

# ---- Arduino 串口发送 ----
SERIAL_PORT = None        # 串口名；None=自动检测 USB 串口（如 COM5）
SERIAL_BAUD = 9600        # 波特率（与板子 Serial.begin(9600) 一致）
IDLE_SEND_SEC = 60.0      # 超过该时长未向板子发送，则补发一次 '1'
SECOND_MIN, SECOND_MAX = 2.0, 5.0   # 命中后补发第二次 '1' 的随机间隔（秒）
WINDOW_SEC = 5.0          # 匹配窗口长度（秒），需 >= 参考时长
TRIM_DB = -10.0           # 参考样本静音裁剪阈值（相对峰值 dB），只保留铃声核心段
SMOOTH_FRAMES = 3         # 帧间平滑窗口（静噪：抑制随机噪声对单帧 MFCC 的扰动）


# ==================== 音频 I/O ====================
def _decode_with_ffmpeg(path, target_sr=None):
    """非 wav（如 mp3）用 ffmpeg 解码为单声道 wav；不可用则返回 (None, None)。"""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None, None
    out_dir = os.environ.get("ENNEWCLAW_TASK_TEMP_DIR") or tempfile.gettempdir()
    sr = int(target_sr or 44100)
    out = os.path.join(out_dir, "decoded_reference.wav")
    import subprocess
    subprocess.run([exe, "-y", "-i", path, "-ac", "1", "-ar", str(sr), out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    import soundfile as sf
    y, srr = sf.read(out, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype("float32"), srr


def read_wav(path, target_sr=None):
    """读取 wav（优先 soundfile，回退标准库 wave），返回 (float32 mono, sr)。"""
    y, sr = None, None
    try:
        import soundfile as sf
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
    except Exception:
        y, sr = _decode_with_ffmpeg(path, target_sr)   # mp3 等压缩格式兜底
        if y is not None:
            return y, sr
        import wave
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            ch = w.getnchannels()
            sw = w.getsampwidth()
            raw = w.readframes(n)
        if sw == 2:
            a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sw == 1:
            a = (np.frombuffer(raw, dtype="<u1").astype(np.float32) - 128.0) / 128.0
        elif sw == 4:
            a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError("unsupported sample width: %s" % sw)
        y = a.reshape(-1, ch).mean(axis=1) if ch > 1 else a
    y = np.asarray(y, dtype=np.float32)
    if target_sr and sr != target_sr:
        y = resample_poly(y, target_sr, sr).astype(np.float32)
        sr = target_sr
    return y, sr


def trim_silence(y, sr, thresh_db=TRIM_DB):
    """按峰值相对 dB 阈值裁剪首尾静音，返回裁剪后的音频。"""
    if len(y) == 0:
        return y
    hop = max(1, int(sr * 0.01))
    win = max(hop, int(sr * 0.02))
    rms = []
    for s in range(0, max(1, len(y) - win), hop):
        rms.append(np.sqrt(np.mean(y[s:s + win] ** 2)))
    rms = np.asarray(rms) if rms else np.asarray([0.0])
    peak = float(rms.max()) or 1e-9
    thr = peak * (10 ** (thresh_db / 20.0))
    idx = np.where(rms > thr)[0]
    if len(idx) == 0:
        return y
    start = max(0, idx[0] * hop - win)
    end = min(len(y), idx[-1] * hop + 2 * win)
    return y[start:end]


# ==================== 特征提取 ====================
_win_cache = {}


def _hamming(win):
    if win not in _win_cache:
        _win_cache[win] = get_window("hamming", win).astype(np.float32)
    return _win_cache[win]


def mel_filterbank(sr, n_fft, n_mels, fmin=20.0, fmax=None):
    """构造 Mel 三角滤波器组，形状 [n_mels, n_fft//2+1]。"""
    if fmax is None:
        fmax = sr / 2.0

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    freqs = mel2hz(mels)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        if r <= l:
            continue
        if c <= l:
            c = l + 1
        if r <= c:
            r = c + 1
        for k in range(l, min(c, len(fb[i - 1]))):
            fb[i - 1, k] = (k - l) / float(c - l)
        for k in range(c, min(r, len(fb[i - 1]))):
            fb[i - 1, k] = (r - k) / float(r - c)
    return fb


def extract_feat(y, sr, fb, smooth=None):
    """返回 (logmel[D=n_mels, T], edb[T])。

    用 log-Mel 谱去趋势（detrend）作为特征：先减去沿 mel 轴的平滑包络，
    去掉“扬声器→麦克风”造成的频谱缓慢倾斜（频响），保留精细峰结构；
    既对声学路径鲁棒（实测真实铃声 dist 低），又能区分不同音高的叮声。
    edb：每帧对数能量(dB)，用于帧能量加权，降低低能量帧干扰。
    """
    if smooth is None:
        smooth = SMOOTH_FRAMES
    y = np.asarray(y, dtype=np.float32)
    empty = (np.zeros((fb.shape[0], 0), dtype=np.float32), np.zeros(0, dtype=np.float32))
    if len(y) < int(sr * FRAME_MS / 1000):
        return empty
    y = np.append(y[0], y[1:] - PREEMPH * y[:-1])
    win = int(sr * FRAME_MS / 1000)
    hop = int(sr * HOP_MS / 1000)
    n_frames = 1 + (len(y) - win) // hop
    if n_frames <= 0:
        return empty
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = y[idx] * _hamming(win)[None, :]           # [T, win]
    edb = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    spec = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2   # [T, n_fft//2+1]
    logmel = np.log(spec @ fb.T + 1e-10)               # [T, n_mels]
    if DETREND_W and DETREND_W > 1 and logmel.shape[1] >= DETREND_W:
        logmel = logmel - uniform_filter1d(logmel, size=DETREND_W, axis=1, mode="nearest")
    if smooth and smooth > 1 and logmel.shape[0] >= smooth:
        logmel = uniform_filter1d(logmel, size=smooth, axis=0, mode="nearest")
    return logmel.T.astype(np.float32), edb.astype(np.float32)


def energy_weight(edb, floor_db=25.0):
    """峰值下 floor_db 内线性升到 1，更低则到 0，用于帧能量加权。"""
    e = np.asarray(edb, dtype=np.float64)
    if e.size == 0:
        return e
    peak = float(e.max())
    if not np.isfinite(peak):
        return np.ones_like(e)
    return np.clip((e - (peak - floor_db)) / floor_db, 0.0, 1.0)


# ==================== 匹配（子序列 DTW，行内向量化） ====================
_fb_cache = {}


def get_fb(sr):
    if sr not in _fb_cache:
        _fb_cache[sr] = mel_filterbank(sr, N_FFT, N_MELS)
    return _fb_cache[sr]


def frame_cost(ref_f, win_f, metric="cosine"):
    """帧间代价矩阵 [T_win, T_ref]。

    metric="cosine"：1-余弦相似度（尺度无关，抗幅度/增益差异），范围 [0,2]。
    metric="euclid"：欧氏距离。
    """
    if metric == "cosine":
        rn = ref_f / (np.linalg.norm(ref_f, axis=0, keepdims=True) + 1e-9)
        wn = win_f / (np.linalg.norm(win_f, axis=0, keepdims=True) + 1e-9)
        cos = rn.T @ wn                                   # [Tb, Ta]
        return (1.0 - cos).T                              # [Ta, Tb]
    diff = win_f[:, :, None] - ref_f[:, None, :]          # [D, Ta, Tb]
    return np.sqrt(np.sum(diff * diff, axis=0))


def subsequence_dtw(cost, norm=None):
    """
    cost: [Ta, Tb]，A=监听窗口(Ta 帧)，B=参考(Tb 帧)。
    open-begin/open-end：允许参考对齐到窗口的任意子段。
    norm 为归一化因子（缺省用 Tb）；传入帧能量权重和时即为加权平均。
    行内用 cumsum+cummin 技巧向量化，避免 O(Ta*Tb) 的 Python 双循环。
    """
    Ta, Tb = cost.shape
    if Ta < 1 or Tb < 1:
        return float("inf")
    acc = np.empty((Ta, Tb), dtype=np.float64)
    acc[0, 0] = cost[0, 0]
    if Tb > 1:
        acc[0, 1:] = np.cumsum(cost[0, 1:]) + cost[0, 0]
    for i in range(1, Ta):
        prev = acc[i - 1]
        a0 = float(cost[i, 0])
        acc[i, 0] = a0
        if Tb > 1:
            c = cost[i, 1:]
            b = np.minimum(prev[1:], prev[:-1])
            P = np.cumsum(c)
            Pprev = np.empty_like(P)
            Pprev[0] = 0.0
            Pprev[1:] = P[:-1]
            V = b - Pprev
            acc[i, 1:] = P + np.minimum(np.minimum.accumulate(V), a0)
    best = float(acc[:, Tb - 1].min())
    return best / (float(norm) if norm else max(1.0, float(Tb)))


class RingMatcher:
    def __init__(self, ref_mfcc, sr, threshold=DTW_THRESHOLD, ref_edb=None,
                 energy_floor_db=ENERGY_FLOOR_DB, metric="cosine", smooth=None):
        self.ref = np.asarray(ref_mfcc, dtype=np.float64)
        self.sr = sr
        self.threshold = threshold
        self.ref_frames = self.ref.shape[1]
        self.energy_floor_db = energy_floor_db
        self.metric = metric
        self.smooth = SMOOTH_FRAMES if smooth is None else smooth
        if ref_edb is None:
            self.ref_w = np.ones(self.ref_frames, dtype=np.float64)
        else:
            self.ref_w = energy_weight(ref_edb, energy_floor_db)
        self.norm = max(1.0, float(self.ref_w.sum()))

    def score(self, window_audio):
        """返回 (dist, is_hit)；dist 越小越像。"""
        win_f, win_edb = extract_feat(window_audio, self.sr, get_fb(self.sr), smooth=self.smooth)
        win_f = win_f.astype(np.float64)
        if win_f.shape[1] < max(3, self.ref_frames // 4):
            return float("inf"), False
        base = frame_cost(self.ref, win_f, metric=self.metric)   # [Ta, Tb]
        w_win = energy_weight(win_edb, self.energy_floor_db)[:, None]
        cost = base * (0.5 * (w_win + self.ref_w[None, :]))
        dist = subsequence_dtw(cost, norm=self.norm)
        return dist, dist < self.threshold


# ==================== 命中动作 / 串口发送 ====================
def on_match(distance=None):
    """无串口模式下的默认动作：仅打印标识 1。"""
    print("1")


def find_arduino_port():
    """自动查找 USB 串口（CP210x / CH340 / USB Serial 等）。"""
    try:
        from serial.tools import list_ports
    except Exception:
        return None
    keys = ("CP210", "CH340", "USB-SERIAL", "USB SERIAL", "ARDUINO", "UART", "CH910")
    cands = []
    for p in list_ports.comports():
        desc = ((p.description or "") + " " + (p.hwid or "")).upper()
        if any(k in desc for k in keys):
            cands.append(p.device)
    if cands:
        return cands[0]
    ports = list(list_ports.comports())
    return ports[0].device if ports else None


class SerialSender:
    """向 Arduino 串口发送字符 '1'。dry=True 时只打印不发送（无板子也能跑）。"""

    def __init__(self, port=None, baud=SERIAL_BAUD, dry=False):
        self.dry = dry
        self.baud = baud
        self.ser = None
        self.lock = threading.Lock()
        if dry:
            print("[串口] dry-run：不实际发送，仅打印")
            self.port = port
            return
        import serial
        self.port = port or find_arduino_port()
        if not self.port:
            raise RuntimeError("未找到串口设备，请用 --port 指定（如 --port COM5）")
        self.ser = serial.Serial(self.port, baud, timeout=1)
        time.sleep(2.0)  # 等 Arduino 因打开串口而复位完成
        print("[串口] 已连接 %s @ %d" % (self.port, baud))

    def send(self, payload=b"1"):
        if self.dry or self.ser is None:
            print("[串口][dry] send %r" % payload)
            return True
        try:
            with self.lock:
                self.ser.write(payload)
            return True
        except Exception as e:
            print("[串口] 发送失败:", e)
            return False

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None


class CommandDispatcher:
    """发送时序：
       1) 命中铃声 -> 立即发 '1'，并安排在 random(1~5)s 后补发一次 '1'；
       2) 超过 IDLE_SEND_SEC 没有任何发送 -> 补发一次 '1'（静默心跳）。
    """

    def __init__(self, sender, idle_sec=IDLE_SEND_SEC,
                 second_min=SECOND_MIN, second_max=SECOND_MAX):
        self.sender = sender
        self.idle_sec = idle_sec
        self.second_min = second_min
        self.second_max = second_max
        self.last_sent = time.time()
        self.pending_second = None

    def _send(self, reason):
        if self.sender.send(b"1"):
            self.last_sent = time.time()
            print("[%s] 发送 '1' 到板子  (%s)" % (time.strftime("%H:%M:%S"), reason))

    def on_ring(self, now=None):
        """识别到铃声：立即发一次，并安排随机补发。"""
        now = time.time() if now is None else now
        self._send("识别到铃声")
        self.pending_second = now + random.uniform(self.second_min, self.second_max)

    def tick(self, now=None):
        now = time.time() if now is None else now
        if self.pending_second is not None and now >= self.pending_second:
            self.pending_second = None
            self._send("铃声后随机补发")
        elif now - self.last_sent >= self.idle_sec:
            self._send("静默 %ds 心跳" % int(self.idle_sec))


# ==================== 实时检测器 ====================
class RingDetector:
    def __init__(self, sr, matcher, window_sec=WINDOW_SEC, step=STEP_SEC,
                 cooldown=COOLDOWN_SEC, release=RELEASE_SEC, confirm=HIT_CONFIRM,
                 energy_gate=ENERGY_GATE, verbose=False, dispatcher=None):
        self.sr = sr
        self.matcher = matcher
        self.step = step
        self.cooldown = cooldown
        self.release = release
        self.confirm = max(1, int(confirm))
        self.energy_gate = energy_gate
        self.verbose = verbose
        self.dispatcher = dispatcher
        self.buf_len = int(sr * window_sec)
        self.buf = collections.deque(maxlen=self.buf_len)
        self.lock = threading.Lock()
        self.running = True
        self.last_trigger = 0.0
        self.last_hit = -1e9
        self.hit_streak = 0
        self.armed = True          # True=可以触发；触发后置 False，待铃声明消后重新武装

    def push(self, mono):
        with self.lock:
            self.buf.extend(mono.tolist())

    def callback(self, indata, frames, time_info, status):
        if status:
            print("audio status:", status, file=sys.stderr)
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata.ravel()
        self.push(mono)

    def process_window(self, win, now=None):
        if now is None:
            now = time.time()
        rms = float(np.sqrt(np.mean(win ** 2))) if len(win) else 0.0
        if rms < self.energy_gate:
            hit, dist = False, float("nan")          # 静音也计入“未命中”，用于复位
        else:
            dist, hit = self.matcher.score(win)
        if hit:
            self.last_hit = now
            self.hit_streak += 1
            if self.armed and self.hit_streak >= self.confirm:
                print("[%s] 识别到目标铃声  dist=%.3f rms=%.4f" %
                      (time.strftime("%H:%M:%S"), dist, rms))
                if self.dispatcher is not None:
                    self.dispatcher.on_ring(now)
                else:
                    on_match(dist)
                self.armed = False
        else:
            self.hit_streak = 0
            if (not self.armed) and (now - self.last_hit) > self.release:
                self.armed = True                    # 铃声明消，重新武装
            if self.verbose and dist == dist:        # dist 非 NaN
                print("[%s] dist=%.3f rms=%.4f" % (time.strftime("%H:%M:%S"), dist, rms))

    def run(self):
        while self.running:
            time.sleep(self.step)
            now = time.time()
            if self.dispatcher is not None:
                self.dispatcher.tick(now)
            with self.lock:
                if len(self.buf) < int(self.buf_len * 0.6):
                    continue
                win = np.fromiter(self.buf, dtype=np.float32, count=len(self.buf))
            self.process_window(win, now=now)

    def stop(self):
        self.running = False


# ==================== 输入源 ====================
def run_from_file(detector, path):
    """把音频文件当作输入流回放（用模拟时钟，便于离线调试）。"""
    y, sr = read_wav(path, target_sr=detector.sr)
    chunk = max(1, int(detector.sr * detector.step))
    print("回放文件: %s  时长 %.2fs" % (path, len(y) / float(detector.sr)))
    for s in range(0, len(y), chunk):
        detector.push(y[s:s + chunk])
        with detector.lock:
            win = np.fromiter(detector.buf, dtype=np.float32, count=len(detector.buf))
        detector.process_window(win, now=s / float(detector.sr))


# ==================== 设备与主流程 ====================
def list_devices():
    import sounddevice as sd
    print("默认设备(输入,输出):", sd.default.device)
    for i, d in enumerate(sd.query_devices()):
        print("%2d | %s | in:%d out:%d | sr:%d" % (
            i, d["name"], d["max_input_channels"],
            d["max_output_channels"], int(d["default_samplerate"])))


def find_loopback_device():
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        name = d["name"].lower()
        if d["max_input_channels"] > 0 and ("\u7acb\u4f53\u58f0\u6df7\u97f3" in d["name"] or "stereo mix" in name):
            return i
    return None


def resolve_sr(device):
    """优先 16k，不行就用设备默认采样率。"""
    import sounddevice as sd
    targets = [CAPTURE_SR]
    try:
        dev = device if device is not None else sd.default.device[0]
        info = sd.query_devices(dev, "input")
        targets += [int(info["default_samplerate"]), 48000, 44100]
    except Exception:
        targets += [44100, 48000]
    for cand in targets:
        try:
            sd.check_input_settings(device=device, channels=1, samplerate=cand, dtype="float32")
            return cand
        except Exception:
            continue
    return CAPTURE_SR


def build_reference(path, sr):
    y, _ = read_wav(path, target_sr=sr)
    y = trim_silence(y, sr)
    ref, ref_edb = extract_feat(y, sr, get_fb(sr))
    print("参考样本: %s  有效时长 %.2fs  帧数 %d" % (path, len(y) / float(sr), ref.shape[1]))
    return y, ref, ref_edb


def run_realtime(args, sr, matcher, dispatcher=None):
    import sounddevice as sd
    detector = RingDetector(sr, matcher, window_sec=args.window, step=args.step,
                            cooldown=args.cooldown, confirm=args.confirm,
                            energy_gate=args.energy_gate,
                            verbose=args.verbose, dispatcher=dispatcher)
    print("开始监听... 按 Ctrl+C 停止")
    try:
        with sd.InputStream(device=args.device, channels=1, samplerate=sr,
                            dtype="float32", blocksize=int(sr * 0.05),
                            callback=detector.callback):
            detector.run()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        detector.stop()


def selftest(args):
    sr = CAPTURE_SR
    y, ref, ref_edb = build_reference(args.reference, sr)
    matcher = RingMatcher(ref, sr, threshold=args.threshold, ref_edb=ref_edb)
    rng = np.random.default_rng(0)

    print("\n[正样本] 参考铃声叠加噪声 + 随机静音背景")
    d0, h0 = matcher.score(y)
    print("  参考自身: dist=%.4f  hit=%s" % (d0, h0))
    for i in range(3):
        pa = rng.normal(0, 0.0005, int(sr * rng.uniform(0.5, 2.0))).astype(np.float32)
        pb = rng.normal(0, 0.0005, int(sr * rng.uniform(0.5, 2.0))).astype(np.float32)
        noise = (0.02 * rng.normal(0, 1, len(y))).astype(np.float32)
        clip = np.concatenate([pa, y + noise, pb])
        d, hit = matcher.score(clip)
        print("  正样本%d: dist=%.3f  hit=%s" % (i + 1, d, hit))

    print("\n[负样本] 应保持不命中")
    t = np.arange(int(sr * 4)) / float(sr)

    def nrm(x, rms):
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean()
        return (x / (np.sqrt(np.mean(x ** 2)) + 1e-12) * rms).astype(np.float32)

    cl = np.zeros(len(t))
    cl[rng.integers(0, len(t), 40)] = rng.normal(0, 1, 40)
    negs = {
        "白噪声": (0.1 * rng.normal(0, 1, len(t))).astype(np.float32),
        "正弦440Hz": (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
        "扫频": (0.2 * np.sin(2 * np.pi * (200 + 400 * t) * t)).astype(np.float32),
        "叮1200Hz": nrm(np.sin(2 * np.pi * 1200 * t) * np.exp(-3 * t), 0.02),
        "叮1800Hz": nrm(np.sin(2 * np.pi * 1800 * t) * np.exp(-3 * t), 0.02),
        "叮2500Hz": nrm(np.sin(2 * np.pi * 2500 * t) * np.exp(-3 * t), 0.02),
        "键盘点击": nrm(cl, 0.03),
    }
    for name, clip in negs.items():
        d, hit = matcher.score(clip)
        print("  %s: dist=%.3f  hit=%s" % (name, d, hit))


def main():
    ap = argparse.ArgumentParser(description="\u94c3\u58f0\u8bc6\u522b\uff1a\u8bc6\u522b\u5230\u76ee\u6807\u94c3\u58f0\u8f93\u51fa 1")
    ap.add_argument("--reference", default=REFERENCE_FILE)
    ap.add_argument("--device", type=int, default=None, help="\u8f93\u5165\u8bbe\u5907\u7f16\u53f7\uff08\u89c1 --list\uff09")
    ap.add_argument("--samplerate", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=DTW_THRESHOLD)
    ap.add_argument("--energy-gate", type=float, default=ENERGY_GATE)
    ap.add_argument("--cooldown", type=float, default=COOLDOWN_SEC)
    ap.add_argument("--window", type=float, default=WINDOW_SEC)
    ap.add_argument("--step", type=float, default=STEP_SEC)
    ap.add_argument("--confirm", type=int, default=HIT_CONFIRM, help="\u547d\u4e2d\u786e\u8ba4\u6b21\u6570")
    ap.add_argument("--port", default=SERIAL_PORT, help="Arduino \u4e32\u53e3\uff08\u5982 COM5\uff09\uff0c\u7f3a\u7701\u81ea\u52a8\u68c0\u6d4b")
    ap.add_argument("--baud", type=int, default=SERIAL_BAUD, help="\u4e32\u53e3\u6ce2\u7279\u7387")
    ap.add_argument("--no-serial", action="store_true", help="\u4e0d\u53d1\u4e32\u53e3\uff0c\u4ec5\u6253\u5370\uff08\u65e0\u677f\u5b50\u65f6\u6d4b\u8bd5\uff09")
    ap.add_argument("--loopback", action="store_true", help="\u5185\u5f55\u7cfb\u7edf\u58f0\u97f3\uff08\u7acb\u4f53\u58f0\u6df7\u97f3\uff09")
    ap.add_argument("--input", default=None, help="\u4ece\u97f3\u9891\u6587\u4ef6\u8bfb\u5165\uff08\u8c03\u8bd5\uff09")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_devices()
        return
    if args.selftest:
        selftest(args)
        return

    if args.loopback and args.device is None:
        args.device = find_loopback_device()
        print("loopback \u8bbe\u5907\u7f16\u53f7:", args.device)

    sr = args.samplerate or resolve_sr(args.device)
    print("\u91c7\u96c6\u91c7\u6837\u7387: %d Hz" % sr)
    _, ref, ref_edb = build_reference(args.reference, sr)
    matcher = RingMatcher(ref, sr, threshold=args.threshold, ref_edb=ref_edb)
    print("\u9608\u503c dist<%.2f  \u80fd\u91cf\u95e8\u63a7 %.4f  \u7a97\u53e3 %.1fs" %
          (args.threshold, args.energy_gate, args.window))

    # 准备向 Arduino 发送 '1'
    sender = None
    try:
        sender = SerialSender(args.port, args.baud, dry=args.no_serial)
    except Exception as e:
        print("[\u4e32\u53e3] \u521d\u59cb\u5316\u5931\u8d25: %s" % e)
        print("[\u4e32\u53e3] \u964d\u7ea7\u4e3a dry-run\uff08\u4ec5\u6253\u5370\uff0c\u4e0d\u53d1\u9001\uff09\u3002\u8bf7\u5173\u95ed\u5360\u7528\u8be5\u4e32\u53e3\u7684\u7a0b\u5e8f\u540e\u91cd\u8bd5\u3002")
        sender = SerialSender(dry=True)
    dispatcher = CommandDispatcher(sender)
    print("\u53d1\u9001\u65f6\u5e8f: \u547d\u4e2d\u5373\u53d1 '1' + random(%.0f~%.0fs) \u8865\u53d1\uff1b\u9759\u9ed8 %.0fs \u8865\u53d1\u4e00\u6b21"
          % (SECOND_MIN, SECOND_MAX, IDLE_SEND_SEC))

    try:
        if args.input:
            detector = RingDetector(sr, matcher, window_sec=args.window, step=args.step,
                                    cooldown=args.cooldown, confirm=args.confirm,
                                    energy_gate=args.energy_gate,
                                    verbose=args.verbose, dispatcher=None)
            run_from_file(detector, args.input)
        else:
            run_realtime(args, sr, matcher, dispatcher=dispatcher)
    finally:
        sender.close()
        print("[\u4e32\u53e3] \u5df2\u5173\u95ed")


if __name__ == "__main__":
    main()