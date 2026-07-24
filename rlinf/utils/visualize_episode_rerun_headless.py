#!/usr/bin/env python3
"""无显示器服务器上的 dobot episode Rerun 可视化。

═══════════════════════ 快速开始（推荐：connect 模式） ════════════════════════

1. 终端 A — 启动 rerun 代理（一次性，保持运行，可复用给多个 episode）：
       .venv/bin/rerun --serve-web --bind 0.0.0.0 \
           --port 9876 --web-viewer-port 9090 --hide-welcome-screen

2. 终端 B — 推送 episode 数据（每次可视化一个 episode）：
       .venv/bin/python rlinf/utils/visualize_episode_rerun_headless.py \\
           <HDF5 或 parquet 文件>

3. 笔记本浏览器打开脚本打印的 URL（已自动填入本机 LAN IP）。

══════════════════════ 为什么需要两步？ ═══════════════════════

Rerun 的 web viewer 是一个浏览器 WASM 应用，它需要同时访问：
- HTTP (9090)：加载网页 UI
- gRPC (9876)：获取数据流

CLI 的 `--serve-web` 同时托管这两个服务并内部打通代理，SDK 只需用
`rr.connect_grpc("rerun+http://...")` 把数据推进去即可。

══════════════════════ 工作模式 ═══════════════════════

connect（默认）  SDK 连到已在另一终端运行的 `rerun --serve-web` 代理。
                 最灵活：server 常驻，可反复推送多个 episode 叠加对比。

serve           脚本自动 spawn `rerun --serve-web` 代理子进程，然后连接。
                 用完即走：数据推完后阻塞等待，Ctrl+C 退出时自动清理代理。
                 注意：依赖 venv 里有 rerun CLI（`uv pip install rerun-sdk`）。

save            离线模式：不连任何 viewer，直接落盘 .rrd 文件。
                 scp 到本地笔记本，`rerun ep.rrd` 打开查看。
                 最稳，零网络配置，但无流式体验。

══════════════════════ 示例 ═══════════════════════

  # HDF5（推荐，有 probe/wrench/segment_marks 全部信息）
  .venv/bin/python examples/dobot_clean/utils/visualize_episode_rerun_headless.py \\
      examples/dobot_clean/recorded_data/20260718/hdf5/episode_20260718_161023_plating_hanging.hdf5

  # 目录自动选最新 episode
  .venv/bin/python examples/dobot_clean/utils/visualize_episode_rerun_headless.py \\
      examples/dobot_clean/recorded_data/20260718/

  # LeRobot parquet（转换后的训练数据）
  .venv/bin/python examples/dobot_clean/utils/visualize_episode_rerun_headless.py \\
      datasets/dobot_cf5af_t265_joint/data/chunk-000/episode_000000.parquet

  # 截取帧范围
  .venv/bin/python examples/dobot_clean/utils/visualize_episode_rerun_headless.py \\
      <path> --start 100 --end 400

  # 离线 rrd（无需网络）
  .venv/bin/python examples/dobot_clean/utils/visualize_episode_rerun_headless.py \\
      <path> --mode save --out /tmp/ep.rrd

  # serve 模式（自动起代理，用完 Ctrl+C 退出）
  .venv/bin/python examples/dobot_clean/utils/visualize_episode_rerun_headless.py \\
      <path> --mode serve
"""
from __future__ import annotations

import sys
from pathlib import Path

# When this file is executed directly, Python prepends ``rlinf/utils`` to
# sys.path. That makes third-party ``import logging`` statements resolve to
# ``rlinf/utils/logging.py`` instead of the standard library and breaks
# pyarrow/cloudpickle during ``import rerun``. Keep the supported direct-file
# invocation while restoring normal package/module resolution.
if __package__ in (None, ""):
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _SCRIPT_DIR.parents[1]
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != _SCRIPT_DIR
    ]
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import io
import shutil
import socket
import subprocess
import time
import urllib.parse

import numpy as np

QPOS_LABELS = ["J1", "J2", "J3", "J4", "J5", "J6", "gripper"]
POSE_LABELS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
WRENCH_LABELS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]


# ----------------------------------------------------------------------------
# 输入解析：HDF5 或 parquet 都统一抽象成 Episode 结构
# ----------------------------------------------------------------------------
class Episode:
    """统一的 episode 数据载体（按需懒加载字段）。"""

    def __init__(self):
        self.n = 0
        self.qpos = None          # (N,7) 或 None
        self.action = None        # (N,7) 或 None
        self.pose = None          # (N,7/8) 或 None
        self.wrench = None        # (N,6) 或 None
        self.images = {}          # cam_name -> (N,H,W,3) uint8
        self.times = None         # (N,) 秒
        self.task = ""
        self.attrs = {}           # 文件级元数据
        self.segment_marks = None  # (K,) int  阶段分界帧
        self.source = ""


def resolve_input_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_dir():
        # 优先 hdf5，其次 h5，最后 parquet
        for ext in ("*.hdf5", "*.h5", "*.parquet"):
            files = sorted(p.rglob(ext), key=lambda x: x.stat().st_mtime, reverse=True)
            if files:
                print(f"[input] 目录输入，自动选最新 {ext}: {files[0]}")
                return files[0]
        print(f"[input] 错误：目录里没有 hdf5/h5/parquet: {p}", file=sys.stderr)
        sys.exit(1)
    return p


def load_hdf5(path: Path, start: int, end: int | None) -> Episode:
    import h5py
    ep = Episode()
    ep.source = f"hdf5:{path.name}"
    with h5py.File(path, "r") as f:
        if "observations/qpos" in f:
            n = f["observations/qpos"].shape[0]
        elif "action" in f:
            n = f["action"].shape[0]
        else:
            n = 0
        end = n if end is None else min(end, n)
        start = max(0, min(start, n))
        if end <= start:
            print(f"[input] 错误：帧范围为空 start={start} end={end} n={n}", file=sys.stderr)
            sys.exit(1)
        sl = slice(start, end)
        ep.n = end - start

        if "observations/qpos" in f:
            ep.qpos = np.asarray(f["observations/qpos"][sl], dtype=np.float32)
        if "action" in f:
            ep.action = np.asarray(f["action"][sl], dtype=np.float32)
        if "observations/pose" in f:
            ep.pose = np.asarray(f["observations/pose"][sl], dtype=np.float32)
        if "observations/wrench" in f:
            ep.wrench = np.asarray(f["observations/wrench"][sl], dtype=np.float32)

        if "observations/images" in f:
            for cam in sorted(f["observations/images"].keys()):
                arr = f[f"observations/images/{cam}"][sl]
                arr = np.asarray(arr)
                if arr.ndim == 4 and arr.shape[1] == 3:  # CHW -> HWC
                    arr = np.transpose(arr, (0, 2, 3, 1))
                ep.images[cam] = np.ascontiguousarray(arr)

        # 时间戳
        if "observations/timestamps/start" in f:
            ts = np.asarray(f["observations/timestamps/start"][sl], dtype=np.float64)
        elif "observations/timestamps/robot" in f:
            ts = np.asarray(f["observations/timestamps/robot"][sl], dtype=np.float64)
        else:
            ts = None
        if ts is not None and ts.size == ep.n:
            ep.times = ts - ts[0]
        else:
            ep.times = np.arange(ep.n, dtype=np.float64) / 31.0

        # task
        if "task" in f and f["task"].shape[0] > 0:
            v = f["task"][min(start, f["task"].shape[0] - 1)]
            ep.task = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)

        # attrs
        for k in f.attrs:
            v = f.attrs[k]
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            ep.attrs[k] = v

        # segment_marks：文件级 attr，帧索引相对全 episode
        sm = ep.attrs.get("segment_marks")
        if sm is not None:
            try:
                marks = np.asarray(sm).reshape(-1).astype(int)
                # 过滤掉区间外的，并平移到本切片坐标系
                in_range = marks[(marks >= start) & (marks < end)] - start
                ep.segment_marks = in_range if len(in_range) else None
            except Exception:
                pass
    return ep


def load_parquet(path: Path, start: int, end: int | None) -> Episode:
    import pyarrow.parquet as pq

    ep = Episode()
    ep.source = f"parquet:{path.name}"
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    end = n if end is None else min(end, n)
    start = max(0, min(start, n))
    if end <= start:
        print(f"[input] 错误：帧范围为空 start={start} end={end} n={n}", file=sys.stderr)
        sys.exit(1)
    # 分批读，避免一次加载大图像数据
    schema_names = set(pf.schema_arrow.names)

    def first_available(*candidates: str) -> str | None:
        return next((name for name in candidates if name in schema_names), None)

    state_field = first_available("observation.state", "state")
    action_field = first_available("action", "actions")
    wrench_field = first_available("observation.wrench", "wrench")
    image_field = first_available("observation.images.cam_left_wrist", "image")
    cols = [
        name
        for name in (
            state_field,
            action_field,
            wrench_field,
            "timestamp" if "timestamp" in schema_names else None,
            image_field,
            "frame_index" if "frame_index" in schema_names else None,
        )
        if name is not None
    ]
    # 实际上 read_row_group 只读一组；改用 read_table with range
    batches = pf.iter_batches(columns=cols, row_groups=list(range(pf.metadata.num_row_groups)))
    rows = []
    for b in batches:
        rows.append(b.to_pydict())
    # 合并
    merged = {}
    for k in cols:
        merged[k] = []
    for b in rows:
        for k in cols:
            merged[k].extend(b[k])

    sl = slice(start, end)
    ep.n = end - start

    if state_field is not None and merged[state_field]:
        state = np.asarray(merged[state_field][sl], dtype=np.float32)
        # Current Dobot pose datasets store [x,y,z,qw,qx,qy,qz,gripper]
        # directly in ``state``. Older joint datasets contain 7-D qpos.
        if state.ndim == 2 and state.shape[1] >= 8:
            ep.pose = state
        else:
            ep.qpos = state
    if action_field is not None and merged[action_field]:
        ep.action = np.asarray(merged[action_field][sl], dtype=np.float32)
    if wrench_field is not None and merged[wrench_field]:
        ep.wrench = np.asarray(merged[wrench_field][sl], dtype=np.float32)
    if "timestamp" in merged and merged["timestamp"]:
        ts = np.asarray(merged["timestamp"][sl], dtype=np.float64)
        ep.times = ts - ts[0] if ts.size else np.arange(ep.n) / 31.0
    else:
        ep.times = np.arange(ep.n, dtype=np.float64) / 31.0

    # parquet 里图像是 PNG bytes
    if image_field is not None and merged[image_field]:
        from PIL import Image

        imgs = merged[image_field][sl]
        arrs = []
        for rec in imgs:
            if isinstance(rec, dict):
                b = rec.get("bytes")
            else:
                b = rec
            if not b:
                arrs.append(np.zeros((224, 224, 3), dtype=np.uint8))
                continue
            im = Image.open(io.BytesIO(b)).convert("RGB")
            arrs.append(np.asarray(im, dtype=np.uint8))
        ep.images["cam_left_wrist"] = np.stack(arrs)
    return ep


def load_episode(path: Path, start: int, end: int | None) -> Episode:
    suf = path.suffix.lower()
    if suf in (".hdf5", ".h5"):
        return load_hdf5(path, start, end)
    if suf == ".parquet":
        return load_parquet(path, start, end)
    print(f"[input] 错误：不支持的文件类型 {path}", file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------------
# Rerun server 管理
# ----------------------------------------------------------------------------
def _detect_lan_ip() -> str:
    """返回本机第一个非回环的 IPv4 地址，失败时返回 127.0.0.1。"""
    try:
        # 方法 1：创建一个 UDP socket 连外部地址，读取本端 IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.5)
            # 不真正发包，只是让 OS 选择出口网卡
            s.connect(("192.168.3.224", 1))
            ip = s.getsockname()[0]
            if ip and ip != "127.0.0.1":
                return ip
    except OSError:
        pass
    # 方法 2：fallback 到 hostname -I
    try:
        out = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=2
        )
        first = out.stdout.strip().split()[0] if out.stdout.strip() else ""
        if first and first != "127.0.0.1":
            return first
    except Exception:
        pass
    return "127.0.0.1"


def _make_browser_url(host: str, web_port: int, grpc_port: int) -> str:
    """构造带 ?url= 参数的浏览器 URL（已 URL 编码）。"""
    grpc_url = f"rerun+http://{host}:{grpc_port}/proxy"
    encoded = urllib.parse.quote(grpc_url, safe="")
    return f"http://{host}:{web_port}?url={encoded}"


def _find_rerun_cli() -> str:
    """定位 rerun CLI：优先 PATH，其次 Python 解释器同目录（venv 场景）。"""
    found = shutil.which("rerun")
    if found:
        return found
    # venv 场景：rerun 通常装在 <venv>/bin/rerun，与 python 同目录
    candidates = [
        Path(sys.executable).parent / "rerun",
        Path(sys.prefix) / "bin" / "rerun",
        Path(sys.prefix) / "rerun",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return ""


def spawn_web_viewer(web_port: int, grpc_port: int) -> subprocess.Popen:
    """在 headless 服务器上 spawn `rerun --serve-web --bind 0.0.0.0` 代理进程。

    CLI 的 --serve-web 同时托管 HTTP（web viewer）和 gRPC（数据通道），
    并且内部自动打通 gRPC-web 代理——这是 remote access 能工作的关键。
    """
    rerun_bin = _find_rerun_cli()
    if not rerun_bin:
        print("[server] 错误：找不到 rerun CLI（PATH 里没有，venv bin 里也没有）",
              file=sys.stderr)
        print("[server] 请确认已安装：uv pip install rerun-sdk", file=sys.stderr)
        sys.exit(1)
    cmd = [
        rerun_bin,
        "--serve-web",
        "--bind=0.0.0.0",
        f"--port={grpc_port}",
        f"--web-viewer-port={web_port}",
        "--hide-welcome-screen",
    ]
    print(f"[server] spawn: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    # 等 server 起来
    for _ in range(40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect(("127.0.0.1", grpc_port))
                return proc
            except OSError:
                time.sleep(0.25)
    print("[server] 警告：等待 gRPC server 就绪超时，仍尝试连接", file=sys.stderr)
    return proc


# ----------------------------------------------------------------------------
# Rerun 记录
# ----------------------------------------------------------------------------
def _set_time(rr, seconds: float) -> None:
    # 0.23 起推荐 set_time(timestamp=...)；老版本回落到 set_time_seconds。
    if hasattr(rr, "set_time"):
        try:
            rr.set_time("time", duration=float(seconds))
            return
        except TypeError:
            rr.set_time("time", timestamp=float(seconds))
            return
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("time", float(seconds))


def _log_series(rr, entity_root: str, row: np.ndarray, labels: list[str]) -> None:
    row = np.asarray(row).reshape(-1)
    for dim, v in enumerate(row):
        if np.isfinite(v):
            label = labels[dim] if dim < len(labels) else str(dim)
            rr.log(f"{entity_root}/{label}", rr.Scalars(float(v)))


def log_episode(rr, ep: Episode, recording_name: str | None = None) -> None:
    """把整个 episode 流入 rerun（按 ep.times 时间轴）。"""
    if recording_name:
        # 新建独立 recording，避免与其它 episode 混在一条时间线
        try:
            rr.spawn(recording_name)  # 老版本兼容，但通常我们已外部起 server
        except Exception:
            pass

    # 静态元数据
    attr_lines = [f"source: {ep.source}", f"frames: {ep.n}",
                  f"duration: {ep.times[-1]:.2f}s", f"task: {ep.task or '(none)'}"]
    for k, v in ep.attrs.items():
        attr_lines.append(f"{k}: {v}")
    rr.log("metadata", rr.TextDocument("\n".join(attr_lines), media_type="text/plain"),
           static=True)

    # TCP 轨迹（一次画整条）
    if ep.pose is not None and ep.pose.shape[1] >= 3:
        xyz = ep.pose[:, :3].astype(np.float32)
        finite = np.isfinite(xyz).all(axis=1)
        if np.any(finite):
            rr.log("robot/tcp_trajectory", rr.LineStrips3D([xyz[finite]]), static=True)

    last_task = None
    for i in range(ep.n):
        t = float(ep.times[i])
        _set_time(rr, t)

        # 图像
        for cam, arr in ep.images.items():
            rr.log(f"images/{cam}", rr.Image(arr[i]))

        # 标量曲线
        if ep.qpos is not None:
            _log_series(rr, "state/qpos", ep.qpos[i], QPOS_LABELS)
        if ep.action is not None:
            _log_series(rr, "state/action", ep.action[i], QPOS_LABELS)
        if ep.pose is not None:
            _log_series(rr, "state/pose", ep.pose[i], POSE_LABELS)
            if ep.pose.shape[1] >= 3 and np.isfinite(ep.pose[i, :3]).all():
                rr.log("robot/tcp_position", rr.Points3D(ep.pose[i, :3].reshape(1, 3).astype(np.float32)))
        if ep.wrench is not None:
            _log_series(rr, "wrench", ep.wrench[i], WRENCH_LABELS)

        # 时间间隔
        if i > 0:
            rr.log("timing/dt", rr.Scalars(float(ep.times[i] - ep.times[i - 1])))

        # action vs next-state 偏差（动作/状态错位检查）
        if ep.action is not None and ep.qpos is not None and i < ep.n - 1:
            off = float(np.max(np.abs(ep.action[i] - ep.qpos[i + 1])))
            rr.log("quality/action_qpos_offset", rr.Scalars(off))

        # 阶段标记
        if ep.segment_marks is not None and i in ep.segment_marks:
            rr.log("info/segment", rr.TextLog(f"frame {i}: segment boundary"))

        # 任务文本（仅变化时）
        if ep.task and ep.task != last_task:
            rr.log("info/task", rr.TextLog(f"frame {i}: {ep.task}"))
            last_task = ep.task

        if (i + 1) % 200 == 0:
            print(f"  logged {i + 1}/{ep.n} frames")

    print(f"[done] 已记录 {ep.n} 帧，时长 {ep.times[-1]:.2f}s")


# ----------------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Headless Rerun 可视化 dobot episode（无显示器服务器友好）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", help="HDF5/H5/parquet 文件，或目录（自动选最新）")
    ap.add_argument("--start", type=int, default=0, help="起始帧")
    ap.add_argument("--end", type=int, default=None, help="结束帧（不含）")
    ap.add_argument(
        "--mode",
        choices=["connect", "serve", "save"],
        default="connect",
        help="connect=连到已在另一终端运行的 rerun --serve-web 代理（默认）；"
             "serve=脚本自动 spawn 代理+连接（用完 Ctrl+C）；"
             "save=落盘 .rrd 文件离线查看",
    )
    ap.add_argument("--web-port", type=int, default=9090, help="web viewer HTTP 端口")
    ap.add_argument("--grpc-port", type=int, default=9876, help="gRPC server 端口")
    ap.add_argument(
        "--host",
        default=None,
        help="服务器的网卡 IP 地址（笔记本浏览器访问用）。默认自动检测局域网 IP。",
    )
    ap.add_argument("--out", default=None, help="save 模式下的 .rrd 输出路径")
    ap.add_argument("--recording-name", default=None, help="rerun recording 名（多 episode 叠加时有用）")
    args = ap.parse_args()

    try:
        import rerun as rr
    except ImportError:
        print("错误：需要 rerun-sdk，运行 `uv pip install rerun-sdk`", file=sys.stderr)
        sys.exit(1)

    path = resolve_input_path(args.input)
    ep = load_episode(path, args.start, args.end)
    print(f"[loaded] {ep.source}: {ep.n} frames, "
          f"{sorted(ep.images.keys()) or 'no-cam'}, "
          f"qpos={ep.qpos is not None}, action={ep.action is not None}, "
          f"pose={ep.pose is not None}, wrench={ep.wrench is not None}, "
          f"segment_marks={ep.segment_marks is not None}")

    host = args.host or _detect_lan_ip()
    browser_url = _make_browser_url(host, args.web_port, args.grpc_port)

    server_proc = None
    try:
        if args.mode == "serve":
            # serve 模式：脚本 spawn CLI 代理，完成后阻塞等待 Ctrl+C
            server_proc = spawn_web_viewer(args.web_port, args.grpc_port)
            rr.init(args.recording_name or "dobot_episode", spawn=False)
            rr.connect_grpc(url=f"rerun+http://127.0.0.1:{args.grpc_port}/proxy")
        elif args.mode == "connect":
            # connect 模式：连到已手动启动的 CLI 代理
            rr.init(args.recording_name or "dobot_episode", spawn=False)
            rr.connect_grpc(url=f"rerun+http://127.0.0.1:{args.grpc_port}/proxy")
            print(f"[connect] 已连到 gRPC server port={args.grpc_port}")
        else:  # save
            out_file = args.out or "/tmp/dobot_episode.rrd"
            rr.init(args.recording_name or "dobot_episode", spawn=False)
            rr.save(out_file)
            print(f"[save] 将写入 {out_file}")

        log_episode(rr, ep, args.recording_name)

        # 打印浏览器访问链接
        if args.mode in ("serve", "connect"):
            print(f"\n{'═' * 60}")
            print(f"[viewer] ✅ 数据已推送完毕（{ep.n} 帧，{ep.times[-1]:.1f}s）")
            print("[viewer]")
            print("[viewer] 📍 笔记本浏览器打开：")
            print(f"[viewer]    {browser_url}")
            if host in ("127.0.0.1", "localhost"):
                print("[viewer]")
                print("[viewer] ⚠️ 自动检测本机 IP 失败，如果上面 URL 连不上，")
                print("[viewer]    请用 --host <服务器IP> 手动指定。")
            if args.mode == "serve":
                print("[viewer]")
                print("[viewer] 按 Ctrl+C 退出（会自动关闭代理 server）。")
            print(f"{'═' * 60}")

        if args.mode == "serve":
            import threading
            threading.Event().wait()
        elif args.mode == "save":
            out_file = args.out or "/tmp/dobot_episode.rrd"
            size_mb = Path(out_file).stat().st_size / (1024 * 1024)
            print(f"[save] ✅ 完成：{out_file}  ({size_mb:.1f} MB)")
            print(f"[save]    scp 到笔记本后运行：rerun {out_file}")
    except KeyboardInterrupt:
        print("\n[viewer] 收到中断。")
    finally:
        try:
            rr.disconnect()
        except Exception:
            pass
        if server_proc is not None:
            print("[viewer] 关闭代理 server ...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
