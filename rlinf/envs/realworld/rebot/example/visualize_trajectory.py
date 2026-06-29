#!/usr/bin/env python3
"""轨迹可视化工具。

输入一条或多条 .npy 轨迹文件，计算末端位姿并在三维空间可视化。

用法:
    # 单条轨迹
    uv run python example/visualize_trajectory.py records/teach_trajectory_20260611_154039.npy

    # 多条轨迹对比
    uv run python example/visualize_trajectory.py records/*.npy

    # 指定标签
    uv run python example/visualize_trajectory.py records/*.npy --labels "示教" "回放1"

    # 保存图片
    uv run python example/visualize_trajectory.py records/*.npy -o comparison.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from reBotArm_control_py.kinematics import load_robot_model, compute_fk, pad_q_for_model


def set_axes_equal(ax: Axes3D) -> None:
    """强制 3D 坐标轴等比例，确保轨迹形状不失真。"""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]

    max_range = max(x_range, y_range, z_range) / 2.0

    x_mid = (xlim[0] + xlim[1]) / 2.0
    y_mid = (ylim[0] + ylim[1]) / 2.0
    z_mid = (zlim[0] + zlim[1]) / 2.0

    ax.set_xlim3d([x_mid - max_range, x_mid + max_range])
    ax.set_ylim3d([y_mid - max_range, y_mid + max_range])
    ax.set_zlim3d([z_mid - max_range, z_mid + max_range])


def load_trajectory(path: str) -> np.ndarray:
    """加载轨迹文件。"""
    traj = np.load(path)
    if traj.ndim != 2:
        raise ValueError(f"轨迹应为二维数组，实际为 {traj.ndim}D: {path}")
    if traj.shape[1] != 6:
        raise ValueError(f"轨迹 shape[1] 应为 6，实际为 {traj.shape[1]}: {path}")
    return traj


def compute_end_effector_positions(trajectory: np.ndarray) -> np.ndarray:
    """将关节角序列转换为末端位置序列。"""
    model = load_robot_model()
    n_ctrl = trajectory.shape[1] if trajectory.ndim == 2 else model.nq
    positions = []
    for q in trajectory:
        # RS 模型含夹爪自由度，须将记录的臂关节角补齐到模型维度
        q_full = pad_q_for_model(model, np.asarray(q, dtype=np.float64), n_ctrl)
        pos, _, _ = compute_fk(model, q_full)
        positions.append(pos)
    return np.array(positions)


def main():
    parser = argparse.ArgumentParser(description="轨迹末端位姿三维可视化")
    parser.add_argument("paths", nargs="+", type=str, help="轨迹 .npy 文件路径")
    parser.add_argument(
        "--labels", nargs="*", type=str, help="轨迹标签（与文件数量一致）"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="保存图片路径（不指定则弹窗显示）"
    )
    parser.add_argument(
        "--no-points", action="store_true", help="不显示采样点标记"
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths]
    labels = args.labels or [p.stem for p in paths]

    if len(labels) != len(paths):
        print(f"错误: 标签数量 ({len(labels)}) 与轨迹数量 ({len(paths)}) 不一致")
        sys.exit(1)

    # 加载并计算所有轨迹的末端位置
    all_positions = []
    for path in paths:
        if not path.exists():
            print(f"错误: 文件不存在: {path}")
            sys.exit(1)
        traj = load_trajectory(str(path))
        positions = compute_end_effector_positions(traj)
        all_positions.append(positions)
        print(f"加载: {path.name} → {len(traj)} 点")

    # 创建 3D 图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10(np.linspace(0, 1, len(paths)))

    for positions, label, color in zip(all_positions, labels, colors):
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        # 绘制轨迹连线
        ax.plot(x, y, z, color=color, linewidth=2, label=label)

        # 标记起点
        ax.scatter(x[0], y[0], z[0], color=color, s=100, marker="o")
        ax.text(x[0], y[0], z[0], "起点", color=color, fontsize=8)

        # 标记终点
        ax.scatter(x[-1], y[-1], z[-1], color=color, s=100, marker="^")
        ax.text(x[-1], y[-1], z[-1], "终点", color=color, fontsize=8)

        # 采样点（可选）
        if not args.no_points:
            ax.scatter(x, y, z, color=color, s=10, alpha=0.3)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("末端执行器轨迹")
    ax.legend()

    set_axes_equal(ax)

    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"\n图片已保存: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
