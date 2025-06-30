#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据生成器
运行：python3 gen_data.py
"""

import random
import json
import time
from pathlib import Path

# ===================== 可自行修改的参数 =====================
OUTPUT_PREFIX = "n3_g1_k1_a20_b200_m1000_cnt1000.3000"          # 最终会生成 sample.in / sample.json

RAND_SEED     = int(time.time())  # 随机种子；想复现实验可手动改成固定数
random.seed(RAND_SEED)

# ---------- 单一数值：直接给定 ----------
N = 2           # 服务器种类数   (1 ≤ N ≤ 10)
M = 500         # 用户数量       (1 ≤ M ≤ 500)
a = 20          # 显存与 batchsize 的关系 (10 ≤ a ≤ 20)
b = 200         # 同上 (100 ≤ b ≤ 200)

# ---------- 多个数值：给定范围，均匀随机 ----------
# 服务器参数
G_MIN, G_MAX = 1, 1              # g_i   NPU 个数 (1, 10)
K_MIN, K_MAX = 1, 1              # k_i   推理速度参数 (1, 5)
MEMP_MIN, MEMP_MAX = 1200, 1200  # m_i   NPU 显存大小 (1000, 2000)

# 用户请求
CNT_MIN, CNT_MAX   = 1000, 3000     # cnt_i 样本数 （1, 6000）
TIME_MIN, TIME_MAX = 0, 60000    # s_i / e_i 的整体区间 (0, 60000)
START_TIME_MIN, START_TIME_MAX = 0, 60000 # 请求开始时间 s_i (0， 60000)
CNT_RATE_MIN, CNT_RATE_MAX = 5, 60000  # cnt_i 倍率 (5, 60000) 自定义的

# 通信时延
LAT_MIN, LAT_MAX = 10, 20        # latency_{i,j}  (10, 20)

# 若因 cnt_i 太大导致无法满足 5 * cnt_i ≤ e_i - s_i，可放宽搜索次数
MAX_RETRY_PER_USER = 2000
# ==========================================================


def gen_servers():
    """生成 N 行服务器参数 (g_i, k_i, m_i)"""
    servers = []
    for _ in range(N):
        g = random.randint(G_MIN, G_MAX)
        k = random.randint(K_MIN, K_MAX)
        memp = random.randint(MEMP_MIN, MEMP_MAX)
        servers.append((g, k, memp))
    return servers


def gen_users():
    """生成 M 个用户的 (s_i, e_i, cnt_i)"""
    users = []
    for _ in range(M):
        cnt = random.randint(CNT_MIN, CNT_MAX)   # 1. 先确定 cnt_i

        # 2. 不断尝试随机 s_i、e_i 直到满足 5*cnt ≤ e-s
        for _retry in range(MAX_RETRY_PER_USER):
            s = random.randint(START_TIME_MIN, START_TIME_MAX)
            if s + 5 * cnt > TIME_MAX:
                continue
            dur = int(cnt * random.uniform(CNT_RATE_MIN, CNT_RATE_MAX))
            e = s + dur
            if e > TIME_MAX:
                e = random.randint(s + 5 * cnt, TIME_MAX)
            users.append((s, e, cnt))
            break
        else:
            # 理论上很难触发；若触发说明 TIME_MAX 太小或 cnt 过大
            raise RuntimeError(f"无法为 cnt={cnt} 找到合规的 (s,e)")
    return users


def gen_latency():
    """生成 N 行 × M 列的 latency_{i,j}"""
    latency = []
    for _ in range(N):
        row = [random.randint(LAT_MIN, LAT_MAX) for _ in range(M)]
        latency.append(row)
    return latency


def write_input_file(path: Path, servers, users, latency):
    """写入 *.in 文件"""
    with path.open("w") as f:
        # 第一行 N
        f.write(f"{N}\n")

        # 接着 N 行服务器
        for g, k, memp in servers:
            f.write(f"{g} {k} {memp}\n")

        # 一行 M
        f.write(f"{M}\n")

        # M 行用户 (s_i, e_i, cnt_i)
        for s, e, cnt in users:
            f.write(f"{s} {e} {cnt}\n")

        # N 行 latency
        for row in latency:
            f.write(" ".join(map(str, row)) + "\n")

        # 最后一行 a b
        f.write(f"{a} {b}\n")


def write_json_file(path: Path, servers, users, latency):
    """写入 *.json 配置文件（含随机种子、范围等信息）"""
    cfg = {
        "version": "0612",
        "seed": RAND_SEED,
        "N": N,
        "M": M,
        "a": a,
        "b": b,
        "ranges": {
            "g": [G_MIN, G_MAX],
            "k": [K_MIN, K_MAX],
            "m": [MEMP_MIN, MEMP_MAX],
            "cnt": [CNT_MIN, CNT_MAX],
            "time": [TIME_MIN, TIME_MAX],
            "start_time": [START_TIME_MIN, START_TIME_MAX],
            "latency": [LAT_MIN, LAT_MAX]
        },
        # 也可以直接把生成结果写进去，便于调试或评测
        "servers": servers,
        "first_10_users": users[:10],  # 只放前十条，避免过大
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)


def main():
    servers  = gen_servers()
    users    = gen_users()
    latency  = gen_latency()

    in_path  = Path(f"{OUTPUT_PREFIX}.in")
    json_path = Path(f"{OUTPUT_PREFIX}.json")

    write_input_file(in_path, servers, users, latency)
    write_json_file(json_path, servers, users, latency)

    print(f"生成完毕：{in_path}  {json_path}")


if __name__ == "__main__":
    main()
