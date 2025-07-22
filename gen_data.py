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
OUTPUT_PREFIX = "server1_n3_a19_b190_m1000_cnt100_s0_rate5_lat20"          # 指定文件名，最终会生成 OUTPUT_PREFIX.in / OUTPUT_PREFIX.json
FUSAI = 1         # 0：生成初赛数据  1：生成复赛数据  
# 0：正常生成  1：将初赛数据转变为复赛数据(使用与原文件一致的a、b)  2: 使用本脚本a、b范围参数
CONVERT_DATA = 2  
CONVERT_FILE_PATH = "tmp_bk/server1_n3_a20_b200_m1000_cnt100_s0_rate5_lat20.in"  # 要转换的文件路径，不会删除原文件
CONVERT_CONFIG_PATH = "data/data_config/server1_n3_a20_b200_m1000_cnt100_s0_rate5_lat20.json"  # 要转换的配置文件路径

RAND_SEED     = int(time.time())  # 随机种子；想复现实验可手动改成固定数
random.seed(RAND_SEED)

# ---------- 单一数值：直接给定 ----------
N = 3           # 服务器种类数   (1 ≤ N ≤ 10)
M = 500         # 用户数量       (1 ≤ M ≤ 500)
a = 20          # 显存与 batchsize 的关系 (10 ≤ a ≤ 20) - 初赛用
b = 200         # 同上 (100 ≤ b ≤ 200) - 初赛用

# ---------- 多个数值：给定范围，均匀随机 ----------
# 服务器参数
G_MIN, G_MAX = 1, 1              # g_i   NPU 个数 (1, 10)
K_MIN, K_MAX = 1, 1              # k_i   推理速度参数 (1, 5)
MEMP_MIN, MEMP_MAX = 1200, 1800  # m_i   NPU 显存大小 (1000, 2000)

# 用户请求
CNT_MIN, CNT_MAX   = 6000, 6000     # cnt_i 样本数 （1, 6000）
TIME_MIN, TIME_MAX = 0, 60000    # s_i / e_i 的整体区间 (0, 60000)
START_TIME_MIN, START_TIME_MAX = 0, 60000 # 请求开始时间 s_i (0， 60000)
CNT_RATE_MIN, CNT_RATE_MAX = 5, 60000  # cnt_i 倍率 (5, 60000) 自定义的

# 通信时延
LAT_MIN, LAT_MAX = 10, 20        # latency_{i,j}  (10, 20)

# 模型参数 - 复赛用
A_MIN, A_MAX = 18, 20          # a_i 显存与 batchsize 的关系 (10, 20)
B_MIN, B_MAX = 180, 200        # b_i 同上 (100, 200)

# 手动指定服务器参数
MANUAL_SERVER = 0  
manual_servers = [[(1,5),(2,2),(3,1)]]

# 若因 cnt_i 太大导致无法满足 5 * cnt_i ≤ e_i - s_i，可放宽搜索次数
MAX_RETRY_PER_USER = 2000
# ==========================================================


def gen_servers():
    """生成 N 行服务器参数 (g_i, k_i, m_i)"""
    servers = []
    if MANUAL_SERVER:
        global N
        N = len(manual_servers[MANUAL_SERVER - 1])
    for i in range(N):
        g = random.randint(G_MIN, G_MAX)
        k = random.randint(K_MIN, K_MAX)
        if MANUAL_SERVER:
            g, k = manual_servers[MANUAL_SERVER - 1][i]
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


def gen_user_models():
    """生成M个用户的模型参数 (a_i, b_i) - 复赛用"""
    user_models = []
    for _ in range(M):
        ai = random.randint(A_MIN, A_MAX)
        bi = random.randint(B_MIN, B_MAX)
        user_models.append((ai, bi))
    return user_models


def read_chusai_data(file_path):
    """读取初赛数据文件"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    line_idx = 0
    
    # 读取 N
    N = int(lines[line_idx].strip())
    line_idx += 1
    
    # 读取 N 行服务器数据
    servers = []
    for _ in range(N):
        g, k, m = map(int, lines[line_idx].strip().split())
        servers.append((g, k, m))
        line_idx += 1
    
    # 读取 M
    M = int(lines[line_idx].strip())
    line_idx += 1
    
    # 读取 M 行用户数据
    users = []
    for _ in range(M):
        s, e, cnt = map(int, lines[line_idx].strip().split())
        users.append((s, e, cnt))
        line_idx += 1
    
    # 读取 N 行通信时延
    latency = []
    for _ in range(N):
        row = list(map(int, lines[line_idx].strip().split()))
        latency.append(row)
        line_idx += 1
    
    # 读取 a, b
    a, b = map(int, lines[line_idx].strip().split())
    
    return N, servers, M, users, latency, a, b


def read_config_file(file_path):
    """读取配置文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"警告：配置文件 {file_path} 不存在，将创建新配置")
        return None
    except json.JSONDecodeError:
        print(f"警告：配置文件 {file_path} 格式错误，将创建新配置")
        return None


def write_input_file(path: Path, servers, users, latency, user_models=None):
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

        # 最后的模型参数
        if FUSAI == 1:  # 复赛格式：M 行 a_i, b_i
            if user_models is None:
                raise RuntimeError("复赛数据需要用户模型参数")
            for ai, bi in user_models:
                f.write(f"{ai} {bi}\n")
        else:  # 初赛格式：一行 a, b
            f.write(f"{a} {b}\n")


def write_json_file(path: Path, servers, users, latency, user_models=None, original_config=None):
    """写入 *.json 配置文件（含随机种子、范围等信息）"""
    # 如果有原配置，先保留其中的信息
    if original_config:
        cfg = original_config.copy()
        # 只更新与a、b模型参数相关的字段
        cfg.update({
            "fusai": FUSAI,  # 标记为复赛数据
            "converted_from_chusai": True,
            "conversion_timestamp": time.time(),
            "conversion_mode": CONVERT_DATA,
            "A_MIN": A_MIN,
            "A_MAX": A_MAX,
            "B_MIN": B_MIN,
            "B_MAX": B_MAX
        })
        # 在转换模式下，使用配置中的a,b设置范围参数
        if CONVERT_DATA in [1] and "a" in cfg and "b" in cfg:
            cfg.update({
                "A_MIN": cfg["a"],
                "A_MAX": cfg["a"],
                "B_MIN": cfg["b"],
                "B_MAX": cfg["b"]
            })
        # 更新模型参数相关信息
        if FUSAI == 1 and user_models:
            cfg["first_10_user_models"] = user_models[:10]
        print(f"保持原配置文件结构，仅更新模型参数相关字段")
    else:
        # 正常生成模式，创建完整配置
        cfg = {
            "version": "0721",
            "seed": RAND_SEED,
            "fusai": FUSAI,
            "convert_data": CONVERT_DATA,
            "manual_server": MANUAL_SERVER,
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
                "latency": [LAT_MIN, LAT_MAX],
                "a_range": [A_MIN, A_MAX],
                "b_range": [B_MIN, B_MAX]
            },
            "servers": servers,
            "first_10_users": users[:10],  # 只放前十条，避免过大
        }
        
        if FUSAI == 1 and user_models:
            cfg["first_10_user_models"] = user_models[:10]
 
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)


def convert_chusai_to_fusai():
    """将初赛数据转换为复赛数据"""
    print(f"开始转换初赛数据: {CONVERT_FILE_PATH}")
    
    # 读取初赛数据
    N_orig, servers, M_orig, users, latency, a_orig, b_orig = read_chusai_data(CONVERT_FILE_PATH)
    
    # 读取原配置文件
    original_config = read_config_file(CONVERT_CONFIG_PATH)
    
    # 确定a、b参数来源
    if original_config and "a" in original_config and "b" in original_config:
        # 优先使用原配置文件中的a、b
        config_a, config_b = original_config["a"], original_config["b"]
        print(f"使用原配置文件中的参数 a={config_a}, b={config_b}")
    else:
        # 使用从初赛数据文件读取的a、b
        config_a, config_b = a_orig, b_orig
        print(f"使用初赛数据文件中的参数 a={config_a}, b={config_b}")
    
    # 更新全局变量
    global N, M
    N, M = N_orig, M_orig
    
    # 生成用户模型参数
    if CONVERT_DATA == 1:
        # 使用原文件的 a, b 为所有用户生成相同参数
        user_models = [(config_a, config_b) for _ in range(M)]
        print(f"为所有用户生成相同模型参数 a={config_a}, b={config_b}")
    elif CONVERT_DATA == 2:
        # 使用脚本中的范围参数为每个用户随机生成
        user_models = gen_user_models()
        print(f"使用脚本范围参数 a∈[{A_MIN},{A_MAX}], b∈[{B_MIN},{B_MAX}] 为每个用户随机生成模型参数")
    
    return servers, users, latency, user_models, original_config


def main():
    user_models = None
    original_config = None

    if CONVERT_DATA > 0:
        # 转换模式，不修改原文件，始终生成新的 OUTPUT_PREFIX 文件
        servers, users, latency, user_models, original_config = convert_chusai_to_fusai()
    else:
        # 正常生成模式
        servers = gen_servers()
        users = gen_users()
        latency = gen_latency()

        if FUSAI == 1:
            user_models = gen_user_models()

    # 始终使用 OUTPUT_PREFIX 生成输出文件
    in_path = Path(f"{OUTPUT_PREFIX}.in")
    json_path = Path(f"{OUTPUT_PREFIX}.json")

    write_input_file(in_path, servers, users, latency, user_models)
    write_json_file(json_path, servers, users, latency, user_models, original_config)

    if FUSAI == 1:
        print(f"生成复赛数据完毕：{in_path}  {json_path}")
    else:
        print(f"生成初赛数据完毕：{in_path}  {json_path}")

    if CONVERT_DATA > 0:
        print(f"已生成转换后的数据文件：{in_path}")
        print(f"已生成转换后的配置文件：{json_path}")


if __name__ == "__main__":
    main()
