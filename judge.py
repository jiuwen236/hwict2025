import os
import sys
import subprocess
import glob
import time
import math

# output file: score.log

def compile_cpp():
    cpp_files = ["main.cpp"]
    cmd = ["g++","-std=c++11"] + cpp_files + ["-o", "main"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        with open("score.log", "w", encoding="utf-8") as f:
            f.write("Compilation Error\n")
            f.write(res.stderr.decode())
        print("Compilation Error")
        sys.exit(1)

def compute_score(in_file, stdout):
    data = open(in_file, encoding="utf-8").read().split()
    idx = 0
    N = int(data[idx]); idx += 1
    g = []; k = []; m = []
    for _ in range(N):
        g.append(int(data[idx])); 
        k.append(float(data[idx+1])); 
        m.append(int(data[idx+2])); 
        idx += 3
    M = int(data[idx]); idx += 1
    s = []; e = []; cnt = []
    for _ in range(M):
        s.append(int(data[idx])); 
        e.append(int(data[idx+1])); 
        cnt.append(int(data[idx+2])); 
        idx += 3
    latency = [[0]*M for _ in range(N)]
    for i in range(N):
        for j in range(M):
            latency[i][j] = int(data[idx]); idx += 1
    a = int(data[idx]); b = int(data[idx+1]); idx += 2

    # 解析选手输出
    out_lines = [ln for ln in stdout.strip().splitlines() if ln.strip()]
    if len(out_lines) != 2*M:
        return 0.0

    requests = []
    V_list = []

    for u in range(M):
        # T_i
        parts = out_lines[2*u].split()
        if len(parts) != 1:
            return 0.0
        T = int(parts[0])
        if T < 1 or T > 300:
            return 0.0
        items = out_lines[2*u+1].split()
        if len(items) != 4*T:
            return 0.0

        prev_time = -1
        sumB = 0
        V = []
        for j in range(T):
            time_j   = int(items[4*j + 0])
            server_j = int(items[4*j + 1])
            npu_j    = int(items[4*j + 2])
            B_j      = int(items[4*j + 3])

            # 基本合法性检查
            if time_j <= prev_time or time_j < 0 or time_j > 1000000:
                return 0.0
            if server_j < 1 or server_j > N:
                return 0.0
            if npu_j < 1 or npu_j > g[server_j-1]:
                return 0.0
            if B_j < 1 or B_j > 1000 or a*B_j + b > m[server_j-1]:
                return 0.0
            if time_j < s[u]:
                return 0.0

            prev_time = time_j
            sumB += B_j
            V.append(npu_j-1)

            requests.append({
                'user':   u,
                'time':   time_j,
                'server': server_j-1,
                'npu':    npu_j-1,
                'B':      B_j
            })
        if sumB != cnt[u]:
            return 0.0
        V_list.append(V)

    # 检查通信间隔约束
    user_reqs = [[] for _ in range(M)]
    for r in requests:
        user_reqs[r['user']].append(r)
    for u in range(M):
        user_reqs[u].sort(key=lambda x: x['time'])
        for i in range(len(user_reqs[u])-1):
            cur = user_reqs[u][i]
            nxt = user_reqs[u][i+1]
            delay = latency[cur['server']][u]
            if nxt['time'] < cur['time'] + delay + 1:
                return 0.0

    # 准备仿真数据
    arrival_dict = {}
    for r in requests:
        u = r['user']; sid = r['server']
        r['arrival']  = r['time'] + latency[sid][u]
        r['proc_time']= math.ceil(r['B'] / (k[sid] * math.sqrt(r['B'])))
        r['mem']      = a * r['B'] + b
        r['started']  = False
        r['completed']= False
        arrival_dict.setdefault(r['arrival'], []).append(r)

    # Server 状态
    servers = []
    for i in range(N):
        servers.append({
            'mem_cap': m[i],
            'mem_used': 0,
            'active': [],              # list of (completion_time, request)
            'queues': [[] for _ in range(g[i])]
        })

    total_reqs = len(requests)
    if total_reqs == 0:
        return 0.0

    done = 0
    user_end = [0]*M

    # 设定一个足够大的仿真上界，防止死循环
    max_arr = max(arrival_dict.keys())
    max_proc = max(r['proc_time'] for r in requests)
    max_time = max_arr + 2*max_proc + 5  # +5 作为余量

    # 从最早到达时刻开始
    t = min(arrival_dict.keys())

    # 事件驱动主循环：直到所有请求完成或 t 超过上界
    while done < total_reqs and t <= max_time:
        # 1) 完成任务
        for srv in servers:
            rem = []
            for comp_t, rq in srv['active']:
                if comp_t == t:
                    srv['mem_used'] -= rq['mem']
                    rq['completed'] = True
                    done += 1
                    user_end[rq['user']] = max(user_end[rq['user']], comp_t)
                else:
                    rem.append((comp_t, rq))
            srv['active'] = rem

        if done >= total_reqs:
            break

        # 2) 新到达
        if t in arrival_dict:
            for rq in arrival_dict[t]:
                servers[rq['server']]['queues'][rq['npu']].append(rq)

        # 3) 对刚到达的队列做排序
        if t in arrival_dict:
            for srv in servers:
                for q in srv['queues']:
                    if q:
                        q.sort(key=lambda x: (x['arrival'], x['user']))

        # 4) 调度
        scheduled = False
        for srv in servers:
            for q in srv['queues']:
                new_q = []
                for rq in q:
                    if rq['started']:
                        continue
                    if srv['mem_used'] + rq['mem'] <= srv['mem_cap']:
                        rq['started'] = True
                        comp = t + rq['proc_time']
                        srv['active'].append((comp, rq))
                        srv['mem_used'] += rq['mem']
                        scheduled = True
                    else:
                        new_q.append(rq)
                srv['queues'][srv['queues'].index(q)] = new_q

        # 5) 跳到下一个事件时刻（到达或完成）
        next_times = []
        # 下一个到达
        nxt_arr = min((tt for tt in arrival_dict.keys() if tt > t), default=None)
        if nxt_arr is not None:
            next_times.append(nxt_arr)
        # 下一个完成
        for srv in servers:
            for comp_t, _ in srv['active']:
                if comp_t > t:
                    next_times.append(comp_t)
        if not next_times:
            # 理论上不应该出现，但防止死循环
            t += 1
        else:
            t = min(next_times)

    # 如果没完成就 0 分
    if done < total_reqs:
        return 0.0

    # 计算分数
    def h(x): return 2 ** (-x / 100.0)
    def p(x): return 2 ** (-x / 200.0)

    K = sum(1 for i in range(M) if user_end[i] > e[i])
    total = 0.0
    for i in range(M):
        xi = (user_end[i] - e[i]) / (e[i] - s[i])
        mi = sum(1 for j in range(len(V_list[i])-1) if V_list[i][j] != V_list[i][j+1])
        total += h(xi) * p(mi)
    return h(K) * total * 10000.0


def main():
    compile_cpp()
    in_files = sorted(glob.glob(os.path.join('data', '*.in')))
    num_cases = len(in_files)
    total_score = 0.0
    total_time = 0.0
    log_lines = []
    log_lines.append(f"测试用例数量: {num_cases}")
    for infile in in_files:
        name = os.path.basename(infile)
        start = time.perf_counter()
        try:
            res = subprocess.run(["./main"], stdin=open(infile), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            dur = time.perf_counter() - start
            total_time += dur
            out = res.stdout.decode()
            err = res.stderr.decode()
            if res.returncode != 0:
                score = 0.0
                log_lines.append(f"文件名: {name}")
                log_lines.append(f"分数: {score:.2f}, 时间: {dur:.2f}s")
                if err:
                    log_lines.append("错误:")
                    log_lines.append(err)
            else:
                score = compute_score(infile, out)
                total_score += score
                log_lines.append(f"文件名: {name}")
                log_lines.append(f"分数: {score:.2f}, 时间: {dur:.2f}s")
                if err:
                    log_lines.append("错误:")
                    log_lines.append(err)
        except subprocess.TimeoutExpired:
            dur = 30.0
            log_lines.append(f"文件名: {name}")
            log_lines.append(f"分数: 0.00, 时间: {dur:.2f}s")
            log_lines.append("错误: Timeout")
        log_lines.append("\n")
    avg_score = total_score / num_cases if num_cases else 0.0
    avg_time = total_time / num_cases if num_cases else 0.0
    log_lines.append(f"总分: {total_score:.2f}")
    log_lines.append(f"平均分: {avg_score:.2f}, 平均时间: {avg_time:.2f}s")
    with open("score.log", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"总分: {total_score:.2f}")
    print(f"平均分: {avg_score:.2f}, 平均时间: {avg_time:.2f}s")

if __name__ == '__main__':
    main() 