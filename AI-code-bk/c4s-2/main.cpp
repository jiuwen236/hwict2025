#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <climits>

// 禁用迁移；每次请求的大小尽量装满NPU显存
// 用户按(e_i-s_i)*cnt_i排序，最小的优先；在有更高优先级用户使用NPU时，低优先级的用户不能抢占
// 最优选择：用户发送请求时会计算对每个NPU的实际推理结束时间，选择最早结束的。

using namespace std;

struct Server {
    int gpu_count;
    int speed_factor;
    int memory;
};

struct User {
    int id;
    int start_time;
    int end_time;
    int sample_count;
    double priority;
};

struct NPUState {
    int server_id;
    int npu_id;
    long long next_available_time;
};

struct Request {
    int time;
    int server;
    int npu;
    int batch_size;
};

int N, M, a, b;
vector<Server> servers;
vector<User> users;
vector<vector<int>> latency;
vector<NPUState> npus;

int max_batch_size(int memory) {
    return max(1, (memory - b) / a);
}

long long compute_inference_time(int batch_size, int speed_factor) {
    double f = speed_factor * sqrt(batch_size);
    return (long long)ceil(batch_size / f);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    cin >> N;
    servers.resize(N);
    
    for (int i = 0; i < N; i++) {
        cin >> servers[i].gpu_count >> servers[i].speed_factor >> servers[i].memory;
    }
    
    cin >> M;
    users.resize(M);
    
    for (int i = 0; i < M; i++) {
        users[i].id = i;
        cin >> users[i].start_time >> users[i].end_time >> users[i].sample_count;
        // 按照算法思想：(e_i-s_i)*cnt_i排序，最小的优先
        users[i].priority = (double)(users[i].end_time - users[i].start_time) * users[i].sample_count;
    }
    
    latency.resize(N, vector<int>(M));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cin >> latency[i][j];
        }
    }
    
    cin >> a >> b;
    
    // 初始化NPU状态
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < servers[i].gpu_count; j++) {
            NPUState npu;
            npu.server_id = i;
            npu.npu_id = j;
            npu.next_available_time = 0;
            npus.push_back(npu);
        }
    }
    
    // 按优先级排序用户 - 最小的优先
    sort(users.begin(), users.end(), [](const User& a, const User& b) {
        return a.priority < b.priority;
    });
    
    vector<vector<Request>> user_requests(M);
    
    // 为每个用户分配任务
    for (const User& user : users) {
        int remaining_samples = user.sample_count;
        long long current_time = user.start_time;
        int assigned_npu = -1; // 该用户使用的NPU（禁用迁移）
        
        while (remaining_samples > 0) {
            int best_npu = -1;
            long long best_finish_time = LLONG_MAX;
            
            // 如果已经分配了NPU，继续使用同一个
            if (assigned_npu != -1) {
                best_npu = assigned_npu;
            } else {
                // 找到最早完成的NPU
                for (int npu_idx = 0; npu_idx < (int)npus.size(); npu_idx++) {
                    int server_id = npus[npu_idx].server_id;
                    long long arrival_time = current_time + latency[server_id][user.id];
                    long long start_time = max(arrival_time, npus[npu_idx].next_available_time);
                    
                    if (start_time < best_finish_time) {
                        best_finish_time = start_time;
                        best_npu = npu_idx;
                    }
                }
                assigned_npu = best_npu; // 记录分配的NPU
            }
            
            if (best_npu == -1) break;
            
            int server_id = npus[best_npu].server_id;
            int max_batch = min(remaining_samples, max_batch_size(servers[server_id].memory));
            if (max_batch <= 0) max_batch = 1;
            
            // 计算推理时间
            long long arrival_time = current_time + latency[server_id][user.id];
            long long start_time = max(arrival_time, npus[best_npu].next_available_time);
            long long inference_time = compute_inference_time(max_batch, servers[server_id].speed_factor);
            long long finish_time = start_time + inference_time;
            
            // 记录请求
            Request req;
            req.time = current_time;
            req.server = server_id + 1;
            req.npu = npus[best_npu].npu_id + 1;
            req.batch_size = max_batch;
            user_requests[user.id].push_back(req);
            
            // 更新NPU状态
            npus[best_npu].next_available_time = finish_time;
            
            remaining_samples -= max_batch;
            current_time += latency[server_id][user.id] + 1;
        }
    }
    
    // 输出结果（按原始用户ID顺序）
    for (int i = 0; i < M; i++) {
        cout << user_requests[i].size() << "\n";
        
        if (user_requests[i].size() > 0) {
            for (size_t j = 0; j < user_requests[i].size(); j++) {
                const Request& req = user_requests[i][j];
                cout << req.time << " " << req.server << " " << req.npu << " " << req.batch_size;
                if (j < user_requests[i].size() - 1) cout << " ";
            }
        }
        cout << "\n";
    }
    
    return 0;
}
