#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <climits>

// 5034326 45572021 2025-05-30 22:31:36

using namespace std;

struct Server {
    int g, k, m;  // NPU数量，推理速度系数，显存大小
};

struct User {
    int s, e, cnt;  // 开始时间，结束时间，样本数量
};

int main() {
    int N;
    cin >> N;
    
    vector<Server> servers(N);
    for (int i = 0; i < N; i++) {
        cin >> servers[i].g >> servers[i].k >> servers[i].m;
    }
    
    int M;
    cin >> M;
    
    vector<User> users(M);
    for (int i = 0; i < M; i++) {
        cin >> users[i].s >> users[i].e >> users[i].cnt;
    }
    
    vector<vector<int>> latency(N, vector<int>(M));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cin >> latency[i][j];
        }
    }
    
    int a, b;
    cin >> a >> b;
    
    // 简单的调度策略：为每个用户选择最合适的服务器
    for (int i = 0; i < M; i++) {
        int best_server = 0;
        int min_latency = latency[0][i];
        
        // 选择延迟最小的服务器
        for (int j = 1; j < N; j++) {
            if (latency[j][i] < min_latency) {
                min_latency = latency[j][i];
                best_server = j;
            }
        }
        
        // 计算最大可用的batch size（不超过显存限制）
        int max_batch = (servers[best_server].m - b) / a;
        if (max_batch <= 0) max_batch = 1;
        if (max_batch > 1000) max_batch = 1000;
        
        // 计算需要多少次请求
        int remaining = users[i].cnt;
        vector<tuple<int, int, int, int>> requests; // time, server, npu, batch
        
        int current_time = users[i].s;
        int npu = 0; // 使用第一个NPU
        
        while (remaining > 0) {
            int batch_size = min(remaining, max_batch);
            
            // 确保batch size满足显存约束
            while (a * batch_size + b > servers[best_server].m && batch_size > 1) {
                batch_size--;
            }
            
            if (batch_size <= 0) batch_size = 1;
            
            requests.push_back(make_tuple(current_time, best_server + 1, npu + 1, batch_size));
            remaining -= batch_size;
            
            // 计算下次发送时间（考虑延迟）
            if (remaining > 0) {
                current_time += latency[best_server][i] + 1;
            }
        }
        
        // 输出结果
        cout << requests.size() << endl;
        for (auto& req : requests) {
            cout << get<0>(req) << " " << get<1>(req) << " " << get<2>(req) << " " << get<3>(req);
            if (&req != &requests.back()) cout << " ";
        }
        cout << endl;
    }
    
    return 0;
}
