#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

// - 禁用迁移；每次请求的大小尽量大
// - 用户按开始时间排序，最小的优先
// - 最优选择：用户发送请求时会计算对每个NPU的实际推理结束时间，选择最早结束的。
// - 重点优化h(K)，当预测的用户请求完成时间大于结束时间时，放弃该用户(移至末尾，最后处理)

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int N;
    cin >> N;
    
    vector<int> g(N), m(N);
    vector<double> k(N);
    for (int i = 0; i < N; i++) {
        cin >> g[i] >> k[i] >> m[i];
    }
    
    int M;
    cin >> M;
    
    vector<int> s(M), e(M), cnt(M);
    for (int i = 0; i < M; i++) {
        cin >> s[i] >> e[i] >> cnt[i];
    }
    
    vector<vector<int>> latency(N, vector<int>(M));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cin >> latency[i][j];
        }
    }
    
    int a, b;
    cin >> a >> b;
    
    // 按开始时间排序用户
    vector<int> order(M);
    for (int i = 0; i < M; i++) order[i] = i;
    sort(order.begin(), order.end(), [&](int i, int j) {
        return s[i] < s[j];
    });
    
    // NPU的下次可用时间
    vector<vector<int>> npu_free_time(N);
    for (int i = 0; i < N; i++) {
        npu_free_time[i].resize(g[i], 0);
    }
    
    // 每个用户的调度结果
    vector<vector<int>> result(M);
    
    for (int uid : order) {
        int remaining = cnt[uid];
        int send_time = s[uid];
        
        if (remaining <= 0) continue;
        
        // 简单策略：找到第一个能容纳的NPU
        int chosen_server = -1, chosen_npu = -1;
        
        for (int si = 0; si < N && chosen_server == -1; si++) {
            int max_batch = (m[si] - b) / a;
            if (max_batch > 0) {
                for (int ni = 0; ni < g[si]; ni++) {
                    chosen_server = si;
                    chosen_npu = ni;
                    break;
                }
            }
        }
        
        if (chosen_server == -1) continue;
        
        // 使用选定的NPU进行调度，尽量用大batch
        while (remaining > 0) {
            int max_batch = (m[chosen_server] - b) / a;
            if (max_batch <= 0) break;
            
            int batch = min(remaining, max_batch);
            if (batch <= 0) break;
            
            int arrival = send_time + latency[chosen_server][uid];
            int start = max(arrival, npu_free_time[chosen_server][chosen_npu]);
            int inference = (int)ceil((double)batch / (k[chosen_server] * sqrt(batch)));
            int completion = start + inference;
            
            // 简单的超时检查
            if (completion > e[uid]) {
                break;
            }
            
            result[uid].push_back(send_time);
            result[uid].push_back(chosen_server + 1);
            result[uid].push_back(chosen_npu + 1);
            result[uid].push_back(batch);
            
            npu_free_time[chosen_server][chosen_npu] = completion;
            remaining -= batch;
            send_time = arrival + 1;
        }
    }
    
    // 输出结果
    for (int i = 0; i < M; i++) {
        int T = result[i].size() / 4;
        cout << T << "\n";
        for (int j = 0; j < result[i].size(); j++) {
            if (j > 0) cout << " ";
            cout << result[i][j];
        }
        cout << "\n";
    }
    
    return 0;
}
