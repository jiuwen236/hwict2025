#include <bits/stdc++.h>
using namespace std;

// 30196751.30  74575211  2025-06-01 21:51:10
// - 禁用迁移；用户按e_i-s_i排序，持续时间最短的优先；每次请求的大小尽量装满NPU显存
// - 最优选择：用户发送请求时会计算每个NPU的推理结束时间，选择最短的。

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N;
    if (!(cin >> N)) return 0;
    vector<int> g(N), k(N), m(N);
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

    vector<vector<long long>> avail(N);
    for (int i = 0; i < N; i++) {
        avail[i].assign(g[i], 0LL);
    }

    vector<int> order(M);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int u, int v) {
        int du = e[u] - s[u], dv = e[v] - s[v];
        return du != dv ? du < dv : u < v;
    });

    vector<int> T_res(M), srv_res(M), npu_res(M);
    vector<vector<int>> B_res(M);
    vector<vector<long long>> time_res(M);

    for (int idx : order) {
        int cnt_i = cnt[idx];
        int s_i = s[idx];
        long long best_finish = LLONG_MAX;
        int best_srv = 0, best_npu = 0, best_lat = 0;
        vector<int> best_B;

        for (int i = 0; i < N; i++) {
            int Bmax = (m[i] - b) / a;
            if (Bmax <= 0) continue;
            int T = (cnt_i + Bmax - 1) / Bmax;
            vector<int> B(T);
            for (int j = 0; j < T; j++) {
                B[j] = (j + 1 < T ? Bmax : (cnt_i - (T - 1) * Bmax));
            }
            int lat = latency[i][idx];
            long long rec_time = s_i + lat;
            for (int r = 0; r < g[i]; r++) {
                long long t_available = avail[i][r], t_finish = 0;
                for (int j = 0; j < T; j++) {
                    double speed = k[i] * sqrt((double)B[j]);
                    long long t_compute = (long long)ceil(B[j] / speed);
                    long long start_t = max(t_available, rec_time);
                    t_finish = start_t + t_compute;
                    t_available = t_finish;
                }
                if (t_available < best_finish) {
                    best_finish = t_available;
                    best_srv = i;
                    best_npu = r;
                    best_B = B;
                    best_lat = lat;
                }
            }
        }
        srv_res[idx] = best_srv + 1;
        npu_res[idx] = best_npu + 1;
        B_res[idx] = best_B;
        T_res[idx] = best_B.size();
        vector<long long> times(T_res[idx]);
        long long t = s_i;
        for (int j = 0; j < T_res[idx]; j++) {
            times[j] = t;
            t += best_lat + 1;
        }
        time_res[idx] = times;
        avail[best_srv][best_npu] = best_finish;
    }

    for (int i = 0; i < M; i++) {
        int T = T_res[i];
        cout << T << "\n";
        for (int j = 0; j < T; j++) {
            cout << time_res[i][j] << " "
                 << srv_res[i] << " "
                 << npu_res[i] << " "
                 << B_res[i][j];
            if (j + 1 < T) cout << " ";
        }
        cout << "\n";
    }

    return 0;
}
