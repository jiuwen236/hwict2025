#include <bits/stdc++.h>
using namespace std;

// 4762600.51  63599203 2025-05-31 02:12:34

int main(){
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    int N;
    if(!(cin >> N)) return 0;
    vector<int> g(N), k(N), m(N);
    for(int i = 0; i < N; i++) cin >> g[i] >> k[i] >> m[i];
    int M; cin >> M;
    vector<int> s(M), e(M), cnt(M);
    for(int i = 0; i < M; i++) cin >> s[i] >> e[i] >> cnt[i];
    vector<vector<int>> latency(N, vector<int>(M));
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++) cin >> latency[i][j];
    }
    int a, b; cin >> a >> b;

    // compute max batch per server and per-batch processing time
    vector<int> Bmax(N);
    for(int i = 0; i < N; i++){
        Bmax[i] = (m[i] - b) / a;
        if(Bmax[i] < 1) Bmax[i] = 1;
    }
    vector<int> pb_time(N);
    for(int i = 0; i < N; i++){
        double v = k[i] * sqrt((double)Bmax[i]);
        pb_time[i] = (int)ceil(Bmax[i] / v);
    }
    // select best server for each user by minimal per-sample cost
    vector<int> best_server(M), best_Bmax(M), best_lat(M);
    for(int j = 0; j < M; j++){
        double best_cost = 1e300;
        int bi = 0;
        for(int i = 0; i < N; i++){
            double cost = (pb_time[i] + latency[i][j]) / (double)Bmax[i];
            if(cost < best_cost){ best_cost = cost; bi = i; }
        }
        best_server[j] = bi;
        best_Bmax[j] = Bmax[bi];
        best_lat[j] = latency[bi][j];
    }

    // schedule for each user with dynamic server and round-robin NPU
    for(int j = 0; j < M; j++){
        int remain = cnt[j];
        int lat = best_lat[j];
        int server_id = best_server[j];
        int Bper = best_Bmax[j];
        int gpu_cnt = g[server_id];
        int t = s[j];
        vector<int> times, servers, npus, Bs;
        int npu_idx = 0;
        while(remain > 0){
            int B = min(remain, Bper);
            times.push_back(t);
            servers.push_back(server_id + 1);
            npus.push_back((npu_idx % gpu_cnt) + 1);
            Bs.push_back(B);
            remain -= B;
            npu_idx++;
            if(remain > 0) t += lat + 1;
        }
        int T = times.size();
        cout << T << '\n';
        for(int i = 0; i < T; i++){
            cout << times[i] << ' ' << servers[i] << ' ' << npus[i] << ' ' << Bs[i];
            if(i + 1 < T) cout << ' ';
        }
        cout << '\n';
    }
    return 0;
}
