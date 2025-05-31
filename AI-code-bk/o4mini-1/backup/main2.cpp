#include <bits/stdc++.h>
using namespace std;

// 5034337.01  59674160  2025-05-31 02:06:19

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

    // choose fastest server type
    int idx0 = 0;
    for(int i = 1; i < N; i++){
        if(k[i] > k[idx0] || (k[i] == k[idx0] && m[i] > m[idx0])) idx0 = i;
    }
    int Bper = (m[idx0] - b) / a;
    if(Bper < 1) Bper = 1;
    int gpu_cnt = g[idx0];
    // assign each user a fixed NPU to avoid migration penalty
    vector<int> user_npu(M);
    for(int j = 0; j < M; j++){
        user_npu[j] = (j % gpu_cnt) + 1;
    }
    // schedule for each user on the fastest server and fixed NPU
    for(int j = 0; j < M; j++){
        int remain = cnt[j];
        int lat = latency[idx0][j];
        int t = s[j];
        vector<int> times, servers, npus, Bs;
        while(remain > 0){
            int B = min(remain, Bper);
            times.push_back(t);
            servers.push_back(idx0 + 1);
            npus.push_back(user_npu[j]);
            Bs.push_back(B);
            remain -= B;
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
