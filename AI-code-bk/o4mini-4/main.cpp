#include <bits/stdc++.h>
using namespace std;

// 线下：2887443   线上：72070727  
// 结论1：线上数据不是rate限制
// 结论2：线上按(e_i-s_i)排序更好，线下按开始时间更好。按时间排序对限制窗口效果很好

// - 禁用迁移；每次请求的大小尽量装满NPU显存
// - 用户按开始时间排序，最小的优先
// - 最优选择：用户发送请求时会计算对每个NPU的实际推理结束时间，选择最早结束的。

int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    int N;
    cin >> N;
    vector<int> g(N), k(N), m(N);
    for (int i = 0; i < N; ++i) {
        cin >> g[i] >> k[i] >> m[i];
    }
    int M;
    cin >> M;
    vector<int> s(M), e(M), cnt(M);
    for (int i = 0; i < M; ++i) {
        cin >> s[i] >> e[i] >> cnt[i];
    }
    vector<vector<int>> lat(N, vector<int>(M));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            cin >> lat[i][j];
        }
    }
    int a, b;
    cin >> a >> b;

    // 每个服务器的最大批次大小
    vector<int> Bmax(N);
    for (int i = 0; i < N; ++i) {
        Bmax[i] = max(1, (m[i] - b) / a);
    }

    struct User { int id, s, e, cnt; };
    vector<User> users;
    users.reserve(M);
    for (int i = 0; i < M; ++i) {
        users.push_back({i, s[i], e[i], cnt[i]});
    }
    sort(users.begin(), users.end(), [](auto &u1, auto &u2) {
        return u1.s < u2.s;
    });

    vector<vector<long long>> sendTimes(M);
    vector<vector<int>> batches(M);
    vector<int> T(M), serverChoice(M), npuChoice(M);
    vector<vector<long long>> nextFree(N);
    for (int i = 0; i < N; ++i) {
        nextFree[i].assign(g[i], 0LL);
    }

    // 为每个用户选择最佳Server和NPU，禁用迁移
    for (auto &u : users) {
        int uid = u.id;
        int start = u.s;
        int total = u.cnt;

        double bestFinish = 1e18;
        int bestS = 0, bestNP = 0, bestT = 0;

        // 枚举所有Server和NPU
        for (int i = 0; i < N; ++i) {
            int Bm = Bmax[i];
            int Ti = (total + Bm - 1) / Bm;
            if (Ti > 300) Ti = 300;
            for (int np = 0; np < g[i]; ++np) {
                double t_free = nextFree[i][np];
                double currSend = start;
                double finishTime = t_free;
                for (int t = 0; t < Ti; ++t) {
                    int batch = (t < Ti - 1 ? Bm : total - Bm * (Ti - 1));
                    double arrival = currSend + lat[i][uid];
                    double startProc = max(arrival, finishTime);
                    double procTime = ceil(batch / ((double)k[i] * sqrt(batch)));
                    finishTime = startProc + procTime;
                    currSend = currSend + lat[i][uid] + 1;
                }
                if (finishTime < bestFinish) {
                    bestFinish = finishTime;
                    bestS = i;
                    bestNP = np;
                    bestT = (total + Bm - 1) / Bm;
                    if (bestT > 300) bestT = 300;
                }
            }
        }

        // 生成最终的发送方案
        int Bm = Bmax[bestS];
        int Ti = bestT;
        vector<long long> sts;
        vector<int> bs;
        long long currSend = start;
        for (int t = 0; t < Ti; ++t) {
            int batch = (t < Ti - 1 ? Bm : total - Bm * (Ti - 1));
            sts.push_back(currSend);
            bs.push_back(batch);
            currSend = currSend + lat[bestS][uid] + 1;
        }
        sendTimes[uid] = move(sts);
        batches[uid] = move(bs);
        T[uid] = Ti;
        serverChoice[uid] = bestS;
        npuChoice[uid] = bestNP;

        // 更新NPU的可用时间
        double t_free = nextFree[bestS][bestNP];
        double curr = start;
        double finishTime = t_free;
        for (int t = 0; t < Ti; ++t) {
            int batch = batches[uid][t];
            double arrival = curr + lat[bestS][uid];
            double startProc = max(arrival, finishTime);
            double procTime = ceil(batch / ((double)k[bestS] * sqrt(batch)));
            finishTime = startProc + procTime;
            curr = curr + lat[bestS][uid] + 1;
        }
        nextFree[bestS][bestNP] = (long long)finishTime;
    }

    // 输出结果
    for (int i = 0; i < M; ++i) {
        cout << T[i] << "\n";
        for (int t = 0; t < T[i]; ++t) {
            cout << sendTimes[i][t] << " " << (serverChoice[i] + 1)
                 << " " << (npuChoice[i] + 1) << " " << batches[i][t];
            if (t + 1 < T[i]) cout << " ";
        }
        cout << "\n";
    }

    return 0;
}
