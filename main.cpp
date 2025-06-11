#include <bits/stdc++.h>
using namespace std;
using ll = long long;

// o4mini-3

struct User {
  int id, s, e, cnt;
  ll weight;
};

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  int N;
  cin >> N;
  vector<int> cores(N), speedCoef(N), memSize(N);  // g, k, m
  for (int i = 0; i < N; i++) {
    cin >> cores[i] >> speedCoef[i] >> memSize[i];
  }

  int M;
  cin >> M;
  vector<User> users(M);
  for (int i = 0; i < M; i++) {
    users[i].id = i;
    cin >> users[i].s >> users[i].e >> users[i].cnt;
    users[i].weight = ll(users[i].e - users[i].s) * users[i].cnt;
  }

  vector<vector<int>> latency(N, vector<int>(M)); // latency[server][user]
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      cin >> latency[i][j];
    }
  }

  int A, B;
  cin >> A >> B;

  // 预先算出每台服务器的“最大批次”与单次推理耗时
  vector<int> maxBatch(N);
  vector<ll> procTime(N);
  for (int i = 0; i < N; i++) {
    int batch_size = (memSize[i] - B) / A;
    assert(batch_size > 0);
    maxBatch[i] = batch_size;
    double sp = speedCoef[i] * sqrt((double)batch_size);
    procTime[i] = (ll)ceil(batch_size / sp);
  }

  // 用户按 weight 排序
  vector<int> order(M);
  iota(order.begin(), order.end(), 0);
  sort(order.begin(), order.end(), [&](int x, int y) {
    if (users[x].weight != users[y].weight)
      return users[x].weight < users[y].weight;
    return x < y;
  });

  // 每台服务器每个 NPU 的下次可用时间
  vector<vector<ll>> freeAt(N);
  for (int i = 0; i < N; i++) {
    freeAt[i].assign(cores[i], 0LL);
  }

  // 最终输出结构
  vector<vector<array<int, 4>>> schedule(M);
  vector<int> T_out(M);

  // 按优先级调度
  for (int idx : order) {
    auto &u = users[idx];
    ll bestFinish = LLONG_MAX;
    int bestSrv = -1, bestNpu = -1, bestB = 0, bestT = 0, bestLat = 0;
    ll bestPt = 0;

    // 尝试每台服务器
    for (int j = 0; j < N; j++) {
      int B = maxBatch[j];
      int T = (u.cnt + B - 1) / B; // 批次数
      // 找最早空闲的 NPU
      int sel = 0;
      ll ft = freeAt[j][0];
      for (int x = 1; x < cores[j]; x++) {
        if (freeAt[j][x] < ft) {
          ft = freeAt[j][x];
          sel = x;
        }
      }
      int lat = latency[j][u.id];
      // 预测完成时间
      ll start = max<ll>(u.s + lat, ft);
      ll p = procTime[j];
      ll fin = start + p;
      if (T > 1) {
        ll step = max<ll>(p, lat + 1);
        fin += (T - 1) * step;
      }
      if (fin < bestFinish) {
        bestFinish = fin;
        bestSrv = j;
        bestNpu = sel;
        bestB = B;
        bestT = T;
        bestLat = lat;
        bestPt = p;
      }
    }

    // 记录该用户的发送方案
    T_out[idx] = bestT;
    schedule[idx].reserve(bestT);
    int rem = u.cnt;
    for (int t = 0; t < bestT; t++) {
      int batch = min(bestB, rem);
      int tm = u.s + t * (bestLat + 1);
      schedule[idx].push_back({tm, bestSrv + 1, bestNpu + 1, batch});
      rem -= batch;
    }
    // 更新该 NPU 的空闲时间
    freeAt[bestSrv][bestNpu] = bestFinish;
  }

  // 输出
  for (int i = 0; i < M; i++) {
    cout << schedule[i].size() << "\n";
    for (auto &job : schedule[i]) {
      cout << job[0] << ' ' << job[1] << ' ' << job[2] << ' ' << job[3] << ' ';
    }
    cout << "\n";
  }
  return 0;
}