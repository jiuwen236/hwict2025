#include <bits/stdc++.h>
using namespace std;

// 线下: 3984132  线上: 79583261

const bool POSTPONE = 1;  // 延迟超时请求
const bool IMMEDIATE = 0; // 无视一切，立即发送请求

using ll = long long;
using vi = vector<int>;

// debug
int computing_power, cnt_sum, start_sum;

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
    start_sum += users[i].s;
    cnt_sum += users[i].cnt;
    // users[i].weight = ll(users[i].e - users[i].s) * users[i].cnt;
    // users[i].weight = users[i].s;
    users[i].weight = users[i].cnt;
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
  vector<int> procTime(N);
  for (int i = 0; i < N; i++) {
    int batch_size = (memSize[i] - B) / A;
    assert(batch_size > 0);
    maxBatch[i] = batch_size;
    double sp = speedCoef[i] * sqrt((double)batch_size);
    procTime[i] = (int)ceil(batch_size / sp);
    // procTime[i] = 1;
  }

  // 用户按 weight 排序
  deque<int> q_user(M);
  deque<int> postponed;
  iota(q_user.begin(), q_user.end(), 0);
  sort(q_user.begin(), q_user.end(), [&](int x, int y) {
    if (users[x].weight != users[y].weight)
      return users[x].weight < users[y].weight;
    return x < y;
  });

  // 每台服务器每个 NPU 的下次可用时间
  vector<vector<vector<bool>>> freeAt(N);
  int npu_num = 0;
  for (int i = 0; i < N; i++) {
    npu_num += cores[i];
    computing_power += cores[i] * speedCoef[i];
  }
  for (int i = 0; i < N; i++) {
    freeAt[i].resize(cores[i]);
    for (int j = 0; j < cores[i]; j++) {
      freeAt[i][j].resize(1000 * 1000 * 9 / npu_num, true);
    }
  }

  // 观测线上数据
  // assert(npu_num > 1);  // 初赛线上有 npu_num = 1 的情况, 没有 npu_num = 1 and speedCoef[0] = 1 的情况
  // assert(computing_power > 2);  // 初赛线上有 computing_power = 2 的情况
  double avg_cnt = (double)cnt_sum / M;
  // assert(avg_cnt <= 5998); // 初赛线上存在 avg_cnt < 1500 和 avg_cnt > 5998 (computing_power > 3) 的情况
  double avg_start = (double)start_sum / M;
  double variance = 0; // 平均L1距离
  for (int i = 0; i < M; i++)
    variance += abs(users[i].s - avg_start);
  variance /= M;
  // cerr << "variance: " << variance << endl;
  // assert(variance >= 1 || M < 100);  // 初赛线上存在 variance < 1 的情况

  // 最终输出结构
  vector<vector<array<int, 4>>> schedule(M);
  vector<int> T_out(M);

  // 按优先级调度
  int timeout_cnt = 0;

  while (!q_user.empty() || !postponed.empty()) {
    // 选择普通用户或延迟用户
    int idx = -1;
    bool is_late_user = false;
    if (!q_user.empty()) {
      idx = q_user.front();
      q_user.pop_front();
    } else {
      is_late_user = true;
      idx = postponed.front();
      postponed.pop_front();
    }
    auto &u = users[idx];
    ll bestFinish = LLONG_MAX;
    int bestSrv = -1, bestNpu = -1, bestB = 0, bestBnum = 0, bestLat = 0;
    vi bestStartTime, bestUseTime;
    int bestPt = 0, bestSpecTime = 0;

    // 尝试每台服务器
    for (int j = 0; j < N; j++) {
      int B = maxBatch[j];
      int B_num = (u.cnt + B - 1) / B; // 批次数
      // 尝试每台服务器的每个 NPU
      for (int x = 0; x < cores[j]; x++) {
        int lat = latency[j][u.id];
        // 预测完成时间
        int start = max(u.s, lat);
        int p = procTime[j];
        int spec_size = (u.cnt - 1) % B + 1;
        int spec_time = (int)ceil(sqrt((double)spec_size) / speedCoef[j]);
        int finish = start;
        int need_time = spec_time;
        int step_time = max<int>(p, lat + 1);
        if (B_num > 1) {
          need_time += (B_num - 1) * p;
        }
        vi startTime, use_time;
        use_time.reserve(need_time);
        // 计算每批次的开始时间与使用时间(有抢占问题)
        for (int i = 0; i < B_num; i++) {
          bool is_spec = B_num - 1 == i;
          int proc_time = is_spec ? spec_time : p;
          for (int t = 0; t < proc_time;) {
            if (freeAt[j][x][finish]) {
              if (t == 0) {
                startTime.push_back(finish);
              }
              use_time.push_back(finish);
              t++;
            }
            finish++;
          }
          if (!is_spec && finish - startTime.back() <= lat) {
            finish = startTime.back() + lat + 1;
          }
        }
        // 更新最早完成的npu
        if (finish < bestFinish) {
          bestFinish = finish;
          bestSrv = j;
          bestNpu = x;
          bestB = B;
          bestBnum = B_num;
          bestLat = lat;
          bestPt = p;
          bestSpecTime = spec_time;
          bestStartTime = move(startTime);
          bestUseTime = move(use_time);
        }
      }
    }

    // 当预测的用户请求完成时间大于结束时间时，放弃该用户(移至末尾，最后处理)
    if (POSTPONE && !is_late_user && bestFinish > u.e) {
      postponed.push_back(idx);
      continue;
    }
    // 记录该用户的发送方案
    T_out[idx] = bestBnum;
    schedule[idx].reserve(bestBnum);
    int rem = u.cnt;
    assert(bestStartTime.size() == bestBnum);
    int last_time = -100;
    for (int t = 0; t < bestBnum; t++) {
      int batch = min(bestB, rem);
      int tm = max(last_time + bestLat + 1, u.s);
      if (!IMMEDIATE) {
        tm = max(tm, bestStartTime[t] - bestLat);
      }
      last_time = tm;
      schedule[idx].push_back({tm, bestSrv + 1, bestNpu + 1, batch});
      rem -= batch;
    }
    if (bestFinish > u.e) {
      timeout_cnt++;
      // if (idx < 10) cerr << "timeout user: " << idx << " " << bestFinish << " " << u.e << endl;
    }
    // 更新该 NPU 的空闲时间
    for (int t : bestUseTime) {
      freeAt[bestSrv][bestNpu][t] = false;
    }
  }

  // 输出
  for (int i = 0; i < M; i++) {
    assert(schedule[i].size() > 0);
    cout << schedule[i].size() << "\n";
    for (auto &job : schedule[i]) {
      cout << job[0] << ' ' << job[1] << ' ' << job[2] << ' ' << job[3] << ' ';
    }
    cout << "\n";
  }
  // cerr << "timeout_cnt: " << timeout_cnt << ", timeout_rate: " << (double)timeout_cnt / M * 100 << "%" << endl;
  return 0;
}