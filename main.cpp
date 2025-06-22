#include <bits/stdc++.h>
using namespace std;

// 方法1  
// 最优batch_size / cnt排序 / 初始版本
// 线上: 82928031 / 79583261 / 74742303  线下: 22481058 / 21481910 / 11103833

const int METHOD = 1;

// 方法1
const bool POSTPONE = 1;  // 延迟超时请求
const bool IMMEDIATE = 0; // 无视一切，立即发送请求
const bool BEST_BS = 1; // 是否使用更优 batch size

using ll = long long;
using vi = vector<int>;

// debug
int computing_power, cnt_sum, start_sum, timeout_cnt;

// 题目常数
const int MAX_M = 500;

// 最终输出结构
vector<vector<array<int, 4>>> schedule;
vector<int> T_out;

struct User {
  int id, s, e, cnt, duration;
  ll weight;
};

// 输入
int N, M, A, B;
vector<int> cores, speedCoef, memSize;
vector<User> users;
deque<int> q_user;
vector<vector<int>> latency;

// 方法1变量
int npu_num;

// 方法2变量

// 方法1，非并行
void solve1() {
  // 预处理NPU的batch size和时间
  vector<int> maxBatch, procTime;
  maxBatch.resize(N);
  procTime.resize(N);
  for (int i = 0; i < N; i++) {
    int batch_size = (memSize[i] - B) / A;
    assert(batch_size > 0);
    maxBatch[i] = batch_size;
    int sp = speedCoef[i];
    procTime[i] = (int)ceil(sqrt((double)batch_size) / sp);
    // 非并行的最优 batch size
    double throughput1 = (double)maxBatch[i] / procTime[i];
    int t = 1;
    while(!(t * t * sp * sp <= batch_size && (t + 1) * (t + 1) * sp * sp > batch_size)) {
      t++;
      if (t > 100) {
        t = 1;
        break;
      }
    }
    batch_size = min(batch_size, t * t * sp * sp);
    double proc_time = (int)ceil(sqrt((double)batch_size) / sp);
    double throughput2 = (double)batch_size / proc_time;
    if(BEST_BS && throughput1 < throughput2) {
      maxBatch[i] = batch_size;
      procTime[i] = proc_time;
    }
  }

  // resize
  vector<vector<vector<bool>>> freeAt;
  freeAt.resize(N);
  for (int i = 0; i < N; i++) {
    freeAt[i].resize(cores[i]);
    for (int j = 0; j < cores[i]; j++) {
      freeAt[i][j].resize(1000 * 1000 * 9 / npu_num, true);
    }
  }

  deque<int> postponed;
  // 按优先级调度
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
}

// bool is_exceed_time(int user_id) {
// }

// 方法2，允许 npu 并行处理多个用户请求
void solve2() {
  // 必然超时的用户


  // 逐个添加用户，使用最大吞吐量方案
  for(int user_id : q_user) {
    
  }

  // 处理超时用户
  
}


int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  cin >> N;
  cores.resize(N);
  speedCoef.resize(N);
  memSize.resize(N);
  for (int i = 0; i < N; i++) {
    cin >> cores[i] >> speedCoef[i] >> memSize[i];
  }

  cin >> M;
  users.resize(M);
  for (int i = 0; i < M; i++) {
    users[i].id = i;
    cin >> users[i].s >> users[i].e >> users[i].cnt;
    users[i].duration = users[i].e - users[i].s;
    start_sum += users[i].s;
    cnt_sum += users[i].cnt;
    // users[i].weight = ll(users[i].e - users[i].s) * users[i].cnt;
    // users[i].weight = users[i].s;
    users[i].weight = users[i].cnt;
  }

  latency.assign(N, vector<int>(M)); // latency[server][user]
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      cin >> latency[i][j];
    }
  }

  cin >> A >> B;

  for (int i = 0; i < M; ++i) q_user.push_back(i);
  // Sort user queue by weight
  sort(q_user.begin(), q_user.end(), [&](int x, int y) {
    if (users[x].weight != users[y].weight)
      return users[x].weight < users[y].weight;
    return x < y;
  });
  
  for (int i = 0; i < N; i++) {
    npu_num += cores[i];
    computing_power += cores[i] * speedCoef[i];
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
  schedule.resize(M);
  T_out.resize(M);

  if (METHOD == 1) solve1();
  else solve2();

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