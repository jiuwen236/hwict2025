#include <bits/stdc++.h>
using namespace std;

// 方法1  
// 最优batch_size / cnt排序 / 初始版本
// 线上: 82928031 / 79583261 / 74742303  线下: 2430318 / 2311709 / 1343744

// 自动选择更高分方法 + 削峰
// 线上: 83653628 / 83707134    线下: 2869256 / 3048753

// 方法封装
struct Method {
  int METHOD;
  // 方法1
  bool POSTPONE;  // 延迟超时请求
  bool IMMEDIATE; // 无视一切，立即发送请求
  bool BEST_BS; // 是否使用更优 batch size
  // 方法2
  int max_parallel;
  int min_bs;
  bool allow_reverse; // 进行削峰

  // 方法1
  static Method method1(bool postpone = 1, bool immediate = 0, bool best_bs = 1) {
    Method m;
    m.METHOD = 1;
    m.POSTPONE = postpone;
    m.IMMEDIATE = immediate;
    m.BEST_BS = best_bs;
    return m;
  }

  // 方法2
  static Method method2(int max_parallel = 2, int min_bs = 5, bool allow_reverse = 1) {
    Method m;
    m.METHOD = 2;
    m.max_parallel = max_parallel;
    m.min_bs = min_bs;
    m.allow_reverse = allow_reverse;
    return m;
  }
};

// 自动选择最高分方法
vector<Method> methods = {
  // 方法1
  // Method::method1(0, 1, 0), // 初始版本
  // Method::method1(1, 0, 0), // cnt排序
  Method::method1(),        // 最优batch_size
  // 方法2
  Method::method2(2, 5, 0),
  Method::method2(),        // 削峰
  Method::method2(1, 5, 1), // 削峰 且 非并行
};

using ll = long long;
using vi = vector<int>;

// debug
int computing_power, cnt_sum, start_sum, reverse_cnt;

// 题目常数
const int MAX_N = 10;
const int MAX_M = 500;
const int MAX_T_NUM = 300;
const int MAX_END_TIME = 60000;

// 最终输出结构
struct Schedule {
  vector<vector<array<int, 4>>> schedule;
  vector<int> T_out;
  double timeout_rate;
  double score;
  Schedule() : timeout_rate(0.0), score(0.0) {}
  Schedule(int M) : schedule(M), T_out(M), timeout_rate(0.0), score(0.0) {}
};

struct User {
  int id, s, e, cnt, duration;
  ll weight;
};

// 输入
int N, M, A, B;
vector<int> cores, speedCoef, memSize;
vector<User> users;
deque<int> q_user_ori;
vector<vector<int>> latency;

// 全局变量
int npu_num;

// 方法1变量

// 方法2变量
int max_mem_size;
vector<double> avg_cnt(MAX_END_TIME + 1);

// 分数计算函数
double h_func(double x) {
  return pow(2, -x / 100);
}
double p_func(double x) {
  return pow(2, -x / 200);
}

// 方法1，非并行
Schedule solve1(bool POSTPONE, bool IMMEDIATE, bool BEST_BS) {
  Schedule res(M);
  deque<int> q_user = q_user_ori;
  int timeout_cnt = 0;
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

    // 尝试每台服务器
    for (int i = 0; i < N; i++) {
      int B = maxBatch[i];
      int B_num = (u.cnt + B - 1) / B; // 批次数
      // 尝试每台服务器的每个 NPU
      for (int x = 0; x < cores[i]; x++) {
        int lat = latency[i][u.id];
        // 预测完成时间
        int start = u.s + lat;
        int p = procTime[i];
        int spec_size = (u.cnt - 1) % B + 1;
        int spec_time = (int)ceil(sqrt((double)spec_size) / speedCoef[i]);
        int finish = start;
        int need_time = spec_time;
        int step_time = max<int>(p, lat + 1);
        if (B_num > 1) {
          need_time += (B_num - 1) * p;
        }
        vi startTime, use_time;
        use_time.reserve(need_time);
        // 计算每批次的开始时间与使用时间(有抢占问题)
        for (int b = 0; b < B_num; b++) {
          bool is_spec = B_num - 1 == b;
          int proc_time = is_spec ? spec_time : p;
          for (int t = 0; t < proc_time;) {
            if (freeAt[i][x][finish]) {
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
          bestSrv = i;
          bestNpu = x;
          bestB = B;
          bestBnum = B_num;
          bestLat = lat;
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
    res.T_out[idx] = bestBnum;
    res.schedule[idx].reserve(bestBnum);
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
      res.schedule[idx].push_back({tm, bestSrv + 1, bestNpu + 1, batch});
      rem -= batch;
    }
    res.score += h_func(double(bestFinish - u.e) / (u.e - u.s));
    if (bestFinish > u.e) {
      timeout_cnt++;
      // if (idx < 10) cerr << "timeout user: " << idx << " " << bestFinish << " " << u.e << endl;
    }
    // 更新该 NPU 的空闲时间
    for (int t : bestUseTime) {
      freeAt[bestSrv][bestNpu][t] = false;
    }
  }
  res.timeout_rate = (double)timeout_cnt / M;
  res.score *= h_func(timeout_cnt) * p_func(0.0) * 10000;
  return res;
}

// bool is_exceed_time(int user_id) {
// }

void get_avg_cnt() {
  for(auto &u : users) {
    double avg = (double)u.cnt / u.duration;
    for(int i = u.s; i <= u.e; ++i) {
      avg_cnt[i] += avg;
    }
  }
  int dur = 16;
  for(int i = MAX_END_TIME; i >= dur - 1; --i) {
    for(int j = i - dur + 1; j < i; ++j) {
      avg_cnt[i] += avg_cnt[j];
    }
    avg_cnt[i] /= dur;
  }
}

struct BS_Plan {
  vi batch_size;  // 有序
  vi time;        // batch_size对应消耗的时间
  int loop_time;  // time的最小公倍数
  double throughput; // 吞吐量(每毫秒完成请求的batchsize的和)
  int left_mem; // 剩余显存
};

// DFS枚举方案，满足条件的所有方案，按吞吐量降序排序
// 允许指定bs比例，如要求bs[20]=0.1，合法方案中bs>=20的数量/batch_size.size()应至少为0.1
// 构造一个递归函数，遍历不大于之前的bs，直到显存不支持该bs
// max_parallel = 1，比方法1差一点点
vector<BS_Plan> get_bs_plan_dfs(int k, int m, int a, int b, vector<double> &bs_ratio, int max_parallel = 2, int min_bs = 5) {
    vector<BS_Plan> plans;
    
    function<void(int, vector<int>, int)> dfs = 
        [&](int current_mem, vector<int> current_bs, int max_bs) {
        
        if (!current_bs.empty()) {
            bool ok = true;
            if (!bs_ratio.empty()) {
                int n = current_bs.size();
                for (size_t i = 0; i < bs_ratio.size(); ++i) {
                    if (bs_ratio[i] > 1e-9) {
                        int cnt = 0;
                        for (int bs : current_bs) {
                            if (bs >= (int)i) {
                                cnt++;
                            }
                        }
                        if ((double)cnt / n < bs_ratio[i]) {
                            ok = false;
                            break;
                        }
                    }
                }
            }
            
            if (ok) {
                BS_Plan plan;
                double total_throughput = 0;
                int loop = 1;
                
                vector<pair<int, int>> vp;
                for (int bs : current_bs) {
                    int t = (int)ceil(sqrt((double)bs) / k);
                    vp.push_back({bs, t});
                    total_throughput += (double)bs / t;
                    loop = loop / gcd(loop, t) * t;
                }
                
                sort(vp.begin(), vp.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
                    return a.first < b.first;
                });
                for(auto& p : vp) {
                    plan.batch_size.push_back(p.first);
                    plan.time.push_back(p.second);
                }

                plan.throughput = total_throughput;
                plan.loop_time = loop;
                plan.left_mem = m - current_mem;
                plans.push_back(plan);
            } else {
              return;
            }
        }
        
        if(current_bs.size() >= max_parallel) return;  // 降低并行度

        for (int bs = max_bs; bs >= min_bs; --bs) {
            int mem_needed = a * bs + b;
            if (current_mem + mem_needed <= m) {
                vector<int> next_bs = current_bs;
                next_bs.push_back(bs);
                dfs(current_mem + mem_needed, next_bs, bs);
            }
        }
    };
    
    int max_possible_bs = (m - b) / a;
    if (max_possible_bs > 0) {
        dfs(0, {}, max_possible_bs);
    }
    
    sort(plans.begin(), plans.end(), [](const BS_Plan& p1, const BS_Plan& p2) {
        if (abs(p1.throughput - p2.throughput) > 1e-9) {
            return p1.throughput > p2.throughput;
        }
        return p1.batch_size < p2.batch_size;
    });

    plans.erase(unique(plans.begin(), plans.end(), [](const BS_Plan& p1, const BS_Plan& p2) {
        return p1.batch_size == p2.batch_size;
    }), plans.end());

    assert(plans.size() > 0);

    return plans;
}

// 方法2，允许 npu 并行处理多个用户请求
Schedule solve2(int max_parallel, int min_bs_dfs, bool allow_reverse) {
  deque<int> q_user = q_user_ori;
  Schedule res(M);
  int timeout_cnt = 0;
  // 必然超时的用户
  
  // 满足最多300请求
  vector<int> min_bs(M);
  for(int user_id : q_user) {
    min_bs[user_id] = (users[user_id].cnt - 1) / MAX_T_NUM + 1;
  }

  // 计算最优bs方案
  bool log_bs_plan = 0;
  vector<BS_Plan> bs_plans;
  for(int i = 0; i < N; ++i) {
    // 总体bs要求
    vector<double> bs_ratio((max_mem_size - B) / A * 2);
    for(int user_id : q_user) {
      bs_ratio[min_bs[user_id]] += 1;
    }
    int max_nz_idx = -1;
    for(int i = 0; i < bs_ratio.size(); ++i) {
      if(bs_ratio[i] > 0) 
        max_nz_idx = max(max_nz_idx, i);
      bs_ratio[i] /= M;
    }
    bs_ratio.resize(max_nz_idx + 1);
    // 调用函数获取最优bs方案
    bs_plans.emplace_back(get_bs_plan_dfs(speedCoef[i], memSize[i], A, B, bs_ratio, max_parallel, min_bs_dfs)[0]);
    if(log_bs_plan) {
      cerr << "bs_plans[" << i << "].throughput: " << bs_plans[i].throughput << " loop_time: " << bs_plans[i].loop_time << " left_mem: " << bs_plans[i].left_mem << endl;
      cerr << "bs_plans[" << i << "].batch_size: ";
      for(int bs : bs_plans[i].batch_size) cerr << bs << " ";
      cerr << endl;
      cerr << "bs_plans[" << i << "].time: ";
      for(int t : bs_plans[i].time) cerr << t << " ";
      cerr << endl;
    }
  }
  if(log_bs_plan) cerr << endl;

  // 占用情况初始化
  vector<vector<vector<vector<bool>>>> freeAt(N);
  for(size_t i = 0; i < N; ++i) {
    freeAt[i].resize(cores[i]);
    for(size_t j = 0; j < cores[i]; ++j) {
      freeAt[i][j].resize(bs_plans[i].batch_size.size());
      // 将一个npu划分为多个并行的通道
      for(size_t k = 0; k < bs_plans[i].batch_size.size(); ++k) {
        freeAt[i][j][k].resize(1000 * 1000 * 27 / npu_num / bs_plans[i].time[k], true);
      }
    }
  }

  // 逐个添加用户
  deque<int> postponed;
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
    bool is_reverse = false;
    if(allow_reverse && !is_late_user && avg_cnt[u.s] > avg_cnt[u.e]) {
      is_reverse = true;
      reverse_cnt++;
    }
    int bestFinish = is_reverse ? -1 : INT_MAX;
    int bestSrv = -1, bestNpu = -1, bestLat = 0, bestActualFinish;
    deque<pair<int, int>> best_bs_plan;
    vector<pair<size_t, size_t>> bestUseTime;

    for(size_t i = 0; i < N; ++i) {
      // if (idx % N != i) continue;
      int lat = latency[i][u.id];
      int current_cnt = u.cnt;
      for(size_t j = 0; j < cores[i]; ++j) {
        current_cnt = u.cnt;
        deque<pair<int, int>> bs_plan;
        vector<pair<size_t, size_t>> use_time;  // freeAt的最后两个维度
        size_t min_bs_idx = lower_bound(bs_plans[i].batch_size.begin(), bs_plans[i].batch_size.end(), min_bs[u.id]) - bs_plans[i].batch_size.begin();
        if(min_bs_idx == bs_plans[i].batch_size.size()) continue;
        int min_bs_time = bs_plans[i].time[min_bs_idx];
        int start = u.s + lat;
        int finish = is_reverse ? -1 : INT_MAX;
        int actual_finish = -1;
        int end_time = is_late_user ? INT_MAX : u.e;
        // 逐毫秒检查是否可以发送请求
        for(int t = is_reverse ? u.e : start; is_reverse ? t >= start : t <= end_time; is_reverse ? --t : ++t) {
          if(current_cnt <= 0) break;
          if(current_cnt < bs_plans[i].batch_size[min_bs_idx]) {
            min_bs_idx = lower_bound(bs_plans[i].batch_size.begin(), bs_plans[i].batch_size.end(), current_cnt) - bs_plans[i].batch_size.begin();
          }
          for(size_t k = min_bs_idx; k < bs_plans[i].batch_size.size(); ++k) {
            if(t % bs_plans[i].time[k] == 0) {
              if(is_reverse && t + bs_plans[i].time[k] > u.e) break;
              size_t time_idx = t / bs_plans[i].time[k];
              if(freeAt[i][j][k][time_idx]) {
                if(is_reverse)
                  bs_plan.push_front({t, min(bs_plans[i].batch_size[k], current_cnt)});
                else 
                  bs_plan.push_back({t, min(bs_plans[i].batch_size[k], current_cnt)});
                use_time.push_back({k, time_idx});
                if(is_reverse && bs_plan.size() == 1)
                  actual_finish = t + bs_plans[i].time[k];
                current_cnt -= bs_plans[i].batch_size[k];
                finish = is_reverse ? t : t + bs_plans[i].time[k];
                int plus_time = lat;
                t = is_reverse ? t - plus_time : t + plus_time;
                break;
              }
            }
          }
        }
        if(current_cnt > 0) continue;
        if(is_reverse ? finish > bestFinish : finish < bestFinish) {
          bestFinish = finish;
          bestSrv = i;
          bestNpu = j;
          bestLat = lat;
          bestActualFinish = is_reverse ? actual_finish : finish;
          best_bs_plan = move(bs_plan);
          bestUseTime = move(use_time);
        }
      }
    }

    assert(!is_late_user || bestSrv != -1);
    if(!is_late_user && (is_reverse ? bestFinish < u.s : bestFinish > u.e)) {
      postponed.push_back(idx);
      continue;
    }

    res.score += h_func(double(bestActualFinish - u.e) / (u.e - u.s));
    if (bestActualFinish > u.e) {
      timeout_cnt++;
    }

    res.T_out[idx] = best_bs_plan.size();
    res.schedule[idx].reserve(best_bs_plan.size());
    for(auto& p : best_bs_plan) {
      res.schedule[idx].push_back({p.first - bestLat, bestSrv + 1, bestNpu + 1, p.second});
    }
    for(auto& p : bestUseTime) {
      freeAt[bestSrv][bestNpu][p.first][p.second] = false;
    }
  }
  // cerr << "reverse_cnt: " << reverse_cnt << endl;
  res.timeout_rate = (double)timeout_cnt / M;
  res.score *= h_func(timeout_cnt) * p_func(0.0) * 10000;
  return res;
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
    max_mem_size = max(max_mem_size, memSize[i]);
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

  for (int i = 0; i < M; ++i) q_user_ori.push_back(i);
  // Sort user queue by weight
  sort(q_user_ori.begin(), q_user_ori.end(), [&](int x, int y) {
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
  // for (int i = 0; i < N; i++) assert(memSize[i] > 1000); // 初赛线上存在 memSize[i] = 1000 的情况

  // 削峰预处理
  get_avg_cnt();

  // 获取解决方案
  vector<Schedule> results;
  size_t best_idx = -1;
  double best_score = 0;
  for(size_t i = 0; i < methods.size(); ++i) {
    if(methods[i].METHOD == 1) {
      results.push_back(solve1(methods[i].POSTPONE, methods[i].IMMEDIATE, methods[i].BEST_BS));
    } 
    if(methods[i].METHOD == 2) {
      results.push_back(solve2(methods[i].max_parallel, methods[i].min_bs, methods[i].allow_reverse));
    }
    if(results[i].score > best_score) {
      best_score = results[i].score;
      best_idx = i;
    }
  }
  assert(best_idx != -1);
  auto schedule = results[best_idx].schedule;

  // 观测线上数据
  // assert(best_score < 2000000 || best_score > 3000000); // 不存在200w-300w

  // 输出
  for (int i = 0; i < M; i++) {
    assert(schedule[i].size() > 0);
    cout << schedule[i].size() << "\n";
    for (auto &job : schedule[i]) {
      cout << job[0] << ' ' << job[1] << ' ' << job[2] << ' ' << job[3] << ' ';
    }
    cout << "\n";
  }
  // cerr << "timeout_rate: " << results[best_idx].timeout_rate * 100 << "%" << " score: " << (int)results[best_idx].score << endl;
  return 0;
}