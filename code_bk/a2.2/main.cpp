#include <bits/stdc++.h>
using namespace std;

// Hyperparameter search configuration
const bool HP_SEARCH = 1;
const double HP_TIME_LIMIT = 5; // 单位：秒
const bool RANDOM_SEARCH = 1;  // 关闭时，仅进行爬山法优化
static const vector<vector<int>> HPARAM_VALUES = {
  {1,2,3}, // max_parallel  1好像没必要
  {5},     // min_bs_dfs
  {0,1,2}, // reverse_mode 1: 进行削峰，2: 按概率决定是否反向
  {0,1,2}, // order_mode  不迁移时如何选择npu：0: 选择最快的，1: 选择第1个合法解，2: 选择最后一个
  {0,1}    // move_mode  是否迁移
};

bool log_method = 0;  // 是否打印不同方法分数信息
const int TOP_N_METHOD2 = 4;  // 打印前N名method2参数设置

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
  int reverse_mode;
  int order_mode;
  int move_mode;

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
  static Method method2(int max_parallel = 2, int min_bs = 5, int reverse_mode = 1, int order_mode = 0, int move_mode = 0) {
    Method m;
    m.METHOD = 2;
    m.max_parallel = max_parallel;
    m.min_bs = min_bs;
    m.reverse_mode = reverse_mode;
    m.order_mode = order_mode;
    m.move_mode = move_mode;
    return m;
  }
};

// 初始方法
vector<Method> methods = {
  // 方法1
  // Method::method1(0, 1, 0), // 初始版本
  // Method::method1(1, 0, 0), // cnt排序
  Method::method1(),        // 最优batch_size
  // 方法2
  // Method::method2(1, 5, 0),   // 模拟方法1
  // Method::method2(2, 5, 0),   // 初始并行版本
  // Method::method2(2, 5, 1),   // 削峰
  // Method::method2(1, 5, 1),   // 削峰 且 非并行
  Method::method2(2, 5, 0, 0, 0),
  Method::method2(2, 5, 1, 1, 1),
  Method::method2(2, 5, 1, 2, 0),
};

using ll = long long;
using vi = vector<int>;

// mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
mt19937 rng(1);

// debug
int computing_power, cnt_sum, start_sum, reverse_cnt, max_latency;
int min_cnt_dbg = INT_MAX, min_latency = INT_MAX;
double avg_rate;

// 题目常数
const int MAX_N = 10;
const int MAX_M = 500;
const int MAX_T_NUM = 300;
const int MAX_END_TIME = 60000;
const int MIN_LATENCY = 10;

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
  double weight;
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

int get_process_time(int bs, int sp) {
  return (int)ceil(sqrt((double)bs) / sp);
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
Schedule solve2(int max_parallel, int min_bs_dfs, int reverse_mode, int order_mode, int move_mode) {
  deque<int> q_user = q_user_ori;
  Schedule res(M);
  int timeout_cnt = 0;
  uniform_real_distribution<float> dist(0.0, 1.0);
  
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
      // for(int i = 0; i < bs_ratio.size(); ++i) {
      //   if(bs_ratio[i] > 0) {
      //     cerr << "bs_ratio[" << i << "]: " << bs_ratio[i] << endl;
      //   }
      // }
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
    // 通过反向遍历削峰
    if(reverse_mode && !is_late_user) {
      if(reverse_mode == 1 && avg_cnt[u.s] > avg_cnt[u.e]) {
        is_reverse = true;
        reverse_cnt++;
      }
      if(reverse_mode == 2) {
        double p1 = pow(avg_cnt[u.s] * 100, 1);
        double p2 = pow(avg_cnt[u.e] * 100, 1);
        double p = dist(rng);
        if(p < p1 / (p1 + p2)) {
          is_reverse = true;
          reverse_cnt++;
        }
      }
    }
    int bestFinish = is_reverse ? -1 : INT_MAX;
    int bestActualFinish, bestMoveCnt = 0;
    deque<array<int, 4>> bestSchedule;
    vector<array<int, 4>> bestUseTime;
    bool success = false;

    // 遍历顺序
    vector<int> server_order(N);
    iota(server_order.begin(), server_order.end(), 0);
    if(order_mode == 2) reverse(server_order.begin(), server_order.end());
    vector<vi> core_order(N);
    for(int i = 0; i < N; ++i) {
      core_order[i].resize(cores[i]);
      iota(core_order[i].begin(), core_order[i].end(), 0);
      if(order_mode == 2) reverse(core_order[i].begin(), core_order[i].end());
    }
    // 不迁移的并行，会考虑削峰
    for(int i : server_order) {
      // if (idx % N != i) continue;
      int lat = latency[i][u.id];
      int current_cnt = u.cnt;
      for(int j : core_order[i]) {
        current_cnt = u.cnt;
        deque<array<int, 4>> schedule;
        vector<array<int, 4>> use_time;  // freeAt的最后两个维度
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
          // 最后的余数可以用更小的bs
          if(current_cnt < bs_plans[i].batch_size[min_bs_idx]) {
            min_bs_idx = lower_bound(bs_plans[i].batch_size.begin(), bs_plans[i].batch_size.end(), current_cnt) - bs_plans[i].batch_size.begin();
          }
          for(int k = min_bs_idx; k < bs_plans[i].batch_size.size(); ++k) {
            if(t % bs_plans[i].time[k] == 0) {
              if(is_reverse && t + bs_plans[i].time[k] > u.e) break;
              int time_idx = t / bs_plans[i].time[k];
              if(freeAt[i][j][k][time_idx]) {
                if(is_reverse)
                  schedule.push_front({t - lat, i + 1, j + 1, min(bs_plans[i].batch_size[k], current_cnt)});
                else 
                  schedule.push_back({t - lat, i + 1, j + 1, min(bs_plans[i].batch_size[k], current_cnt)});
                use_time.push_back({i, j, k, time_idx});
                if(is_reverse && schedule.size() == 1)
                  actual_finish = t + bs_plans[i].time[k];
                // finish = is_reverse ? t : t + bs_plans[i].time[k];
                finish = is_reverse ? t : t + get_process_time(min(bs_plans[i].batch_size[k], current_cnt), speedCoef[i]);
                current_cnt -= bs_plans[i].batch_size[k];
                int plus_time = lat;
                t = is_reverse ? t - plus_time : t + plus_time;
                break;
              }
            }
          }
        }
        if(current_cnt > 0) continue;
        if(is_reverse ? finish > bestFinish : (finish < bestFinish && (is_late_user || finish <= u.e))) {
          success = true;
          bestFinish = finish;
          bestActualFinish = is_reverse ? actual_finish : finish;
          bestSchedule = move(schedule);
          bestUseTime = move(use_time);
          if(order_mode > 0) break;
        }
      }
      if(order_mode > 0 && success) break;
    }

    // 允许迁移的并行，解决超时的用户，采用迁移冷却期来使迁移次数尽量小
    if(!success && move_mode == 1) {
      for(int exponent = 4; exponent >= 0; --exponent) {
        int cooling = 1 << exponent;
        bestActualFinish = u.s;
        bestMoveCnt = 0;
        bestSchedule.clear();
        bestUseTime.clear();
        int current_cnt = u.cnt;
        pair<int, int> last_npu = {-1, -1};
        int cooling_time_end = 0;
        for(int time = u.s; time - MIN_LATENCY < u.e; ++time) {
          if(current_cnt <= 0) {
            success = true;
            // cerr << " cooling: " << cooling << " success" << endl;
            break;
          }
          array<int, 4> schedule_item = {INT_MAX, -1, -1, 0};
          array<int, 4> use_time_item;
          for(int i = 0; i < N; ++i) {
            if(i != last_npu.first && time < cooling_time_end) continue;
            int lat = latency[i][u.id];
            size_t min_bs_idx = lower_bound(bs_plans[i].batch_size.begin(), bs_plans[i].batch_size.end(), min_bs[u.id]) - bs_plans[i].batch_size.begin();
            if(min_bs_idx == bs_plans[i].batch_size.size()) continue;
            for(int j = 0; j < cores[i]; ++j) {
              if(i != last_npu.first && time < cooling_time_end) continue;
              for(int k = min_bs_idx; k < bs_plans[i].batch_size.size(); ++k) {
                int t = time + lat;
                while(t % bs_plans[i].time[k] != 0) ++t;
                int time_idx = t / bs_plans[i].time[k];
                while(!freeAt[i][j][k][time_idx] && time_idx * bs_plans[i].time[k] < u.e) 
                  ++time_idx;
                t = time_idx * bs_plans[i].time[k];
                if(freeAt[i][j][k][time_idx]) {
                  if(t - lat < schedule_item[0]) {
                    schedule_item = {t - lat, i + 1, j + 1, min(bs_plans[i].batch_size[k], current_cnt)};
                    use_time_item = {i, j, k, time_idx};
                  }
                }
              }
            }
          }
          if(schedule_item[0] >= u.e) break;
          int server = schedule_item[1] - 1;
          int npu = schedule_item[2] - 1;
          int lat = latency[server][u.id];
          bestActualFinish = max(bestActualFinish, schedule_item[0] + lat + get_process_time(schedule_item[3], speedCoef[server]));
          bestSchedule.push_back(schedule_item);
          bestUseTime.push_back(use_time_item);
          current_cnt -= schedule_item[3];
          time = schedule_item[0] + lat;
          if(last_npu != make_pair(server, npu)) {
            if(last_npu.first != -1) {
              cooling_time_end = schedule_item[0] + cooling * lat;
              bestMoveCnt += 1;
            }
            last_npu = make_pair(server, npu);
          }
        }
        if(success) break;
      }
    }

    assert(!is_late_user || success);
    if(!is_late_user && !success) {
      postponed.push_back(idx);
      continue;
    }

    res.score += h_func(double(bestActualFinish - u.e) / (u.e - u.s)) * p_func(bestMoveCnt);
    if (bestActualFinish > u.e) {
      timeout_cnt++;
    }

    res.T_out[idx] = bestSchedule.size();
    res.schedule[idx].reserve(bestSchedule.size());
    for(auto& p : bestSchedule) {
      res.schedule[idx].push_back(p);
    }
    for(auto& p : bestUseTime) {
      freeAt[p[0]][p[1]][p[2]][p[3]] = false;
    }
  }
  // cerr << "reverse_cnt: " << reverse_cnt << endl;
  res.timeout_rate = (double)timeout_cnt / M;
  res.score *= h_func(timeout_cnt) * 10000;
  return res;
}

// 添加超参搜索辅助结构
struct HyperParams {
  int max_parallel;
  int min_bs_dfs;
  int reverse_mode;
  int order_mode;
  int move_mode;
};

using HPVec = vector<int>;
struct VecHash {
  size_t operator()(HPVec const &v) const noexcept {
    size_t h = 0;
    for (int x : v) h = h * 31 + hash<int>()(x);
    return h;
  }
};
struct VecEq {
  bool operator()(HPVec const &a, HPVec const &b) const noexcept {
    return a == b;
  }
};

// Hyperparameter search state
unordered_map<double, vector<HPVec>> score2params;
unordered_set<HPVec, VecHash, VecEq> seen;

// Top N method2 parameters tracking
vector<pair<double, HPVec>> topN_method2;

// Function to update top N method2 parameters
void update_topN_method2(double score, const HPVec& hp) {
  // Check if this parameter set already exists
  for(auto& p : topN_method2) {
    if(p.second == hp) {
      if(score > p.first) {
        p.first = score; // Update score if better
      }
      return;
    }
  }
  
  // Add new parameter set
  topN_method2.push_back({score, hp});
  
  // Sort by score (descending) and keep only top N
  sort(topN_method2.begin(), topN_method2.end(), [](const pair<double, HPVec>& a, const pair<double, HPVec>& b) {
    return a.first > b.first;
  });
  
  if(topN_method2.size() > TOP_N_METHOD2) {
    topN_method2.resize(TOP_N_METHOD2);
  }
}

// 程序经过的时间
int get_duration(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

Schedule solve() {
  // ========== 超参数搜索 ==========
  using namespace std::chrono;
  const double TIME_LIMIT = HP_TIME_LIMIT;
  int totalCombos = 1;
  for(const auto &vals : HPARAM_VALUES) totalCombos *= vals.size();
  // 初始化搜索集合
  vector<HPVec> init_hp;
  for(auto &m : methods) {
    if(m.METHOD == 2)
      init_hp.push_back({m.max_parallel, m.min_bs, m.reverse_mode, m.order_mode, m.move_mode});
  }
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double bestScore2 = -1;
  HPVec bestHP2;
  Schedule bestSol2;
  for(auto &hp : init_hp) {
    if(get_duration(t0) >= TIME_LIMIT) break;
    Schedule sol = solve2(hp[0], hp[1], hp[2], hp[3], hp[4]);
    double s = sol.score;
    score2params[s].push_back(hp);
    seen.insert(hp);
    update_topN_method2(s, hp);
    if(s > bestScore2) { bestScore2 = s; bestHP2 = hp; bestSol2 = sol; }
  }
  while(HP_SEARCH && get_duration(t0) < TIME_LIMIT && (int)seen.size() < totalCombos) {
    // 局部爬山优化
    bool improved = true;
    while(improved && get_duration(t0) < TIME_LIMIT) {
      improved = false;
      for(int pi = 0; pi < (int)HPARAM_VALUES.size(); ++pi) {
        for(int val : HPARAM_VALUES[pi]) {
          if(val == bestHP2[pi]) continue;
          auto cand = bestHP2;
          cand[pi] = val;
          if(seen.count(cand)) continue;
          if(get_duration(t0) >= TIME_LIMIT) break;
          Schedule sol = solve2(cand[0], cand[1], cand[2], cand[3], cand[4]);
          double s = sol.score;
          score2params[s].push_back(cand);
          seen.insert(cand);
          update_topN_method2(s, cand);
          if(s > bestScore2) {
            bestScore2 = s;
            bestHP2 = cand;
            bestSol2 = sol;
            improved = true;
            break;
          }
        }
        if(improved || get_duration(t0) >= TIME_LIMIT) break;
      }
    }
    if(!RANDOM_SEARCH)
      break;
    // 轮盘赌 + 交叉 + 变异
    uniform_real_distribution<double> prob01(0.0,1.0);
    while(get_duration(t0) < TIME_LIMIT && (int)seen.size() < totalCombos) {
      // 构建轮盘赌分布
      vector<pair<double,HPVec>> pool; pool.reserve(score2params.size());
      double sumScore = 0.0;
      for(auto &kv : score2params) {
        double sc = kv.first;
        const auto &vec = kv.second;
        if(vec.empty()) continue;
        HPVec hp = vec[rng() % vec.size()];
        pool.push_back({sc, hp});
        sumScore += sc;
      }
      if(pool.size() < 2 || sumScore <= 0) break;

      auto roulette_pick = [&](const HPVec &avoid)->HPVec {
        while(true) {
          double r = prob01(rng) * sumScore;
          double running = 0.0;
          for(auto &p : pool) {
            running += p.first;
            if(running >= r) {
              if(p.second != avoid) return p.second;
              else break;
            }
          }
        }
      };

      HPVec parent1 = roulette_pick({});
      HPVec parent2 = roulette_pick(parent1);
      for(int childIdx=0; childIdx<4; ++childIdx) {
        if(get_duration(t0) >= TIME_LIMIT) break;
        HPVec child = parent1;
        for(int k=0; k<(int)HPARAM_VALUES.size(); ++k) {
          if(rng() & 1) child[k] = parent2[k];
        }
        // 0.05 概率变异
        if(prob01(rng) < 0.05) {
          int paramIdx = rng() % (int)HPARAM_VALUES.size();
          const auto &vals = HPARAM_VALUES[paramIdx];
          if(vals.size()>1) {
            int newVal;
            do {
              newVal = vals[rng()%vals.size()];
            } while(newVal == child[paramIdx]);
            child[paramIdx] = newVal;
          }
        }
        if(seen.count(child)) continue;
        Schedule sol = solve2(child[0], child[1], child[2], child[3], child[4]);
        double s = sol.score;
        score2params[s].push_back(child);
        seen.insert(child);
        update_topN_method2(s, child);
        if(s > bestScore2) {
          bestScore2 = s;
          bestHP2 = child;
          bestSol2 = sol;
          // 回到爬山步骤
          improved = true;
          break;
        }
      }
      if(improved) {
        // 重新进入爬山搜索
        break;  // 跳出遗传循环，返回外层重新执行爬山优化
      }
    }
  }

  // 最终方案
  Schedule sol2, sol1;
  double score2Final = -1, score1 = -1;
  
  sol2 = bestSol2;
  score2Final = bestScore2;
  
  // <500w好像没必要
  sol1 = solve1(methods[0].POSTPONE, methods[0].IMMEDIATE, methods[0].BEST_BS);
  score1 = sol1.score;
  
  Schedule finalSol;
  if(score2Final > score1) {
    finalSol = sol2;
  } else {
    finalSol = sol1;
    if(log_method)
      cerr << "method1: (" << (int)sol1.score << ")" << endl;
  }
  return finalSol;
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
    assert(5 * users[i].cnt <= users[i].duration);
    avg_rate += double(users[i].duration) / users[i].cnt;
    start_sum += users[i].s;
    cnt_sum += users[i].cnt;
    min_cnt_dbg = min(min_cnt_dbg, users[i].cnt);
    // users[i].weight = users[i].duration * users[i].cnt;
    // users[i].weight = users[i].s;
    users[i].weight = users[i].cnt;
    // users[i].weight = double(users[i].cnt) / users[i].duration;
  }
  avg_rate /= M;
  // cerr << "avg_rate: " << avg_rate << endl;

  latency.assign(N, vector<int>(M)); // latency[server][user]
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      cin >> latency[i][j];
      max_latency = max(max_latency, latency[i][j]);
      min_latency = min(min_latency, latency[i][j]);
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

  // 削峰预处理
  get_avg_cnt();

  // 超参搜索
  auto schedule = solve().schedule;

  // 观测线上数据
  // assert(npu_num > 1);  // 初赛线上有 npu_num = 1 的情况, 没有 npu_num = 1 and speedCoef[0] = 1 的情况
  // assert(computing_power > 2);  // 初赛线上有 computing_power = 2 的情况
  double avg_cnt = (double)cnt_sum / M;
  // assert(avg_cnt <= 5998); // 初赛线上存在 avg_cnt < 1500 和 avg_cnt > 5998 (computing_power > 3) 的情况
  double avg_start = (double)start_sum / M;
  double variance = 0; // 开始时间的平均L1距离
  for (int i = 0; i < M; i++)
    variance += abs(users[i].s - avg_start);
  variance /= M;
  // cerr << "variance: " << variance << endl;
  // assert(variance >= 1 || M < 100);  // 初赛线上存在 variance < 1 的情况
  // for (int i = 0; i < N; i++) assert(memSize[i] > 1000); // 初赛线上存在 memSize[i] = 1000 的情况
  
  // assert(best_score < 4000000 || best_score > 5000000); // 存在<200w、400w-500w，不存在200w-400w
  // if(best_score < 2000000)  // <200w时，存在npu_num=1、6，不存在其他
  //   // 存在npu_num=6，computing_power=12，a=20且b=200且max_mem_size=1000且variance<1且avg_rate=5，同时不存在cnt最小值>1000或avg_cnt>4000或M<400或min_latency==20
  //   if(npu_num == 6 && computing_power == 12 && A == 20 && B == 200 && max_mem_size == 1000 && variance < 1 && avg_rate < 5+1e-5) {
  //     assert(0);
  //   }

  // 输出
  for (int i = 0; i < M; i++) {
    assert(schedule[i].size() > 0);
    cout << schedule[i].size() << "\n";
    for (auto &job : schedule[i]) {
      cout << job[0] << ' ' << job[1] << ' ' << job[2] << ' ' << job[3] << ' ';
    }
    cout << "\n";
  }
  
  // Print top N method2 parameters
  if(log_method) {
    cerr << "Top" << TOP_N_METHOD2 << " method2: ";
    for(int i = 0; i < topN_method2.size(); ++i) {
      if(i > 0) cerr << " | ";
      const auto& hp = topN_method2[i].second;
      cerr << "(" << hp[0] << "," << hp[1] << "," << hp[2] << "," << hp[3] << "," << hp[4] << ":" << (int)topN_method2[i].first << ")";
    }
    cerr << endl;
  }
  
  // cerr << "timeout_rate: " << results[best_idx].timeout_rate * 100 << "%" << " score: " << (int)results[best_idx].score << endl;
  return 0;
}