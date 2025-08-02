#include <bits/stdc++.h>
using namespace std;

// 本地评分(linux)执行 ./judge
// 简单运行：g++ main.cpp -o main && ./main < data/n2_g1_k1_a20_b200.in > tmp.out

// Hyperparameter search configuration
const bool HP_SEARCH = 1;
const double HP_TIME_LIMIT = 10; // 单位：秒
const bool RANDOM_SEARCH = 1;  // 关闭时，仅进行爬山法优化
static const vector<vector<int>> HPARAM_VALUES = {
  {1,2,3}, // parallel_num/max_parallel  1好像没必要
  {1,2}, // reverse_mode 1: 进行削峰，2: 按概率决定是否反向
  {0,1,2}, // order_mode  不迁移时如何选择npu：0: 选择最快的，1: 选择第1个合法解，2: 选择最后一个
  {1}    // move_mode  是否迁移
};

const bool use_method1 = 1; // 是否使用方法1
const bool USE_LAST_SEND = 0; // 最后发送的 or 最晚完成的

const bool fusai = 1;
bool log_method = 1;  // 是否打印不同方法分数信息
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
  static Method method2(int max_parallel = 2, int reverse_mode = 1, int order_mode = 0, int move_mode = 0) {
    Method m;
    m.METHOD = 2;
    m.max_parallel = max_parallel;
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
  // Method::method2(1, 0),   // 模拟方法1
  // Method::method2(2, 0),   // 初始并行版本
  Method::method2(2, 1, 1, 1),
  // Method::method2(1, 1, 1, 1), // debug
};

// 非并行
// 1545451
// 非并行 + 并行 (10秒)
// 1363073 (方法2分数不准)

// 是否加入随机性
// mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
mt19937 rng(1);

using ll = long long;
using vi = vector<int>;
using pii = pair<int, int>;

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
vector<int> user_a, user_b;

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
  vector<vi> maxBatch, procTime;
  maxBatch.resize(N);
  procTime.resize(N);
  for (int i = 0; i < N; i++) {
    maxBatch[i].resize(M);
    procTime[i].resize(M);
    for (int j = 0; j < M; j++) {
      int batch_size = (memSize[i] - user_b[j]) / user_a[j];
      assert(batch_size > 0);
      maxBatch[i][j] = batch_size;
      int sp = speedCoef[i];
      procTime[i][j] = (int)ceil(sqrt((double)batch_size) / sp);
      // 非并行的最优 batch size
      double throughput1 = (double)maxBatch[i][j] / procTime[i][j];
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
        maxBatch[i][j] = batch_size;
        procTime[i][j] = proc_time;
      }
    }
  }

  // resize
  vector<vector<vector<bool>>> freeAt;
  vector<vector<vector<int>>> left_mem;
  freeAt.resize(N);
  left_mem.resize(N);
  for (int i = 0; i < N; i++) {
    freeAt[i].resize(cores[i]);
    left_mem[i].resize(cores[i]);
    for (int j = 0; j < cores[i]; j++) {
      freeAt[i][j].resize(1000 * 1000 * 9 / npu_num, true);
      left_mem[i][j].resize(1000 * 1000 * 9 / npu_num, 0);
    }
  }

  deque<int> postponed;
  // 按优先级调度
  while (!q_user.empty() || !postponed.empty()) {
    // 选择普通用户或延迟用户
    int u_id = -1;
    bool is_late_user = false;
    if (!q_user.empty()) {
      u_id = q_user.front();
      q_user.pop_front();
    } else {
      is_late_user = true;
      u_id = postponed.front();
      postponed.pop_front();
    }
    auto &u = users[u_id];
    ll bestFinish = LLONG_MAX;
    int bestSrv = -1, bestNpu = -1, bestB = 0, bestBnum = 0, bestLat = 0;
    vi bestStartTime, bestUseTime;
    vi bestLeftMem, bestLeftMemTime;

    // 尝试每台服务器
    for (int i = 0; i < N; i++) {
      int B = maxBatch[i][u.id];
      int B_num = (u.cnt + B - 1) / B; // 批次数
      // 尝试每台服务器的每个 NPU
      for (int x = 0; x < cores[i]; x++) {
        int lat = latency[i][u.id];
        // 预测完成时间
        int start = u.s + lat;
        int p = procTime[i][u.id];
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
      postponed.push_back(u_id);
      continue;
    }
    // 记录该用户的发送方案
    res.T_out[u_id] = bestBnum;
    res.schedule[u_id].reserve(bestBnum);
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
      res.schedule[u_id].push_back({tm, bestSrv + 1, bestNpu + 1, batch});
      rem -= batch;
    }
    res.score += h_func(double(bestFinish - u.e) / (u.e - u.s));
    if (bestFinish > u.e) {
      timeout_cnt++;
      // if (u_id < 10) cerr << "timeout user: " << u_id << " " << bestFinish << " " << u.e << endl;
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

pii get_bs(int memory, int a, int b, int sp) {
  int bs = (memory - b) / a;
  double time = sqrt((double)bs) / sp;
  int procTime = (int)ceil(time);
  double throughput1 = (double)bs / procTime;
  int int_time = (int)floor(time);
  if(int_time == 0) {
    return {bs, procTime};
  }
  int bs2 = int_time * int_time * sp * sp;
  double throughput2 = (double)bs2 / int_time;
  if(throughput1 > throughput2) {
    return {bs, procTime};
  } else {
    return {bs2, int_time};
  }
}

// 方法2，允许 npu 并行处理多个用户请求
Schedule solve2(int parallel_num, int reverse_mode, int order_mode, int move_mode) {
  deque<int> q_user = q_user_ori;
  Schedule res(M);
  int timeout_cnt = 0;
  uniform_real_distribution<float> dist(0.0, 1.0);
  vector<int> debug_finish(M);

  // 占用情况初始化
  vector<vector<vector<uint16_t>>> freeAt(N);
  for(size_t i = 0; i < N; ++i) {
    freeAt[i].resize(cores[i]);
    for(size_t j = 0; j < cores[i]; ++j) {
      freeAt[i][j].resize(1000 * 1000 * 27 / npu_num, memSize[i]);
    }
  }

  // 逐个添加用户
  deque<int> postponed;
  while (!q_user.empty() || !postponed.empty()) {
    // 选择普通用户或延迟用户
    int u_id = -1;
    bool is_late_user = false;
    if (!q_user.empty()) {
      u_id = q_user.front();
      q_user.pop_front();
    } else {
      is_late_user = true;
      u_id = postponed.front();
      postponed.pop_front();
    }
    auto &u = users[u_id];
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
    int bestActualFinish = -2, bestMoveCnt = 0;
    deque<array<int, 4>> bestSchedule;
    vector<array<int, 5>> bestUseTime;
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
      // if (u_id % N != i) continue;
      int lat = latency[i][u.id];
      int current_cnt = u.cnt;
      for(int j : core_order[i]) {
        current_cnt = u.cnt;
        deque<array<int, 4>> schedule;
        vector<array<int, 5>> use_time; 
        int start = u.s + lat;
        int finish = is_reverse ? INT_MAX : -1;
        int actual_finish = -1;
        int end_time = is_late_user ? INT_MAX : u.e;
        // 逐毫秒检查是否可以发送请求
        for(int t = is_reverse ? u.e - 1 : start; is_reverse ? t >= start : t <= end_time; is_reverse ? --t : ++t) {
          if(freeAt[i][j][t] < user_b[u.id] + user_a[u.id]) continue;
          if(current_cnt <= 0) break;
          if(t == 200000) 
            bool debug = true;
          // 决策batch_size
          bool valid = true;
          int loop_cnt = 3, bs, proc_time, space;
          while(loop_cnt--) {
            int mem_size = -1;
            if(loop_cnt == 2) 
              mem_size = min(memSize[i] / parallel_num + memSize[i] / 10, (int)freeAt[i][j][t]);  // 可调参
            else if(loop_cnt == 1) 
              mem_size = min(int(memSize[i] / parallel_num * 1.3 + memSize[i] * 0.15), (int)freeAt[i][j][t]);
            else
              mem_size = (int)freeAt[i][j][t];
            auto [bs_tmp, proc_time_tmp] = get_bs(mem_size, user_a[u.id], user_b[u.id], speedCoef[i]);
            bs = bs_tmp;
            proc_time = proc_time_tmp;
            int minus_num = 1; // 0: 是最后一个请求 1: 正常
            if(current_cnt <= bs) {
              bs = current_cnt;
              proc_time = get_process_time(bs, speedCoef[i]);
              minus_num = 0;
            }
            if(double(u.cnt - current_cnt + bs) / (schedule.size() + 1) * (MAX_T_NUM - minus_num) < u.cnt) {
              if(loop_cnt >= 1) continue;
              if(loop_cnt == 0) {
                valid = false;
                break;
              }
            }
            if(!valid) continue;
            // 探测后续是否有空间
            bool has_space = false;
            if(proc_time == 1) {
              has_space = true;
            }
            space = bs * user_a[u.id] + user_b[u.id];
            while(!has_space) {
              if(!is_reverse) {
                int tmp_t = t + 1;
                for(; tmp_t < t + proc_time; ++tmp_t) {
                  if(freeAt[i][j][tmp_t] < space) {
                    break;
                  }
                }
                if(tmp_t == t + proc_time) {
                  has_space = true;
                }
              } else {
                int tmp_t = t - 1;
                for(; tmp_t > t - proc_time; --tmp_t) {
                  if(freeAt[i][j][tmp_t] < space) {
                    break;
                  }
                }
                if(tmp_t == t - proc_time) {
                  has_space = true;
                }
              }
              if(!has_space) {
                proc_time--;
                if(proc_time == 1) has_space = true;
                bs = proc_time * proc_time * speedCoef[i] * speedCoef[i];
                space = bs * user_a[u.id] + user_b[u.id];
                minus_num = 1;
              }
            }
            if(double(u.cnt - current_cnt + bs) / (schedule.size() + 1) * (MAX_T_NUM - minus_num) < u.cnt) {
              if(loop_cnt >= 1) continue;
              if(loop_cnt == 0) {
                valid = false;
                break;
              }
            }
            break;
          }
          if(!valid) continue;
          // 记录下来
          if(!is_reverse){
            schedule.push_back({t - lat, i + 1, j + 1, bs});
            use_time.push_back({i, j, space, t, t + proc_time});
          }
          else {
            if(t - proc_time + 1 - lat < start) break;
            schedule.push_front({t - proc_time + 1 - lat, i + 1, j + 1, bs});
            use_time.push_back({i, j, space, t - proc_time + 1, t + 1});
          }
          for(int k = use_time.back()[3]; k < use_time.back()[4]; ++k) {
            freeAt[i][j][k] -= space;
          }
          if(is_reverse && schedule.size() == 1)
            actual_finish = t + 1;
          if(!is_reverse) {
            if(USE_LAST_SEND)
              finish = t + proc_time;
            else
              finish = max(t + proc_time, finish);
          } else {
            finish = min(t - proc_time + 1, finish);
          }
          current_cnt -= bs;
          int plus_time = lat;
          t = is_reverse ? t - proc_time + 1 - plus_time : t + plus_time;
        }
        for(auto& p : use_time) {
          for(int k = p[3]; k < p[4]; ++k) {
            freeAt[p[0]][p[1]][k] += p[2];
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
      vector<double> cooling_list = {1.5, 1};
      for(double cooling : cooling_list) {
        bestActualFinish = u.s;
        bestMoveCnt = 0;
        bestSchedule.clear();
        for(auto& p : bestUseTime) {
          for(int k = p[3]; k < p[4]; ++k) {
            freeAt[p[0]][p[1]][k] += p[2];
          }
        }
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
          array<int, 5> use_time_item;
          // 找从time起，能最快发出请求的npu
          for(int i = 0; i < N; ++i) {
            if(i != last_npu.first && time < cooling_time_end) continue;
            int lat = latency[i][u.id];
            for(int j = 0; j < cores[i]; ++j) {
              if(i != last_npu.first && time < cooling_time_end) continue;
              int t = time + lat;
              // 逐毫秒检查是否可以发送请求
              for(; t <= u.e; ++t) {
                if(freeAt[i][j][t] < user_b[u.id] + user_a[u.id]) continue;
                int mem_size = (int)freeAt[i][j][t];  // 尽可能大
                auto [bs, proc_time] = get_bs(mem_size, user_a[u.id], user_b[u.id], speedCoef[i]);
                int minus_num = 1;
                if(current_cnt <= bs) {
                  bs = current_cnt;
                  minus_num = 0;
                  proc_time = get_process_time(bs, speedCoef[i]);
                }
                if(double(u.cnt - current_cnt + bs) / (bestSchedule.size() + 1) * (MAX_T_NUM - minus_num) < u.cnt) continue;
                // 探测后续是否有空间
                bool has_space = false;
                if(proc_time == 1) {
                  has_space = true;
                }
                int space = bs * user_a[u.id] + user_b[u.id];
                while(!has_space) {
                  int tmp_t = t + 1;
                  for(; tmp_t < t + proc_time; ++tmp_t) {
                    if(freeAt[i][j][tmp_t] < space) {
                      break;
                    }
                  }
                  if(tmp_t == t + proc_time) {
                    has_space = true;
                  }
                  if(!has_space) {
                    proc_time--;
                    if(proc_time == 1) has_space = true;
                    bs = proc_time * proc_time * speedCoef[i] * speedCoef[i];
                    minus_num = 1;
                    space = bs * user_a[u.id] + user_b[u.id];
                  }
                }
                if(double(u.cnt - current_cnt + bs) / (bestSchedule.size() + 1) * (MAX_T_NUM - minus_num) < u.cnt) continue;
                // 是否更快
                if(t - lat < schedule_item[0]) {
                  schedule_item = {t - lat, i + 1, j + 1, bs};
                  use_time_item = {i, j, space, t, t + proc_time};
                }
                break;
              }
            }
          }
          if(time < cooling_time_end && schedule_item[0] > cooling_time_end + 1) {
            cooling_time_end = time;
            time--;
            continue;
          }
          if(schedule_item[0] >= u.e) break;
          int server = schedule_item[1] - 1;
          int npu = schedule_item[2] - 1;
          int lat = latency[server][u.id];
          if(USE_LAST_SEND)
            bestActualFinish = use_time_item[4];
          else
            bestActualFinish = max(bestActualFinish, use_time_item[4]);
          bestSchedule.push_back(schedule_item);
          bestUseTime.push_back(use_time_item);
          for(int k = use_time_item[3]; k < use_time_item[4]; ++k) {
            freeAt[server][npu][k] -= use_time_item[2];
          }
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
        if(success)
          break;
      }
      for(auto& p : bestUseTime) {
        for(int k = p[3]; k < p[4]; ++k) {
          freeAt[p[0]][p[1]][k] += p[2];
        }
      }
    }

    assert(!is_late_user || success);
    if(!is_late_user && !success) {
      postponed.push_back(u_id);
      continue;
    }

    res.score += h_func(double(bestActualFinish - u.e) / (u.e - u.s)) * p_func(bestMoveCnt);
    if (bestActualFinish > u.e) {
      timeout_cnt++;
    }
    debug_finish[u_id] = bestActualFinish;

    res.T_out[u_id] = bestSchedule.size();
    res.schedule[u_id].reserve(bestSchedule.size());
    for(auto& p : bestSchedule) {
      res.schedule[u_id].push_back(p);
    }
    for(auto& p : bestUseTime) {
      for(int k = p[3]; k < p[4]; ++k) {
        freeAt[p[0]][p[1]][k] -= p[2];
      }
    }
  }
  // cerr << "reverse_cnt: " << reverse_cnt << endl;
  // 打印用户时间
  // for(int i = 0; i < M; ++i) {
  //   int finish_time = debug_finish[i];
  //   // if(finish_time > users[i].e) {
  //     cerr << "user " << i << " finish time: " << finish_time << " " << users[i].e << endl;
  //   // }
  // }
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
      init_hp.push_back({m.max_parallel, m.reverse_mode, m.order_mode, m.move_mode});
  }
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double bestScore2 = -1;
  HPVec bestHP2;
  Schedule bestSol2;
  for(auto &hp : init_hp) {
    if(get_duration(t0) >= TIME_LIMIT) break;
    Schedule sol = solve2(hp[0], hp[1], hp[2], hp[3]);
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
          Schedule sol = solve2(cand[0], cand[1], cand[2], cand[3]);
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
        Schedule sol = solve2(child[0], child[1], child[2], child[3]);
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
  if(use_method1) {
    sol1 = solve1(methods[0].POSTPONE, methods[0].IMMEDIATE, methods[0].BEST_BS);
    score1 = sol1.score;
  }
  
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

  if(!fusai){
    cin >> A >> B;
    user_a.resize(M);
    user_b.resize(M);
    for(int i=0; i<M; ++i) {
      user_a[i] = A;
      user_b[i] = B;
    }
  }
  else {
    user_a.resize(M);
    user_b.resize(M);
    for(int i=0; i<M; ++i) {
      cin >> user_a[i] >> user_b[i];
      A += user_a[i];
      B += user_b[i];
    }
    // A = round(((double)A / M + 20) / 2);
    // B = round(((double)B / M + 200) / 2);
    A = 20;
    B = 200;
  }

  for (int i = 0; i < M; i++) {
    // users[i].weight = users[i].duration * users[i].cnt;
    // users[i].weight = users[i].s;
    // users[i].weight = users[i].cnt;
    // users[i].weight = double(users[i].cnt) / users[i].duration;
    users[i].weight = double(users[i].cnt) * user_a[i] * user_b[i];
  }

  for (int i = 0; i < M; ++i) q_user_ori.push_back(i);
  // Sort user queue by weight
  sort(q_user_ori.begin(), q_user_ori.end(), [&](int x, int y) {
    if (users[x].weight != users[y].weight)
      return users[x].weight < users[y].weight;
    return users[x].s < users[y].s;
  });
  
  for (int i = 0; i < N; i++) {
    npu_num += cores[i];
    computing_power += cores[i] * speedCoef[i];
  }

  // 削峰预处理
  get_avg_cnt();

  // 超参搜索
  auto full_schedule = solve();
  auto schedule = full_schedule.schedule;

  // 观测线上数据
  // assert(npu_num > 1);  // 初赛线上有 npu_num = 1 的情况, 没有 npu_num = 1 and speedCoef[0] = 1 的情况
  // assert(computing_power > 2);  // 初赛线上有 computing_power = 2 的情况
  double avg_cnt = (double)cnt_sum / M;
  // assert(avg_cnt <= 5998); // 初赛线上存在 avg_cnt < 1500 和 avg_cnt > 5998 (computing_power > 3) 的情况。复赛有<=200的
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
      cerr << "(" << hp[0] << "," << hp[1] << "," << hp[2] << "," << hp[3] << ":" << (int)topN_method2[i].first << ")";
    }
    cerr << " total: " << seen.size();
    cerr << endl;
  }
  
  // cerr << "timeout_rate: " << full_schedule.timeout_rate * 100 << "%" << " score: " << (int)full_schedule.score << endl;
  return 0;
}