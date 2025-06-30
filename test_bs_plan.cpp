#include <bits/stdc++.h>
using namespace std;

// g++ -std=c++17 -O2 test_bs_plan.cpp -o test_bs_plan && ./test_bs_plan
// main.cpp如有修改，记得复制过来
int k = 1, m = 2000, a = 10, b = 100;
int top_k = 4;
const int MAX_PARALLEL = 2;
const int MIN_BS = 5;
vector<double> bs_ratio(50);
void init_bs_ratio() {
    // bs_ratio[20] = 0.1;
    // bs_ratio[40] = 1;
}

using ll = long long;
using vi = vector<int>;

struct BS_Plan {
  vi batch_size;  // 有序
  vi time;        // batch_size对应消耗的时间
  int loop_time;  // time的最小公倍数
  double throughput; // 吞吐量(每毫秒完成请求的batchsize的和)
  int left_mem; // 剩余显存
};

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

// 背包DP，返回每个显存下最优的一个batch size方案，按吞吐量降序排序 (AI生成)
// 缺点：无法满足bs比例要求
vector<BS_Plan> get_bs_plan_dp(int k, int m, int a, int b) {
    // 枚举所有可能的批大小，计算对应时间和吞吐量贡献
    int maxBS = (m - b) / a;
    vector<int> bsList, timeList, weightList;
    vector<double> profitList;
    for (int bs = 1; bs <= maxBS; ++bs) {
        int mem = a * bs + b;
        if (mem > m) break;
        int t = (int)ceil(sqrt((double)bs) / k);
        double p = (double)bs / t;
        bsList.push_back(bs);
        timeList.push_back(t);
        weightList.push_back(mem);
        profitList.push_back(p);
    }

    int cap = m, n = bsList.size();
    vector<double> dp(cap + 1, -1e18);
    vector<int> prev(cap + 1, -1), last_item(cap + 1, -1);
    dp[0] = 0.0;
    for (int i = 0; i < n; ++i) {
        int w = weightList[i];
        double p = profitList[i];
        for (int wgt = w; wgt <= cap; ++wgt) {
            if (dp[wgt - w] + p > dp[wgt]) {
                dp[wgt] = dp[wgt - w] + p;
                prev[wgt] = wgt - w;
                last_item[wgt] = i;
            }
        }
    }

    vector<BS_Plan> plans;
    // 对每个可达到的内存容量，重建对应的最佳方案
    for (int wgt = 1; wgt <= cap; ++wgt) {
        if (last_item[wgt] < 0) continue;
        vector<int> cnt(n, 0);
        int cur = wgt;
        while (cur > 0) {
            int idx = last_item[cur];
            cnt[idx]++;
            cur = prev[cur];
        }
        BS_Plan plan;
        for (int i = 0; i < n; ++i) {
            for (int c = 0; c < cnt[i]; ++c) {
                plan.batch_size.push_back(bsList[i]);
                plan.time.push_back(timeList[i]);
            }
        }
        // 将batch_size和time按batch_size升序排列
        vector<pair<int,int>> vp;
        for (size_t i = 0; i < plan.batch_size.size(); ++i) {
            vp.emplace_back(plan.batch_size[i], plan.time[i]);
        }
        sort(vp.begin(), vp.end());
        plan.batch_size.clear();
        plan.time.clear();
        for (auto &pr : vp) {
            plan.batch_size.push_back(pr.first);
            plan.time.push_back(pr.second);
        }
        // 计算循环周期和吞吐量
        int loop = 1;
        for (int t : plan.time) {
            loop = loop / gcd(loop, t) * t;
        }
        plan.loop_time = loop;
        plan.throughput = dp[wgt];
        plan.left_mem = m - wgt;
        plans.push_back(plan);
    }
    // 按吞吐量降序排序
    sort(plans.begin(), plans.end(), [](const BS_Plan& a, const BS_Plan& b){ return a.throughput > b.throughput; });

    return plans;
}

int main() {
    init_bs_ratio();
    // auto plans = get_bs_plan_dp(k, m, a, b);
    auto plans = get_bs_plan_dfs(k, m, a, b, bs_ratio, MAX_PARALLEL, MIN_BS);
    cout << "共返回 " << plans.size() << " 个方案\n";
    for (int idx = 0; idx < min(top_k, (int)plans.size()); ++idx) {
        const auto &plan = plans[idx];
        cout << "方案 " << idx + 1 << ": 吞吐量=" << plan.throughput << ", loop_time=" << plan.loop_time << ", left_mem=" << plan.left_mem << "\n";
        cout << "  batch sizes:";
        for (int bs : plan.batch_size) cout << ' ' << bs;
        cout << "\n  times:";
        for (int t : plan.time) cout << ' ' << t;
        cout << "\n";
    }
    return 0;
} 