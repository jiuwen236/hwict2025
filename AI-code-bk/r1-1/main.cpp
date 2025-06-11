#include <iostream>
#include <vector>
#include <queue>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <iomanip>

using namespace std;

// 禁用迁移；每次请求的大小尽量装满NPU显存
// 用户按(e_i-s_i)*cnt_i排序，最小的优先；在有更高优先级用户使用NPU时，低优先级的用户不能抢占
// 最优选择：用户发送请求时会计算对每个NPU的实际推理结束时间，选择最早结束的。

struct NPUState {
    int current_time;
    int used_mem;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> processing;
    vector<tuple<int, int, int>> waiting_queue;
    int total_mem;
    int k;
    int server_type;
    int npu_index;
};

struct Request {
    int send_time;
    int server;
    int npu;
    int batchsize;
};

struct User {
    int s, e, cnt;
    int id;
    vector<int> latencies;
    vector<Request> requests;
    double priority;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);

    int N;
    cin >> N;
    vector<int> server_g(N), server_k(N), server_m(N);
    for (int i = 0; i < N; i++) {
        cin >> server_g[i] >> server_k[i] >> server_m[i];
    }

    int M;
    cin >> M;
    vector<User> users(M);
    for (int i = 0; i < M; i++) {
        cin >> users[i].s >> users[i].e >> users[i].cnt;
        users[i].id = i;
        users[i].priority = (users[i].e - users[i].s) * (double)users[i].cnt;
    }

    vector<vector<int>> latency_matrix(N, vector<int>(M));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cin >> latency_matrix[i][j];
        }
    }

    int a, b;
    cin >> a >> b;

    for (int i = 0; i < M; i++) {
        users[i].latencies.resize(N);
        for (int j = 0; j < N; j++) {
            users[i].latencies[j] = latency_matrix[j][i];
        }
    }

    sort(users.begin(), users.end(), [](const User& u1, const User& u2) {
        return u1.priority < u2.priority;
    });

    vector<NPUState> npuStates;
    unordered_map<int, int> npu_state_index_map;
    for (int server_type = 0; server_type < N; server_type++) {
        for (int j = 0; j < server_g[server_type]; j++) {
            NPUState state;
            state.current_time = 0;
            state.used_mem = 0;
            state.total_mem = server_m[server_type];
            state.k = server_k[server_type];
            state.server_type = server_type;
            state.npu_index = j;
            npuStates.push_back(state);
            int idx = npuStates.size() - 1;
        }
    }
    int total_npus = npuStates.size();

    vector<User> scheduled_users = users;
    for (int u_idx = 0; u_idx < scheduled_users.size(); u_idx++) {
        User& user = scheduled_users[u_idx];
        int remaining = user.cnt;
        long long next_send_time = user.s;
        int last_server = -1;
        int last_npu = -1;
        int last_latency = 0;

        while (remaining > 0) {
            int max_batch = min(remaining, 1000);
            double best_finish_time = 1e18;
            int best_batch = 0;
            int best_server = -1;
            int best_npu = -1;
            int best_send_time = -1;
            int best_npu_state_index = -1;
            NPUState best_state_copy;

            for (int ns_index = 0; ns_index < total_npus; ns_index++) {
                NPUState npu = npuStates[ns_index];
                int maxB = min(max_batch, (npu.total_mem - b) / a);
                if (maxB <= 0) continue;
                int batch = maxB;
                long long send_time = next_send_time;
                long long arrival_time = send_time + user.latencies[npu.server_type];

                if (npu.current_time > arrival_time) {
                    continue;
                }

                while (!npu.processing.empty() && npu.processing.top().first <= arrival_time) {
                    int finish_time = npu.processing.top().first;
                    int released_mem = 0;
                    while (!npu.processing.empty() && npu.processing.top().first == finish_time) {
                        released_mem += npu.processing.top().second;
                        npu.processing.pop();
                    }
                    npu.used_mem -= released_mem;
                    npu.current_time = finish_time;

                    sort(npu.waiting_queue.begin(), npu.waiting_queue.end(), [](const tuple<int, int, int>& t1, const tuple<int, int, int>& t2) {
                        if (get<0>(t1) != get<0>(t2)) {
                            return get<0>(t1) < get<0>(t2);
                        }
                        return get<1>(t1) < get<1>(t2);
                    });

                    vector<tuple<int, int, int>> new_waiting;
                    for (auto& req : npu.waiting_queue) {
                        int arr_time = get<0>(req);
                        int uid = get<1>(req);
                        int bs = get<2>(req);
                        int mem_required = a * bs + b;
                        if (npu.used_mem + mem_required <= npu.total_mem) {
                            npu.used_mem += mem_required;
                            double time_cost = (double)bs / (npu.k * sqrt(bs));
                            int processing_time = ceil(time_cost);
                            int finish = npu.current_time + processing_time;
                            npu.processing.push({finish, mem_required});
                        } else {
                            new_waiting.push_back(req);
                        }
                    }
                    npu.waiting_queue = new_waiting;
                }

                if (npu.current_time < arrival_time) {
                    npu.current_time = arrival_time;
                }

                npu.waiting_queue.push_back({arrival_time, user.id, batch});
                sort(npu.waiting_queue.begin(), npu.waiting_queue.end(), [](const tuple<int, int, int>& t1, const tuple<int, int, int>& t2) {
                    if (get<0>(t1) != get<0>(t2)) {
                        return get<0>(t1) < get<0>(t2);
                    }
                    return get<1>(t1) < get<1>(t2);
                });

                bool allocated = false;
                int start_time = -1;
                double finish_time_val = 1e18;
                vector<tuple<int, int, int>> new_waiting;
                for (auto it = npu.waiting_queue.begin(); it != npu.waiting_queue.end(); ) {
                    int arr_time = get<0>(*it);
                    int uid = get<1>(*it);
                    int bs = get<2>(*it);
                    int mem_required = a * bs + b;
                    if (npu.used_mem + mem_required <= npu.total_mem) {
                        npu.used_mem += mem_required;
                        double time_cost = (double)bs / (npu.k * sqrt(bs));
                        int processing_time = ceil(time_cost);
                        int finish = npu.current_time + processing_time;
                        npu.processing.push({finish, mem_required});
                        if (arr_time == arrival_time && uid == user.id && bs == batch) {
                            allocated = true;
                            start_time = npu.current_time;
                            finish_time_val = finish;
                        }
                        it = npu.waiting_queue.erase(it);
                    } else {
                        new_waiting.push_back(*it);
                        it++;
                    }
                }
                for (auto& req : new_waiting) {
                    npu.waiting_queue.push_back(req);
                }
                new_waiting.clear();

                if (!allocated) {
                    while (!allocated && !npu.processing.empty()) {
                        int finish_time = npu.processing.top().first;
                        npu.current_time = finish_time;
                        int released_mem = 0;
                        while (!npu.processing.empty() && npu.processing.top().first == finish_time) {
                            released_mem += npu.processing.top().second;
                            npu.processing.pop();
                        }
                        npu.used_mem -= released_mem;

                        sort(npu.waiting_queue.begin(), npu.waiting_queue.end(), [](const auto& t1, const auto& t2) {
                            if (get<0>(t1) != get<0>(t2)) return get<0>(t1) < get<0>(t2);
                            return get<1>(t1) < get<1>(t2);
                        });

                        new_waiting.clear();
                        for (auto it = npu.waiting_queue.begin(); it != npu.waiting_queue.end(); ) {
                            int arr_time = get<0>(*it);
                            int uid = get<1>(*it);
                            int bs = get<2>(*it);
                            int mem_required = a * bs + b;
                            if (npu.used_mem + mem_required <= npu.total_mem) {
                                npu.used_mem += mem_required;
                                double time_cost = (double)bs / (npu.k * sqrt(bs));
                                int processing_time = ceil(time_cost);
                                int finish = npu.current_time + processing_time;
                                npu.processing.push({finish, mem_required});
                                if (arr_time == arrival_time && uid == user.id && bs == batch) {
                                    allocated = true;
                                    start_time = npu.current_time;
                                    finish_time_val = finish;
                                }
                                it = npu.waiting_queue.erase(it);
                            } else {
                                new_waiting.push_back(*it);
                                it++;
                            }
                        }
                        for (auto& req : new_waiting) {
                            npu.waiting_queue.push_back(req);
                        }
                        new_waiting.clear();
                    }
                }

                if (allocated && finish_time_val < best_finish_time) {
                    best_finish_time = finish_time_val;
                    best_batch = batch;
                    best_server = npu.server_type;
                    best_npu = npu.npu_index;
                    best_send_time = send_time;
                    best_npu_state_index = ns_index;
                    best_state_copy = npu;
                }
            }

            if (best_server == -1) {
                best_batch = min(remaining, 1000);
                int maxB = 0;
                for (int ns_index = 0; ns_index < total_npus; ns_index++) {
                    maxB = min(best_batch, (npuStates[ns_index].total_mem - b) / a);
                    if (maxB > 0) {
                        best_batch = maxB;
                        best_server = npuStates[ns_index].server_type;
                        best_npu = npuStates[ns_index].npu_index;
                        best_send_time = next_send_time;
                        best_npu_state_index = ns_index;
                        break;
                    }
                }
                if (best_server == -1) {
                    best_batch = 1;
                    for (int ns_index = 0; ns_index < total_npus; ns_index++) {
                        if ((npuStates[ns_index].total_mem - b) / a >= 1) {
                            best_server = npuStates[ns_index].server_type;
                            best_npu = npuStates[ns_index].npu_index;
                            best_send_time = next_send_time;
                            best_npu_state_index = ns_index;
                            break;
                        }
                    }
                }
                best_state_copy = npuStates[best_npu_state_index];
            }

            user.requests.push_back({best_send_time, best_server+1, best_npu+1, best_batch});
            npuStates[best_npu_state_index] = best_state_copy;
            remaining -= best_batch;
            last_latency = user.latencies[best_server];
            next_send_time = best_send_time + last_latency + 1;
        }
    }

    sort(scheduled_users.begin(), scheduled_users.end(), [](const User& u1, const User& u2) {
        return u1.id < u2.id;
    });

    for (int i = 0; i < scheduled_users.size(); i++) {
        User& user = scheduled_users[i];
        cout << user.requests.size() << "\n";
        for (int j = 0; j < user.requests.size(); j++) {
            Request& req = user.requests[j];
            if (j > 0) {
                cout << " ";
            }
            cout << req.send_time << " " << req.server << " " << req.npu << " " << req.batchsize;
        }
        cout << "\n";
    }

    return 0;
}