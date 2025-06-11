#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <climits>

using namespace std;

// 38896496.79  63599203 2025-06-01 21:38:10

struct Server {
    int npu_count;
    int k_factor;
    int memory;
};

struct User {
    int start_time;
    int end_time;
    int sample_count;
    vector<int> latency;
};

struct Task {
    int time;
    int server_id;
    int npu_id;
    int batch_size;
};

int N, M, a, b;
vector<Server> servers;
vector<User> users;

int calc_inference_time(int batch_size, int k_factor) {
    if (batch_size <= 0) return 0;
    double speed = k_factor * sqrt(batch_size);
    return max(1, (int)ceil(batch_size / speed));
}

int get_max_batch_size(int server_id) {
    return min(1000, max(1, (servers[server_id].memory - b) / a));
}

double calc_efficiency(int user_id, int server_id, int batch_size) {
    int latency = users[user_id].latency[server_id];
    int inference_time = calc_inference_time(batch_size, servers[server_id].k_factor);
    int total_time = latency + inference_time;
    
    if (total_time <= 0) return 0;
    return (double)batch_size / total_time;
}

int find_optimal_batch_size(int user_id, int server_id, int remaining_samples, int time_budget) {
    int max_batch = min(remaining_samples, get_max_batch_size(server_id));
    if (max_batch <= 0) return 0;
    
    int latency = users[user_id].latency[server_id];
    if (time_budget <= latency) return 0;
    
    int available_time = time_budget - latency;
    
    int best_batch = 1;
    double best_efficiency = 0;
    
    // Key candidate batch sizes
    vector<int> candidates = {1, 2, 5, 10, 20, 50, 100, 200, 500};
    
    // Add adaptive candidates
    candidates.push_back(remaining_samples);
    candidates.push_back(remaining_samples / 2);
    candidates.push_back(remaining_samples / 5);
    candidates.push_back(remaining_samples / 10);
    candidates.push_back(max_batch);
    
    for (int batch : candidates) {
        if (batch <= 0 || batch > max_batch) continue;
        
        int inference_time = calc_inference_time(batch, servers[server_id].k_factor);
        if (inference_time > available_time) continue;
        
        double efficiency = calc_efficiency(user_id, server_id, batch);
        if (efficiency > best_efficiency) {
            best_efficiency = efficiency;
            best_batch = batch;
        }
    }
    
    return best_batch;
}

vector<Task> schedule_user(int user_id) {
    vector<Task> tasks;
    User& user = users[user_id];
    
    int remaining_samples = user.sample_count;
    int current_time = user.start_time;
    
    vector<int> npu_counters(N, 0);
    int last_server = -1;
    
    while (remaining_samples > 0 && current_time < user.end_time) {
        int time_budget = user.end_time - current_time;
        
        int best_server = -1;
        int best_batch = 0;
        double best_score = -1;
        
        for (int s = 0; s < N; s++) {
            int batch = find_optimal_batch_size(user_id, s, remaining_samples, time_budget);
            if (batch <= 0) continue;
            
            double efficiency = calc_efficiency(user_id, s, batch);
            
            // Migration bonus - strongly prefer same server
            double migration_bonus = (s == last_server) ? efficiency * 0.4 : 0;
            
            // Quality bonus
            double quality_bonus = servers[s].k_factor * 0.02;
            
            double score = efficiency + migration_bonus + quality_bonus;
            
            if (score > best_score) {
                best_score = score;
                best_server = s;
                best_batch = batch;
            }
        }
        
        if (best_server == -1 || best_batch <= 0) break;
        
        Task task;
        task.time = current_time;
        task.server_id = best_server + 1;
        task.npu_id = (npu_counters[best_server] % servers[best_server].npu_count) + 1;
        task.batch_size = best_batch;
        
        tasks.push_back(task);
        remaining_samples -= best_batch;
        current_time += users[user_id].latency[best_server] + 1;
        npu_counters[best_server]++;
        last_server = best_server;
    }
    
    return tasks;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    cin >> N;
    servers.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> servers[i].npu_count >> servers[i].k_factor >> servers[i].memory;
    }
    
    cin >> M;
    users.resize(M);
    for (int i = 0; i < M; i++) {
        cin >> users[i].start_time >> users[i].end_time >> users[i].sample_count;
    }
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int lat;
            cin >> lat;
            users[j].latency.push_back(lat);
        }
    }
    
    cin >> a >> b;
    
    for (int i = 0; i < M; i++) {
        vector<Task> user_tasks = schedule_user(i);
        
        cout << user_tasks.size() << "\n";
        if (user_tasks.size() > 0) {
            for (size_t j = 0; j < user_tasks.size(); j++) {
                const Task& task = user_tasks[j];
                cout << task.time << " " << task.server_id << " " 
                     << task.npu_id << " " << task.batch_size;
                if (j < user_tasks.size() - 1) cout << " ";
            }
        }
        cout << "\n";
    }
    
    cout.flush();
    return 0;
}

