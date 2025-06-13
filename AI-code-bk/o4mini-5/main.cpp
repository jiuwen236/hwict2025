#include <bits/stdc++.h>
using namespace std;

// - 禁用迁移；每次请求的大小尽量大
// - 用户按开始时间排序，最小的优先
// - 最优选择：用户发送请求时会计算对每个NPU的实际推理结束时间，选择最早结束的。
// - 重点优化h(K)，当预测的用户请求完成时间大于结束时间时，放弃该用户(移至末尾，最后处理)

using Arr = array<int,4>;
int main(){
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    int N;
    if(!(cin>>N)) return 0;
    vector<int> g(N), k(N), m(N);
    for(int i=0;i<N;i++) cin>>g[i]>>k[i]>>m[i];
    int M; cin>>M;
    vector<int> s(M), e(M), cnt(M);
    for(int i=0;i<M;i++) cin>>s[i]>>e[i]>>cnt[i];
    vector<vector<int>> latency(N, vector<int>(M));
    for(int i=0;i<N;i++) for(int j=0;j<M;j++) cin>>latency[i][j];
    int a,b; cin>>a>>b;
    vector<int> order(M);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int u,int v){ return s[u]<s[v]; });
    vector<vector<Arr>> result(M);
    vector<vector<double>> avail(N);
    for(int i=0;i<N;i++) avail[i] = vector<double>(g[i], 0.0);
    vector<int> maxB(N);
    for(int i=0;i<N;i++) maxB[i] = max(1, (m[i]-b)/a);
    queue<int> q;
    for(int u: order) q.push(u);
    vector<int> postponed;
    while(!q.empty()){
        int u=q.front(); q.pop();
        auto tmp = avail;
        vector<Arr> local;
        int rem = cnt[u];
        double prevArrival = s[u] - 1, predEnd = s[u];
        bool ok = true;
        while(rem>0){
            double bestF=1e18;
            int bs=-1, bp=-1, bb=0;
            double bt=0;
            double bestArr = 0;
            for(int i=0;i<N;i++){
                int Bi = min(maxB[i], rem);
                double sendT = max(prevArrival + 1, (double)s[u]);
                double arrT = sendT + latency[i][u];
                double servT = ceil(Bi/(k[i]*sqrt((double)Bi)));
                for(int p=0;p<g[i];p++){
                    double st = max(tmp[i][p], arrT);
                    double ft = st + servT;
                    if(ft < bestF){ bestF = ft; bs = i; bp = p; bb = Bi; bt = sendT; bestArr = arrT; }
                }
            }
            if(bs < 0){ ok = false; break; }
            local.push_back({(int)bt, bs+1, bp+1, bb});
            tmp[bs][bp] = bestF;
            prevArrival = bestArr;
            rem -= bb;
            predEnd = bestF;
        }
        if(!ok || predEnd > e[u]) postponed.push_back(u);
        else{ avail = tmp; result[u] = local; }
    }
    for(int u: postponed){
        auto tmp = avail;
        vector<Arr> local;
        int rem = cnt[u];
        double prevArrival = s[u] - 1;
        while(rem>0){
            double bestF=1e18;
            int bs=-1, bp=-1, bb=0;
            double bt=0;
            double bestArr = 0;
            for(int i=0;i<N;i++){
                int Bi = min(maxB[i], rem);
                double sendT = max(prevArrival + 1, (double)s[u]);
                double arrT = sendT + latency[i][u];
                double servT = ceil(Bi/(k[i]*sqrt((double)Bi)));
                for(int p=0;p<g[i];p++){
                    double st = max(tmp[i][p], arrT);
                    double ft = st + servT;
                    if(ft < bestF){ bestF = ft; bs = i; bp = p; bb = Bi; bt = sendT; bestArr = arrT; }
                }
            }
            if(bs < 0) break;
            local.push_back({(int)bt, bs+1, bp+1, bb});
            tmp[bs][bp] = bestF;
            prevArrival = bestArr;
            rem -= bb;
        }
        avail = tmp;
        result[u] = local;
    }
    for(int i=0;i<M;i++){
        auto &v = result[i];
        int T = v.size();
        cout<<T<<'\n';
        for(int j=0;j<T;j++){ auto &r = v[j]; cout<<r[0]<<' '<<r[1]<<' '<<r[2]<<' '<<r[3]<<(j+1<T?' ':'\n'); }
    }
    return 0;
}
