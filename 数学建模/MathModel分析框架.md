$$\boxed{\text{优化问题分析框架}}$$

↓

$$\boxed{\text{明确优化目标}}$$

确定问题最终需要：

$$\min f(x)$$

或：

$$\max f(x)$$

进一步判断：

$$\begin{array}{c} \text{单目标优化}\\ \text{多目标优化}\\ \text{层次目标优化}\\ \text{动态目标优化}\\ \text{鲁棒目标优化} \end{array}$$

↓

$$\boxed{\text{识别决策变量}}$$

设：

$$x=(x_1,x_2,\cdots,x_n)$$

判断变量类型：

$$\begin{array}{c|c} \text{变量类型}&\text{数学形式}\\ \hline \text{连续变量}&x_i\in\mathbb R\\ \text{非负连续变量}&x_i\ge0\\ \text{整数变量}&x_i\in\mathbb Z\\ \text{非负整数变量}&x_i\in\mathbb Z_{\ge0}\\ \text{二元变量}&x_i\in\{0,1\}\\ \text{离散变量}&x_i\in\{a_1,a_2,\cdots,a_k\}\\ \text{序数变量}&x_i\in\{1,2,\cdots,k\}\\ \text{混合变量}&x=(x_c,x_i,x_b)\\ \text{函数型变量}&x=x(t)\\ \text{矩阵变量}&X\in\mathbb R^{m\times n} \end{array}$$

↓

$$\boxed{\text{构造目标函数}}$$
$$f(x)$$

判断目标函数的数学性质：

$$\begin{array}{c} \text{线性}\\ \text{仿射}\\ \text{二次}\\ \text{多项式}\\ \text{凸}\\ \text{严格凸}\\ \text{非凸}\\ \text{可微}\\ \text{不可微}\\ \text{光滑}\\ \text{非光滑}\\ \text{可分离}\\ \text{非可分离}\\ \text{确定性}\\ \text{随机性}\\ \text{显式}\\ \text{隐式}\\ \text{黑箱} \end{array}$$

↓

$$\boxed{\text{建立约束系统}}$$

标准形式：

$$\begin{aligned} \min_x\quad & f(x)\\ \text{s.t.}\quad &g_i(x)\le0,\quad i=1,\cdots,m\\ &h_j(x)=0,\quad j=1,\cdots,p\\ &l\le x\le u \end{aligned}$$

判断约束结构：

$$\begin{array}{c} \text{线性约束}\\ \text{非线性约束}\\ \text{等式约束}\\ \text{不等式约束}\\ \text{边界约束}\\ \text{整数约束}\\ \text{逻辑约束}\\ \text{互斥约束}\\ \text{容量约束}\\ \text{平衡约束}\\ \text{流量守恒约束}\\ \text{资源约束}\\ \text{时间约束}\\ \text{顺序约束}\\ \text{概率约束}\\ \text{动态约束}\\ \text{微分方程约束}\\ \text{互补约束}\\ \text{锥约束}\\ \text{半正定约束} \end{array}$$

↓

$$\boxed{\text{确定可行域}}$$
$$\Omega= \{x\mid g_i(x)\le0,\ h_j(x)=0,\ l\le x\le u\}$$

判断：

$$\begin{array}{c} \text{可行域是否为空}\\ \text{可行域是否有界}\\ \text{可行域是否凸}\\ \text{可行域是否连通}\\ \text{可行域是否连续}\\ \text{可行域是否离散}\\ \text{可行域是否包含整数结构} \end{array}$$

↓

$$\boxed{\text{识别数学结构}}$$

判断问题是否具有：

$$\begin{array}{c} \text{线性结构}\\ \text{二次结构}\\ \text{凸结构}\\ \text{网络结构}\\ \text{图结构}\\ \text{树结构}\\ \text{匹配结构}\\ \text{流结构}\\ \text{路径结构}\\ \text{分配结构}\\ \text{排序结构}\\ \text{调度结构}\\ \text{选址结构}\\ \text{分阶段结构}\\ \text{递推结构}\\ \text{时间动态结构}\\ \text{稀疏结构}\\ \text{块结构}\\ \text{可分解结构}\\ \text{随机结构}\\ \text{对称结构}\\ \text{低秩结构}\\ \text{组合结构} \end{array}$$

↓

$$\boxed{\text{判断确定性与不确定性}}$$
$$\begin{array}{c|c} \text{参数性质}&\text{优化类型}\\ \hline \text{参数完全已知}&\text{确定性优化}\\ \text{参数服从概率分布}&\text{随机优化}\\ \text{参数位于不确定集合}&\text{鲁棒优化}\\ \text{概率分布本身不确定}&\text{分布鲁棒优化}\\ \text{信息逐步获得}&\text{在线优化}\\ \text{参数随时间变化}&\text{动态优化}\\ \text{不同未来状态}&\text{情景优化} \end{array}$$

↓

$$\boxed{\text{判断时间结构}}$$
$$\begin{array}{c} \text{静态优化}\\ \text{单阶段优化}\\ \text{多阶段优化}\\ \text{动态优化}\\ \text{序贯决策}\\ \text{在线决策}\\ \text{滚动优化}\\ \text{最优控制} \end{array}$$

动态系统一般表示为：

$$x_{t+1}=F(x_t,u_t)$$

或：

$$\dot{x}(t)=F(x(t),u(t),t)$$

↓

$$\boxed{\text{判断目标之间的关系}}$$

若存在多个目标：

$$f_1(x),f_2(x),\cdots,f_k(x)$$

进一步判断：

$$\begin{array}{c} \text{目标一致}\\ \text{目标冲突}\\ \text{目标有优先级}\\ \text{目标具有权重}\\ \text{目标不可直接比较} \end{array}$$

对应：

$$\begin{array}{c} \text{加权和}\\ \text{目标规划}\\ \varepsilon\text{-约束}\\ \text{词典序优化}\\ \text{Pareto优化} \end{array}$$

↓

$$\boxed{\text{判断优化问题规模}}$$

关注：

$$\begin{array}{c} n=\text{决策变量数量}\\ m=\text{约束数量}\\ n_I=\text{整数变量数量}\\ n_B=\text{二元变量数量}\\ |V|=\text{节点数量}\\ |E|=\text{边数量}\\ T=\text{决策阶段数量}\\ |\Omega|=\text{搜索空间规模} \end{array}$$

进一步划分：

$$\begin{array}{c} \text{小规模}\\ \text{中等规模}\\ \text{大规模}\\ \text{超大规模} \end{array}$$

↓

$$\boxed{\text{判断问题复杂度}}$$

分析：

$$\begin{array}{c} \text{是否存在多项式时间算法}\\ \text{是否属于组合爆炸问题}\\ \text{是否属于NP-hard问题}\\ \text{是否存在指数级搜索空间}\\ \text{是否可以利用问题结构降复杂度}\\ \text{是否可以进行分解求解} \end{array}$$

↓

$$\boxed{\text{判断最优性要求}}$$
$$\begin{array}{c} \text{全局最优解}\\ \text{局部最优解}\\ \text{近似最优解}\\ \text{可行解}\\ \text{满意解} \end{array}$$

全局最优：

$$f(x^*)\le f(x),\quad \forall x\in\Omega$$

局部最优：

$$f(x^*)\le f(x),\quad \forall x\in N(x^*)\cap\Omega$$

↓

$$\boxed{\text{判断梯度与导数信息}}$$
$$\begin{array}{c} \text{目标函数梯度可得}\\ \text{Hessian矩阵可得}\\ \text{只能计算函数值}\\ \text{函数不可微}\\ \text{函数带噪声}\\ \text{函数求值成本高} \end{array}$$

对应算法方向：

$$\begin{array}{c} \text{一阶方法}\\ \text{二阶方法}\\ \text{无梯度方法}\\ \text{随机优化方法}\\ \text{代理模型方法} \end{array}$$

↓

$$\boxed{\text{判断凸性}}$$

若：

$$f(\lambda x+(1-\lambda)y) \le \lambda f(x)+(1-\lambda)f(y)$$

并且：

$$\Omega\text{为凸集}$$

则属于：

$$\boxed{\text{凸优化}}$$

凸优化中：

$$\boxed{\text{局部最优}=\text{全局最优}}$$

否则：

$$\boxed{\text{非凸优化}}$$

需要重点考虑：

$$\begin{array}{c} \text{局部极值}\\ \text{初值敏感性}\\ \text{全局搜索}\\ \text{多起点策略}\\ \text{松弛策略} \end{array}$$

↓

$$\boxed{\text{判断能否模型转化}}$$

考虑：

$$\begin{array}{c} \text{线性化}\\ \text{凸化}\\ \text{整数化}\\ \text{连续松弛}\\ \text{拉格朗日松弛}\\ \text{半正定松弛}\\ \text{二阶锥松弛}\\ \text{变量替换}\\ \text{辅助变量}\\ \text{维度压缩}\\ \text{约束消除}\\ \text{惩罚化}\\ \text{障碍化}\\ \text{对偶化}\\ \text{分解}\\ \text{离散化}\\ \text{情景化} \end{array}$$

↓

$$\boxed{\text{优化问题类型识别}}$$

依据：

$$\boxed{ \text{决策变量} + \text{目标函数} + \text{约束结构} + \text{数学结构} + \text{随机性} + \text{时间结构} }$$

确定问题类型。

↓

$$\boxed{\text{问题类型与常见方法}}$$
$$\begin{array}{c|l} \text{问题类型}&\text{常见方法}\\ \hline \text{线性规划 LP} & \text{单纯形法、对偶单纯形法、内点法} \\ \hline \text{大规模线性规划} & \text{内点法、修正单纯形法、分解法} \\ \hline \text{整数规划 IP} & \text{分支定界法、割平面法、Branch-and-Cut} \\ \hline \text{0-1整数规划 BIP} & \text{分支定界、割平面、Branch-and-Cut、逻辑剪枝} \\ \hline \text{混合整数线性规划 MILP} & \text{Branch-and-Bound、Branch-and-Cut、Branch-and-Price} \\ \hline \text{混合整数非线性规划 MINLP} & \text{分支定界、外逼近法、广义Benders分解} \\ \hline \text{无约束优化} & \text{梯度下降、牛顿法、拟牛顿法、共轭梯度法、信赖域法} \\ \hline \text{非线性规划 NLP} & \text{梯度法、牛顿法、BFGS、SQP、信赖域法} \\ \hline \text{约束非线性规划} & \text{SQP、罚函数法、障碍函数法、增广拉格朗日法} \\ \hline \text{二次规划 QP} & \text{主动集法、内点法、梯度投影法、共轭梯度法} \\ \hline \text{二次约束规划 QCQP} & \text{内点法、半正定松弛、二阶锥松弛} \\ \hline \text{凸优化} & \text{内点法、梯度法、加速梯度法、近端梯度法、ADMM} \\ \hline \text{强凸优化} & \text{梯度下降、Nesterov加速梯度、牛顿法} \\ \hline \text{非凸优化} & \text{梯度法、信赖域、多起点法、全局优化、元启发式算法} \\ \hline \text{非光滑优化} & \text{次梯度法、近端梯度法、Bundle Method、ADMM} \\ \hline \text{稀疏优化} & \text{坐标下降、ISTA、FISTA、近端梯度、ADMM} \\ \hline L_1\text{优化} & \text{近端梯度、坐标下降、ADMM、软阈值算法} \\ \hline \text{最小二乘优化} & \text{正规方程、QR分解、SVD、共轭梯度} \\ \hline \text{非线性最小二乘} & \text{Gauss-Newton、Levenberg-Marquardt、信赖域法} \\ \hline \text{锥规划 CP} & \text{内点法、原始-对偶法} \\ \hline \text{二阶锥规划 SOCP} & \text{原始-对偶内点法、锥优化算法} \\ \hline \text{半正定规划 SDP} & \text{内点法、一阶方法、半正定松弛} \\ \hline \text{几何规划 GP} & \text{对数变换、凸优化、内点法} \\ \hline \text{网络优化} & \text{最短路、最大流、最小费用流、网络单纯形法} \\ \hline \text{最短路径问题} & \text{Dijkstra、Bellman-Ford、Floyd-Warshall、A^*} \\ \hline \text{单源最短路} & \text{Dijkstra、Bellman-Ford、SPFA} \\ \hline \text{全源最短路} & \text{Floyd-Warshall、Johnson算法} \\ \hline \text{最大流问题} & \text{Ford-Fulkerson、Edmonds-Karp、Dinic、Push-Relabel} \\ \hline \text{最小割问题} & \text{最大流算法、Stoer-Wagner算法} \\ \hline \text{最小费用最大流} & \text{SSAP、Cost Scaling、网络单纯形法} \\ \hline \text{最小费用流} & \text{SSAP、Cycle Canceling、网络单纯形、Cost Scaling} \\ \hline \text{二分图匹配} & \text{增广路算法、匈牙利算法、Hopcroft-Karp} \\ \hline \text{加权匹配} & \text{Hungarian Algorithm、KM算法、线性规划} \\ \hline \text{一般图匹配} & \text{Blossom算法} \\ \hline \text{指派问题} & \text{匈牙利算法、线性规划、最小费用流} \\ \hline \text{运输问题} & \text{运输单纯形法、线性规划、网络流算法} \\ \hline \text{转运问题} & \text{最小费用流、网络单纯形法} \\ \hline \text{最小生成树} & \text{Kruskal、Prim、Borůvka} \\ \hline \text{最大生成树} & \text{Kruskal、Prim} \\ \hline \text{组合优化} & \text{动态规划、分支定界、整数规划、局部搜索、元启发式算法} \\ \hline \text{背包问题} & \text{动态规划、分支定界、整数规划、贪心松弛} \\ \hline \text{集合覆盖问题} & \text{整数规划、贪心算法、分支定界、近似算法} \\ \hline \text{集合划分问题} & \text{整数规划、列生成、Branch-and-Price} \\ \hline \text{最大独立集} & \text{整数规划、分支定界、局部搜索} \\ \hline \text{最大团问题} & \text{分支定界、整数规划、启发式算法} \\ \hline \text{图着色问题} & \text{回溯法、整数规划、DSATUR、启发式算法} \\ \hline \text{旅行商问题 TSP} & \text{Held-Karp、Branch-and-Bound、Branch-and-Cut、局部搜索} \\ \hline \text{车辆路径问题 VRP} & \text{整数规划、列生成、Branch-and-Price、局部搜索、元启发式算法} \\ \hline \text{选址问题} & \text{整数规划、分支定界、拉格朗日松弛、Benders分解} \\ \hline \text{设施选址问题} & \text{MILP、Benders分解、拉格朗日松弛、启发式算法} \\ \hline \text{调度优化} & \text{整数规划、动态规划、约束规划、分支定界、启发式算法} \\ \hline \text{作业车间调度 JSSP} & \text{MILP、约束规划、分支定界、遗传算法、禁忌搜索} \\ \hline \text{流水车间调度 FSSP} & \text{动态规划、NEH、整数规划、元启发式算法} \\ \hline \text{机器调度} & \text{整数规划、动态规划、优先规则、启发式算法} \\ \hline \text{生产计划优化} & \text{线性规划、MILP、动态规划、滚动优化} \\ \hline \text{库存优化} & \text{动态规划、随机规划、马尔可夫决策、鲁棒优化} \\ \hline \text{供应链优化} & \text{LP、MILP、网络流、随机规划、鲁棒优化} \\ \hline \text{多目标优化} & \text{加权和法、}\varepsilon\text{-约束法、目标规划、NSGA-II、MOEA/D} \\ \hline \text{Pareto优化} & \text{NSGA-II、NSGA-III、MOEA/D、Pareto搜索} \\ \hline \text{目标规划} & \text{加权目标规划、优先级目标规划、词典序优化} \\ \hline \text{双层优化} & \text{KKT转化、单层化、Branch-and-Bound、启发式算法} \\ \hline \text{多层优化} & \text{分层规划、KKT条件、启发式算法} \\ \hline \text{动态规划 DP} & \text{Bellman递推、状态压缩、记忆化搜索、近似动态规划} \\ \hline \text{多阶段决策} & \text{动态规划、随机动态规划、滚动时域优化} \\ \hline \text{动态优化} & \text{动态规划、最优控制、模型预测控制} \\ \hline \text{最优控制} & \text{Pontryagin极大值原理、动态规划、直接法、间接法} \\ \hline \text{模型预测控制 MPC} & \text{滚动时域优化、QP、NLP} \\ \hline \text{随机优化} & \text{随机规划、SAA、随机梯度、随机动态规划} \\ \hline \text{两阶段随机规划} & \text{L-shaped Method、Benders分解、SAA} \\ \hline \text{多阶段随机规划} & \text{随机动态规划、SDDP、场景树方法} \\ \hline \text{机会约束规划} & \text{确定性等价、场景法、Monte Carlo、凸近似} \\ \hline \text{鲁棒优化} & \text{鲁棒对偶、列约束生成、可调鲁棒优化} \\ \hline \text{分布鲁棒优化 DRO} & \text{对偶化、Wasserstein鲁棒优化、矩信息方法} \\ \hline \text{情景优化} & \text{Scenario Approach、样本优化、情景削减} \\ \hline \text{在线优化} & \text{Online Gradient Descent、Mirror Descent、FTRL} \\ \hline \text{在线凸优化} & \text{OGD、Online Newton Step、Mirror Descent} \\ \hline \text{黑箱优化} & \text{贝叶斯优化、遗传算法、PSO、SA、DE} \\ \hline \text{无梯度优化} & \text{Nelder-Mead、Powell、模式搜索、CMA-ES} \\ \hline \text{昂贵函数优化} & \text{贝叶斯优化、代理模型、响应面法} \\ \hline \text{全局优化} & \text{Branch-and-Bound、区间算法、DIRECT、空间分支定界} \\ \hline \text{大规模优化} & \text{分解法、坐标下降、随机梯度、ADMM、一阶方法} \\ \hline \text{分布式优化} & \text{ADMM、分布式梯度下降、Consensus Optimization} \\ \hline \text{可分解优化} & \text{Benders分解、Dantzig-Wolfe分解、拉格朗日分解} \\ \hline \text{块结构优化} & \text{Block Coordinate Descent、分解协调法、ADMM} \\ \hline \text{列生成} & \text{Column Generation、Dantzig-Wolfe分解} \\ \hline \text{Branch-and-Price} & \text{列生成、分支定界} \\ \hline \text{Benders结构问题} & \text{经典Benders、逻辑Benders、广义Benders} \\ \hline \text{拉格朗日松弛问题} & \text{次梯度法、Bundle Method、拉格朗日启发式} \\ \hline \text{约束满足问题 CSP} & \text{回溯搜索、约束传播、Branch-and-Bound} \\ \hline \text{约束规划 CP} & \text{约束传播、域削减、搜索剪枝} \\ \hline \text{多项式优化} & \text{SOS优化、半正定松弛、Moment Relaxation} \\ \hline \text{互补优化} & \text{互补条件、MPEC、正则化法} \\ \hline \text{变分不等式} & \text{投影法、Extragradient、Interior Point} \\ \hline \text{均衡优化} & \text{变分不等式、固定点算法、互补问题方法} \\ \hline \text{博弈优化} & \text{Nash均衡、Best Response、Mirror Descent、均衡规划} \\ \hline \text{资源分配优化} & \text{LP、MILP、网络流、拉格朗日分解} \\ \hline \text{排序优化} & \text{整数规划、动态规划、局部搜索} \\ \hline \text{路径规划} & \text{Dijkstra、A^*、D^*、动态规划、采样优化} \\ \hline \text{连续轨迹优化} & \text{最优控制、直接配点法、SQP、MPC} \\ \hline \text{参数估计优化} & \text{最小二乘、极大似然、梯度法、Gauss-Newton} \\ \hline \text{机器学习优化} & \text{SGD、Momentum、Adam、RMSProp、L-BFGS} \\ \hline \text{大规模机器学习优化} & \text{Mini-batch SGD、Adam、Adagrad、分布式优化} \\ \hline \text{超参数优化} & \text{网格搜索、随机搜索、贝叶斯优化、Hyperband} \\ \hline \text{正则化优化} & \text{坐标下降、近端梯度、ADMM} \\ \hline \text{遗传算法 GA} & \text{选择、交叉、变异、精英保留} \\ \hline \text{粒子群 PSO} & \text{速度更新、位置更新、个体最优、群体最优} \\ \hline \text{模拟退火 SA} & \text{邻域搜索、Metropolis准则、降温策略} \\ \hline \text{蚁群算法 ACO} & \text{概率转移、信息素更新、信息素挥发} \\ \hline \text{差分进化 DE} & \text{差分变异、交叉、选择} \\ \hline \text{禁忌搜索 TS} & \text{邻域搜索、禁忌表、藐视准则} \\ \hline \text{CMA-ES} & \text{协方差矩阵自适应、进化策略} \\ \hline \text{人工蜂群 ABC} & \text{雇佣蜂搜索、观察蜂搜索、侦察蜂搜索} \\ \hline \text{局部搜索} & \text{邻域搜索、爬山法、迭代改进} \\ \hline \text{变邻域搜索 VNS} & \text{邻域切换、局部搜索、扰动} \\ \hline \text{大邻域搜索 LNS} & \text{破坏算子、修复算子、局部搜索} \\ \hline \text{自适应大邻域搜索 ALNS} & \text{自适应算子选择、破坏、修复} \\ \hline \text{GRASP} & \text{随机贪心构造、局部搜索} \\ \hline \text{混合元启发式} & \text{全局搜索、局部搜索、问题特定算子} \end{array}$$

↓

$$\boxed{\text{根据问题结构选择求解层级}}$$
$$\boxed{\text{专用算法}}$$

优先于：

$$\boxed{\text{通用优化算法}}$$

优先于：

$$\boxed{\text{元启发式算法}}$$

一般遵循：

$$\text{存在特殊结构} \Rightarrow \text{利用特殊结构}$$
$$\text{结构弱但数学表达明确} \Rightarrow \text{通用数学规划}$$
$$\text{高度非凸、复杂、黑箱} \Rightarrow \text{全局搜索或元启发式}$$

↓

$$\boxed{\text{选择精确算法或近似算法}}$$
$$\begin{array}{c|c} \text{算法类别}&\text{核心目标}\\ \hline \text{精确算法}&\text{证明或逼近全局最优}\\ \text{近似算法}&\text{获得有理论误差界的近似解}\\ \text{启发式算法}&\text{快速获得高质量可行解}\\ \text{元启发式算法}&\text{进行更广泛的全局搜索}\\ \text{混合算法}&\text{结合精确搜索与启发式搜索} \end{array}$$

↓

$$\boxed{\text{算法初始化}}$$

确定：

$$\begin{array}{c} \text{初始解}\\ \text{初始可行解}\\ \text{初始种群}\\ \text{初始步长}\\ \text{学习率}\\ \text{惩罚参数}\\ \text{温度参数}\\ \text{搜索邻域}\\ \text{随机种子} \end{array}$$

↓

$$\boxed{\text{算法迭代}}$$

一般形式：

$$x^{(k+1)} = \Phi(x^{(k)})$$

或：

$$x^{(k+1)} = x^{(k)}+\Delta x^{(k)}$$

基本过程：

$$\boxed{ \text{生成候选解} \rightarrow \text{计算目标函数} \rightarrow \text{检查约束} \rightarrow \text{更新搜索方向} \rightarrow \text{接受或拒绝} }$$

↓

$$\boxed{\text{约束处理}}$$

对于不可行解，可以采用：

$$\begin{array}{c} \text{直接可行域搜索}\\ \text{投影法}\\ \text{罚函数法}\\ \text{障碍函数法}\\ \text{修复算子}\\ \text{拉格朗日乘子}\\ \text{增广拉格朗日}\\ \text{可行性优先规则} \end{array}$$

↓

$$\boxed{\text{算法停止条件}}$$

判断：

$$\begin{array}{c} |f(x^{(k+1)})-f(x^{(k)})|\le\varepsilon\\ \|x^{(k+1)}-x^{(k)}\|\le\varepsilon\\ \|\nabla f(x^{(k)})\|\le\varepsilon\\ \text{最优间隙}\le\varepsilon\\ k\ge k_{\max}\\ \text{运行时间达到限制}\\ \text{长期无改善} \end{array}$$

↓

$$\boxed{\text{获得候选最优解}}$$
$$x^*$$

以及：

$$f(x^*)$$

↓

$$\boxed{\text{可行性检验}}$$

检查：

$$g_i(x^*)\le0,\quad h_j(x^*)=0,\quad l\le x^*\le u$$

以及：

$$x_i^*\in\mathbb Z$$

或：

$$x_i^*\in\{0,1\}$$

↓

$$\boxed{\text{最优性检验}}$$

根据问题类型判断：

$$\begin{array}{c} \text{KKT条件}\\ \text{对偶间隙}\\ \text{最优间隙}\\ \text{下界与上界}\\ \text{分支定界证明}\\ \text{全局最优证明}\\ \text{局部最优条件} \end{array}$$

↓

$$\boxed{\text{对偶分析}}$$

构造：

$$\boxed{\text{原问题 Primal}}$$

↓

$$\boxed{\text{对偶问题 Dual}}$$

研究：

$$\begin{array}{c} \text{弱对偶}\\ \text{强对偶}\\ \text{对偶间隙}\\ \text{影子价格}\\ \text{拉格朗日乘子}\\ \text{约束边际价值} \end{array}$$

↓

$$\boxed{\text{敏感性分析}}$$

分析：

$$\begin{array}{c} \text{目标函数系数变化}\\ \text{约束右端项变化}\\ \text{参数变化}\\ \text{权重变化}\\ \text{边界变化}\\ \text{数据扰动}\\ \text{模型参数误差} \end{array}$$

研究：

$$p\rightarrow p+\Delta p$$

导致：

$$x^*\rightarrow x^*+\Delta x$$

以及：

$$f(x^*)\rightarrow f(x^*)+\Delta f$$

↓

$$\boxed{\text{鲁棒性分析}}$$

判断：

参数扰动

是否导致：

方案发生大幅改变

分析：

$$\begin{array}{c} \text{参数鲁棒性}\\ \text{数据鲁棒性}\\ \text{结构鲁棒性}\\ \text{算法鲁棒性}\\ \text{最优解鲁棒性} \end{array}$$

↓

$$\boxed{\text{稳定性分析}}$$

研究：

$$\Delta p\rightarrow0$$

时：

$$\Delta x^*\rightarrow0$$

以及：

$$\Delta f^*\rightarrow0$$

↓

$$\boxed{\text{算法收敛性分析}}$$

关注：

$$\begin{array}{c} \text{是否收敛}\\ \text{收敛到局部最优还是全局最优}\\ \text{线性收敛}\\ \text{超线性收敛}\\ \text{二次收敛}\\ \text{概率收敛}\\ \text{渐近收敛} \end{array}$$

↓

$$\boxed{\text{计算复杂度分析}}$$

分析：

$$\begin{array}{c} \text{时间复杂度}\\ \text{空间复杂度}\\ \text{迭代次数}\\ \text{函数评价次数}\\ \text{梯度计算次数}\\ \text{搜索节点数}\\ \text{内存占用} \end{array}$$

↓

$$\boxed{\text{算法性能评价}}$$

综合评价：

$$\begin{array}{c} \text{目标函数值}\\ \text{可行性}\\ \text{最优间隙}\\ \text{收敛速度}\\ \text{运行时间}\\ \text{稳定性}\\ \text{鲁棒性}\\ \text{可扩展性}\\ \text{初值敏感性}\\ \text{参数敏感性} \end{array}$$

↓

$$\boxed{\text{多算法比较}}$$

建立：

$$\begin{array}{c|c} \text{评价维度}&\text{比较内容}\\ \hline \text{解质量}&f(x^*)\\ \text{最优性}&\text{最优间隙}\\ \text{效率}&\text{运行时间}\\ \text{稳定性}&\text{多次运行结果波动}\\ \text{可扩展性}&\text{规模增加后的性能}\\ \text{鲁棒性}&\text{数据扰动后的性能}\\ \text{实现复杂度}&\text{算法实现与调参难度} \end{array}$$

↓

$$\boxed{\text{最终优化分析主线}}$$
$$\boxed{ \text{明确目标} \rightarrow \text{确定决策变量} \rightarrow \text{建立目标函数} \rightarrow \text{建立约束} }$$

↓

$$\boxed{ \text{分析变量类型} + \text{目标函数结构} + \text{约束结构} + \text{可行域} }$$

↓

$$\boxed{ \text{判断线性 / 非线性} + \text{连续 / 离散} + \text{凸 / 非凸} }$$

↓

$$\boxed{ \text{判断网络 / 图 / 调度 / 路径 / 分配 / 动态等特殊结构} }$$

↓

$$\boxed{ \text{判断确定性 / 随机性 / 鲁棒性 / 动态性} }$$

↓

$$\boxed{\text{确定优化问题类别}}$$

↓

$$\boxed{\text{判断能否进行模型转化}}$$

↓

$$\boxed{ \text{选择专用算法} \rightarrow \text{数学规划算法} \rightarrow \text{启发式 / 元启发式算法} }$$

↓

$$\boxed{\text{初始化与迭代求解}}$$

↓

$$\boxed{\text{得到候选最优解}}$$

↓

$$\boxed{ \text{可行性检验} + \text{最优性检验} + \text{收敛性检验} }$$

↓

$$\boxed{ \text{敏感性分析} + \text{稳定性分析} + \text{鲁棒性分析} }$$

↓

$$\boxed{ \text{算法复杂度分析} + \text{多算法比较} }$$

↓

$$\boxed{\text{形成最终优化方案}}$$