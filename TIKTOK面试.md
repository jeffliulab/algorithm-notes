整体情况：

- **1 - 2 round**: Basic coding questions. Fundamentals of CS, strong enough for a background of basic CS, data structures, and algorithms. Leetcode, easy to medium difficulty and **discussion of your previous experience/projects.**
- **The third round** is a system design with the hiring manager that'll involve high level technical questions
- **The fourth round** may be optional depending on the hiring team's signals/needs from your interview process.

(job description for AML - Orchestration): [https://jobs.bytedance.com/en/position/7516656741909793042/detail](https://jobs.bytedance.com/en/position/7516656741909793042/detail)

1、TIKTOK没有题库，每个面试官自己出题

2、AML应该偏infra一点

3、leetcode

4、MLE八股（最多写个梯度下降求根号看看基本功）

5、项目：开发优化Ops，训练效率如何增加

6、ML system design

## 1 - 2轮

---

### **1-2 轮技术面 leetcode重点**

* bug-free/最优时间/最优空间
* 哈希表，Tree，Graph
* BFS/DFS/Sorting
* 线程安全的生产者-消费者队列

---

### **1-2轮 知识重点**

* MLE八股
* Adam优化器和梯度下降函数能做一个简单的解

---

### **1-2轮 项目和工作经验BQ**

* 项目背景和目标，负责部分的工作
* 使用了Python的哪些特性，遇到了什么技术挑战
* 是否用过PyTorch，对内部机制有什么了解？
* 是否对模型训练性能进行过优化？比如数据加载、分布式训练等
* ML编排和Infra基础架构的Kubernets，Spark，Docker用过吗？（这个可以说Docker，项目通过Docker部署在Google Cloud Run上了）

---



## 🛠️ 面试准备建议（针对 JD）

### 1. **Coding Round（LC中等+）**

* Focus: **Python + Algorithms + DS**
* 推荐刷题方向：
  * Heap / Priority Queue
  * Graph / Topological Sort
  * System Design for distributed model serving

### 2. **System Design（ML Infra方向）**

* 如问你：**如何部署一个可扩展的 ML Pipeline？**
  * 要提及：Docker + CI/CD + GCP/AWS + Feature Store + Online/Offline Serving

### 3. **ML Infrastructure 面试内容**

* 如何优化大规模训练系统（内存优化、分布式训练、resource scheduling）
* 对比：PyTorch vs TensorFlow 在部署中的优缺点
* 实践项目中 MLOps 设计如何考虑复用性与监控（MLflow, logging）

## 3轮

#### **第三轮：系统设计 (System Design with Hiring Manager)**

这一轮依然是决定性的，由招聘经理亲自面试。对硕士生，他/她可能不会期望你从零设计一个全新的、开创性的系统，但会非常看重你的设计思路、技术广度和对核心问题的把握能力。

**考察重点:**

1. **系统设计基础:** 对负载均衡、缓存、数据库、消息队列等基本组件的理解和运用。
2. **ML系统领域的理解:** 知道机器学习系统（特别是训练和推理系统）的特殊性，比如对GPU的需求、长时间运行的任务、数据依赖等。
3. **逻辑思维与沟通:** 能否在引导和压力下，有条理地分析问题，并清晰地阐述自己的设计方案和各种权衡（Trade-offs）。

**可能遇到的问题类型:**

问题的主题和博士岗类似，但考察的深度和期望会有所调整。

* **“让你来设计一个简单的图片分享网站的后台，用户可以上传图片和浏览。”** (可能用一个相对常规的题目作为热身，看你的基础设计能力)
  * **逐步深入:** “如果现在要加入一个AI功能，比如自动给图片打标签，你该如何修改你的系统架构？” 这个问题就把你拉回了ML的领域。
* **“我们想为内部算法工程师提供一个自助式的模型训练服务，你会如何设计这个系统？”** (与JD高度相关的核心题目)
  * **面试官可能会引导你思考:**
    * “用户如何提交一个训练任务？（比如通过Web界面还是命令行？）”
    * “后台如何管理这些任务？（比如用一个任务队列？）”
    * “如何分配GPU资源给不同的任务？”
    * “训练产生的数据和模型保存在哪里？”
    * “如何让用户看到训练的进度和日志？”
* **“你如何设计一个系统来监控我们线上几百个机器学习模型的健康状况？”**
  * **追问:** 需要监控哪些指标？（QPS, 延迟, 内存/GPU使用率, 预测结果分布等）。数据如何收集、存储和可视化？出现问题如何报警？

对硕士生而言，面试官更想看到你 **知识的广度和思考的框架** 。能说出使用哪些开源组件（如用Prometheus做监控，用Celery/RabbitMQ做任务队列，用Docker做环境隔离）并解释原因，会非常加分。

---

#### **第四轮：加面或团队匹配 (Optional Round)**

对硕士生来说，这一轮的情况与博士生类似，但目的可能更偏向于确认综合素质。

1. **技术加面:** 如果前几轮面试官的评价有分歧，或者觉得需要更强的信号，可能会安排另一位资深工程师再进行一轮coding或系统设计面试。
2. **行为面试与团队匹配 (更常见):** 由更高级别的经理或总监来进行。
   * **问题:** “你为什么选择我们公司/团队？” “你未来3-5年的职业规划是什么？” “你遇到的最大的挫折是什么？如何克服的？” “你如何学习一门新技术？”
   * **目的:** 评估你的求职动机、学习能力、抗压能力和文化匹配度。他们希望招到的是一个有潜力、有热情、能快速成长为团队核心的成员。

### 综合准备建议 (硕士版)

1. **全面梳理你的项目经验:** 把简历上每一个实习和课程项目都当作一个产品来准备。用STAR法则（Situation, Task, Action, Result）讲清楚来龙去脉，尤其是你的技术贡献和思考。
2. **夯实算法和CS基础:** LeetCode中等难度的题目要刷得滚瓜烂熟。对于网络、操作系统等基础知识，要能讲清核心概念。
3. **重点学习JD中的技术栈:** 你不需要成为 `Kubernetes` 或 `Spark` 的专家，但至少要去了解它们是做什么的，解决了什么问题，以及它们是如何在机器学习场景中被使用的。可以看一些入门教程或官方文档，理解其核心架构。
4. **系统学习ML系统设计:** 这是你和普通后端工程师拉开差距的关键。学习如何设计一个完整的ML Pipeline，从数据到模型部署。可以阅读头部科技公司的技术博客（Uber Engineering, Netflix TechBlog, aribnb.ai等）。
5. **准备提问:** 准备有深度的问题，展现你对该领域的思考和热情。例如：“我看到JD中提到了Orchestration，团队目前是基于开源方案（如Kubeflow/Airflow）做二次开发，还是在自研相关的系统？”
