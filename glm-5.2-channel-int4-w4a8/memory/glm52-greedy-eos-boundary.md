---
name: glm52-greedy-eos-boundary
description: greedy下部分问答输入首token=EOS空输出,非乱码,sampling正常,是w4a8量化精度边界
metadata:
  type: project
---

修复乱码(两个根因都修)后,残留现象:greedy(temp=0)下部分问答类输入首 token = EOS,
输出空字符串(completion_tokens=1, finish_reason=stop)。

**Why:** 这**不是乱码 bug**(乱码已 0/10 修复)。是 w4a8 量化精度使 EOS token 与首个回答
token 的概率非常接近,greedy 选最高概率时碰巧选了 EOS。证据:knowledge 类输入
sampling(temp=0.7)输出完全正常("1.符号主义...2.连接主义...3.深度学习...4.大模型"),
仅 greedy 空。qa 类(期望单字回答)更极端,sampling 也可能 EOS。

**How to apply:** 用 sampling(temperature>0)缓解;或调整 EOS token 概率逻辑。不要当作
indexer/算子 bug 继续排查——乱码(数字符号垃圾)与 greedy EOS(干净空输出)是不同性质。
区分:乱码=输出无意义符号串;EOS=输出空但 finish_reason=stop 且 sampling 正常。
