# SemEval-2026-2.1_summary

经过几个月的学习，从学习吴恩达机器学习基础课、算法练习到具体的 imdb 深度学习教程学习，再到SemEval-2026-task-1任务。我的学习内容逐步从传统深度学习模型过渡到LLM优化，这一过程收获良多。特别感谢王津导师的支持和耐心教导！到目前，SemEval-2026-task-1马上结束了。

在实验过程中，由于我学习的不够深入，实验和工程经验不足，踩了很多坑，另外对于大模型微调需要时间成本，实验进度推进缓慢，目前还没有得到理想的结果。


# 流程和问题分析

## 目标

task-1-A 的任务目标是条件幽默生成任务：给 headline 或 two words ，生成一段幽默文本，分别在英文、中文、西班牙语环境下完成。老师提供给我的思路是，用 ppo 算法调整 qwen3 模型，让模型生成更幽默的内容。

&nbsp;
&nbsp;

## 一、数据集构建

### 1.数据来源和说明

本次任务的数据源来自其它 humor detection 任务的数据集：

英文：https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection/tree/master/Data

中文：https://github.com/DUTIR-Emotion-Group/CCL2019-Chinese-Humor-Computation/tree/master/task1

西班牙语：https://github.com/pln-fing-udelar/pln-inco-resources/tree/master/humor/haha2021

这些数据总结来看，包含两个属性 jokes：is_humor ，是二元标签，这份数据能够用来训练deberta打分函数，但并没有直接与我们的目标任务对齐，目标任务需要的训练数据形式应该为： headline：two words：jokes。

### 2.数据处理

为了得到与目标任务对齐的训练数据，可以通过手动和自动的方式实现。显然，手动总结 jokes的headline/two words 的成本太高，不现实，尝试自动构建。

**方法**

参考SemEval-2026-task-1-a提供的数据格式，使用LLM通过 Instruction-Learing 生成headline/two words。


&nbsp;

**实现**

首先通过脚本把原数据的“幽默”（is_humor=1）的文本提取出来，然后逐条通过 prompt = instruction + input(jokes) 输入 Qwen2.5-7B-Instruction 模型，把模型生成的headline/two words保存下来得到本次任务的训练集和测试集。

```python
# prompt模板
PROMPT_TEMPLATE = """你是一个幽默分析专家。请分析下面这个笑话，提取或生成：
1. 一个相关的新闻标题（headline）：可以是笑话的背景、主题或相关新闻，10-40字
2. 两个关键词（word1, word2）：笑话中最重要的两个词，必须一个为动词另一个为名词，每个2-5字

笑话：{joke}

要求：
- headline要简洁，像新闻标题
- word1和word2要是笑话中的词汇
- 严格按照JSON格式输出，不要有其他内容

输出格式：
{{"headline": "新闻标题", "word1": "关键词1", "word2": "关键词2"}}"""
```

## 二、PPO算法

### 过程记录

#### 实验

按照老师提供的思路：本次任务用deberta分数作为critic.根据老师给的ppo代码，我尝试完成一份用于本次任务的ppo代码，在运行时遇到一个问题：deberta与Qwen的tokenizer不同，在token-level上无法对齐，导致无法逐token打分，也就无法计算优势函数和损失。我尝试在sequnce-level上计算奖励于是把奖励设计成这样：

$$
r_t =
\begin{cases}
0, & t < T \\
R_{\text{seq}}, & t = T
\end{cases}
$$


```python
#reward结构：
reward_kl = 0 # everywhere
reward_kl[end] = R_seq = RewardModel(x) - β * KL(x)
```

实验后，我意识到一个问题：

对于 t < T,GAE 实际在算:

$$
A_t = \gamma^{T - t} \cdot \left( R_{\text{seq}} - V_t \right)
$$

也就是说：
前面的 token 都在影响最后一句话是否"幽默"。语法 / 结构 token被迫学习 reward

另外，奖励函数设置单一，正面奖励信号只来自deberta_score，但deberta只是在幽默数据集上训练的分类模型，它可能学习不到深层内容，导致模型训练过程崩塌：

```text
PPO 100 step生成样例：

Prompt: 请根据以下新闻标题，创作一段幽默的中文文本：

标题：幼师课堂萌娃哭诉包子无肉

生成文本:小班上老师教幼儿认识蔬菜，老师拿着青菜问：“这是什么？” 幼儿回答：“这是白菜！” 老师又拿出一个红红的苹果：“这是什么？” 幼儿说：“老师，这是包子吗？它没肉啊！” 全班同学哄堂大笑。老师哑然。 网友点评：老师问对了 老师教对了 幼儿想对了！ 老师无奈了





PPO 500 step 生成样例

Prompt: 请根据以下新闻标题，创作一段幽默的中文文本：

标题：啤酒广告为何无人有啤酒肚

生成文本：爸爸：为什么电视上广告那么多？ 爸爸：因为广告能赚钱，小明想知道一下，那么电视上为什么广告那么多啤酒呢 小：因为啤酒卖得便宜，爸爸～爹～「爷去哪里？【呜┉」～ breathable breathable我妈 ridden slate Operand(Theurança(dep DocumentaryFederal tolua Css__,__ Chúng możli/******** 잘못 tolua createStackNavigatorFecha Allows	suite宦	com	Input gbooleanurrencですかussion Emacs Emacsussion	FieldГ hashed mudança剑 setContentบท HomelandPhiladelphia Countdown Leap/********	want 잘못

```

可以看到，文本长度大于某个阈值时，模型开始胡言乱语了。所以，PPO的结果算是失败的，我只跑出中文的版本，生成的文本也用不了。


&nbsp;

#### 工程上遇到的问题

**PPO训练失效！**

得到经过sft的三种语言的Qwen3模型后，我进行PPO训练得到三种语言对应的模型。起初我一直没发现模型的问题，还用该PPO模型完成了SemEval-2026-task-1-A的条件幽默生成任务。在尝试GRPO算法前，我尝试计算模型的dist-n指标和 **ppl(困惑度)** 指标。在对比指标时发现， **sft模型的ppl和ppo模型的ppl竟然相等！** 也就是说模型PPO根本没有更新参数！

重写Debug代码，几经周折，发现不用unsloth参数就能正常更新了(这个问题的具体原因还没清楚)，可能是我没有严格按照token-level的reward计算，unsloth里的优化把我的参数更新给调整了？这里也让训练时间成倍增加......

&nbsp;


## 三、GRPO算法

GRPO算法开始的已经比较晚了，距离SemEval-2026-task-1截止只剩2天，所以这个算法没有跑完，目前还在实验中......

### 奖励函数设计

吸取了前面PPO的教训，我尝试构建更完善的奖励函数

参考deepseek-R1的设计、网上的一些教程和chatGPT的建议，我的reward设计如下：

headline输入时：


$$
R_{\text{headline}}(x)=0.75 \ R_{\text{deb}}(x)\+0.20 \ R_{\text{len}}(x)\-0.15 \ R_{\text{rep}}(x)
$$


```python
# for prompt_type == 'headline'
WEIGHT_DEBERTA    = 0.75
WEIGHT_LENGTH     = 0.20
WEIGHT_REPETITION = -0.15

```


&nbsp;

two words作为输入时：

$$
R_{\text{words}}(x)=0.55 \ R_{\text{deb}}(x)\+0.25 \ R_{\text{wc}}(x)\+0.15 \ R_{\text{len}}(x)\-0.10 \ R_{\text{rep}}(x)
$$


```python
# for prompt_type == 'words'
WEIGHT_DEBERTA       = 0.55
WEIGHT_WORD_COVERAGE = 0.25
WEIGHT_LENGTH        = 0.15
WEIGHT_REPETITION    = -0.10
```

其中：

$$
R_{\text{coverage}}\left(y \mid w_1, w_2\right)=
\begin{cases}
1.0, & \text{if } y \text{ contains both } w_1 \text{ and } w_2, \\
0.5, & \text{if } y \text{ contains exactly one of } \{w_1, w_2\}, \\
0.0, & \text{if } y \text{ contains neither } w_1 \text{ nor } w_2.
\end{cases}
$$

目前还在实验中，是否可行需要通过实验验证。

目前的实验情况：

running中文的GRPO：总共3个Epoch，每个Epoch预估4小时；

总时长预估12小时跑完中文GRPO。

&nbsp;

**目前的实验结果：**

```text
KL系数=0.01
(epoch=1;step=100)
Prompt类型: headline
Prompt: 请根据以下新闻标题，创作一段幽默的中文文本：
标题：经济好转却一切变糟

生成文本: 随着经济好转，一切都变得糟糕起来。随着经济恶化，一切都变得容易了。这就是生活。这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是生活！这就是
  DeBERTa: 0.977
  Word Coverage: 1.000
  Length: 0.750
  Repetition Penalty: 0.297

KL系数=0.01
(epoch=1;step=400)
Prompt类型: headline
Prompt: 请根据以下新闻标题，创作一段幽默的中文文本：
标题：长大不等于成熟

生成文本: 长大并不要求变得更成熟，但要敢于不去睡觉！......哦等等 علين......抱歉。 دمش它不起作用了... دمش有期徒5到7天有期徒。不要杀人... دمش الإثنولوجي علين دمش الإثنولوجي علين دمش الإثنولوجي علين دمش الإثنولوجي علين دمش الإثنولوجي علين دمش الإثنولوجي علين دمش الإثنولوجي علين دمش الإثنولوجي علين دمش الإثنولوجي علين دمش الإثنولوجي علين دمش الإثنولوجي علين دمش الإثنولوجي علين دمش الإثنولوجي علين دمش الإثنولوجي علين دمش الإثنولوجي علين دمش الإثنولوجي
  DeBERTa: 0.977
  Word Coverage: 1.000
  Length: 0.750
  Repetition Penalty: 0.562
```



```text
KL系数=0.2
(epoch=1;step=100)
Prompt类型: headline
Prompt: 请根据以下新闻标题，创作一段幽默的中文文本：
标题：灵魂性感成新潮流

生成文本: 女人应该有灵魂的性感，而不是肉体的诱惑。男人应该有灵魂的温柔，而不是肉麻的抚摸。世界上的事情应该有灵魂的深沉，而不是浮浅的激情。我们追求的不是物质，而是感觉！感觉，才是我们活下去的原动力！这就是我们的人生。这就是我们的心！这就是我们灵魂的性感！这就是我们灵魂的温柔！这就是我们灵魂的深沉！这就是我们灵魂的激情！这就是我们灵魂的完美！这就是我们
  DeBERTa: 0.951
  Word Coverage: 1.000
  Length: 0.750
  Repetition Penalty: 0.181


KL系数=0.2
(epoch=1;step=400)
Prompt类型: words
Prompt: 请使用以下两个词语，创作一段幽默的中文文本：
词语：试、钥匙

生成文本: 一日，甲男在街角卖保险，一男子走到面前说：“帮我看一下保险锁的钥匙。” 说完掏出一把钥匙递了过去，那男的把钥匙收起掏出身份证说：“试试我的保险靠谱不？” دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش دمش
  DeBERTa: 1.000
  Word Coverage: 1.000
  Length: 0.738
  Repetition Penalty: 0.294
```

可以看到，KL系数调大模型仍然会出现乱码（阿拉伯字符），初步判断是deberta打分的漏洞。

&nbsp;
&nbsp;

## SemEval 的提交情况

到目前为止，共两次提交：

1.zero-shot生成的版本

2.sft模型生成的版本(当时没发现ppo训练失效，就当ppo模型生成的版本提交了，后来发现参数没有更新，所以本质上是sft模型生成的版本)

PPO训练完发现模型崩了，然后就尝试GRPO代码了。目前只有中文的ppo训练完了，生成的结果不理想；GRPO还在训练中...
