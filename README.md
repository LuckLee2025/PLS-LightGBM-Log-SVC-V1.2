# PLS数据处理与分析系统

## 1. pls_data_processor.py - 数据获取与预处理

### 主要功能
这个脚本负责从网络上获取排列三历史开奖数据，并将其保存为标准化的CSV文件。

### 核心逻辑流程
1. **数据获取**：
   - 从网络文本文件获取数据（`fetch_pl3_data`）- 从 https://data.17500.cn/pl3_asc.txt 获取
   - 支持自动去除日期信息，保留期号和3位数字

2. **数据解析**：
   - 解析文本数据（`parse_txt_data`）- 将原始文本转换为结构化数据
   - 验证数字范围（0-9）和数据完整性

3. **数据存储**：
   - 保存为CSV文件（`pls.csv`）
   - 按期号升序排序
   - 数据格式：Seq, red_1, red_2, red_3

4. **错误处理**：
   - 多种编码尝试（utf-8, gbk, latin-1）
   - 网络连接错误处理
   - 数据格式验证

### 技术特点
- 使用requests库进行网络请求
- 使用pandas进行数据处理和CSV操作
- 实现了日志系统和进度显示
- 自动备份现有数据文件

## 2. pls_analyzer.py - 数据分析与预测

### 主要功能
这个脚本负责分析排列三历史数据，识别模式，训练机器学习模型，并生成下一期的推荐号码组合。

### 核心逻辑流程
1. **数据加载与预处理**：
   - 加载CSV数据（`load_data`）
   - 数据清理和结构化（`clean_and_structure`）
   - 特征工程（`feature_engineer`）- 创建和值、跨度、奇偶计数、大小分布等特征

2. **历史统计分析**：
   - 频率和遗漏分析（`analyze_frequency_omission`）- 分析各位置号码的出现频率和遗漏情况
   - 模式分析（`analyze_patterns`）- 分析奇偶比、大小比、和值分布、跨度分布等
   - 关联规则挖掘（`analyze_associations`）- 使用Apriori算法挖掘号码关联关系

3. **机器学习模型训练**：
   - 创建滞后特征（`create_lagged_features`）
   - 训练多种模型（`train_prediction_models`）：
     - LightGBM分类器 - 为每个位置的每个数字训练独立模型
   - 预测下一期号码概率（`predict_next_draw_probabilities`）

4. **号码评分与组合生成**：
   - 计算综合得分（`calculate_scores`）- 结合频率、遗漏和ML预测概率
   - 生成推荐组合（`generate_combinations`）- 基于得分和历史模式
   - 应用多样性控制确保推荐组合的差异性

5. **特色功能**：
   - 支持反向思维策略（可选）
   - 模式匹配奖励（奇偶比、大小比、和值范围）
   - 关联规则奖励

6. **结果输出**：
   - 生成详细分析报告
   - 推荐10注号码组合
   - 包含详细的评分分析

### 排列三特色特征
- **和值特征**：三位数字之和（0-27）
- **跨度特征**：最大数字与最小数字的差值
- **奇偶特征**：奇数个数统计
- **大小特征**：大数（5-9）个数统计
- **质合特征**：质数（2,3,5,7）个数统计
- **形态特征**：组六、组三、豹子的分类
- **连号特征**：连续数字的个数
- **重复特征**：重复数字的个数

### 技术特点
- 使用LightGBM进行机器学习预测
- 实现了特征工程和滞后特征创建
- 使用关联规则挖掘（Apriori算法）
- 采用模块化设计，功能分离清晰
- 支持Optuna参数优化（可选）

## 3. pls_bonus_calculation.py - 奖金计算器

### 主要功能
这个脚本负责验证推荐号码的实际表现，计算中奖情况和奖金。

### 核心逻辑流程
1. **数据匹配**：
   - 读取历史开奖数据
   - 查找对应期号的分析报告
   - 解析推荐号码

2. **中奖验证**：
   - 直选验证：三个位置完全匹配
   - 组选验证：三个数字匹配（不考虑顺序）

3. **奖金计算**：
   - 直选：1040元
- 组选3：346元（中奖号码中任意两位数字相同，所选号码与中奖号码相同且顺序不限）
- 组选6：173元（所选号码与中奖号码相同且顺序不限）

4. **报告生成**：
   - 记录验证结果到 `latest_pls_calculation.txt`
   - 保留最近10次验证记录

## 4. pls_wxpusher.py - 微信推送功能

### 主要功能
提供微信推送功能，用于推送排列三分析报告和验证结果。

### 核心功能
1. **分析报告推送**：
   - 推送每期的预测报告
   - 包含推荐号码和分析摘要
   - 显示上期验证结果

2. **验证报告推送**：
   - 推送中奖验证结果
   - 显示开奖号码和中奖情况

3. **错误通知**：
   - 系统异常时发送通知
   - 日常运行状态报告

## 系统整体效果

这个排列三分析系统具有以下特点：

1. **数据流向**：
   - `pls_data_processor.py` 负责获取和预处理数据，生成标准化CSV
   - `pls_analyzer.py` 读取CSV，进行分析和预测

2. **系统优势**：
   - **数据获取的健壮性**：稳定的数据源，多编码支持，错误处理
   - **分析的全面性**：结合统计分析和机器学习
   - **预测的科学性**：基于7000+期历史数据训练
   - **可验证性**：通过实际开奖结果验证预测效果

3. **排列三专业特色**：
   - **适配排列三规则**：3位数字，每位0-9，允许重复
   - **专业特征工程**：和值、跨度、奇偶比、大小比、质合比等
   - **形态分析**：组六、组三、豹子号的识别和分析
   - **位置特化**：为每个位置独立分析和预测

4. **实际应用效果**：
   - 系统能够基于历史数据识别排列三的特有模式
   - 使用机器学习预测各位置各数字的出现概率
   - 生成综合评分较高的号码组合
   - 提供详细的分析报告和可视化结果

## 快速开始

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

2. **获取数据**：
   ```bash
   python pls_data_processor.py
   ```

3. **运行分析**：
   ```bash
   python pls_analyzer.py
   ```

4. **验证结果**（在有新开奖后）：
   ```bash
   python pls_bonus_calculation.py
   ```

## 配置说明

- 修改 `pls_analyzer.py` 中的 `DEFAULT_WEIGHTS` 可调整算法权重
- 修改 `ENABLE_OPTUNA_OPTIMIZATION` 可启用参数优化
- 微信推送需要配置 `WXPUSHER_APP_TOKEN` 等环境变量

## GitHub 同步与部署指南

### 将项目同步到GitHub仓库

如果您需要将本地项目同步到GitHub仓库 `https://github.com/LJQ-HUB-cmyk/PLS-LightGBM-Log-SVC`，请按照以下步骤操作：

#### 第一次同步（初始化）

1. **初始化本地Git仓库**：
   ```bash
   git init
   ```

2. **添加远程仓库**：
   ```bash
   git remote add origin https://github.com/LJQ-HUB-cmyk/PLS-LightGBM-Log-SVC.git
   ```

3. **添加所有文件到Git**：
   ```bash
   git add .
   ```

4. **创建首次提交**：
   ```bash
   git commit -m "Initial commit: 排列三分析系统优化版"
   ```

5. **推送到GitHub**：
   ```bash
   git push -u origin main
   ```

#### 日常更新同步

1. **查看修改状态**：
   ```bash
   git status
   ```

2. **添加修改的文件**：
   ```bash
   git add .
   # 或者添加特定文件
   git add pls_analyzer.py pls_data_processor.py
   ```

3. **提交修改**：
   ```bash
   git commit -m "更新描述，例如：优化分析算法，提升预测准确率"
   ```

4. **推送到GitHub**：
   ```bash
   git push origin main
   ```

#### 从GitHub拉取最新更改

如果仓库有其他人的更新或GitHub Actions的自动提交：

```bash
git pull origin main
```

### GitHub Actions 自动化工作流

本项目已配置GitHub Actions自动化工作流（`.github/workflows/daily-analysis.yml`），具有以下功能：

#### 🕰️ 运行计划
- **每天早上8点（北京时间）** 自动运行
- 也可以通过GitHub网页手动触发

#### 🔄 自动化流程
1. **数据获取**：运行 `pls_data_processor.py` 获取最新排列三数据
2. **奖金计算**：运行 `pls_bonus_calculation.py` 验证上期推荐
3. **分析预测**：运行 `pls_analyzer.py` 生成新一期分析和推荐
4. **文件管理**：
   - 创建固定名称的最新报告文件 `latest_pls_analysis.txt`
   - 自动清理历史报告（保留最新3份）
   - 提交所有更新到GitHub

#### 📁 自动提交的文件
- `pls.csv` - 最新的排列三历史数据
- `latest_pls_analysis.txt` - 最新分析报告的固定名称副本
- `latest_pls_calculation.txt` - 最新验证计算结果
- `pls_analysis_output_*.txt` - 带时间戳的详细分析报告
- `weights_config.json` - 优化后的权重配置

#### 🎯 实现效果
- **无人值守运行**：每天自动获取数据、分析、推荐
- **数据同步**：所有结果自动同步到GitHub仓库
- **历史追踪**：完整的分析历史和版本控制
- **持续集成**：代码更新后自动测试运行

### 环境变量配置（可选）

如果要启用微信推送功能，需要在GitHub仓库设置中添加以下Secrets：

1. 进入GitHub仓库页面
2. 点击 `Settings` → `Secrets and variables` → `Actions`
3. 添加以下环境变量：
   ```
   WXPUSHER_APP_TOKEN=your_app_token
   WXPUSHER_UID=your_uid
   ```

### 故障排查

#### 常见问题及解决方案

1. **推送被拒绝（push rejected）**：
   ```bash
   git pull origin main --rebase
   git push origin main
   ```

2. **合并冲突（merge conflicts）**：
   ```bash
   git status  # 查看冲突文件
   # 手动编辑冲突文件，然后：
   git add .
   git commit -m "解决合并冲突"
   git push origin main
   ```

3. **GitHub Actions失败**：
   - 检查仓库的Actions页面查看错误日志
   - 常见原因：网络连接、依赖安装、权限问题

4. **本地文件与远程不同步**：
   ```bash
   git fetch origin
   git reset --hard origin/main  # 注意：这会丢失本地未提交的修改
   ```

### 最佳实践建议

1. **定期同步**：建议每次重要修改后及时推送到GitHub
2. **提交信息**：使用清晰的提交信息，方便追踪历史
3. **分支管理**：重大功能开发建议使用分支
4. **备份重要数据**：重要的配置文件建议本地备份

## 注意事项

- 彩票具有随机性，本系统仅供学习和研究使用
- 请理性投注，量力而行
- 系统预测结果不构成投注建议

后续更新同步
查看更改状态

git status
添加更改的文件

git add 更改的文件
# 或添加所有更改
git add .
提交更改

git commit -m "优化输出日志文档"
推送到GitHub

git push
常见问题解决
如果遇到分支名称问题（如master与main）：

git branch -M main  # 将当前分支重命名为main
如果需要强制推送（谨慎使用）：

git push -f origin main
如果需要从GitHub拉取最新更改：

git pull origin main
