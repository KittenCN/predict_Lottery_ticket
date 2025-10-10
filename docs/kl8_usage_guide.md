# KL8 快速使用指南

## 模块概览

| 模块名 | 主要功能 | 性能特征 | 适用场景 |
|--------|----------|----------|----------|  
| `kl8_analysis.py` | 基础统计分析和号码生成 | 单线程，内存友好 | 小规模测试，算法验证 |
| `kl8_analysis_plus.py` | 多进程并行分析 | 多进程，高性能 | 大批量号码生成 |
| `kl8_cash.py` | 中奖收益分析 | 单线程，精确计算 | 单期收益分析 |  
| `kl8_cash_plus.py` | 批量收益分析 | 多进程，批量处理 | 历史收益回测 |
| `kl8_running.py` | 任务调度管理 | 多任务编排 | 自动化批量执行 |

## 快速开始

### 1. 基础号码生成
```bash
# 生成10选5的号码，共50注
python src/analysis/kl8_analysis.py --cal_nums 5 --total_create 50 --limit_line 20

# 关键参数说明：
# --cal_nums: 选择号码个数 (1-10)  
# --total_create: 生成注数
# --limit_line: 分析历史数据期数
# --current_nums: 指定期数 (-1为最新期)
```

### 2. 高性能批量生成  
```bash
# 多进程生成大批量号码
python src/analysis/kl8_analysis_plus.py --cal_nums 10 --total_create 1000 --max_workers 8

# 性能提升参数：
# --max_workers: 并行进程数
# --simple_mode 1: 简化输出模式
```

### 3. 收益分析
```bash
# 分析单个预测文件的中奖情况
python src/analysis/kl8_cash.py --cash_file_name result_20231201_10_2023241

# 批量分析目录下所有文件
python src/analysis/kl8_cash.py --path "test_path"
```

### 4. 全自动批量任务
```bash  
# 自动执行多参数组合的分析任务
python src/analysis/kl8_running.py \
  --cal_nums_list "5,7,10" \
  --total_create_list "100,500" \
  --nums_range "2023200,2023210" \
  --running_mode 0
```

## 主要参数详解

### 通用参数
- `--name`: 彩票类型，默认"kl8"  
- `--download`: 是否下载最新数据 (0/1)
- `--current_nums`: 指定分析期数，-1表示最新期
- `--path`: 输出路径标识
- `--simple_mode`: 简化模式 (0=详细输出, 1=简化输出, 2=仅盈利输出)

### 生成控制参数
- `--cal_nums`: 每注选择号码数量 (1-10)
- `--total_create`: 生成总注数
- `--limit_line`: 分析历史数据期数  
- `--err_nums`: 错误阈值，超过则调整参数
- `--repeat`: 重复执行次数

### 高级参数
- `--analysis_history`: 是否进行历史偏差分析 (0/1)
- `--calculate_rate`: 是否计算动态概率率 (0/1)  
- `--check_in_main`: 是否启用主循环验证 (0/1)
- `--random_mode`: 随机模式 (0=智能生成, 1=纯随机)

## 输出文件说明

### 生成结果文件
```
results/result_20231201120000_10_2023241.csv
格式: result_{时间戳}_{选号数}_{期数}.csv
内容: 每行一注号码，按选号数排列
```

### 收益分析结果
```
kl8_running_results.txt  
内容: 各文件投注收益统计
格式: 路径, 第X张, 投入Y元, 奖金Z元, 返奖率W%
```

## 典型使用流程

### 场景1: 快速测试验证
```bash
# 1. 生成少量号码快速验证
python src/analysis/kl8_analysis.py --cal_nums 5 --total_create 10 --simple_mode 1

# 2. 分析生成结果收益
python src/analysis/kl8_cash.py --simple_mode 0
```

### 场景2: 大规模生产分析  
```bash
# 1. 启动多进程大批量生成
python src/analysis/kl8_analysis_plus.py \
  --cal_nums 10 --total_create 5000 \
  --max_workers 8 --simple_mode 1 \
  --path "production_run"

# 2. 并行分析所有结果
python src/analysis/kl8_cash_plus.py --path "production_run" --simple_mode 1
```

### 场景3: 历史回测分析
```bash
# 使用running模块进行全面历史回测
python src/analysis/kl8_running.py \
  --cal_nums_list "5,7,10" \
  --total_create_list "100,500,1000" \
  --nums_range "2023100,2023200" \
  --running_mode 0 \
  --max_workers 4
```

## 性能优化建议

### 内存优化
- 大批量任务使用 `--simple_mode 1` 减少内存占用
- 设置合理的 `--limit_line` 避免加载过多历史数据
- 定期清理 `results/` 目录下的临时文件

### CPU优化  
- 根据CPU核心数设置 `--max_workers`
- 使用 `kl8_analysis_plus.py` 替代单线程版本
- 避免同时运行多个高CPU消耗任务

### 磁盘IO优化
- 使用SSD存储加速文件读写
- 批量任务完成后再进行收益分析
- 考虑使用内存盘存储临时结果

## 常见问题排查

### 1. 生成速度慢
**原因**: 概率约束过严，大量号码被拒绝  
**解决**: 调整 `--limit_line` 和 `--err_nums` 参数

### 2. 内存不足
**原因**: 同时加载过多历史数据  
**解决**: 减少 `--limit_line` 或启用 `--simple_mode 1`

### 3. 收益分析失败
**原因**: 预测文件格式不匹配或路径错误  
**解决**: 检查文件命名格式和 `--path` 参数

### 4. 多进程任务卡死
**原因**: 进程数设置过高或系统资源不足  
**解决**: 减少 `--max_workers` 数量

## 结果解读

### 概率验证指标
系统会输出各维度的概率偏差：
- 重复率偏差: 与历史重复概率的差异
- 冷热号偏差: 冷热号比例的差异  
- 奇偶比偏差: 奇偶数比例的差异
- 分组偏差: 各号码组分布的差异
- 连续号偏差: 连续号码出现的差异

### 收益分析指标
- 投注成本: 总注数 × 2元/注
- 中奖分布: 各奖级中奖注数统计  
- 返奖率: 总奖金 / 总投注 × 100%
- 盈亏平衡: 返奖率 ≥ 100% 为盈利

**注意**: 系统仅供学习和研究使用，彩票投注有风险，请理性对待。