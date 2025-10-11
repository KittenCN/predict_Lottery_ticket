# 运维手册（Ops Guide）

## 1. 运行环境要求
- Python 3.11（推荐通过 `conda create -n python311 python=3.11` 创建环境）；
- 依赖锁定在 `requirements.txt`，包含 TensorFlow 2.15.1；
- Linux / macOS 原生支持，Windows 建议在 WSL2 或 Docker 中运行；
- GPU 环境需预装 CUDA ≥ 12.2（可选）。

> **KL8（快乐8）相关功能已于2025-10迁移至独立项目 [KL8-Lottery-Analyzer](https://github.com/KittenCN/kl8-lottery-analyzer)。本仓库仅支持双色球、大乐透、排列三、七星彩、福彩3D等玩法。**

> **2024-06起，红球预测结果已确保每注号码唯一，彻底避免重复。运维时如遇预测异常请优先检查 `src/pipeline.py` 相关逻辑。**

## 2. 启动与管理
### 本地启动
```bash
conda activate python311
make setup
make train   # 训练默认模型
make predict # 生成预测并落盘
```

### Docker 运行
```bash
docker build -t lottery-predict .
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/predict:/app/predict \
  lottery-predict python scripts/train.py --name ssq --window-size 5
```

### 定时任务建议
- 使用 `cron` / `systemd timer` 触发 `python scripts/get_data.py` 更新历史数据；
- 训练脚本建议每日/每周执行，根据数据规模调整窗口与 epoch；
- 预测脚本支持 `--save` 参数，自动落盘 JSON 结果，可进一步推送至自定义通知渠道。

## 3. 日志与监控
- 使用 `loguru` 输出，默认 INFO 级别，日志目录：`logs/`；
- 关键事件：
  - 数据下载成功/失败与总期数；
  - 训练过程指标：loss、accuracy、学习率调整；
  - 模型保存路径与元数据；
  - 预测结果与窗口信息；
- 可通过 `export LOGURU_LEVEL=DEBUG` 或配置 `.env` 调整日志级别；
- Docker 运行时建议使用 `docker logs` 结合日志挂载。

## 4. 健康检查
- Docker 镜像内置 `HEALTHCHECK`，确保 Python 运行时可用；
- 训练完成后会生成 `model/<code>/window_<n>/metadata.json`，可作为成功标识；
- 可扩展脚本：检查 `metadata.json` 中的时间戳与 `trained_on_issues` 是否覆盖最新期号。

## 5. 故障排查
| 场景 | 排查步骤 | 解决建议 |
| ---- | -------- | -------- |
| 下载失败 | 检查网络连通性、域名是否被墙；查看 `logs/` 中的报错 | 配置代理或切换备用数据源 |
| 训练异常 | 查看日志中的数据维度、loss 变化；确认 `data.csv` 是否足够期数 | 调整 `window_size`、降低 batch_size、增加数据 |
| 预测返回空 | 检查模型目录是否存在、window 是否与训练一致 | 重新训练或指定正确的 `--window-size` |
| TensorFlow 报错 | 确认 Python/NumPy/TensorFlow 版本匹配；GPU 场景检查 CUDA | 重新创建 `python311` 环境并执行 `make setup` |

## 6. 备份与恢复
- 数据：`data/<code>/data.csv` 可定期备份到对象存储或版本库；
- 模型：建议保存 `model/<code>/window_<n>/` 整个目录，包括 `.keras` 与 `metadata.json`；
- 预测：`predict/<code>/prediction_*.json` 可用于复盘，必要时推送至备份存储。

## 7. 安全注意事项
- 所有 HTTP 请求通过白名单校验，不建议随意修改目标域名；
- 避免将 `.env`、密钥或训练得到的敏感结果上传仓库；
- 若部署到公网，建议在外层添加身份认证与速率限制。

## 8. 指标建议（可选扩展）
- 训练时记录 loss/accuracy 曲线，可输出到 TensorBoard 或 Prometheus；
- 预测结果统计（命中率、号码分布）可定期写入外部数据库，便于可视化；
- 若扩展 REST API，可加入请求耗时、错误码、QPS 等监控。

## 9. 常用命令速查
| 目的 | 命令 |
| ---- | ---- |
| 更新依赖 | `make setup` |
| 代码检查 | `make fmt && make lint` |
| 运行测试 | `make test` |
| 全量 CI | `make ci` |
| 下载历史数据 | `python scripts/get_data.py --name ssq` |
| 训练模型 | `python scripts/train.py --name ssq --window-size 5 --red-epochs 40` |
| 预测并保存 | `python scripts/predict.py --name ssq --window-size 5 --save` |

如遇未覆盖的问题，请参考 `ASSUMPTIONS.md` 与 `docs/decision_record.md`，或在 `agent_report.md` 中记录新发现的运维要点。***
