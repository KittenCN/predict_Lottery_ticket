# 环境复现与依赖管理

建议使用 conda 管理二进制依赖（如 numpy、tensorflow-intel、pytorch 等），并使用 pip 安装剩余纯 Python 包。

推荐流程（在仓库根目录）：

1. 使用 conda 创建环境：

```bash
conda env create -f environment.yml
conda activate predict_lottery
```

2. 使用 pip 安装可移植锁定的 pip 包：

```bash
python -m pip install -r requirements.lock.txt
```

3. 验证导入（参见 docs/verify.md）

常见问题与解决：

- 如果 `pip install -r requirements.lock.txt` 出现依赖冲突（ResolutionImpossible），请确保通过 conda 先安装核心二进制依赖（numpy、mkl、tensorflow-intel、torch）。
- 如果遇到本地 file:/// 或 conda-build 路径（代表 pip freeze 中包含本地构建），请使用 `requirements.lock.txt`（已过滤本地路径）而非仓库根的 `requirements.txt`。
- 如需包含私有 VCS 包，请在激活环境后手动 `pip install -e git+https://...`。
