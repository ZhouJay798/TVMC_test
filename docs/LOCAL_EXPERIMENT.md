# 本地实验操作说明

SmolLM2-135M 权重实验请阅读 [专用操作与技术说明](SMOLLM2_EXPERIMENT.md)。本页介绍通用小文件演示。

本说明适用于仓库 `ZhouJay798/TVMC_test` 的 `main` 分支。使用 Git、Python 3.12 和网络连接安装依赖即可运行；不需要 GPU、数据库、API 密钥或五台机器。

## 1. 下载项目并安装依赖

先安装 Git 和 Python 3.12，然后打开终端。Windows 推荐 PowerShell。

```bash
git clone https://github.com/ZhouJay798/TVMC_test.git
cd TVMC_test
git switch main
```

如果已经下载过项目，进入现有目录后执行 `git pull --ff-only` 即可。若有本地修改，先保留自己的改动，不要强制覆盖。

### Windows PowerShell

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.12 --version
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

如果 `python3 --version` 已显示 Python 3.12，也可以用 `python3` 创建环境。以下命令均从项目根目录执行。直接调用虚拟环境里的 Python，无需执行激活脚本。

## 2. 先执行自动测试

Windows：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

macOS / Linux：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

预期结尾为：

```text
Ran 20 tests in ...s

OK
```

这里的 20 是测试方法数（含 7 项离线模型工作流测试及 5 项 Tardos 编码与指标测试），其中一个方法遍历全部 10 种两方组合。测试还检查空文件、二进制文件、份额越界、密文/nonce/tag/承诺篡改、错误身份、错误上下文和证明重放。若出现 `FAILED` 或 `ERROR`，保留错误堆栈排查，不要当作实验成功。

## 3. 运行完整演示

Windows：

```powershell
.\.venv\Scripts\python.exe demo.py
```

macOS / Linux：

```bash
.venv/bin/python demo.py
```

默认生成 47,104 字节的内存样本，然后依次完成：

1. 随机生成秘密材料，经 HKDF 派生 AES-256-GCM 密钥并加密样本。
2. 将秘密材料分成五份，生成 Feldman 公开承诺。
3. 枚举 `(1,2)、(1,3)、(1,4)、(1,5)、(2,3)、(2,4)、(2,5)、(3,4)、(3,5)、(4,5)`。
4. 对每个组合生成两方身份知识证明，验证证明与份额，再解密并逐字节比较原文。
5. 测试单份、份额篡改、重复份额、认证标签篡改与证明重放是否被拒绝。

终端会打印 JSON，并在 `artifacts/` 下生成两个文件：

| 文件 | 如何使用 |
| --- | --- |
| `report.json` | 查看恢复正确性、拒绝结果、耗时和证据大小 |
| `encrypted.json` | 查看密文、nonce、认证标签以及公开承诺；二进制字段采用十六进制表示 |

可用 VS Code、记事本或其他文本编辑器打开这些文件。重复运行会覆盖同一输出目录的同名文件；需要保留结果时使用不同的 `--output` 目录。

## 4. 使用自己的文件

例如把 `sample.txt` 放在项目根目录，再执行：

Windows：

```powershell
.\.venv\Scripts\python.exe demo.py "sample.txt" --output artifacts/my-file
Get-Content artifacts/my-file/report.json
```

macOS / Linux：

```bash
.venv/bin/python demo.py "sample.txt" --output artifacts/my-file
cat artifacts/my-file/report.json
```

也可以传入带引号的绝对路径，输入可以是文本、图片或任意二进制文件。先从小文件开始，程序会把整个文件读入内存。原始文件不会被修改，但不要将输入文件指定为输出目录中的 `report.json` 或 `encrypted.json`，以免输出覆盖输入。

**当前演示不会写出恢复后的明文文件。** 它在内存中对每个两方组合检查 `restored == data`，然后把结果写入 `correct`。份额和身份私钥也不落盘，因此关闭进程后无法仅凭 `encrypted.json` 再次解密。这是完整流程实验，不是持久化文件保管工具。

## 5. 判断实验是否成功

### 恢复正确性

`pairs` 应有 10 项，每项 `correct` 均为 `true`。以下仅为结构示例，实际耗时每次不同：

```json
{
  "parties": [1, 2],
  "correct": true,
  "proof_generation_ms": 3.5,
  "verified_recovery_ms": 7.2
}
```

### 攻击拒绝结果

`attacks_rejected` 的以下六项（非空输入）都应为 `true`。这里 `true` 表示攻击被拒绝，是预期结果。

| 字段 | 对应实验 |
| --- | --- |
| `single_share` | 只提供一份，恢复接口拒绝 |
| `tampered_share` | 修改第一方份额数值，份额验证拒绝 |
| `duplicate_share` | 将同一份提交两次，重复编号检查拒绝 |
| `tampered_ciphertext` | 翻转密文本体一位，解密认证失败；空文件无可翻转密文，此项不执行 |
| `tampered_tag` | 修改 GCM 认证标签，解密认证失败 |
| `proof_replay` | 有效证明先通过一次，再用同一挑战重复提交，被拒绝 |

### 耗时

| 字段 | 单位与范围 |
| --- | --- |
| `input_bytes` | 输入文件字节数 |
| `encryption_ms` | 毫秒；包含秘密生成、承诺、加密和份额生成 |
| `pairs[].proof_generation_ms` | 毫秒；该组合两方证明生成耗时之和，使用预先计算的文件上下文 |
| `pairs[].verified_recovery_ms` | 毫秒；两方证明校验、上下文计算、份额验证、秘密恢复和解密的总耗时 |
| `attack_check_ms` | 毫秒；各拒绝检查的耗时，键与攻击结果对应 |

这些是本机单次墙钟测量，不是各底层算法的独立基准。文件大小、CPU、系统负载都会影响结果；不要求与你看到的示例耗时相等。实验结果汇总使用显式错误检查，任一恢复或拒绝结果不符合预期就报错。

### 证据大小

`evidence_bytes` 统计约定二进制编码的大小，不是 JSON 文件占用空间，也不是完整网络协议的通信量。

| 字段 | 字节数 | 内容 |
| --- | ---: | --- |
| `share` | 33 | 1 字节编号 + 32 字节标量 |
| `feldman_commitments` | 130 | 文件级两个未压缩 P-256 点 |
| `identity_public_key` | 65 | 单方注册公钥 |
| `schnorr_proof` | 97 | 单方 65 字节点 + 32 字节响应 |
| `challenge_nonce` | 32 | 单次挑战 |
| `file_context_hash` | 32 | 文件上下文摘要 |

每个两方组合包含两份证明；公钥、挑战、上下文和承诺需按你的通信模型单独核算。报告只记录证明大小，不导出证明原文。

## 6. 重复观察结果

使用不同目录保存多次运行，例如：

Windows：

```powershell
1..5 | ForEach-Object {
    .\.venv\Scripts\python.exe demo.py "sample.txt" --output "artifacts/run-$_"
    if ($LASTEXITCODE -ne 0) { throw "实验失败" }
}
```

macOS / Linux：

```bash
for i in 1 2 3 4 5; do
    .venv/bin/python demo.py "sample.txt" --output "artifacts/run-$i" || break
done
```

先确认每次恢复与拒绝结果都正确，再比较耗时，可汇总多次测量的中位数。由于每次重新生成随机秘密、身份私钥和 nonce，密文不同是正常现象。实验输出被 `.gitignore` 忽略，不会自动提交到仓库。

## 7. 常见问题与范围

| 现象 | 处理 |
| --- | --- |
| 找不到 `py` 或 `python3.12` | 安装 Python 3.12，重新打开终端，检查 PATH |
| Linux 提示缺少 `venv` / `ensurepip` | 安装与 Python 版本匹配的系统 venv 包，再创建环境 |
| `No module named Crypto` | 用上述虚拟环境的 Python 重新执行 `-m pip install -r requirements.txt`；不要安装名为 `crypto` 的其他包 |
| 找不到 `demo.py` 或依赖文件 | 确认当前目录是包含 README 和 demo.py 的项目根目录 |
| `FileNotFoundError` | 检查输入文件路径；有空格时加引号 |
| 下载依赖失败 | 检查 PyPI 网络/代理设置；保留 TLS 校验，不要用关闭校验绕过错误 |
| 找不到恢复文件 | 当前版本只在内存比较恢复结果，不导出恢复文件，见第 4 节 |

五方由单进程模拟，不涉及跨机器通信。身份知识证明验证的是“持有预注册公钥对应的私钥”，不是实名身份。单份拒绝是接口测试，不是秘密共享安全性的数学证明。本项目用于研究实验，尚未经过生产安全审计；协议细节与限制见根目录 README。
