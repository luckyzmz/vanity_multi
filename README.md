# vanity_multi — 多链 BIP39 vanity 地址生成器

单文件源码 + 单个 exe，把 5 条链的 vanity 搜索合并成一个程序：`eth` / `btc` / `trx` / `sol`（随机 seed 快速版）/ `solbip`（BIP39 助记词版）。

不带参数运行会进交互菜单；带 `-h` 看完整帮助；带 `<chain> <mode> ...` 直接进某条链。

---

## 1. 文件与依赖

| 文件 | 作用 |
|---|---|
| `vanity_multi.cu` | 唯一源文件，五条链各在一个命名空间（`eth` / `btc` / `trx` / `sol` / `solbip`），末尾是统一入口 `main` |
| `bip39_words.h` | BIP39 2048 词表（`__constant__ char BIP39_WORDS[2048][9]`），**必须和 .cu 放同一目录** |
| `build_multi.bat` | Windows 编译脚本（vcvars64 + nvcc） |
| `vanity_multi.exe` | 编译产物 |

运行时会在当前目录生成命中记录和 secret 文件，见第 7 节。

编译依赖：CUDA Toolkit（nvcc，本机为 13.3）、MSVC BuildTools 2022（vcvars64.bat）、NVIDIA GPU（本机 RTX 4080 SUPER，`sm_89`）。

---

## 2. 编译

```bat
build_multi.bat
```

等价的核心命令：

```bat
nvcc -O2 -arch=sm_89 -Xcompiler=/utf-8 -o vanity_multi.exe vanity_multi.cu
```

### 几个必须知道的坑

1. **单次编译要 25 分钟以上**。`ptxas` 阶段会长时间满载、`multi_build.log` 长时间是 0 字节，这不是卡死，别中途重开。
2. **PATH 里没有 cmd**。用 PowerShell 调脚本要写全路径：
   ```powershell
   & "C:\Windows\System32\cmd.exe" /c "build_multi.bat"
   ```
3. **重编前先删旧 exe**（否则 `LNK1104: cannot open file`），并确认没有残留的 `nvcc / cicc / cudafe++ / ptxas / cl` 进程。
4. **源文件的坑（合并时踩过）**：
   - 含 UTF-8 BOM 的源文件不能直接插进 `namespace` 中间，会变成非法字符；
   - nvcc 的 `cudafe++` 处理 `R"(...)"` 原始字符串会 internal error（0xC0000409）；
   - nvcc 13.3 对**含非 ASCII 字节的字符串字面量不展开转义**：`"中文\n"` 会把 `\n` 原样输出成反斜杠 + n；而含中文又不带任何转义又会报 `missing closing quote`。
     所以曾有版本把含中文的 `printf / fputs` 字面量整段改成 **纯 ASCII 的 `\NNN` 八进制转义**（`merge_multi.py` 的 `esc_nonascii_literals()`；八进制最多 3 位，不会像 `\x` 那样贪婪吞掉后随字符）。现在控制台文案已全部英文化，这条只是保底机制，源码里已不存在含非 ASCII 的输出行。
5. `bip39_words.h` 不在同目录会直接编译失败。
6. **控制台文案一律英文**。Windows 控制台默认代码页 936（GBK），程序输出 UTF-8 中文会整屏乱码（`澶氶摼` 这种就是 UTF-8 被当 GBK 解出来的结果）。所以 `-h`、菜单、进度行、提示语全部是纯 ASCII 英文；这份 README 仍是中文。

---

## 3. 快速开始

```bat
vanity_multi.exe                          :: 交互菜单
vanity_multi.exe -h                       :: 完整帮助
vanity_multi.exe eth vanity 0x12345 8 1024
vanity_multi.exe btc vanity 1abc 0 5 1024
vanity_multi.exe trx vanity Txxx 4 1024
vanity_multi.exe sol vanity Ase 1024
vanity_multi.exe solbip vanity Ase 1024
```

菜单模式：先打印帮助 → 选链（1-5）→ 输入该链的参数（例 `vanity Ase 1024`）→ 跑完自动回到菜单；直接回车返回菜单，`q` 退出，`h` 再打一遍帮助。

---

## 4. 链与派生口径

| 子命令 | 链 | 派生 | 曲线/哈希 | 地址形态 |
|---|---|---|---|---|
| `eth` | Ethereum | BIP39 → PBKDF2 → BIP32/44 `m/44'/60'/0'/0/N` | secp256k1 + keccak256 | `0x` + 40 hex |
| `btc` | Bitcoin | `m/44'/0'/0'/0/N` | secp256k1 + hash160 / bech32 | `1…` P2PKH、`3…` P2SH-P2WPKH、`bc1q…` P2WPKH、`bc1p…` P2TR |
| `trx` | TRON | `m/44'/195'/0'/0/N` | secp256k1 + keccak256 | `T…`（34 字符，Base58Check，0x41 前缀） |
| `sol` | Solana（快速） | 随机 32B seed → SHA512 → clamp | ed25519 | Base58（32B 公钥） |
| `solbip` | Solana（助记词） | BIP39 → PBKDF2(2048) → SLIP-0010 `m/44'/501'/0'/0'` | ed25519 | Base58 |

要点：

- **每条链都是无限跑**：命中不停，只有 Ctrl+C 退出；每批都会重置去重表，同一个前缀会持续产出新命中。
- **nidx**（`eth/btc/trx`）：一个助记词派生多少个 HD 索引地址（`m/44'/…/N`，N=0..nidx-1）。默认 1。
- **BTC 的 addrtype 是数字**，默认 3：
  `0`=P2PKH(`1…`)、`1`=P2SH-P2WPKH(`3…`)、`2`=P2WPKH(`bc1q…`)、`3`=P2TR(`bc1p…`)。
  类型和前缀必须自洽，否则永远搜不到（例如搜 `1abc` 却用类型 3）。
- **P2TR 说明**：本工具的 P2TR 是「原始 x-only 公钥」直接编码成 bech32m，**不是 BIP86 的 tweak 地址**。要往 `bc1p` 地址收币并能花费，请按 BIP86 自行 tweak；本工具只保证私钥/助记词可恢复，不保证该 `bc1p` 地址是 BIP86 标准地址。
- **TRX 只有一个地址**：GasFree 账户地址是服务方按你的 EOA 确定性派生的（资产实际落在 GasFree 账户），本工具不输出它，需用钱包或 provider API 查询。

---

## 5. 子命令与参数

### 5.1 eth / btc / trx

```bat
vanity_multi.exe eth vanity <prefix>          [nidx] [blocks]
vanity_multi.exe eth file   <prefix_file>     [nidx] [blocks]
vanity_multi.exe eth selftest
vanity_multi.exe eth bench                    [iters] [blocks]

vanity_multi.exe btc vanity <prefix>      <addrtype> [nidx] [blocks]
vanity_multi.exe btc file   <prefix_file> <addrtype> [nidx] [blocks]
vanity_multi.exe btc selftest
vanity_multi.exe btc bench                    [iters] [blocks]

vanity_multi.exe trx vanity <prefix>          [nidx] [blocks]
vanity_multi.exe trx file   <prefix_file>     [nidx] [blocks]
vanity_multi.exe trx selftest
vanity_multi.exe trx bench                    [iters] [blocks]
```

- `blocks`：CUDA 网格块数，每块 256 线程，默认 1024。
- `iters`：`bench` 的迭代数，每迭代 = `blocks*256` 个候选。
- `nidx` 上限 1,000,000（实际用 1~8 就够）。

### 5.2 sol / solbip

```bat
vanity_multi.exe sol    vanity <prefix>  [blocks]
vanity_multi.exe sol    file   <prefix_file> [blocks]
vanity_multi.exe sol    bench  [iters] [blocks]
vanity_multi.exe sol    selftest

vanity_multi.exe solbip vanity <prefix>  [blocks]
vanity_multi.exe solbip file   <prefix_file> [blocks]
vanity_multi.exe solbip bench  [iters] [blocks]
vanity_multi.exe solbip selftest
```

`blocks` 默认 512。

### 5.3 前缀规则

| 链 | 允许的前缀 | 说明 |
|---|---|---|
| ETH | `0x` + 十六进制（`0x` 可省略，`12345` 与 `0x12345` 等价） | 只取前 `NL` 位十六进制数字 |
| BTC | `1` / `3` / `bc1q` / `bc1p` 开头 | Base58 部分区分大小写；bech32 全小写 |
| TRX | 必须以 `T` 开头 | 之后是 Base58 字符集 |
| SOL | Base58 字符集，最长 44 | 区分大小写 |

前缀长度不限，且**前缀文件支持 2GB 以上**（流式解析、去重、末行无换行也能读）。

---

## 6. 输出与实时速率

- 命中**立刻**无缓冲追加写入 `<chain>_hits.txt`（`setvbuf(_IONBF)`），Ctrl+C 强杀也不会丢已命中的记录。
- 每批结束刷新一行进度：`[ 12.3 s] 340703 mnem/s | 340703 addr/s | hit N`，**累计口径**（总候选 / 总耗时），不是瞬时值。
- ETH 的批量固定为 1 批 1 迭代（`blocks*256` 个候选），速率就是实测值。

各链命中字段：

| 链 | 字段 |
|---|---|
| eth | `HIT [p] 0x… -> m/44'/60'/0'/0/N`、mnemonic、address、private(hex) |
| btc | `HIT [p] "<地址>" -> m/44'/0'/0'/0/N`、mnemonic、address、private(hex) |
| trx | `HIT [p] "T…" -> m/44'/195'/0'/0/N`、mnemonic、address、private(hex) |
| sol | `HIT [p] prefix "…"`、Address、Secret key(64B base58)、Seed(32B base58)、Keypair JSON |
| solbip | 同上 + `Mnemonic (BIP39 12 words)` + `Path m/44'/501'/0'/0'` |

---

## 7. 运行期生成的文件

| 文件 | 说明 |
|---|---|
| `eth_hits.txt` / `btc_hits.txt` / `trx_hits.txt` | 命中记录（含助记词与私钥） |
| `sol_hits.txt` | `sol` 快速版命中记录 |
| `sol_bip39_hits.txt` | `solbip` 命中记录 |
| `.eth_vanity_secret` / `.btc_vanity_secret` / `.trx_vanity_secret` / `.sol_vanity_secret` | 32B 随机 secret，首次运行自动生成，决定助记词序列；不打印 |

- **secret 决定序列**：同一 secret + 同一起点才会复现同一批地址。丢了不影响从 hits 里的助记词恢复钱包，但无法复现同一序列 —— 建议离线备份。
- `sol` 快速版**不落 secret**：每次运行随机生成 run salt（启动时打印 32B hex），想复现就自己记下这串 salt。
- 想改路径：环境变量 `ETH_VANITY_SECRET` / `BTC_VANITY_SECRET` / `TRX_VANITY_SECRET` / `SOL_VANITY_SECRET`。
- **Windows 上 hits 文件不能 chmod 0600**（源码里该调用被 `#ifndef _WIN32` 跳过）。文件里是明文助记词 + 私钥，请放在私有目录，别放同步盘/共享目录。

---

## 8. 实测速率

（RTX 4080 SUPER / `sm_89` / `-O2`；2026-11-11 终版 exe 实测）

| 链 | 命令 | 结果 |
|---|---|---|
| eth | `eth bench 8 1024` | 2,097,152 次 / 6.17 s = **339,653 addr/s** |
| btc (P2TR) | `btc bench 3 8 1024` | 2,097,152 次 / 5.98 s = **350,504 addr/s** |
| trx | `trx bench 8 1024` | 2,097,152 次 / 5.99 s = **350,300 addr/s** |
| sol | `sol bench 64 1024` | 16,777,216 keys / 2.23 s = **7,519,436 keys/s** |
| solbip | `solbip bench 1 8192` | 2,097,152 keys / 3.93 s = **534,132 keys/s** |

真实搜索（vanity 模式，8~10 s 采样）的累计速率与上表同量级：`eth vanity 0x1` 稳定 320 k mnem/s、`btc vanity 1abc 0 1 1024` 340~362 k、`trx vanity T1 1 1024` 339 k。

> 早期版本在「短前缀、命中频繁」时速率会显示成 2 M mnem/s（约 6.6 倍虚高）。原因是 kernel 开头有 `if (*d_found) return;`：一个线程命中后整批其余线程立即退出，而主机端仍按整批 `blocks*256` 计数。终版已删掉这句，每批一定跑满，速率就是真实吞吐。

速率量级参考（历史实测，单链版）：

- `sol` 快速版：`bench 64 1024` → 16,777,216 keys / 2.12 s ≈ 7.9 M keys/s；长测 `bench 512 2048` → 268,435,456 keys / 70.06 s ≈ 3.8 M keys/s；真实搜索稳态约 13 M addr/s（短批）。
- `solbip`：`bench 1 32768` → 8,388,608 keys / 15.39 s ≈ 545 k keys/s（每候选 2048 轮 PBKDF2 = 8192 次 SHA-512 压缩，慢是硬成本）。

---

## 9. 验证

### 9.1 内置自检（不联网、不跑搜索）

```bat
vanity_multi.exe eth selftest
vanity_multi.exe btc selftest
vanity_multi.exe trx selftest
vanity_multi.exe sol selftest
vanity_multi.exe solbip selftest
```

三条 secp256k1 链各打印 11 行曲线/标量向量（`1G`、`2G`、`3G`、`(p-1)+1`、`(n-1)+1`、`(n-2)+3`、`a*a^-1`、`123456789*987654321` 等），按已知常量核对（例如 `1G x = 59f2815b16f81798`、`2G x = abac09b95c709ee5`、`3G x = 8601f113bce036f9`）；`sol` 与 `solbip` 打印显式结论 `selftest: 10 pass, 0 fail` / `selftest: 12 pass, 0 fail`（ed25519，RFC 8032 的 `d75a9801…`、`3d4017c3…` 等向量）。

### 9.2 BIP39 标准向量

```bat
vanity_multi.exe eth test
```

固定熵 → 助记词，`00×16` 得 `abandon … about`，`7f×16` 得 `legal winner … yellow`，`80×16` 得 `letter advice … above`，与 BIP39 官方一致。

### 9.3 BTC 派生逐行对照

```bat
vanity_multi.exe btc btctest
```

输出 N=0 / N=3 的 priv / pub_comp / hash160 / P2PKH / P2WPKH / P2TR / mnemonic，可与单链版 `btc_vanity_bip39_v2.exe` 逐行 diff（本轮已做过：两份 17 行完全一致）。

### 9.4 命中记录的独立复算

仓库同目录的 `verify_multi_hits.py` 不复用被验证的程序，直接用 `hashlib / hmac / ecdsa / pycryptodome` 从助记词重算：

```powershell
python verify_multi_hits.py eth eth_hits.txt
python verify_multi_hits.py btc btc_hits.txt
python verify_multi_hits.py trx trx_hits.txt
```

本轮结果：ETH 47/47、BTC 2/2、TRX 7/7 全部 ok，0 mismatch。

> 提醒：这个脚本自己也要先做锚点自校验（它最初漏了 BIP44 的硬化位 `0x80000000`，导致全员 mismatch）。用 BIP39 标准助记词算出的 ETH 地址 `0x9858effd232b4033e47d90003d41ec34ecaeda94` 可作锚点。

---

### 9.5 本轮（终版 exe）验证记录

| 检查 | 结果 |
|---|---|
| `-h` | 4,230 字节、**纯 ASCII**（`bytes.decode('ascii')` 通过）、75 行、字面 `\n` 计数 0、无字面 `\t` |
| 菜单（stdin 重定向 `2 → selftest → q`） | 打印帮助 → `select [1-5/h/q]: ` → 选 2 → `[btc finished, back to menu]` → 重排菜单 → `bye`，exit=0，stderr 空，**纯 ASCII**、无字面 `\n`/`\r` |
| `eth/btc/trx/sol/solbip selftest` | secp256k1 三链 11 行向量与已知常量一致（`1G x=59f2815b16f81798`、`2G x=abac09b95c709ee5`、`3G x=8601f113bce036f9`）；`sol` `10 pass 0 fail`；`solbip` `12 pass 0 fail` |
| `eth test` | `00×16` → `abandon … about`；`80×16` → `letter advice … above`，与 BIP39 一致 |
| `btc btctest` | 17 行，与单链版 `bt_multi.txt` 逐行 IDENTICAL |
| `eth vanity 0x1`（7 s） | 315.8 k mnem/s（不再虚高），7 s 累计 8 条命中并实时写入 `eth_hits.txt`，输出纯 ASCII |
| `btc vanity 1abc 0 1 1024`（9 s） | 10 次进度刷新、速率 332 k，命中后继续搜索（不再是「一个就退出」） |
| `trx vanity T1 1 1024`（7 s） | 339 k mnem/s，8 次刷新，无限跑 |

---

## 10. 已知行为与限制

1. **BTC/TRX/SOL 的起点随机**：每次运行都不同序列（ETH/BTC/TRX 靠 `.xxx_vanity_secret` + 时间种子，SOL 快速版靠 run salt）。同一参数两次运行的首个命中不同属正常。
2. **永不自动退出**：五条链的搜索都是死循环，只有 Ctrl+C 能停。BTC 早期版本「找到一个就退出」的问题已在终版改掉（每轮重置 `d_found` / `d_nfound`），现在同一前缀会持续产出新命中。
3. **每个前缀每批最多记 1 条命中**（kernel 里用 `atomicCAS` 门控）。所以短前缀（如 `0x1`）的命中产出 ≈ 每批 1 条；想让命中出得更快，把 `blocks` 调小（每批更短）。
4. **P2TR 非 BIP86 tweak**，见第 4 节。
5. **前缀越长越贵**：SOL 每多 1 个字符约 58 倍；6 字符以上在 0.5 M/s 下要数小时。
6. **调 `blocks` 别太激进**：4080 SUPER 上 512~2048 都可以，再大单批延迟高、Ctrl+C 响应变慢。
7. **nidx 大不等于快**：一个助记词算多次派生，单批命中多但总速率下降。

---

## 11. 安全与合规

- hits 文件是**明文私钥 + 助记词**：放私有目录、别进云同步、别截图发群。
- `.xxx_vanity_secret` 只影响可复现性，不影响资产恢复；仍然建议离线备份。
- 只对自己拥有/控制的地址做 vanity；不要用它做他人地址的碰撞搜索。
- 工具输出的是真链上密钥：任何一次粘贴到在线工具、聊天窗口都等于交出资产。

---

## 12. 源码结构（改代码时看这里）

`merge_multi.py` 是生成脚本：读 5 份单链 `.cu` → 剥 `#include` / `#define` 块（`#define` 按反斜杠续行整块搬到 namespace 外，出 namespace 后 `#undef`）→ 截到 `int main` 之前 → 包进各自的 `namespace` → 尾部接统一入口（`HELP_LINES` / `MENU_LINES` / `main`）。

- 改帮助或菜单：改 `merge_multi.py` 里的 `HELP` / `MENU` 字符串，然后 `python merge_multi.py`，再重编。
- 改某条链的源码：改对应单链 `.cu`，再跑生成脚本。
- `CHAINS` 表里可以给某条链挂 `subs`（字符串替换表），`solbip` 就是靠它把 `sol_hits.txt` 换成 `sol_bip39_hits.txt`。
- 生成出来的 `.cu` 是纯 ASCII：帮助/菜单以字节级 `\xNN` 数组写出（内容为英文），`esc_nonascii_literals()` 作为保底仍会把任何残留的非 ASCII 输出字符转成 `\NNN`。改文案请直接改英文原文，见第 2 节第 6 条。
