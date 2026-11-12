# vanity_multi — Multi-chain BIP39 vanity address generator / 多链 BIP39 vanity 地址生成器

[中文](#中文) · [English](#english)

One source file, one executable, five chains: `eth` / `btc` / `trx` / `sol` (random seed, fast) / `solbip` (BIP39 mnemonic). Run without arguments for the interactive menu, `-h` for full help, or `<chain> <mode> ...` to go straight to one chain.

单份源码、单个 exe，五条链合并：`eth` / `btc` / `trx` / `sol`（随机 seed 快速版）/ `solbip`（BIP39 助记词版）。不带参数进交互菜单，`-h` 看完整帮助，`<chain> <mode> ...` 直接进某条链。

> Console output is ASCII-only English on purpose: Windows consoles default to code page 936 (GBK), which turns UTF-8 Chinese into mojibake, so `-h`, the menu, prompts and progress lines are English.
> 控制台文案故意只用 ASCII 英文：Windows 控制台默认代码页 936（GBK），输出 UTF-8 中文会整屏乱码。所以 `-h`、菜单、提示、进度行都是英文；本 README 中英双语。

---

## 中文

### 1. 文件与依赖

| 文件 | 作用 |
|---|---|
| `vanity_multi.cu` | 唯一源文件。五条链各在一个命名空间（`eth` / `btc` / `trx` / `sol` / `solbip`），末尾是统一入口 `main` |
| `bip39_words.h` | BIP39 2048 词表（`__constant__ char BIP39_WORDS[2048][9]`），**必须与 .cu 同目录** |
| `build_multi.bat` | Windows 编译脚本（vcvars64 + nvcc） |
| `vanity_multi.exe` | 编译产物（约 33 MB，含五套链上密码学与词表） |
| `merge_multi.py` | 生成脚本：把 5 份单链 `.cu` 合并成上面这一个 `.cu` |

编译依赖：CUDA Toolkit（nvcc，实测 13.3）、MSVC BuildTools 2022（`vcvars64.bat`）、NVIDIA GPU（实测 RTX 4080 SUPER，`sm_89`）。

### 2. 编译

```bat
build_multi.bat
```

等价核心命令：

```bat
nvcc -O2 -arch=sm_89 -Xcompiler=/utf-8 -o vanity_multi.exe vanity_multi.cu
```

必须知道的坑：

1. **单次编译 25 分钟以上**。`ptxas` 阶段长时间满载、日志长时间 0 字节都不是卡死，别中途重开。
2. **PATH 里可能没有 `cmd`**。PowerShell 里要写全路径：`& "C:\Windows\System32\cmd.exe" /c "build_multi.bat"`。
3. **重编前先删旧 exe**，否则 `LNK1104: cannot open file`；并确认没有残留的 `nvcc / cicc / cudafe++ / ptxas / cl` 进程。
4. 源码层的坑：
   - 带 UTF-8 BOM 的文件不能直接插进 `namespace` 中间（会变成非法字符）；
   - nvcc 的 `cudafe++` 处理 `R"(...)"` 原始字符串会 internal error（0xC0000409）；
   - nvcc 13.3 对含非 ASCII 字节的字符串字面量不展开 `\n` 转义，而含中文又不带转义会报 `missing closing quote` —— 这也是控制台改英文的原因之一；
   - `bip39_words.h` 不在同目录直接编译失败。
5. **别用带 `&` 的 `cmd /c` 串间接调编译脚本**，容易误触发整轮编译。

### 3. 快速开始

```bat
vanity_multi.exe                          :: 交互菜单
vanity_multi.exe -h                       :: 完整帮助
vanity_multi.exe eth vanity 0x12345 8 1024
vanity_multi.exe btc vanity 1abc 0 5 1024
vanity_multi.exe trx vanity Txxx 4 1024
vanity_multi.exe sol vanity Ase 1024
vanity_multi.exe solbip vanity Ase 1024
```

菜单模式：打印帮助 → 选链（1-5）→ 输入该链参数（例 `vanity Ase 1024`）→ 跑完自动回菜单；回车返回菜单，`q` 退出，`h` 重打帮助。

### 4. 链与派生口径

| 子命令 | 链 | 派生 | 曲线/哈希 | 地址形态 |
|---|---|---|---|---|
| `eth` | Ethereum | BIP39 → PBKDF2 → BIP32/44 `m/44'/60'/0'/0/N` | secp256k1 + keccak256 | `0x` + 40 hex |
| `btc` | Bitcoin | `m/44'/0'/0'/0/N` | secp256k1 + hash160 / bech32 | `1…` P2PKH、`3…` P2SH-P2WPKH、`bc1q…` P2WPKH、`bc1p…` P2TR |
| `trx` | TRON | `m/44'/195'/0'/0/N` | secp256k1 + keccak256 | `T…`（34 字符，Base58Check，0x41 前缀） |
| `sol` | Solana 快速 | 随机 32B seed → SHA512 → clamp | ed25519 | Base58（32B 公钥） |
| `solbip` | Solana 助记词 | BIP39 → PBKDF2(2048) → SLIP-0010 `m/44'/501'/0'/0'` | ed25519 | Base58 |

要点：

- **每条链都无限跑**：命中不停，只有 Ctrl+C 退出；每批重置去重表，同一前缀持续产出新命中。
- **nidx**（`eth/btc/trx`）：一个助记词派生多少个 HD 地址（`m/44'/…/N`，N=0..nidx-1），默认 1。
- **BTC 的 `addrtype` 是数字**，默认 3：`0`=P2PKH(`1…`)、`1`=P2SH-P2WPKH(`3…`)、`2`=P2WPKH(`bc1q…`)、`3`=P2TR(`bc1p…`)。类型必须和前缀自洽，否则永远搜不到。
- **P2TR 不是 BIP86**：本工具把原始 x-only 公钥直接编成 bech32m，没做 taproot tweak。要往该 `bc1p` 地址收币并能花费，请自行按 BIP86 tweak；工具只保证私钥/助记词可恢复。
- **TRX 只输出标准 T 地址**：GasFree 账户地址由服务方按你的 EOA 派生，本工具不输出，需用钱包或 provider API 查询。

### 5. 子命令与参数

```bat
vanity_multi.exe eth vanity <prefix>          [nidx] [blocks]
vanity_multi.exe eth file   <prefix_file>     [nidx] [blocks]
vanity_multi.exe eth selftest
vanity_multi.exe eth bench                    [iters] [blocks]

vanity_multi.exe btc vanity <prefix>      <addrtype> [nidx] [blocks]
vanity_multi.exe btc file   <prefix_file> <addrtype> [nidx] [blocks]
vanity_multi.exe btc selftest
vanity_multi.exe btc bench  <addrtype>        [iters] [blocks]

vanity_multi.exe trx vanity <prefix>          [nidx] [blocks]
vanity_multi.exe trx file   <prefix_file>     [nidx] [blocks]
vanity_multi.exe trx selftest
vanity_multi.exe trx bench                    [iters] [blocks]

vanity_multi.exe sol    vanity <prefix>      [blocks]
vanity_multi.exe sol    file   <prefix_file> [blocks]
vanity_multi.exe sol    bench  [iters] [blocks]
vanity_multi.exe sol    selftest

vanity_multi.exe solbip vanity <prefix>      [blocks]
vanity_multi.exe solbip file   <prefix_file> [blocks]
vanity_multi.exe solbip bench  [iters] [blocks]
vanity_multi.exe solbip selftest
```

- `blocks`：CUDA 网格块数，每块 256 线程，`eth/btc/trx` 默认 1024，`sol/solbip` 默认 512。
- `iters`：`bench` 迭代数，每迭代 = `blocks*256` 个候选。
- `nidx` 上限 1,000,000（实际 1~8 就够）。
- 注意 `btc bench` 的参数顺序与其它链不同：`addrtype` 在最前。

前缀规则：

| 链 | 允许的前缀 | 说明 |
|---|---|---|
| ETH | `0x` + 十六进制（`0x` 可省略） | `12345` 与 `0x12345` 等价 |
| BTC | `1` / `3` / `bc1q` / `bc1p` 开头 | Base58 区分大小写；bech32 全小写 |
| TRX | 必须以 `T` 开头 | 之后是 Base58 字符集 |
| SOL | Base58 字符集，最长 44 | 区分大小写 |

前缀长度不限，前缀文件支持 2GB 以上（流式解析、去重、末行无换行也能读）。

### 6. 输出与实时速率

- 命中**立刻**无缓冲追加到 `<chain>_hits.txt`（`setvbuf(_IONBF)`），Ctrl+C 强杀也不丢已命中记录。
- 每批刷新一行：`[ 12.3 s] 340703 mnem/s | 340703 addr/s | hit N`，**累计口径**（总候选 / 总耗时）。
- ETH 固定 1 批 1 迭代，速率即实测值。

| 链 | 命中字段 |
|---|---|
| eth | `HIT [p] 0x… -> m/44'/60'/0'/0/N`、mnemonic、address、private(hex) |
| btc | `HIT [p] "<地址>" -> m/44'/0'/0'/0/N`、mnemonic、address、private(hex) |
| trx | `HIT [p] "T…" -> m/44'/195'/0'/0/N`、mnemonic、address、private(hex) |
| sol | `HIT [p] prefix "…"`、Address、Secret key(64B)、Seed(32B)、Keypair JSON |
| solbip | 同上 + `Mnemonic (BIP39 12 words)` + `Path m/44'/501'/0'/0'` |

### 7. 运行期生成的文件

| 文件 | 说明 |
|---|---|
| `eth_hits.txt` / `btc_hits.txt` / `trx_hits.txt` | 命中记录（含助记词与私钥） |
| `sol_hits.txt` / `sol_bip39_hits.txt` | `sol` / `solbip` 命中记录 |
| `.eth_vanity_secret` 等 4 个 | 32B 随机 secret，首次运行生成，决定助记词序列；不打印 |

- secret 决定序列：同 secret + 同起点才复现同一批地址。丢了不影响从 hits 恢复钱包，但无法复现序列，建议离线备份。
- `sol` 快速版不落 secret，每次运行随机生成 run salt（启动时打印 32B hex），想复现自己记下。
- 改路径用环境变量 `ETH_VANITY_SECRET` / `BTC_VANITY_SECRET` / `TRX_VANITY_SECRET` / `SOL_VANITY_SECRET`。
- Windows 上 hits 文件无法 `chmod 0600`（源码用 `#ifndef _WIN32` 跳过）。内含明文助记词与私钥，请放私有目录，别放同步盘/共享目录。

### 8. 实测速率

RTX 4080 SUPER / `sm_89` / `-O2`，release 实测：

| 链 | 命令 | 结果 |
|---|---|---|
| eth | `eth bench 8 1024` | 2,097,152 次 / 6.17 s = **339,653 addr/s** |
| btc (P2TR) | `btc bench 3 8 1024` | 2,097,152 次 / 5.98 s = **350,504 addr/s** |
| trx | `trx bench 8 1024` | 2,097,152 次 / 5.99 s = **350,300 addr/s** |
| sol | `sol bench 64 1024` | 16,777,216 keys / 2.23 s = **7,519,436 keys/s** |
| solbip | `solbip bench 1 8192` | 2,097,152 keys / 3.93 s = **534,132 keys/s** |

真实搜索（vanity 模式）与上表同量级：`eth vanity 0x1` ≈ 315~320 k mnem/s、`btc vanity 1abc 0 1 1024` 332~362 k、`trx vanity T1 1 1024` ≈ 339 k。

> 早期版本短前缀时显示约 2 M mnem/s（虚高 6.6 倍），原因是 kernel 开头 `if (*d_found) return;` 让整批提前退出而主机仍按整批计数。现已删除，速率即真实吞吐。

### 9. 验证

```bat
vanity_multi.exe eth selftest
vanity_multi.exe btc selftest
vanity_multi.exe trx selftest
vanity_multi.exe sol selftest
vanity_multi.exe solbip selftest
vanity_multi.exe eth test        :: BIP39 标准向量
vanity_multi.exe btc btctest     :: BTC 派生逐行对照
```

- secp256k1 三链各打印 11 行曲线/标量向量，按已知常量核对：`1G x = 59f2815b16f81798`、`2G x = abac09b95c709ee5`、`3G x = 8601f113bce036f9`。
- `sol` → `selftest: 10 pass, 0 fail`；`solbip` → `selftest: 12 pass, 0 fail`（RFC 8032 ed25519 向量）。
- `eth test`：`00×16` → `abandon … about`，`7f×16` → `legal winner … yellow`，`80×16` → `letter advice … above`，与 BIP39 官方一致。
- `btc btctest`：输出 N=0/N=3 的 priv / pub_comp / hash160 / P2PKH / P2WPKH / P2TR / mnemonic，可与单链版逐行 diff（已验证 17 行完全一致）。
- `verify_multi_hits.py` 用 `hashlib / hmac / ecdsa / pycryptodome` 从助记词独立复算命中记录：ETH 47/47、BTC 2/2、TRX 7/7，0 mismatch。
- `-h` 4,230 字节、75 行、纯 ASCII、字面 `\n` 计数 0；菜单流程 exit=0、stderr 空。

### 10. 已知行为与限制

1. **起点随机**：每次运行序列不同（靠 secret + 时间种子 / run salt），同参数两次的首个命中不同属正常。
2. **永不自动退出**：五条链都是死循环，只有 Ctrl+C 停。
3. **每个前缀每批最多记 1 条命中**（`atomicCAS` 门控）：短前缀命中产出 ≈ 每批 1 条；想更快就把 `blocks` 调小。
4. **P2TR 非 BIP86 tweak**，见第 4 节。
5. **前缀越长越贵**：SOL 每多 1 个字符约 58 倍；6 字符以上在 0.5 M/s 下要数小时。
6. **blocks 别太激进**：4080 SUPER 上 512~2048 都行，再大单批延迟高、Ctrl+C 响应慢。
7. **nidx 大不等于快**：一个助记词算多次派生，单批命中多但总速率下降。

### 11. 安全与合规

- hits 文件是**明文私钥 + 助记词**：放私有目录、别进云同步、别截图发群。
- `.xxx_vanity_secret` 只影响可复现性，不影响资产恢复，仍建议离线备份。
- 只对自己拥有/控制的地址做 vanity；不要用它碰撞他人地址。
- 工具输出真实链上密钥：任何一次粘贴到在线工具或聊天窗口都等于交出资产。
- 请自行确认所在司法辖区对助记词/地址生成工具的规定。

### 12. 源码结构（改代码看这里）

`merge_multi.py` 读 5 份单链 `.cu` → 剥 `#include` / `#define`（`#define` 按反斜杠续行整块搬出 namespace，出 namespace 后 `#undef`）→ 截到 `int main` 之前 → 包进各自 `namespace` → 末尾接统一入口（`HELP_LINES` / `MENU_LINES` / `main`）。

- 改帮助/菜单：改 `merge_multi.py` 的 `HELP` / `MENU`，`python merge_multi.py`，再重编。
- 改某条链：改对应单链 `.cu`，再跑生成脚本。
- `CHAINS` 表的 `subs` 可做字符串替换，`solbip` 就靠它把 `sol_hits.txt` 换成 `sol_bip39_hits.txt`。
- 生成的 `.cu` 是纯 ASCII，改文案请直接写英文。

### 13. 许可

仓库暂未附带 LICENSE 文件。公开前请补一个（MIT / Apache-2.0 等），否则默认保留所有权利，别人无法合法复用。

---

## English

### 1. Files & Dependencies

| File | Purpose |
|---|---|
| `vanity_multi.cu` | The only source file. One namespace per chain (`eth` / `btc` / `trx` / `sol` / `solbip`), unified `main` at the end |
| `bip39_words.h` | BIP39 2048-word table (`__constant__ char BIP39_WORDS[2048][9]`), **must sit next to the .cu** |
| `build_multi.bat` | Windows build script (vcvars64 + nvcc) |
| `vanity_multi.exe` | Build output (~33 MB: five chains of on-chain crypto plus the word table) |
| `merge_multi.py` | Generator that merges the five per-chain `.cu` files into the one above |

Build requirements: CUDA Toolkit (nvcc, tested on 13.3), MSVC BuildTools 2022 (`vcvars64.bat`), an NVIDIA GPU (tested on RTX 4080 SUPER, `sm_89`).

### 2. Build

```bat
build_multi.bat
```

Equivalent core command:

```bat
nvcc -O2 -arch=sm_89 -Xcompiler=/utf-8 -o vanity_multi.exe vanity_multi.cu
```

Pitfalls worth knowing:

1. **A full build takes 25+ minutes.** `ptxas` stays saturated and the log stays at 0 bytes for a long time — that is not a hang, do not restart it.
2. **`cmd` may not be on PATH.** In PowerShell use the full path: `& "C:\Windows\System32\cmd.exe" /c "build_multi.bat"`.
3. **Delete the old exe before rebuilding**, otherwise `LNK1104: cannot open file`; also make sure no `nvcc / cicc / cudafe++ / ptxas / cl` processes are left.
4. Source-level pitfalls:
   - a file with a UTF-8 BOM cannot be pasted into the middle of a `namespace` (illegal character);
   - nvcc's `cudafe++` hits an internal error (0xC0000409) on `R"(...)"` raw string literals;
   - nvcc 13.3 does not expand `\n` inside string literals containing non-ASCII bytes, while non-ASCII without escapes fails with `missing closing quote` — one more reason console text is plain English;
   - `bip39_words.h` missing from the directory is an immediate build failure.
5. **Never chain `cmd /c` commands with `&`** to invoke the build script; it can accidentally start a full rebuild.

### 3. Quick Start

```bat
vanity_multi.exe                          :: interactive menu
vanity_multi.exe -h                       :: full help
vanity_multi.exe eth vanity 0x12345 8 1024
vanity_multi.exe btc vanity 1abc 0 5 1024
vanity_multi.exe trx vanity Txxx 4 1024
vanity_multi.exe sol vanity Ase 1024
vanity_multi.exe solbip vanity Ase 1024
```

Menu mode: prints help → pick a chain (1-5) → type that chain's arguments (e.g. `vanity Ase 1024`) → returns to the menu when done. Enter goes back to the menu, `q` quits, `h` reprints help.

### 4. Chains & Derivation

| Subcommand | Chain | Derivation | Curve / hash | Address form |
|---|---|---|---|---|
| `eth` | Ethereum | BIP39 → PBKDF2 → BIP32/44 `m/44'/60'/0'/0/N` | secp256k1 + keccak256 | `0x` + 40 hex |
| `btc` | Bitcoin | `m/44'/0'/0'/0/N` | secp256k1 + hash160 / bech32 | `1…` P2PKH, `3…` P2SH-P2WPKH, `bc1q…` P2WPKH, `bc1p…` P2TR |
| `trx` | TRON | `m/44'/195'/0'/0/N` | secp256k1 + keccak256 | `T…` (34 chars, Base58Check, 0x41 prefix) |
| `sol` | Solana fast | random 32B seed → SHA512 → clamp | ed25519 | Base58 (32B pubkey) |
| `solbip` | Solana mnemonic | BIP39 → PBKDF2(2048) → SLIP-0010 `m/44'/501'/0'/0'` | ed25519 | Base58 |

Notes:

- **Every chain runs forever**: a hit never stops the search; only Ctrl+C exits. Found flags reset each batch, so the same prefix keeps producing new hits.
- **nidx** (`eth/btc/trx`): how many HD addresses to derive per mnemonic (`m/44'/…/N`, N = 0..nidx-1). Default 1.
- **BTC `addrtype` is numeric**, default 3: `0`=P2PKH(`1…`), `1`=P2SH-P2WPKH(`3…`), `2`=P2WPKH(`bc1q…`), `3`=P2TR(`bc1p…`). The type must match the prefix or nothing ever matches.
- **P2TR is not BIP86**: this tool encodes the raw x-only pubkey straight to bech32m without the taproot tweak. If you want to receive and spend at that `bc1p` address, apply the BIP86 tweak yourself; the tool only guarantees private key / mnemonic recovery.
- **TRX outputs the standard T address only**: the GasFree account address is derived by the service provider from your EOA. This tool does not print it — query it with a wallet or provider API.

### 5. Subcommands & Arguments

```bat
vanity_multi.exe eth vanity <prefix>          [nidx] [blocks]
vanity_multi.exe eth file   <prefix_file>     [nidx] [blocks]
vanity_multi.exe eth selftest
vanity_multi.exe eth bench                    [iters] [blocks]

vanity_multi.exe btc vanity <prefix>      <addrtype> [nidx] [blocks]
vanity_multi.exe btc file   <prefix_file> <addrtype> [nidx] [blocks]
vanity_multi.exe btc selftest
vanity_multi.exe btc bench  <addrtype>        [iters] [blocks]

vanity_multi.exe trx vanity <prefix>          [nidx] [blocks]
vanity_multi.exe trx file   <prefix_file>     [nidx] [blocks]
vanity_multi.exe trx selftest
vanity_multi.exe trx bench                    [iters] [blocks]

vanity_multi.exe sol    vanity <prefix>      [blocks]
vanity_multi.exe sol    file   <prefix_file> [blocks]
vanity_multi.exe sol    bench  [iters] [blocks]
vanity_multi.exe sol    selftest

vanity_multi.exe solbip vanity <prefix>      [blocks]
vanity_multi.exe solbip file   <prefix_file> [blocks]
vanity_multi.exe solbip bench  [iters] [blocks]
vanity_multi.exe solbip selftest
```

- `blocks`: CUDA grid blocks, 256 threads each. Default 1024 for `eth/btc/trx`, 512 for `sol/solbip`.
- `iters`: `bench` iterations; each iteration = `blocks*256` candidates.
- `nidx` caps at 1,000,000 (1~8 is plenty in practice).
- Note `btc bench` takes `addrtype` first — a different argument order from the other chains.

Prefix rules:

| Chain | Allowed prefix | Notes |
|---|---|---|
| ETH | `0x` + hex (`0x` optional) | `12345` equals `0x12345` |
| BTC | starts with `1` / `3` / `bc1q` / `bc1p` | Base58 part is case sensitive; bech32 is lowercase |
| TRX | must start with `T` | Base58 alphabet afterwards |
| SOL | Base58 alphabet, max 44 chars | case sensitive |

Prefixes have no length limit, and prefix files can exceed 2 GB (streamed parsing, de-duplicated, a final line without newline is still read).

### 6. Output & Live Rate

- Hits are appended **immediately** and unbuffered to `<chain>_hits.txt` (`setvbuf(_IONBF)`); a hard Ctrl+C never loses recorded hits.
- One progress line per batch: `[ 12.3 s] 340703 mnem/s | 340703 addr/s | hit N` — a **cumulative** figure (total candidates / total time), not an instantaneous one.
- ETH runs one iteration per batch, so its rate is the measured throughput.

| Chain | Hit fields |
|---|---|
| eth | `HIT [p] 0x… -> m/44'/60'/0'/0/N`, mnemonic, address, private(hex) |
| btc | `HIT [p] "<address>" -> m/44'/0'/0'/0/N`, mnemonic, address, private(hex) |
| trx | `HIT [p] "T…" -> m/44'/195'/0'/0/N`, mnemonic, address, private(hex) |
| sol | `HIT [p] prefix "…"`, Address, Secret key(64B), Seed(32B), Keypair JSON |
| solbip | as above plus `Mnemonic (BIP39 12 words)` and `Path m/44'/501'/0'/0'` |

### 7. Runtime Files

| File | Description |
|---|---|
| `eth_hits.txt` / `btc_hits.txt` / `trx_hits.txt` | Hit records (mnemonics + private keys) |
| `sol_hits.txt` / `sol_bip39_hits.txt` | Hits for `sol` / `solbip` |
| `.eth_vanity_secret` and 3 siblings | 32B random secret created on first run, decides the mnemonic sequence; never printed |

- The secret decides the sequence: the same secret + the same starting point reproduces the same addresses. Losing it does not lose funds (the recorded mnemonics still restore the wallets) but the sequence cannot be replayed — keep an offline backup.
- `sol` (fast mode) stores no secret; it generates a random run salt per run and prints 32B hex at startup. Note that salt down if you want to replay.
- Override paths with `ETH_VANITY_SECRET` / `BTC_VANITY_SECRET` / `TRX_VANITY_SECRET` / `SOL_VANITY_SECRET`.
- On Windows the hits files cannot `chmod 0600` (the call is behind `#ifndef _WIN32`). They hold plaintext mnemonics and private keys: keep them in a private folder, never on a synced or shared drive.

### 8. Measured Performance

RTX 4080 SUPER / `sm_89` / `-O2`, release build:

| Chain | Command | Result |
|---|---|---|
| eth | `eth bench 8 1024` | 2,097,152 in 6.17 s = **339,653 addr/s** |
| btc (P2TR) | `btc bench 3 8 1024` | 2,097,152 in 5.98 s = **350,504 addr/s** |
| trx | `trx bench 8 1024` | 2,097,152 in 5.99 s = **350,300 addr/s** |
| sol | `sol bench 64 1024` | 16,777,216 keys in 2.23 s = **7,519,436 keys/s** |
| solbip | `solbip bench 1 8192` | 2,097,152 keys in 3.93 s = **534,132 keys/s** |

Real search (vanity mode) is in the same range: `eth vanity 0x1` ≈ 315~320 k mnem/s, `btc vanity 1abc 0 1 1024` 332~362 k, `trx vanity T1 1 1024` ≈ 339 k.

> Early versions showed ~2 M mnem/s on short prefixes (about 6.6x inflated). The cause was `if (*d_found) return;` at the top of the kernel: one thread hit and the whole batch returned while the host still counted the full batch. It is removed now, so the rate is the real throughput.

### 9. Verification

```bat
vanity_multi.exe eth selftest
vanity_multi.exe btc selftest
vanity_multi.exe trx selftest
vanity_multi.exe sol selftest
vanity_multi.exe solbip selftest
vanity_multi.exe eth test        :: BIP39 standard vectors
vanity_multi.exe btc btctest     :: BTC derivation line-by-line diff
```

- Each secp256k1 chain prints 11 curve/scalar vectors to check against known constants: `1G x = 59f2815b16f81798`, `2G x = abac09b95c709ee5`, `3G x = 8601f113bce036f9`.
- `sol` → `selftest: 10 pass, 0 fail`; `solbip` → `selftest: 12 pass, 0 fail` (RFC 8032 ed25519 vectors).
- `eth test`: `00×16` → `abandon … about`, `7f×16` → `legal winner … yellow`, `80×16` → `letter advice … above` — matches the BIP39 spec.
- `btc btctest`: prints priv / pub_comp / hash160 / P2PKH / P2WPKH / P2TR / mnemonic for N=0 and N=3; diffable against the single-chain build (verified identical, 17 lines).
- `verify_multi_hits.py` recomputes recorded hits from the mnemonics using `hashlib / hmac / ecdsa / pycryptodome`, without reusing the program under test: ETH 47/47, BTC 2/2, TRX 7/7, 0 mismatches.
- `-h`: 4,230 bytes, 75 lines, pure ASCII, zero literal `\n`. Menu flow exits 0 with empty stderr.

### 10. Known Behaviour & Limits

1. **Random starting point**: sequences differ per run (secret + time seed / run salt), so the first hit for identical arguments differs between runs. That is expected.
2. **Never exits by itself**: all five chains loop forever; only Ctrl+C stops them.
3. **At most 1 hit per prefix per batch** (`atomicCAS` gate): a short prefix yields roughly one hit per batch; lower `blocks` for faster hit output.
4. **P2TR is not a BIP86 tweak**, see section 4.
5. **Longer prefix is exponentially costly**: about 58x per extra SOL character; 6+ characters at 0.5 M/s means hours.
6. **Do not push `blocks` too far**: 512~2048 is fine on a 4080 SUPER; larger batches add latency and slow Ctrl+C response.
7. **A large `nidx` is not faster**: it derives many addresses per mnemonic, so hits per batch rise while total throughput drops.

### 11. Security & Compliance

- The hits files are **plaintext private keys + mnemonics**: keep them in a private folder, out of cloud sync, never screenshot them into a chat.
- `.xxx_vanity_secret` only affects reproducibility, not fund recovery — still back it up offline.
- Only run vanity for addresses you own or control; do not use it to collide with someone else's address.
- The tool emits real on-chain keys: pasting them into any online tool or chat window hands over the funds.
- Check your jurisdiction's rules on mnemonic/address generation tooling before distributing this.

### 12. Source Layout

`merge_multi.py` reads the five per-chain `.cu` files → strips `#include` / `#define` (multi-line `#define` blocks are moved out of the namespace and `#undef`'d afterwards) → truncates before `int main` → wraps each chain in its own `namespace` → appends the unified entry point (`HELP_LINES` / `MENU_LINES` / `main`).

- To change help or menu text: edit `HELP` / `MENU` in `merge_multi.py`, run `python merge_multi.py`, then rebuild.
- To change one chain: edit that single-chain `.cu`, then rerun the generator.
- `subs` in the `CHAINS` table performs string substitutions — that is how `solbip` writes `sol_bip39_hits.txt` instead of `sol_hits.txt`.
- The generated `.cu` is pure ASCII; write new console text in English.

### 13. License

No LICENSE file ships with the repository yet. Add one (MIT / Apache-2.0 / etc.) before publishing — without a license, all rights are reserved and others cannot legally reuse the code.
