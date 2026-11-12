import io, re, sys

# ETH：v2 的自适应批量放大（ms<200 就 *3，最多到 1e6）会把「本轮启动量」当成本轮实算量累加，
# 实测 8 秒内速率被放大到 1.2e12 mnem/s（物理上不可能）。合并版固定 batch=1，逐批实测。
ETH_FIX = [
    ('if (ms > 1000.0f) { int nb=batch_iters/2; if (nb<1) nb=1; batch_iters=nb; }',
     '/* merge v3: 固定 batch=1，速率按实测单批计算 */'),
    ('else if (ms < 200.0f) { int nb=batch_iters*3; if (nb<1) nb=1; if (nb>1000000) batch_iters=nb; }',
     '/* merge v3: 固定 batch=1 */'),
]

CHAINS = [
    ('eth',    'eth_vanity_bip39_v2.cu',   'eth_vanity_bip39.exe',   ETH_FIX),
    ('btc',    'btc_vanity_bip39_v2.cu',   'btc_vanity_bip39.exe',   []),
    ('trx',    'trx_vanity_bip39_v2.cu',   'trx_vanity_bip39.exe',   []),
    ('sol',    'sol_vanity_v2.cu',         'sol_vanity_v2.exe',      []),
    ('solbip', 'sol_vanity_bip39_v3.cu',   'sol_vanity_bip39_v3.exe',
     [('sol_hits.txt', 'sol_bip39_hits.txt')]),
]

HEADER = r'''// =============================================================================
//  vanity_multi.cu  -  multi-chain vanity address generator (single binary)
//  ETH / BTC / TRX / SOL(fast seed) / SOL(BIP39 mnemonic)
//
//  用法:  vanity_multi.exe            -> 交互菜单
//         vanity_multi.exe -h         -> 完整帮助
//         vanity_multi.exe eth  <args>  直接进入某条链
//         vanity_multi.exe btc  <args>
//         vanity_multi.exe trx  <args>
//         vanity_multi.exe sol  <args>       SOL 随机 seed（快）
//         vanity_multi.exe solbip <args>     SOL BIP39 助记词（慢）
//
//  每条链的代码原样封装在独立命名空间里（eth/btc/trx/sol/solbip），共享
//  BIP39 词表；字段运算、内核、命中结构互不冲突，改动任一链都不影响其他链。
//  各链的参数与语义与其单链版本一致。
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <random>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include "bip39_words.h"

'''

def load(fn):
    """读取源文件，剥离 #include 与 #define 块，返回 (body, define_blocks)"""
    text = io.open(fn, 'r', encoding='utf-8').read().replace('\ufeff', '')
    lines = text.split('\n')
    body, blocks = [], []
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith('#include'):
            i += 1
            continue
        if l.startswith('#define'):
            blk = [l]
            while l.rstrip().endswith('\\') and i + 1 < len(lines):
                i += 1
                l = lines[i]
                blk.append(l)
            blocks.append(blk)
            i += 1
            continue
        body.append(l)
        i += 1
    return body, blocks

def split_main(lines):
    for i, l in enumerate(lines):
        if l.startswith('int main(int argc, char** argv'):
            return lines[:i], lines[i:]
    raise SystemExit('main not found')

def define_name(blk):
    return re.match(r'#define\s+([A-Za-z_][A-Za-z0-9_]*)', blk[0]).group(1)

def c_lit_lines(text):
    """把纯文本转成一组 C 字符串字面量（每行一个，字节级 \\xNN 转义，行尾自带 0x0A）。

    为什么必须全字节转义：nvcc 13.3 的 cudafe++ 在处理“含非 ASCII 字节的字符串字面量”时
    不展开转义序列（\n 会原样输出成反斜杠+n），而含非 ASCII 又不带转义的字面量会直接报
    missing closing quote。把整行 UTF-8 字节都写成 \\xNN 后，源文件是纯 ASCII，输出与预期逐字节一致。
    """
    out = []
    for ln in text.split('\n'):
        bs = ln.encode('utf-8') + b'\n'
        esc = ''.join('\\x%02X' % b for b in bs)
        out.append('    "%s"' % esc)
    return ',\n'.join(out)


def esc_nonascii_literals(text):
    r"""把「输出语句行」里字符串字面量中的非 ASCII 字节改成八进制转义。

    nvcc 13.3 的 cudafe++ 对含非 ASCII 字节的字面量不展开 \n / \r 等转义
    （会原样输出成反斜杠+n），所以含中文的 printf/puts 行必须先把中文变成
    纯 ASCII 的 \NNN（3 位八进制，不会像 \x 那样贪婪吞掉后面的 hex 字符）。
    """
    out = []
    for line in text.split('\n'):
        if ('printf(' in line or 'fprintf(' in line or 'fputs(' in line
                or 'sprintf(' in line or 'snprintf(' in line):
            buf, in_str, i = [], False, 0
            while i < len(line):
                c = line[i]
                if c == '"' and (i == 0 or line[i-1] != '\\'):
                    in_str = not in_str
                    buf.append(c)
                elif in_str and ord(c) > 127:
                    buf.append(''.join('\\%03o' % b for b in c.encode('utf-8')))
                else:
                    buf.append(c)
                i += 1
            line = ''.join(buf)
        out.append(line)
    return '\n'.join(out)


def strip_early_exit(text):
    """去掉 kernel 开头那句整批早退 `if (*d_found) return;`。

    原来只要有一个线程命中，整批其余线程立刻 return，但主机端仍按「整批
    blocks*256 个候选」计入统计，导致命中频繁时速率显示虚高（实测 0x1 前缀
    显示 2.05 M mnem/s，而真实吞吐只有约 30 万）。去掉后每批一定跑满，速率
    就是真实吞吐；命中记录仍由 atomicCAS 保证每个前缀每批只记一条。
    """
    out = []
    for line in text.split('\n'):
        if line.strip() == 'if (*d_found) return;':
            indent = line[:len(line) - len(line.lstrip())]
            out.append(indent + '/* v3: 去掉整批早退，保证每批跑满，速率不虚高 */')
        else:
            out.append(line)
    return '\n'.join(out)


def btc_infinite_loop(text):
    """BTC 的搜索循环原本是 while (nfound < uniq)：命中一个就退出。

    这里改成和 eth/trx 一样的无限循环 + 每轮重置命中标记，这样同一个前缀
    会持续产出新的命中记录，只有 Ctrl+C 才停。
    """
    old_head = """    while (nfound < (u64)uniq) {
        int launched = batch_iters;  // v2: 本轮实际启动的批量；统计与区间推进只准用它（batch_iters 会在轮末被自适应调整）"""
    new_head = """    while (1) {
        int launched = batch_iters;  // v2: 本轮实际启动的批量；统计与区间推进只准用它
        CUDA_CHECK(cudaMemset(d_found, 0, (size_t)((u64)uniq*4)));
        CUDA_CHECK(cudaMemset(d_nfound, 0, 4));
        nfound = 0;                  // 每轮重置，同一前缀可持续出新命中"""
    if old_head not in text:
        raise SystemExit('BTC loop head not found')
    return text.replace(old_head, new_head, 1)


def english_outputs(text):
    """把所有会打印到控制台的中文换成英文。

    Windows 控制台默认代码页是 936(GBK)，程序输出 UTF-8 中文会整屏乱码，
    所以控制台文案一律纯 ASCII 英文（文档 README 仍保留中文）。
    """
    pairs = [
        ('"hd mode: 助记词由 secret 文件 + base_nonce 共同决定（v2 起 base_nonce 不再打印到日志）\\n"',
         '"hd mode: mnemonic sequence comes from the secret file + base_nonce (not printed since v2)\\n"'),
        ('"\\n  (每次运行随机生成，种子 seed=SHA256(salt||counter), counter 从 0 起；所以每次运行序列都不同)\\n"',
         '"\\n  (fresh salt each run: seed = SHA256(salt||counter), counter from 0; sequences differ per run)\\n"'),
        ('| 已覆盖 %u/%llu  (Ctrl+C stop)', '| covered %u/%llu  (Ctrl+C stop)'),
        ('"选择 [1-5/h/q]: "', '"select [1-5/h/q]: "'),
        ('"无效选择\\n"', '"invalid choice\\n"'),
        ('"\\n已选 %s。请输入参数（例: vanity Ase 1024 / file list.txt / selftest / bench 64 1024）\\n"',
         '"\\nselected %s. enter args (e.g. vanity Ase 1024 / file list.txt / selftest / bench 64 1024)\\n"'),
        ('"直接回车返回菜单。\\n> "', '"press Enter to return to the menu.\\n> "'),
        ('"\\n[%s 已结束，返回菜单]\\n"', '"\\n[%s finished, back to menu]\\n"'),
    ]
    for a, b in pairs:
        text = text.replace(a, b)
    return text

out = [HEADER]
out.append('// 各链的宏开关不同（PSTR/NLMAX/RESQ_CAP 等）：进入前定义、离开后撤销。')
for name, fn, exe, subs in CHAINS:
    body, blocks = load(fn)
    body, mainl = split_main(body)
    for a, b in subs:
        body = [l.replace(a, b) for l in body]
        mainl = [l.replace(a, b) for l in mainl]
    if name == 'eth':
        # 兜底：任何残留的「批量三倍放大」行一律清掉（batch 固定为 1）
        def _fix(ls):
            return [('            /* merge v3: 固定 batch=1 */'
                    if 'else if (ms < 200.0f)' in l else l) for l in ls]
        body, mainl = _fix(body), _fix(mainl)
    out.append('\n// ============================ chain: %s ============================' % name)
    for blk in blocks:
        out.extend(blk)
    out.append('namespace %s {' % name)
    out.extend(body)
    mainl[0] = 'static int chain_main(int argc, char** argv) {'
    txt = '\n'.join(mainl)
    txt = txt.replace(exe, 'vanity_multi.exe ' + name)
    for a, b in subs:
        txt = txt.replace(a, b)
    out.extend(txt.split('\n'))
    out.append('} // namespace %s' % name)
    for blk in blocks:
        out.append('#undef %s' % define_name(blk))

HELP = '''vanity_multi - multi-chain BIP39 vanity address generator (single binary)

USAGE
  vanity_multi.exe                          no args -> interactive menu
  vanity_multi.exe -h | --help | help       show this help
  vanity_multi.exe <chain> <mode> [args...] run one chain directly

  <chain>
    eth     Ethereum   BIP39 -> m/44'/60'/0'/0/N  -> secp256k1 -> keccak256 -> 0x address
    btc     Bitcoin    BIP39 -> m/44'/0'/0'/0/N   -> secp256k1 -> Base58 / Bech32 address
    trx     TRON       BIP39 -> m/44'/195'/0'/0/N -> secp256k1 -> keccak256 -> T address
    sol     Solana fast      random 32B seed -> ed25519 -> base58 (no PBKDF2, fastest)
    solbip  Solana mnemonic  BIP39 -> PBKDF2(2048) -> SLIP-0010 m/44'/501'/0'/0'
                             -> ed25519 -> base58 (importable into Phantom / Solflare)

MODES
  eth | btc | trx
    vanity <prefix> [nidx] [blocks]         search forever, stop with Ctrl+C
    file   <prefix_file> [nidx] [blocks]    one prefix per line, unlimited size (2GB+ streamed)
    selftest                                built-in vector self-test
    bench  [iters] [blocks]                 throughput benchmark

  btc takes one extra NUMERIC addrtype argument, right after prefix / prefix file (default 3)
    btc vanity <prefix>      [addrtype] [nidx] [blocks]
    btc file   <prefix_file> [addrtype] [nidx] [blocks]
    addrtype: 0 = P2PKH(1...)   1 = P2SH-P2WPKH(3...)   2 = P2WPKH(bc1q...)
              3 = P2TR(bc1p...)  (default)
      prefix starts with 1 -> 0, with 3 -> 1, bc1q -> 2, bc1p -> 3
      a mismatched type simply never matches
    btc bench also takes addrtype first:
    btc bench  <addrtype> [iters] [blocks]

  sol | solbip
    vanity <prefix> [blocks]                search forever, stop with Ctrl+C
    file   <prefix_file> [blocks]           prefix file (2GB+ streamed)
    bench  [iters] [blocks]                 throughput benchmark (solbip pays PBKDF2)
    selftest                                built-in vector self-test

ARGUMENTS
  prefix    address prefix to match (case sensitive). ETH may include 0x; BTC starts with
            1 / 3 / bc1q / bc1p; TRX must start with T; SOL uses base58 only, max 44 chars.
  nidx      addresses derived per mnemonic (HD index 0..N-1), default 1.
            larger -> more hits per batch but slower per address
  blocks    CUDA grid blocks (256 threads each), default 1024 (sol / solbip: 512)
  iters     bench iterations (each iteration = blocks*256 candidates)

HITS AND OUTPUT
  * every hit is appended immediately to <chain>_hits.txt (unbuffered), Ctrl+C keeps them
  * the search never stops on a hit; only Ctrl+C ends it. Found flags are reset every
    batch, so one prefix keeps producing new hits instead of only one
  * live rate is measured per batch and printed as a cumulative figure
  * ETH/BTC/TRX hit block: mnemonic, derivation path, address, private key (32B hex)
  * SOL fast hit block: base58 address, Seed(32B), Secret key(64B), keypair JSON
  * SOL mnemonic hit block: 12-word mnemonic, path, address, Seed, Secret key, keypair JSON

FILES CREATED AT RUNTIME
  eth_hits.txt / btc_hits.txt / trx_hits.txt     hit records
  sol_hits.txt             sol (random seed)
  sol_bip39_hits.txt       solbip (mnemonic)
  .eth_vanity_secret / .btc_vanity_secret / .trx_vanity_secret / .sol_vanity_secret
      32B secret, created on first run, decides the mnemonic sequence. Losing it does not
      lose coins (the recorded mnemonics still recover the wallet) but the sequence
      cannot be reproduced
  sol prints a fresh 32B run salt at startup (not saved); keep it to replay a run

NOTES
  * Windows cannot chmod 0600 the hits files and they hold plaintext mnemonics and
    private keys: keep them in a private folder, never in synced/shared directories
  * longer prefix = exponentially harder: SOL costs about 58x per extra character
  * prefix files are de-duplicated and a last line without newline is still read
  * only run vanity for addresses you own or control
  * console output is ASCII-only on purpose, so a GBK/legacy code page shows it correctly

See README.md for measured rates, build steps and verification.
'''

MENU = '''
=============================================================
 multi-chain vanity address generator   vanity_multi
=============================================================
  1) ETH          BIP39 -> m/44'/60'/0'/0/N  -> 0x address
  2) BTC          BIP39 -> m/44'/0'/0'/0/N   -> 1 / 3 / bc1q / bc1p
  3) TRX          BIP39 -> m/44'/195'/0'/0/N -> T address
  4) SOL fast     random seed -> ed25519 -> base58     (fastest)
  5) SOL mnemonic BIP39 -> SLIP-0010 -> base58         (importable)
  h) help         q) quit
-------------------------------------------------------------
'''

TAIL = r'''
// =============================================================================
//  统一入口：直接分发 + 交互菜单
// =============================================================================
static const char* HELP_LINES[] = {
__HELP__
};

static const char* MENU_LINES[] = {
__MENU__
};

static void print_lines(const char* const* lines, int n) {
    for (int i = 0; i < n; i++) fputs(lines[i], stdout);
}
static void print_help(void) { print_lines(HELP_LINES, (int)(sizeof(HELP_LINES)/sizeof(HELP_LINES[0]))); }
static void print_menu(void) { print_lines(MENU_LINES, (int)(sizeof(MENU_LINES)/sizeof(MENU_LINES[0]))); }

static int run_chain(const char* chain, int argc, char** argv) {
    if      (!strcmp(chain, "eth"))    return eth::chain_main(argc, argv);
    else if (!strcmp(chain, "btc"))    return btc::chain_main(argc, argv);
    else if (!strcmp(chain, "trx"))    return trx::chain_main(argc, argv);
    else if (!strcmp(chain, "sol"))    return sol::chain_main(argc, argv);
    else if (!strcmp(chain, "solbip")) return solbip::chain_main(argc, argv);
    return -1;
}

static const char* chain_of(int choice) {
    switch (choice) {
        case 1: return "eth";
        case 2: return "btc";
        case 3: return "trx";
        case 4: return "sol";
        case 5: return "solbip";
    }
    return NULL;
}

static int split_args(char* line, char** argv, int maxn) {
    int n = 0; char* p = line;
    while (*p && n < maxn) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        argv[n++] = p;
        while (*p && *p != ' ' && *p != '\t') p++;
        if (*p) *p++ = 0;
    }
    return n;
}

static int is_chain(const char* s) {
    return !strcmp(s, "eth") || !strcmp(s, "btc") || !strcmp(s, "trx") ||
           !strcmp(s, "sol") || !strcmp(s, "solbip");
}

int main(int argc, char** argv) {
    if (argc >= 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help") || !strcmp(argv[1], "help"))) {
        print_help();
        return 0;
    }
    if (argc >= 2) {
        if (is_chain(argv[1])) return run_chain(argv[1], argc - 1, argv + 1);
        fprintf(stderr, "unknown chain '%s'\n\n", argv[1]);
        print_help();
        return 1;
    }

    char line[512];
    print_help();
    for (;;) {
        print_menu();
        printf("选择 [1-5/h/q]: ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) return 0;
        char c = line[0];
        if (c == 'q' || c == 'Q' || c == '\n' || c == '\r' || c == 0) { printf("bye\n"); return 0; }
        if (c == 'h' || c == 'H') { print_help(); continue; }
        int pick = (c >= '1' && c <= '5') ? (c - '0') : -1;
        if (pick < 0) { printf("无效选择\n"); continue; }
        const char* chain = chain_of(pick);
        printf("\n已选 %s。请输入参数（例: vanity Ase 1024 / file list.txt / selftest / bench 64 1024）\n", chain);
        printf("直接回车返回菜单。\n> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) return 0;
        line[strcspn(line, "\r\n")] = 0;
        char* s = line;
        while (*s == ' ') s++;
        if (!*s) continue;
        char* args[32];
        int n = split_args(s, args, 31);
        char cname[16];
        snprintf(cname, sizeof(cname), "%s", chain);
        char* av[33];
        av[0] = cname;
        for (int i = 0; i < n; i++) av[i + 1] = args[i];
        printf("\n");
        run_chain(chain, n + 1, av);
        printf("\n[%s 已结束，返回菜单]\n", chain);
    }
}
'''

tail = TAIL.replace('__HELP__', c_lit_lines(HELP)).replace('__MENU__', c_lit_lines(MENU))
out.append(tail)

io.open('vanity_multi.cu', 'w', encoding='utf-8', newline='\n').write(btc_infinite_loop(strip_early_exit(esc_nonascii_literals(english_outputs('\n'.join(out))))))
print('written', len(out), 'lines')
