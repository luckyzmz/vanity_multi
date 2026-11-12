// =============================================================================
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


// 各链的宏开关不同（PSTR/NLMAX/RESQ_CAP 等）：进入前定义、离开后撤销。

// ============================ chain: eth ============================
#define CUDA_CHECK(call) do { cudaError_t err = call; if(err != cudaSuccess) { fprintf(stderr, "CUDA Err: %s\n", cudaGetErrorString(err)); exit(err); } } while(0)
#define MAXP 256
#define NL 40
#define RESQ_CAP (1<<20)
#define RESQ_MASK (RESQ_CAP-1)
namespace eth {
// eth_vanity_bip39.cu  (v2, 2026-11-02 修订版，基于 0_5i15)
// BIP39 mnemonic -> PBKDF2 -> BIP32/44 m/44'/60'/0'/0/0 -> 真 secp256k1 -> keccak256 -> ETH 地址
// vanity 前缀比对生成器（真链路，非 BitCrack）
// 字段算术：8x32 位小端 limb，p = 2^256 - 2^32 - 977
//
// v2 改动清单（详见随附说明）：
//  1) 修复 file/vanity 无限循环的区间推进 bug：start 只按本轮实际启动的 launched 推进，不再用自适应调整后的 batch_iters（原版会跳过或重复搜索区间）
//  2) 修复 selftest/fieldtest 跑完后穿透再跑一遍 bench（两条 if 链合并）；comb 预计算后加 cudaDeviceSynchronize
//  3) 安全：计数器熵改为 SHA256(secret(32B) || counter)，secret 由主机端真随机生成/加载（.eth_vanity_secret，0600，绝不打印）；hd 模式不再打印 base_nonce
//  4) 性能：热路径 CKD 的 HMAC 改为预计算 ipad/opad 复用（cbase 固定时只算一次），SHA512 压缩 4->2 次/地址
//  5) 规范/健壮：ckd_priv 处理 IL>=n（置 0 永不命中）；sha256 通用填充修复 len>=56 的情况；file 模式 tcnt 改用去重后计数、长度分布标签 bit->hex
//  6) eth_hits.txt 创建后 chmod 0600
// 未包含（需在目标 GPU 上实测后再动，避免无验证改坏）：批量求逆（Montgomery）、fe_mul/fe_sqr 重写、comb 表放共享内存


typedef uint32_t u32;
typedef uint64_t u64;
typedef uint8_t u8;
typedef uint16_t u16;
typedef u32 fe[ 8];



// ==========================================
// secp256k1 常数（LE 32 位 limb）
// ==========================================
__constant__ u32 P_FE[ 8] = {0xFFFFFC2Fu,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
__constant__ u32 N_FE[ 8] = {0xD0364141u,0xBFD25E8Cu,0xAF48A03Bu,0xBAAEDCE6u,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
__constant__ u32 C2P256MN[ 8] = {0x2FC9BEBFu,0x402DA173u,0x50B75FC4u,0x45512319u,0x00000001u,0,0,0};
__constant__ u32 GX_FE[ 8] = {0x16F81798u,0x59F2815Bu,0x2DCE28D9u,0x029BFCDBu,0xCE870B07u,0x55A06295u,0xF9DCBBACu,0x79BE667Eu};
__constant__ u32 GY_FE[ 8] = {0xFB10D4B8u,0x9C47D08Fu,0xA6855419u,0xFD17B448u,0x0E1108A8u,0x5DA4FBFCu,0x26A3C465u,0x483ADA77u};
__constant__ u8 SECRET_K[ 32];  // v2: 主机端真随机 secret，由 main() 拷入；与计数器混合决定助记词序列

// ==========================================
// 字段算术（8x32 位，p = 2^256 - 2^32 - 977）
// ==========================================
__device__ __forceinline__ void fe_cpy(u32 r[ 8], const u32 a[ 8]) { for (int i=0;i<8;i++) r[i]=a[i]; }
__device__ __forceinline__ void fe_set0(u32 r[ 8]) { for (int i=0;i<8;i++) r[i]=0; }
__device__ __forceinline__ void fe_set1(u32 r[ 8]) { fe_set0(r); r[ 0]=1; }
__device__ __forceinline__ int fe_iszero(const u32 a[ 8]) { int z=1; for (int i=0;i<8;i++) z &= (a[i]==0); return z; }
__device__ __forceinline__ int fe_isone(const u32 a[ 8]) { int z=(a[ 0]==1); for (int i=1;i<8;i++) z &= (a[i]==0); return z; }
__device__ __forceinline__ int fe_ge(const u32 a[ 8], const u32 b[ 8]) {
    for (int i=7;i>=0;i--) { if (a[i]!=b[i]) return a[i]>b[i]; }
    return 1;
}
__device__ __forceinline__ void fe_sub(u32 r[ 8], const u32 a[ 8], const u32 b[ 8]) {
    u64 bo=0;
    for (int i=0;i<8;i++){ u64 s=(u64)a[i]-b[i]-bo; r[i]=(u32)s; bo=(s>>32)?1:0; }
    if (bo) {
        u64 c=0;
        for (int i=0;i<8;i++){ u64 s=(u64)r[i]+P_FE[i]+c; r[i]=(u32)s; c=s>>32; }
    }
}
// 折叠 10-limb v 的高位 v[8],v[9]（2^256 ≡ 2^32+977），至多 3 次
__device__ __forceinline__ void fe_fold_hi(u32 v[ 10]) {
    for (int it=0; it<3; it++) {
        if ((v[ 8] | v[ 9]) == 0) return;
        u32 h0 = v[ 8], h1 = v[ 9];
        v[ 8] = 0; v[ 9] = 0;
        u64 c = 0;
        for (int i=0;i<10;i++){
            u64 s = c + (i<8 ? (u64)v[i] : 0);
            if (i>=1) {
                if (i-1==0) s += (u64)h0;
                else if (i-1==1) s += (u64)h1;
            }
            if (i==0) s += (u64)h0 * 977u;
            if (i==1) s += (u64)h1 * 977u;
            v[i] = (u32)s;
            c = s >> 32;
        }
        if (c) { u64 s = (u64)v[ 9] + c; v[ 9] = (u32)s; }
    }
}
// r = t mod p，t 为 9 limbs（< 2^257）
__device__ __forceinline__ void fe_reduce9(u32 r[ 8], const u32 t[ 9]) {
    u32 v[ 10];
    u64 c=0;
    for (int i=0;i<10;i++){
        u64 s = c + (i<8 ? (u64)t[i] : 0);
        if (i==1) s += (u64)t[ 8];
        if (i==0) s += (u64)t[ 8]*977u;
        v[i]=(u32)s; c=s>>32;
    }
    if (c){ u64 s=(u64)v[ 9]+c; v[ 9]=(u32)s; }
    fe_fold_hi(v);
    for (int i=0;i<8;i++) r[i]=v[i];
    if (fe_ge(r, P_FE)) fe_sub(r, r, P_FE);
}
__device__ __forceinline__ void fe_add(u32 r[ 8], const u32 a[ 8], const u32 b[ 8]) {
    u64 c=0; u32 t[ 9];
    for (int i=0;i<8;i++){ u64 s=(u64)a[i]+b[i]+c; t[i]=(u32)s; c=s>>32; }
    t[ 8]=(u32)c;
    fe_reduce9(r, t);
}
// r = a * b mod p（32 位 limb 学校式 + 进位链，无 __int128）
__device__ __noinline__ void fe_mul(u32 r[ 8], const u32 a[ 8], const u32 b[ 8]) {
    u32 t[ 16];
    for (int i=0;i<16;i++) t[i]=0;
    for (int i=0;i<8;i++)
        for (int j=0;j<8;j++) {
            u64 p = (u64)a[i]*b[j];
            u64 s = (u64)t[i+j] + (u32)p;
            t[i+j] = (u32)s;
            u64 c = s >> 32;
            int k = i+j+1;
            while (c && k < 16) { u64 s2=(u64)t[k]+c; t[k]=(u32)s2; c=s2>>32; k++; }
            s = (u64)t[i+j+1] + (u32)(p>>32);
            t[i+j+1] = (u32)s;
            c = s >> 32;
            k = i+j+2;
            while (c && k < 16) { u64 s2=(u64)t[k]+c; t[k]=(u32)s2; c=s2>>32; k++; }
        }
    u32 v[ 10];
    u64 c=0;
    for (int i=0;i<10;i++){
        u64 s = c + (i<8 ? (u64)t[i] : 0);
        if (i>=1 && (i-1)<8) s += (u64)t[ 8+(i-1)];
        if (i<8) s += (u64)t[ 8+i]*977u;
        v[i]=(u32)s; c=s>>32;
    }
    if (c){ u64 s=(u64)v[ 9]+c; v[ 9]=(u32)s; }
    fe_fold_hi(v);
    for (int i=0;i<8;i++) r[i]=v[i];
    if (fe_ge(r, P_FE)) fe_sub(r, r, P_FE);
}
__device__ __forceinline__ void fe_sqr(u32 r[ 8], const u32 a[ 8]) { fe_mul(r, a, a); }
// 费马求逆 a^(p-2)
__device__ __forceinline__ void fe_inv(u32 r[ 8], const u32 a[ 8]) {
    u32 e[ 8] = {0xFFFFFC2Du,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
    u32 res[ 8]; fe_set1(res);
    u32 base[ 8]; fe_cpy(base, a);
    for (int i=255;i>=0;i--){
        fe_sqr(res, res);
        if ((e[i>>5] >> (i&31)) & 1) fe_mul(res, res, base);
    }
    fe_cpy(r, res);
}
// 标量加法 mod n（BIP32 CKD：k_new = (IL + k) mod n）
__device__ __forceinline__ void sc_add_mod_n(u32 r[ 8], const u32 a[ 8], const u32 b[ 8]) {
    u64 c=0; u32 t[ 9];
    for (int i=0;i<8;i++){ u64 s=(u64)a[i]+b[i]+c; t[i]=(u32)s; c=s>>32; }
    t[ 8]=(u32)c;
    if (t[ 8]) {
        u64 cc=0;
        for (int i=0;i<8;i++){ u64 s=(u64)t[i]+C2P256MN[i]+cc; r[i]=(u32)s; cc=s>>32; }
    } else if (fe_ge(t, N_FE)) {
        u64 bo=0;
        for (int i=0;i<8;i++){ u64 s=(u64)t[i]-N_FE[i]-bo; r[i]=(u32)s; bo=(s>>32)?1:0; }
    } else {
        for (int i=0;i<8;i++) r[i]=t[i];
    }
}

// ==========================================
// 雅可比坐标点运算
// ==========================================
struct Jacobian { fe X, Y, Z; };
__device__ __forceinline__ void jac_set_inf(Jacobian& P) { fe_set0(P.X); fe_set0(P.Y); fe_set0(P.Z); }
__device__ __forceinline__ int jac_is_inf(const Jacobian& P) { return fe_iszero(P.Z); }

__device__ __forceinline__ void jac_double(Jacobian& R, const Jacobian& P) {
    if (jac_is_inf(P)) { jac_set_inf(R); return; }
    fe A, B, C, D, E, F, t, t2;
    fe_sqr(A, P.X);
    fe_sqr(B, P.Y);
    fe_sqr(C, B);
    fe_add(t, P.X, B); fe_sqr(t, t);
    fe_sub(t, t, A); fe_sub(t, t, C);
    fe_add(D, t, t);
    fe_add(E, A, A); fe_add(E, E, A);
    fe_sqr(F, E);
    fe_add(t, D, D);
    fe_sub(t, F, t);
    fe_sub(t2, D, t);
    fe_mul(t2, t2, E);
    fe_add(F, C, C); fe_add(F, F, F); fe_add(F, F, F);
    fe_sub(t2, t2, F);
    fe_mul(F, P.Y, P.Z); fe_add(F, F, F);
    fe_cpy(R.X, t); fe_cpy(R.Y, t2); fe_cpy(R.Z, F);
}

__device__ __forceinline__ void jac_add_affine(Jacobian& R, const Jacobian& P, const u32 qx[ 8], const u32 qy[ 8]) {
    if (jac_is_inf(P)) { fe_cpy(R.X, qx); fe_cpy(R.Y, qy); fe_set1(R.Z); return; }
    fe Z1Z1, U2, S2, H, Rr, HH, HHH, V, X3, Y3, Z3, t, t2;
    fe_sqr(Z1Z1, P.Z);
    fe_mul(U2, qx, Z1Z1);
    fe_mul(t, P.Z, Z1Z1);
    fe_mul(S2, qy, t);
    fe_sub(H, U2, P.X);
    fe_sub(Rr, S2, P.Y);
    if (fe_iszero(H)) {
        if (fe_iszero(Rr)) { jac_double(R, P); return; }
        else { jac_set_inf(R); return; }
    }
    fe_sqr(HH, H);
    fe_mul(HHH, H, HH);
    fe_mul(V, P.X, HH);
    fe_sqr(X3, Rr);
    fe_sub(X3, X3, HHH);
    fe_add(t, V, V);
    fe_sub(X3, X3, t);
    fe_sub(t, V, X3);
    fe_mul(t, t, Rr);
    fe_mul(t2, P.Y, HHH);
    fe_sub(Y3, t, t2);
    fe_mul(Z3, P.Z, H);
    fe_cpy(R.X, X3); fe_cpy(R.Y, Y3); fe_cpy(R.Z, Z3);
}

// k * G（从 MSB 到 LSB 的 double-and-add）
__device__ void scalar_mult_base(Jacobian& R, const u32 k[ 8]) {
    jac_set_inf(R);
    for (int i=255;i>=0;i--){
        jac_double(R, R);
        if ((k[i>>5] >> (i&31)) & 1) jac_add_affine(R, R, GX_FE, GY_FE);
    }
}


__device__ void jac_to_affine(u32 x[ 8], u32 y[ 8], const Jacobian& P) {
    fe zi, zi2, zi3;
    fe_inv(zi, P.Z);
    fe_sqr(zi2, zi);
    fe_mul(zi3, zi2, zi);
    fe_mul(x, P.X, zi2);
    fe_mul(y, P.Y, zi3);
}
__device__ const u32* g_comb;

// ==========================================
// 4-tooth comb 固定基点标量乘
// T[t][j] = j * 2^(64t) * G, t=0..3, j=1..15 （仿射坐标，16 u32/点）
// 256 次 double -> 64 次 double + ~64 次 affine add
// ==========================================
__global__ void precompute_comb_kernel(u32* d_table) {
    int t = threadIdx.x;      // 0..3
    int j = blockIdx.x + 1;   // 1..15
    if (t > 3) return;
    u32 kf[ 8] = {0,0,0,0,0,0,0,0};
    // k = j * 2^(64t)：j 放入 limb 2t（j<16 不溢出）
    kf[ 2*t] = (u32)j;
    Jacobian P;
    jac_set_inf(P);
    for (int i=255;i>=0;i--){
        jac_double(P, P);
        if ((kf[i>>5]>>(i&31))&1) jac_add_affine(P, P, GX_FE, GY_FE);
    }
    u32 x[ 8], y[ 8];
    jac_to_affine(x, y, P);
    u32* dst = d_table + ((t*15 + (j-1)) * 16);
    for (int i=0;i<8;i++){ dst[i]=x[i]; dst[8+i]=y[i]; }
}

__device__ __forceinline__ void scalar_mult_base_comb(Jacobian& R, const u32 k[ 8]) {
    jac_set_inf(R);
    const u32* T = g_comb;
    for (int r=15; r>=0; r--) {
        jac_double(R,R); jac_double(R,R); jac_double(R,R); jac_double(R,R);
        for (int t=0;t<4;t++) {
            int shift = 64*t + 4*r;
            int d = (k[shift>>5] >> (shift&31)) & 0xF;
            if (d) {
                const u32* p = T + ((t*15 + (d-1)) * 16);
                jac_add_affine(R, R, p, p+8);
            }
        }
    }
}


// ==========================================
// 字节 / limb 转换
// ==========================================
__device__ __forceinline__ void bytes32_to_fe(u32 f[ 8], const u8 b[ 32]) {
    for (int i=0;i<8;i++) f[i] = ((u32)b[28-4*i]<<24)|((u32)b[29-4*i]<<16)|((u32)b[30-4*i]<<8)|b[31-4*i];
}
__device__ __forceinline__ void fe_to_bytes32(u8 b[ 32], const u32 f[ 8]) {
    for (int i=0;i<8;i++){ u32 v=f[i]; b[28-4*i]=(u8)(v>>24); b[29-4*i]=(u8)(v>>16); b[30-4*i]=(u8)(v>>8); b[31-4*i]=(u8)v; }
}
__device__ __forceinline__ int device_strlen(const char* s) { int l=0; while (s[l]) l++; return l; }

// ==========================================
// SHA-256（BIP39 校验和 / vanity 熵）
// ==========================================
__constant__ u32 K256[ 64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
__device__ __forceinline__ u32 rotr32(u32 x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ void sha256_block(u32 h[ 8], const u8 blk[ 64]) {
    u32 w[ 64];
    for (int i=0;i<16;i++) w[i] = ((u32)blk[i*4]<<24)|((u32)blk[i*4+1]<<16)|((u32)blk[i*4+2]<<8)|blk[i*4+3];
    for (int i=16;i<64;i++){
        u32 s0 = rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);
        u32 s1 = rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    u32 a=h[ 0],b=h[ 1],c=h[ 2],d=h[ 3],e=h[ 4],f=h[ 5],g=h[ 6],hh=h[ 7];
    for (int i=0;i<64;i++){
        u32 S1 = rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
        u32 ch = (e&f)^((~e)&g);
        u32 t1 = hh + S1 + ch + K256[i] + w[i];
        u32 S0 = rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
        u32 maj = (a&b)^(a&c)^(b&c);
        u32 t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[ 0]+=a; h[ 1]+=b; h[ 2]+=c; h[ 3]+=d; h[ 4]+=e; h[ 5]+=f; h[ 6]+=g; h[ 7]+=hh;
}
__device__ void sha256(const u8* data, int len, u8 out[ 32]) {
    u32 h[ 8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    u8 blk[ 64];
    int off = 0;
    while (len - off >= 64) { for (int i=0;i<64;i++) blk[i]=data[off+i]; sha256_block(h, blk); off += 64; }
    for (int i=0;i<64;i++) blk[i]=0;
    int rem = len - off;
    for (int i=0;i<rem;i++) blk[i]=data[off+i];
    blk[rem] = 0x80;
    u64 bits = (u64)len * 8;
    if (rem >= 56) {  // v2: 剩余 >=56 字节时长度字段放不进同一块，先冲掉这一块再补长度块（原版此情况算错；旧调用 len<=40 未触发）
        sha256_block(h, blk);
        for (int i=0;i<64;i++) blk[i]=0;
    }
    for (int i=0;i<8;i++) blk[56+i] = (u8)((bits >> (56-8*i)) & 0xFF);
    sha256_block(h, blk);
    for (int i=0;i<8;i++){ out[i*4]=(u8)(h[i]>>24); out[i*4+1]=(u8)(h[i]>>16); out[i*4+2]=(u8)(h[i]>>8); out[i*4+3]=(u8)h[i]; }
}

// v2: 计数器 -> 熵。ent = SHA256(SECRET_K(32B) || counter 小端 8B) 的前 16 字节。
// 没有 secret 的人无法复现/枚举这条助记词序列（原版仅 SHA256(计数器)，等效熵只有计数器本身且顺序可猜）。
__device__ __forceinline__ void counter_to_entropy(u64 n, u8 ent[ 16]) {
    u8 buf[ 40];
    for (int i=0;i<32;i++) buf[i]=SECRET_K[i];
    for (int i=0;i<8;i++) buf[32+i]=(u8)(n>>(8*i));
    u8 h[ 32]; sha256(buf, 40, h);
    for (int i=0;i<16;i++) ent[i]=h[i];
}

// ==========================================
// SHA-512 / HMAC-SHA512（移植自优化版）
// ==========================================
__device__ __constant__ u64 sha512_k[ 80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};
__device__ __forceinline__ u64 rotr64(u64 x, int n) { return (x >> n) | (x << (64 - n)); }
__device__ __forceinline__ u64 Ch(u64 x, u64 y, u64 z) { return (x & y) ^ (~x & z); }
__device__ __forceinline__ u64 Maj(u64 x, u64 y, u64 z) { return (x & y) ^ (x & z) ^ (y & z); }
__device__ __forceinline__ u64 Sigma0(u64 x) { return rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39); }
__device__ __forceinline__ u64 Sigma1(u64 x) { return rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41); }
__device__ __forceinline__ u64 sigma0(u64 x) { return rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7); }
__device__ __forceinline__ u64 sigma1(u64 x) { return rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6); }
__device__ __forceinline__ void sha512_init(u64 st[ 8]) {
    st[ 0]=0x6a09e667f3bcc908ULL; st[ 1]=0xbb67ae8584caa73bULL;
    st[ 2]=0x3c6ef372fe94f82bULL; st[ 3]=0xa54ff53a5f1d36f1ULL;
    st[ 4]=0x510e527fade682d1ULL; st[ 5]=0x9b05688c2b3e6c1fULL;
    st[ 6]=0x1f83d9abfb41bd6bULL; st[ 7]=0x5be0cd19137e2179ULL;
}
__device__ __forceinline__ void sha512_block_words(u64 st[ 8], const u64 blockW[ 16]) {
    u64 W[ 80];
    #pragma unroll
    for (int i = 0; i < 16; i++) W[i] = blockW[i];
    #pragma unroll
    for (int i = 16; i < 80; i++) W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16];
    u64 a=st[ 0], b=st[ 1], c=st[ 2], d=st[ 3], e=st[ 4], f=st[ 5], g=st[ 6], h=st[ 7];
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        u64 T1 = h + Sigma1(e) + Ch(e,f,g) + sha512_k[i] + W[i];
        u64 T2 = Sigma0(a) + Maj(a,b,c);
        h=g; g=f; f=e; e=d + T1; d=c; c=b; b=a; a=T1 + T2;
    }
    st[ 0]+=a; st[ 1]+=b; st[ 2]+=c; st[ 3]+=d; st[ 4]+=e; st[ 5]+=f; st[ 6]+=g; st[ 7]+=h;
}
__device__ __forceinline__ void load_be_words(const u8 blk[ 128], u64 W[ 16]) {
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        u64 v = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) v = (v << 8) | blk[i*8 + j];
        W[i] = v;
    }
}
__device__ __forceinline__ void hmac_sha512_precompute(const u8* key, int key_len,
                                                       u64 ipad_st[ 8], u64 opad_st[ 8]) {
    u8 real_key[ 128] = {0};
    u8 kblk[ 128];
    if (key_len > 128) {
        u64 st[ 8]; sha512_init(st);
        int off = 0;
        while (off + 128 <= key_len) {
            u64 W[ 16]; load_be_words(key + off, W);
            sha512_block_words(st, W); off += 128;
        }
        u8 pad[ 128] = {0};
        int rem = key_len - off;
        for (int i = 0; i < rem; i++) pad[i] = key[off + i];
        pad[rem] = 0x80;
        u64 bits = (u64)key_len * 8;
        for (int i = 0; i < 8; i++) pad[ 120 + i] = (bits >> (56 - 8*i)) & 0xFF;
        u64 W2[ 16]; load_be_words(pad, W2);
        sha512_block_words(st, W2);
        for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) real_key[i*8+j] = (st[i] >> (56 - 8*j)) & 0xFF;
    } else {
        for (int i = 0; i < key_len; i++) real_key[i] = key[i];
    }
    u64 W[ 16];
    for (int i = 0; i < 128; i++) kblk[i] = real_key[i] ^ 0x36;
    load_be_words(kblk, W);
    sha512_init(ipad_st); sha512_block_words(ipad_st, W);
    for (int i = 0; i < 128; i++) kblk[i] = real_key[i] ^ 0x5c;
    load_be_words(kblk, W);
    sha512_init(opad_st); sha512_block_words(opad_st, W);
}
// 通用 HMAC-SHA512（msg_len <= 64，key_len <= 128）
__device__ __forceinline__ void hmac_sha512_gen(const u8* key, int key_len, const u8* msg, int msg_len, u8 out[ 64]) {
    u64 ipad_st[ 8], opad_st[ 8];
    hmac_sha512_precompute(key, key_len, ipad_st, opad_st);
    u8 blk[ 128];
    for (int i=0;i<128;i++) blk[i]=0;
    for (int i=0;i<msg_len && i<128;i++) blk[i]=msg[i];
    blk[msg_len]=0x80;
    u64 bits=(u64)(128+msg_len)*8;
    for (int i=0;i<8;i++) blk[120+i]=(u8)((bits>>(56-8*i))&0xFF);
    u64 W[ 16], st[ 8], ih[ 8];
    load_be_words(blk, W);
    for (int i=0;i<8;i++) st[i]=ipad_st[i];
    sha512_block_words(st, W);
    for (int i=0;i<8;i++) ih[i]=st[i];
    for (int i=0;i<128;i++) blk[i]=0;
    for (int i=0;i<64;i++) blk[i]=(u8)((ih[i>>3]>>(56-8*(i&7)))&0xFF);
    blk[64]=0x80;
    bits=(128+64)*8;
    for (int i=0;i<8;i++) blk[120+i]=(u8)((bits>>(56-8*i))&0xFF);
    load_be_words(blk, W);
    for (int i=0;i<8;i++) st[i]=opad_st[i];
    sha512_block_words(st, W);
    for (int i=0;i<8;i++) for (int j=0;j<8;j++) out[i*8+j]=(u8)((st[i]>>(56-8*j))&0xFF);
}
// v2: 用预计算好的 ipad/opad 状态做 HMAC-SHA512（要求 msg_len<=119），省去每次调用重算密钥填充的 2 次压缩。
// 热路径（CKD 的链码固定）应先 hmac_sha512_precompute 一次，再反复调本函数。
__device__ __forceinline__ void hmac_sha512_precomp_msg(const u64 ipad_st[ 8], const u64 opad_st[ 8], const u8* msg, int msg_len, u8 out[ 64]) {
    u8 blk[ 128];
    for (int i=0;i<128;i++) blk[i]=0;
    for (int i=0;i<msg_len;i++) blk[i]=msg[i];
    blk[msg_len]=0x80;
    u64 bits=(u64)(128+msg_len)*8;
    for (int i=0;i<8;i++) blk[120+i]=(u8)((bits>>(56-8*i))&0xFF);
    u64 W[ 16], st[ 8], ih[ 8];
    load_be_words(blk, W);
    for (int i=0;i<8;i++) st[i]=ipad_st[i];
    sha512_block_words(st, W);
    for (int i=0;i<8;i++) ih[i]=st[i];
    for (int i=0;i<128;i++) blk[i]=0;
    for (int i=0;i<64;i++) blk[i]=(u8)((ih[i>>3]>>(56-8*(i&7)))&0xFF);
    blk[64]=0x80;
    bits=(128+64)*8;
    for (int i=0;i<8;i++) blk[120+i]=(u8)((bits>>(56-8*i))&0xFF);
    load_be_words(blk, W);
    for (int i=0;i<8;i++) st[i]=opad_st[i];
    sha512_block_words(st, W);
    for (int i=0;i<8;i++) for (int j=0;j<8;j++) out[i*8+j]=(u8)((st[i]>>(56-8*j))&0xFF);
}
// PBKDF2 首次迭代：salt = "mnemonic" + 0x00000001
__device__ __forceinline__ void hmac_sha512_salt(const u64 ipad_st[ 8], const u64 opad_st[ 8], u64 out[ 8]) {
    const u64 saltW[ 16] = {
        0x6D6E656D6F6E6963ULL, 0x0000000180000000ULL,
        0,0,0,0,0,0,0,0,0,0,0,0, 0, 0x0000000000000460ULL
    };
    u64 st[ 8], W[ 16], ih[ 8];
    for (int i = 0; i < 8; i++) st[i] = ipad_st[i];
    for (int i = 0; i < 16; i++) W[i] = saltW[i];
    sha512_block_words(st, W);
    for (int i = 0; i < 8; i++) ih[i] = st[i];
    for (int i = 0; i < 8; i++) st[i] = opad_st[i];
    for (int i = 0; i < 8; i++) W[i] = ih[i];
    W[ 8]  = 0x8000000000000000ULL;
    W[ 9]  = 0; W[ 10] = 0; W[ 11] = 0; W[ 12] = 0; W[ 13] = 0; W[ 14] = 0;
    W[ 15] = 0x0000000000000600ULL;
    sha512_block_words(st, W);
    for (int i = 0; i < 8; i++) out[i] = st[i];
}
// HMAC 一步：消息 = 完整 64B 块 msg
__device__ __forceinline__ void hmac_sha512_64(const u64 ipad_st[ 8], const u64 opad_st[ 8],
                                               const u64 msg[ 8], u64 out[ 8]) {
    u64 st[ 8], W[ 16], ih[ 8];
    for (int i = 0; i < 8; i++) st[i] = ipad_st[i];
    for (int i = 0; i < 8; i++) W[i] = msg[i];
    W[ 8]  = 0x8000000000000000ULL;
    W[ 9]  = 0; W[ 10] = 0; W[ 11] = 0; W[ 12] = 0; W[ 13] = 0; W[ 14] = 0;
    W[ 15] = 0x0000000000000600ULL;
    sha512_block_words(st, W);
    for (int i = 0; i < 8; i++) ih[i] = st[i];
    for (int i = 0; i < 8; i++) st[i] = opad_st[i];
    for (int i = 0; i < 8; i++) W[i] = ih[i];
    W[ 8]  = 0x8000000000000000ULL;
    W[ 9]  = 0; W[ 10] = 0; W[ 11] = 0; W[ 12] = 0; W[ 13] = 0; W[ 14] = 0;
    W[ 15] = 0x0000000000000600ULL;
    sha512_block_words(st, W);
    for (int i = 0; i < 8; i++) out[i] = st[i];
}
// 完整 BIP39 Seed 派生（2048 次 PBKDF2-HMAC-SHA512）
__device__ void device_bip39_seed(const char* mnemonic, u8 seed[ 64]) {
    int pass_len = device_strlen(mnemonic);
    u64 ipad_st[ 8], opad_st[ 8];
    hmac_sha512_precompute((const u8*)mnemonic, pass_len, ipad_st, opad_st);
    u64 U[ 8], T[ 8];
    hmac_sha512_salt(ipad_st, opad_st, U);
    for (int i = 0; i < 8; i++) T[i] = U[i];
    for (int i = 1; i < 2048; i++) {
        hmac_sha512_64(ipad_st, opad_st, U, U);
        for (int j = 0; j < 8; j++) T[j] ^= U[j];
    }
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) seed[i*8 + j] = (T[i] >> (56 - 8*j)) & 0xFF;
}

// ==========================================
// Keccak-256（地址）
// ==========================================
__device__ __constant__ u64 keccak_rc[ 24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};
__device__ __forceinline__ u64 rotl64b(u64 x, int n) { n &= 63; return n ? ((x << n) | (x >> (64 - n))) : x; }
__device__ void keccak_f1600(u64 state[ 25]) {
    u64 C[ 5], D[ 5], B[ 25];
    for (int round=0; round<24; round++) {
        for (int i=0;i<5;i++) C[i]=state[i]^state[i+5]^state[i+10]^state[i+15]^state[i+20];
        for (int i=0;i<5;i++) { D[i]=C[(i+4)%5]^rotl64b(C[(i+1)%5],1); for (int j=0;j<25;j+=5) state[j+i]^=D[i]; }
        for (int i=0;i<25;i++) B[i]=state[i];
        int x=1, y=0; u64 cur=B[ 1];
        for (int t=0;t<24;t++) {
            int tmp=x; x=y; y=(2*tmp+3*y)%5;
            u64 tv=B[x+5*y]; B[x+5*y]=rotl64b(cur,(t+1)*(t+2)/2); cur=tv;
        }
        for (int i=1;i<25;i++) state[i]=B[i];
        for (int j=0;j<25;j+=5) {
            for (int i=0;i<5;i++) B[j+i]=state[j+i];
            for (int i=0;i<5;i++) state[j+i]^=(~B[j+(i+1)%5])&B[j+(i+2)%5];
        }
        state[ 0]^=keccak_rc[round];
    }
}
__device__ void device_keccak256(const u8* data, int len, u8 hash[ 32]) {
    u64 state[ 25] = {0}; int rate=136; int off=0;
    while (off+rate<=len) {
        for (int i=0;i<rate/8;i++) {
            u64 val=0;
            for (int j=0;j<8;j++) val |= ((u64)data[off+i*8+j])<<(8*j);
            state[i]^=val;
        }
        keccak_f1600(state); off+=rate;
    }
    u8 blk[ 136] = {0}; int rem=len-off;
    if (rem>0) for (int i=0;i<rem;i++) blk[i]=data[off+i];
    blk[rem]=0x01; blk[rate-1]|=0x80;
    for (int i=0;i<rate/8;i++) {
        u64 val=0;
        for (int j=0;j<8;j++) val |= ((u64)blk[i*8+j])<<(8*j);
        state[i]^=val;
    }
    keccak_f1600(state);
    for (int i=0;i<4;i++) for (int j=0;j<8;j++) hash[i*8+j]=(u8)((state[i]>>(8*j))&0xFF);
}

// ==========================================
// BIP39 编码：128bit 熵 + 4bit 校验和 -> 12 词
// ==========================================
__device__ void entropy_to_mnemonic(const u8 ent[ 16], char* mnem, int* mnem_len) {
    u8 h[ 32];
    sha256(ent, 16, h);
    u16 idx[ 12];
    for (int w=0;w<12;w++) {
        int bp = w*11; u32 val=0;
        for (int b=0;b<11;b++) {
            int p=bp+b; int byte=p>>3, sh=7-(p&7);
            int bit = (byte<16) ? ((ent[byte]>>sh)&1) : ((h[ 0]>>sh)&1);
            val=(val<<1)|bit;
        }
        idx[w]=(u16)val;
    }
    int pos=0;
    for (int w=0;w<12;w++) {
        const char* word = BIP39_WORDS[idx[w]];
        int len=0; while (word[len]) len++;
        for (int i=0;i<len;i++) mnem[pos++]=word[i];
        if (w<11) mnem[pos++]=' ';
    }
    mnem[pos]=0;
    if (mnem_len) *mnem_len=pos;
}

// ==========================================
// BIP32 / BIP44 派生：m/44'/60'/0'/0/0
// ==========================================
__device__ void ckd_priv(u8 k[ 32], u8 c[ 32], u32 index, int hardened) {
    u8 data[ 37], out[ 64];
    if (hardened) {
        data[ 0]=0;
        for (int i=0;i<32;i++) data[1+i]=k[i];
    } else {
        u32 kf[ 8]; bytes32_to_fe(kf, k);
        Jacobian P; scalar_mult_base_comb(P, kf);
        u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
        u8 xb[ 32]; fe_to_bytes32(xb, x);
        data[ 0] = (y[ 0]&1) ? 0x03 : 0x02;
        for (int i=0;i<32;i++) data[1+i]=xb[i];
    }
    data[33]=(u8)(index>>24); data[34]=(u8)(index>>16); data[35]=(u8)(index>>8); data[36]=(u8)index;
    hmac_sha512_gen(c, 32, data, 37, out);
    u32 il[ 8], kf[ 8], kf2[ 8];
    bytes32_to_fe(il, out);
    bytes32_to_fe(kf, k);
    // v2(BIP32 规范): IL>=n 时该索引无效（概率约 2^-127，n 距 2^256 极近）。置子钥为 0，
    // 其点乘为无穷远点、派生地址永不命中前缀，等效丢弃；原版对此未处理、会算出规范外密钥。
    if (fe_ge(il, N_FE)) fe_set0(kf2);
    else sc_add_mod_n(kf2, il, kf);
    fe_to_bytes32(k, kf2);
    for (int i=0;i<32;i++) c[i]=out[32+i];
}
__device__ void bip32_derive(u8 privk[ 32], const u8 seed[ 64]) {
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    ckd_priv(k, c, 0x80000000u+44u, 1);
    ckd_priv(k, c, 0x80000000u+60u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    ckd_priv(k, c, 0u, 0);
    ckd_priv(k, c, 0u, 0);
    for (int i=0;i<32;i++) privk[i]=k[i];
}
__device__ void derive_address(const u8 privk[ 32], u8 addr[ 20]) {
    u32 kf[ 8]; bytes32_to_fe(kf, privk);
    Jacobian P; scalar_mult_base_comb(P, kf);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    u8 pub[ 64];
    fe_to_bytes32(pub, x);
    fe_to_bytes32(pub+32, y);
    u8 h[ 32];
    device_keccak256(pub, 64, h);
    for (int i=0;i<20;i++) addr[i]=h[12+i];
}
__device__ void derive_full(const u8 ent[ 16], char* mnem, u8 addr[ 20], u8 privk[ 32]) {
    char m[ 128];
    entropy_to_mnemonic(ent, m, 0);
    u8 seed[ 64];
    device_bip39_seed(m, seed);
    bip32_derive(privk, seed);
    derive_address(privk, addr);
    if (mnem) { int l=0; while (m[l]){ mnem[l]=m[l]; l++; } mnem[l]=0; }
}

// ==========================================
// Kernel：vec 诊断（逐层输出中间值）
// ==========================================
__global__ void vec_kernel(const u8* d_ent, u8* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 ent[ 16];
    for (int i=0;i<16;i++) ent[i]=d_ent[i];
    char m[ 128];
    entropy_to_mnemonic(ent, m, 0);
    int o=0;
    for (int i=0;i<16;i++) d_out[o++]=ent[i];
    u8 seed[ 64];
    device_bip39_seed(m, seed);
    for (int i=0;i<64;i++) d_out[o++]=seed[i];
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    for (int i=0;i<32;i++) d_out[o++]=c[i];
    ckd_priv(k, c, 0x80000000u+44u, 1);
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    for (int i=0;i<32;i++) d_out[o++]=c[i];
    ckd_priv(k, c, 0x80000000u+60u, 1);
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    for (int i=0;i<32;i++) d_out[o++]=c[i];
    ckd_priv(k, c, 0x80000000u+0u, 1);
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    for (int i=0;i<32;i++) d_out[o++]=c[i];
    ckd_priv(k, c, 0u, 0);
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    ckd_priv(k, c, 0u, 0);
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    u32 kf[ 8]; bytes32_to_fe(kf, k);
    Jacobian P; scalar_mult_base(P, kf);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    u8 xb[ 32], yb[ 32];
    fe_to_bytes32(xb, x); fe_to_bytes32(yb, y);
    for (int i=0;i<32;i++) d_out[o++]=xb[i];
    for (int i=0;i<32;i++) d_out[o++]=yb[i];
    u8 pub[ 64];
    for (int i=0;i<32;i++) pub[i]=xb[i];
    for (int i=0;i<32;i++) pub[32+i]=yb[i];
    u8 h[ 32];
    device_keccak256(pub, 64, h);
    for (int i=0;i<20;i++) d_out[o++]=h[12+i];
    int l=0; while (m[l]){ d_out[o++]=m[l]; l++; }
    d_out[o]=0;
    d_done[ 0]=1;
}

// ==========================================
// Kernel：selftest
// ==========================================
__global__ void selftest_kernel(u64* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu;
    fe b; fe_set1(b);
    fe r; fe_add(r, a, b);
    u64 v0=0; for (int i=7;i>=0;i--) v0=(v0<<32)|r[i];
    d_out[ 0]=v0;
    fe m1; fe_set0(m1); m1[ 0]=123456789u;
    fe m2; fe_set0(m2); m2[ 0]=987654321u;
    fe_mul(r, m1, m2);
    u64 v1=0; for (int i=7;i>=0;i--) v1=(v1<<32)|r[i];
    d_out[ 1]=v1;
    fe two; fe_set0(two); two[ 0]=2u;
    Jacobian P2; scalar_mult_base(P2, two);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P2);
    u64 v2x=0, v2y=0; for (int i=7;i>=0;i--){ v2x=(v2x<<32)|x[i]; v2y=(v2y<<32)|y[i]; }
    d_out[ 2]=v2x; d_out[ 3]=v2y;
    fe one; fe_set1(one);
    Jacobian P1; scalar_mult_base(P1, one);
    u32 x1[ 8], y1[ 8]; jac_to_affine(x1, y1, P1);
    u64 v3x=0, v3y=0; for (int i=7;i>=0;i--){ v3x=(v3x<<32)|x1[i]; v3y=(v3y<<32)|y1[i]; }
    d_out[ 4]=v3x; d_out[ 5]=v3y;
    u32 nm1[ 8]; fe_cpy(nm1, N_FE); nm1[ 0]-=1;
    u32 one32[ 8]; fe_set1(one32);
    u32 sr[ 8]; sc_add_mod_n(sr, nm1, one32);
    u64 v4=0; for (int i=7;i>=0;i--) v4=(v4<<32)|sr[i];
    d_out[ 6]=v4;
    u32 nm2[ 8]; fe_cpy(nm2, N_FE); nm2[ 0]-=2;
    u32 three[ 8]; fe_set0(three); three[ 0]=3;
    sc_add_mod_n(sr, nm2, three);
    u64 v5=0; for (int i=7;i>=0;i--) v5=(v5<<32)|sr[i];
    d_out[ 7]=v5;
    Jacobian P3a; jac_add_affine(P3a, P2, GX_FE, GY_FE);
    u32 x3[ 8], y3[ 8]; jac_to_affine(x3, y3, P3a);
    fe three3; fe_set0(three3); three3[ 0]=3;
    Jacobian P3b; scalar_mult_base(P3b, three3);
    u32 x3b[ 8], y3b[ 8]; jac_to_affine(x3b, y3b, P3b);
    u64 v6=0; for (int i=7;i>=0;i--) v6=(v6<<32)|x3[i];
    u64 v6b=0; for (int i=7;i>=0;i--) v6b=(v6b<<32)|x3b[i];
    d_out[ 8]=v6; d_out[ 9]=v6b;
    fe inv; fe_inv(inv, m1);
    fe_mul(r, m1, inv);
    u64 v7=0; for (int i=7;i>=0;i--) v7=(v7<<32)|r[i];
    d_out[ 10]=v7;
    d_done[ 0]=1;
}

// ==========================================
// Kernel：jac_double(G) 中间量诊断
// ==========================================
__global__ void jdbl_kernel(u32* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    fe A, B, C, D, E, F, t, t2, F0, t0;
    fe_sqr(A, GX_FE);
    fe_sqr(B, GY_FE);
    fe_sqr(C, B);
    fe_add(t, GX_FE, B); fe_sqr(t, t);
    fe_sub(t, t, A); fe_sub(t, t, C);
    fe_add(D, t, t);
    fe_add(E, A, A); fe_add(E, E, A);
    fe_sqr(F, E);
    fe_cpy(F0, F);
    fe_add(t, D, D);
    fe_cpy(t0, t);
    fe_sub(t, F0, t0);
    fe_sub(t2, D, t);
    fe_mul(t2, t2, E);
    fe_add(F, C, C); fe_add(F, F, F); fe_add(F, F, F);
    fe_sub(t2, t2, F);
    fe_mul(F, GY_FE, GX_FE); fe_add(F, F, F);
    int o=0;
    for (int s=0;s<8;s++) d_out[o++]=A[s];
    for (int s=0;s<8;s++) d_out[o++]=B[s];
    for (int s=0;s<8;s++) d_out[o++]=C[s];
    for (int s=0;s<8;s++) d_out[o++]=D[s];
    for (int s=0;s<8;s++) d_out[o++]=E[s];
    for (int s=0;s<8;s++) d_out[o++]=F0[s];  // E^2
    for (int s=0;s<8;s++) d_out[o++]=t0[s];  // 2D
    for (int s=0;s<8;s++) d_out[o++]=t[s];   // X3
    for (int s=0;s<8;s++) d_out[o++]=t2[s];  // Y3
    for (int s=0;s<8;s++) d_out[o++]=F[s];   // Z3
    d_done[ 0]=1;
}

// ==========================================
// Kernel：最小化 scalar_mult_base(k0h) 诊断
// ==========================================

// ==========================================
// Kernel: keccak test
// ==========================================
__global__ void ktest_kernel(u8* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 h[ 32];
    device_keccak256((const u8*)"", 0, h);
    for (int i=0;i<32;i++) d_out[i]=h[i];
    device_keccak256((const u8*)"abc", 3, h);
    for (int i=0;i<32;i++) d_out[32+i]=h[i];
    const u8 pub[ 64] = {
        0x37,0xb0,0xbb,0x7a,0x82,0x88,0xd3,0x8e,0xd4,0x9a,0x52,0x4b,0x5d,0xc9,0x8c,0xff,
        0x3e,0xb5,0xca,0x82,0x4c,0x9f,0x9d,0xc0,0xdf,0xdb,0x3d,0x9c,0xd6,0x00,0xf2,0x99,
        0xa6,0x17,0x99,0x12,0xb7,0x45,0x1c,0x09,0x89,0x6c,0x40,0x98,0xec,0xa7,0xce,0x6b,
        0x2e,0x58,0x33,0x06,0x72,0x79,0x5e,0x84,0x7c,0x4d,0x6a,0xf4,0x4e,0x02,0x42,0x30 };
    device_keccak256(pub, 64, h);
    for (int i=0;i<32;i++) d_out[64+i]=h[i];
    u64 st[ 25] = {0};
    keccak_f1600(st);
    for (int i=0;i<8;i++) d_out[96+i]=(u8)((st[ 0]>>(8*i))&0xFF);
    d_done[ 0]=1;
}

__global__ void sm_kernel(u32* d_out, u32* d_done, int c) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u32 kf[ 8] = {0,0,0,0,0,0,0,0};
    if (c==0) { // k0h
        const u8 kh[ 32] = { 0xe9,0xb9,0x1c,0x42,0x8e,0x67,0x9f,0x85,0x7f,0xbf,0xa0,0x20,0x34,0x40,0xea,0xea,
                             0xab,0xb0,0xca,0x46,0xd8,0x0e,0x70,0xb6,0xaa,0x62,0x50,0x69,0x6a,0xc0,0xdf,0xc7 };
        bytes32_to_fe(kf, kh);
    } else if (c==1) { kf[ 7]=0x80000000u; } // 2^255
    else if (c==2) { kf[ 7]=0xe9b91c42u; }  // top 32 bits of k0h
    else if (c==3) { kf[ 0]=0xe9u; }        // 0xe9
    else if (c==4) { kf[ 7]=0x00000080u; }  // 2^247
    else if (c==5) { kf[ 0]=0x0au; }        // 10G
    Jacobian P; scalar_mult_base(P, kf);
    if (c==1 || c==5) {
        for (int i=0;i<8;i++) d_out[i]=P.X[i];
        for (int i=0;i<8;i++) d_out[8+i]=P.Y[i];
        for (int i=0;i<8;i++) d_out[16+i]=P.Z[i];
        d_done[ 0]=1; return;
    }
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    for (int i=0;i<8;i++) d_out[i]=x[i];
    for (int i=0;i<8;i++) d_out[8+i]=y[i];
    d_done[ 0]=1;
}

// ==========================================
// Kernel：ptest 点运算诊断（2G/3G/pubkey(k0h)）
// ==========================================
__global__ void ptest_kernel(u32* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    int o=0;
    fe two; fe_set0(two); two[ 0]=2u;
    Jacobian P2; scalar_mult_base(P2, two);
    u32 x2[ 8], y2[ 8]; jac_to_affine(x2, y2, P2);
    for (int i=7;i>=0;i--) d_out[o++]=x2[i];
    for (int i=7;i>=0;i--) d_out[o++]=y2[i];
    fe three; fe_set0(three); three[ 0]=3u;
    Jacobian P3; scalar_mult_base(P3, three);
    u32 x3[ 8], y3[ 8]; jac_to_affine(x3, y3, P3);
    for (int i=7;i>=0;i--) d_out[o++]=x3[i];
    for (int i=7;i>=0;i--) d_out[o++]=y3[i];
    fe four; fe_set0(four); four[ 0]=4u;
    Jacobian P4; scalar_mult_base(P4, four);
    u32 x4[ 8], y4[ 8]; jac_to_affine(x4, y4, P4);
    for (int i=7;i>=0;i--) d_out[o++]=x4[i];
    for (int i=7;i>=0;i--) d_out[o++]=y4[i];
    fe five; fe_set0(five); five[ 0]=5u;
    Jacobian P5; scalar_mult_base(P5, five);
    u32 x5[ 8], y5[ 8]; jac_to_affine(x5, y5, P5);
    for (int i=7;i>=0;i--) d_out[o++]=x5[i];
    for (int i=7;i>=0;i--) d_out[o++]=y5[i];
    fe seven; fe_set0(seven); seven[ 0]=7u;
    Jacobian P7; scalar_mult_base(P7, seven);
    u32 x7[ 8], y7[ 8]; jac_to_affine(x7, y7, P7);
    for (int i=7;i>=0;i--) d_out[o++]=x7[i];
    for (int i=7;i>=0;i--) d_out[o++]=y7[i];
    fe eight; fe_set0(eight); eight[ 0]=8u;
    Jacobian P8; scalar_mult_base(P8, eight);
    u32 x8[ 8], y8[ 8]; jac_to_affine(x8, y8, P8);
    for (int i=7;i>=0;i--) d_out[o++]=x8[i];
    for (int i=7;i>=0;i--) d_out[o++]=y8[i];
    d_done[ 0]=1;
}

// ==========================================
// Kernel：sc_add_mod_n 诊断（IL + master_k）
// ==========================================
__global__ void scdiag_kernel(u32* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    // IL (b50e42e12a00b9e0abc8d9a1ced0f570252afcf07ed746e16d9d3a12dcc7d883)
    const u8 ilb[ 32] = { 0xb5,0x0e,0x42,0xe1,0x2a,0x00,0xb9,0xe0,0xab,0xc8,0xd9,0xa1,0xce,0xd0,0xf5,0x70,
                          0x25,0x2a,0xfc,0xf0,0x7e,0xd7,0x46,0xe1,0x6d,0x9d,0x3a,0x12,0xdc,0xc7,0xd8,0x83 };
    // master_k (1837c1be8e2995ec11cda2b066151be2cfb48adf9e47b151d46adab3a21cdf67)
    const u8 kb[ 32] = { 0x18,0x37,0xc1,0xbe,0x8e,0x29,0x95,0xec,0x11,0xcd,0xa2,0xb0,0x66,0x15,0x1b,0xe2,
                         0xcf,0xb4,0x8a,0xdf,0x9e,0x47,0xb1,0x51,0xd4,0x6a,0xda,0xb3,0xa2,0x1c,0xdf,0x67 };
    u32 il[ 8], kf[ 8], r[ 8];
    bytes32_to_fe(il, ilb);
    bytes32_to_fe(kf, kb);
    int o=0;
    for (int i=0;i<8;i++) d_out[o++]=il[i];
    for (int i=0;i<8;i++) d_out[o++]=kf[i];
    u64 c=0; u32 t[ 9];
    for (int i=0;i<8;i++){ u64 s=(u64)il[i]+kf[i]+c; t[i]=(u32)s; c=s>>32; }
    t[ 8]=(u32)c;
    for (int i=0;i<9;i++) d_out[o++]=t[i];
    sc_add_mod_n(r, il, kf);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    // fe_ge check result
    d_out[o++]=fe_ge(t, N_FE);
    d_out[o++]=t[ 8];
    d_done[ 0]=1;
}

// ==========================================
// Kernel：字段测试
// ==========================================
__global__ void fieldtest_kernel(u32* d_out) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    int o=0;
    fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu; a[ 0]=0xFFFFFC2Eu; a[ 1]=0xFFFFFFFEu;
    fe r;
    fe_mul(r, a, a);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe b; for (int i=0;i<8;i++) b[i]=0xFFFFFFFFu; b[ 0]=0xFFFFFC2Du; b[ 1]=0xFFFFFFFEu;
    fe_mul(r, b, a);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe c2; fe_set0(c2); c2[ 7]=0x80000000u;
    fe_mul(r, c2, c2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe d2; for (int i=0;i<8;i++) d2[i]=0xAAAAAAAAu;
    fe_mul(r, d2, d2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe e2; e2[ 0]=0x12345678u; e2[ 1]=0x9abcdef0u; e2[ 2]=0x11223344u; e2[ 3]=0x55667788u;
    e2[ 4]=0x99aabbccu; e2[ 5]=0xddeeff00u; e2[ 6]=0x10203040u; e2[ 7]=0x50607080u;
    fe_mul(r, e2, e2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe f1; for (int i=0;i<8;i++) f1[i]=0xFFFFFFFFu;
    fe f2; fe_set0(f2); f2[ 0]=0x00000001u; f2[ 1]=0x00000001u;
    fe_add(r, f1, f2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe g2; g2[ 0]=0x12345678u; g2[ 1]=0x9abcdef0u; g2[ 2]=0x11223344u; g2[ 3]=0x55667788u;
    g2[ 4]=0x99aabbccu; g2[ 5]=0xddeeff00u; g2[ 6]=0x10203040u; g2[ 7]=0x50607080u;
    fe_inv(r, g2);
    fe_mul(r, r, g2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    u32 nm1[ 8]; fe_cpy(nm1, N_FE); nm1[ 0]-=1;
    u32 two[ 8]; fe_set0(two); two[ 0]=2u;
    u32 sr[ 8]; sc_add_mod_n(sr, nm1, two);
    for (int i=0;i<8;i++) d_out[o++]=sr[i];
}

// ==========================================
// Kernel：fieldtest 进度标记版（定位挂起用例）
// ==========================================
__global__ void ftprog_kernel(u32* d_out, u32* d_prog) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    int o=0;
    fe r;
    d_prog[ 0]=1;
    fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu; a[ 0]=0xFFFFFC2Eu; a[ 1]=0xFFFFFFFEu;
    fe_mul(r, a, a);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=2;
    fe b; for (int i=0;i<8;i++) b[i]=0xFFFFFFFFu; b[ 0]=0xFFFFFC2Du; b[ 1]=0xFFFFFFFEu;
    fe_mul(r, b, a);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=3;
    fe c2; fe_set0(c2); c2[ 7]=0x80000000u;
    fe_mul(r, c2, c2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=4;
    fe d2; for (int i=0;i<8;i++) d2[i]=0xAAAAAAAAu;
    fe_mul(r, d2, d2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=5;
    fe e2; e2[ 0]=0x12345678u; e2[ 1]=0x9abcdef0u; e2[ 2]=0x11223344u; e2[ 3]=0x55667788u;
    e2[ 4]=0x99aabbccu; e2[ 5]=0xddeeff00u; e2[ 6]=0x10203040u; e2[ 7]=0x50607080u;
    fe_mul(r, e2, e2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=6;
    fe f1; for (int i=0;i<8;i++) f1[i]=0xFFFFFFFFu;
    fe f2; fe_set0(f2); f2[ 0]=0x00000001u; f2[ 1]=0x00000001u;
    fe_add(r, f1, f2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=7;
    fe g2; g2[ 0]=0x12345678u; g2[ 1]=0x9abcdef0u; g2[ 2]=0x11223344u; g2[ 3]=0x55667788u;
    g2[ 4]=0x99aabbccu; g2[ 5]=0xddeeff00u; g2[ 6]=0x10203040u; g2[ 7]=0x50607080u;
    fe_inv(r, g2);
    fe_mul(r, r, g2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=8;
    u32 nm1[ 8]; fe_cpy(nm1, N_FE); nm1[ 0]-=1;
    u32 two[ 8]; fe_set0(two); two[ 0]=2u;
    u32 sr[ 8]; sc_add_mod_n(sr, nm1, two);
    for (int i=0;i<8;i++) d_out[o++]=sr[i];
    d_prog[ 0]=9;
}

// ==========================================
// Kernel：单用例 bisect（定位 GPU 挂起用例）
// ==========================================
__global__ void bisect_kernel(u32* d_out, int c) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    fe r;
    if (c==0) { fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu; a[ 0]=0xFFFFFC2Eu; a[ 1]=0xFFFFFFFEu; fe_mul(r, a, a); }
    else if (c==1) { fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu; a[ 0]=0xFFFFFC2Eu; a[ 1]=0xFFFFFFFEu;
                     fe b; for (int i=0;i<8;i++) b[i]=0xFFFFFFFFu; b[ 0]=0xFFFFFC2Du; b[ 1]=0xFFFFFFFEu; fe_mul(r, b, a); }
    else if (c==2) { fe c2; fe_set0(c2); c2[ 7]=0x80000000u; fe_mul(r, c2, c2); }
    else if (c==3) { fe d2; for (int i=0;i<8;i++) d2[i]=0xAAAAAAAAu; fe_mul(r, d2, d2); }
    else if (c==4) { fe e2; e2[ 0]=0x12345678u; e2[ 1]=0x9abcdef0u; e2[ 2]=0x11223344u; e2[ 3]=0x55667788u;
                     e2[ 4]=0x99aabbccu; e2[ 5]=0xddeeff00u; e2[ 6]=0x10203040u; e2[ 7]=0x50607080u; fe_mul(r, e2, e2); }
    else if (c==5) { fe f1; for (int i=0;i<8;i++) f1[i]=0xFFFFFFFFu;
                     fe f2; fe_set0(f2); f2[ 0]=0x00000001u; f2[ 1]=0x00000001u; fe_add(r, f1, f2); }
    else if (c==6) { fe g2; g2[ 0]=0x12345678u; g2[ 1]=0x9abcdef0u; g2[ 2]=0x11223344u; g2[ 3]=0x55667788u;
                     g2[ 4]=0x99aabbccu; g2[ 5]=0xddeeff00u; g2[ 6]=0x10203040u; g2[ 7]=0x50607080u;
                     fe_inv(r, g2); fe_mul(r, r, g2); }
    else if (c==7) { u32 nm1[ 8]; fe_cpy(nm1, N_FE); nm1[ 0]-=1;
           u32 two[ 8]; fe_set0(two); two[ 0]=2u;
           sc_add_mod_n(r, nm1, two); }
    else if (c==8) { // fe_add(D,D); D=cbc1950e... -> expect 97832a1c...
           u32 D[ 8] = { 0x69cfc671u, 0xd2076736u, 0xf147c112u, 0x6bcb0bbcu, 0xcb96483du, 0xd0271c6bu, 0x6fec4514u, 0xcbc1950eu };
           fe_add(r, D, D); }
    else if (c==9) { // fe_sub(F, T); F=E^2 mod p, T=2D mod p -> expect X3
           u32 F[ 8] = { 0x87054446u, 0x35e2d2b3u, 0xd92564edu, 0x8e25a078u, 0xb6d66bd0u, 0xc1dfbd14u, 0xfe816c05u, 0x14985620u };
           u32 T[ 8] = { 0xd39f90b3u, 0xa40ece6du, 0xe28f8225u, 0xd7961779u, 0x972c907au, 0xa04e38d7u, 0xdfd88a29u, 0x97832a1cu };
           fe_sub(r, F, T); }
    else { u32 nm1[ 8]; fe_cpy(nm1, N_FE); nm1[ 0]-=1;
           u32 two[ 8]; fe_set0(two); two[ 0]=2u;
           sc_add_mod_n(r, nm1, two); }
    for (int i=0;i<8;i++) d_out[i]=r[i];
}

// ==========================================
// Kernel：debug（(p-1)^2 中间量转储）
// ==========================================
__global__ void debug_kernel(u32* d_out) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    int o=0;
    fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu; a[ 0]=0xFFFFFC2Eu; a[ 1]=0xFFFFFFFEu;
    u32 t[ 16];
    for (int i=0;i<16;i++) t[i]=0;
    for (int i=0;i<8;i++)
        for (int j=0;j<8;j++) {
            u64 p = (u64)a[i]*a[j];
            u64 s = (u64)t[i+j] + (u32)p;
            t[i+j] = (u32)s;
            u64 c = s >> 32;
            int k = i+j+1;
            while (c && k < 16) { u64 s2=(u64)t[k]+c; t[k]=(u32)s2; c=s2>>32; k++; }
            s = (u64)t[i+j+1] + (u32)(p>>32);
            t[i+j+1] = (u32)s;
            c = s >> 32;
            k = i+j+2;
            while (c && k < 16) { u64 s2=(u64)t[k]+c; t[k]=(u32)s2; c=s2>>32; k++; }
        }
    for (int i=0;i<16;i++) d_out[o++]=t[i];       // 0-15: 原始乘积
    u32 v[ 10];
    u64 c=0;
    for (int i=0;i<10;i++){
        u64 s = c + (i<8 ? (u64)t[i] : 0);
        if (i>=1 && (i-1)<8) s += (u64)t[ 8+(i-1)];
        if (i<8) s += (u64)t[ 8+i]*977u;
        v[i]=(u32)s; c=s>>32;
    }
    if (c){ u64 s=(u64)v[ 9]+c; v[ 9]=(u32)s; }
    for (int i=0;i<10;i++) d_out[o++]=v[i];       // 16-25: 初始折叠 v
    fe_fold_hi(v);
    for (int i=0;i<10;i++) d_out[o++]=v[i];       // 26-35: fe_fold_hi 后 v
    fe r; fe_mul(r, a, a);
    for (int i=0;i<8;i++) d_out[o++]=r[i];        // 36-43: fe_mul 结果
}

// ==========================================
// Kernel：固定熵测试（BIP39 标准向量）
// ==========================================
__global__ void test_kernel(const u8* d_ent, char* d_mnem, u8* d_addr, u8* d_priv, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 ent[ 16];
    for (int i=0;i<16;i++) ent[i]=d_ent[i];
    char mnem[ 128];
    u8 addr[ 20], privk[ 32];
    derive_full(ent, mnem, addr, privk);
    int l=0; while (mnem[l]){ d_mnem[l]=mnem[l]; l++; } d_mnem[l]=0;
    for (int i=0;i<20;i++) d_addr[i]=addr[i];
    for (int i=0;i<32;i++) d_priv[i]=privk[i];
    d_done[ 0]=1;
}

// ==========================================
// Kernel：vanity 搜索
// ==========================================


// ==========================================
// HD 快速模式：固定种子，变索引 m/44'/60'/0'/0/N 搜索
// setup: 熵->助记词->seed->派生到 m/44'/60'/0'/0 (k0n,c0n) + pubkey(k0n)
// search: N 变化 -> k_N = IL + k0n (mod n) -> pubkey -> keccak -> 前缀比对
// ==========================================
__global__ void hd_setup_kernel(u64 base_nonce, u32* d_kbase, u32* d_cbase, u32* d_pubx, u32* d_puby, char* d_mnem) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 ent[ 16]; counter_to_entropy(base_nonce, ent);
    char m[ 128]; entropy_to_mnemonic(ent, m, 0);
    int l=0; while(m[l]){ d_mnem[l]=m[l]; l++; } d_mnem[l]=0;
    u8 seed[ 64]; device_bip39_seed(m, seed);
    // master
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    // 44' 60' 0'
    ckd_priv(k, c, 0x80000000u+44u, 1);
    ckd_priv(k, c, 0x80000000u+60u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    // 0 (non-hardened) -> k0n
    u8 k0n[ 32], c0n[ 32];
    for (int i=0;i<32;i++){ k0n[i]=k[i]; c0n[i]=c[i]; }
    ckd_priv(k0n, c0n, 0u, 0);
    // 先写 kbase/cbase，再算 pubkey（避免 comb 潜在污染）
    u32 kb[ 8], cb[ 8];
    bytes32_to_fe(kb, k0n); bytes32_to_fe(cb, c0n);
    for (int i=0;i<8;i++){ d_kbase[i]=kb[i]; d_cbase[i]=cb[i]; }
    // pubkey(k0n)
    u32 kf[ 8]; bytes32_to_fe(kf, k0n);
    Jacobian P; scalar_mult_base_comb(P, kf);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    for (int i=0;i<8;i++){ d_pubx[i]=x[i]; d_puby[i]=y[i]; }
}

__global__ __launch_bounds__(256, 2) void hd_vanity_kernel(const u8* d_prefix, int plen,
        const u32* d_kbase, const u32* d_cbase, const u32* d_pubx, const u32* d_puby,
        u64 start_nonce, int iters, char* d_mnem, u8* d_addr, u8* d_priv, u32* d_index, u32* d_found) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    // v2: 链码与父公钥在整个 kernel 生命周期内固定，HMAC 预计算和字节转换提到循环外只做一次
    u32 cb0[ 8]; for (int i=0;i<8;i++) cb0[i]=d_cbase[i];
    u8 c8[ 32]; fe_to_bytes32(c8, cb0);
    u64 cipad[ 8], copad[ 8];
    hmac_sha512_precompute(c8, 32, cipad, copad);
    u8 data0[ 33];
    data0[ 0] = (d_puby[ 0]&1) ? 0x03 : 0x02;
    u8 xb[ 32]; fe_to_bytes32(xb, d_pubx);
    for (int i=0;i<32;i++) data0[1+i]=xb[i];
    for (int it=0; it<iters; it++) {
        /* v3: 去掉整批早退，保证每批跑满，速率不虚高 */
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u32 N = (u32)(n & 0x7FFFFFFFu);
        // IL = HMAC-SHA512(c0n, 0x02/03||X || N)
        u8 data[ 37];
        for (int i=0;i<33;i++) data[i]=data0[i];
        data[ 33]=(u8)(N>>24); data[ 34]=(u8)(N>>16); data[ 35]=(u8)(N>>8); data[ 36]=(u8)N;
        u8 out[ 64];
        hmac_sha512_precomp_msg(cipad, copad, data, 37, out);
        u32 il[ 8]; bytes32_to_fe(il, out);
        u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=d_kbase[i];
        sc_add_mod_n(kf, il, kf);
        Jacobian P; scalar_mult_base_comb(P, kf);
        u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
        u8 pub[ 64];
        fe_to_bytes32(pub, x);
        fe_to_bytes32(pub+32, y);
        u8 hh[ 32];
        device_keccak256(pub, 64, hh);
        u8 addr[ 20]; for (int i=0;i<20;i++) addr[i]=hh[12+i];
        int ok=1;
        for (int i=0;i<plen;i++) {
            u8 nib = (i&1) ? (addr[i>>1]&0xF) : (addr[i>>1]>>4);
            if (nib != d_prefix[i]) { ok=0; break; }
        }
        if (ok) {
            u32 old = atomicCAS(d_found, 0, 1);
            if (old==0) {
                u8 privk[ 32]; fe_to_bytes32(privk, kf);
                for (int i=0;i<20;i++) d_addr[i]=addr[i];
                for (int i=0;i<32;i++) d_priv[i]=privk[i];
                d_index[ 0]=N;
            }
            return;
        }
    }
}

__global__ void hd_diag_kernel(u64 base_nonce, u8* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 ent[ 16]; counter_to_entropy(base_nonce, ent);
    char m[ 128]; entropy_to_mnemonic(ent, m, 0);
    for (int i=0;i<128;i++) d_out[i]=(u8)m[i]; // mnemonic bytes 0..127
    u8 seed[ 64]; device_bip39_seed(m, seed);
    for (int i=0;i<64;i++) d_out[128+i]=seed[i];
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    ckd_priv(k, c, 0x80000000u+44u, 1);
    ckd_priv(k, c, 0x80000000u+60u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    for (int i=0;i<32;i++) d_out[192+i]=k[i];  // k0h
    for (int i=0;i<32;i++) d_out[224+i]=c[i];  // c0h
    u8 k0n[ 32], c0n[ 32];
    for (int i=0;i<32;i++){ k0n[i]=k[i]; c0n[i]=c[i]; }
    ckd_priv(k0n, c0n, 0u, 0);
    for (int i=0;i<32;i++) d_out[256+i]=k0n[i]; // k0n
    for (int i=0;i<32;i++) d_out[288+i]=c0n[i]; // c0n
    u32 kf[ 8]; bytes32_to_fe(kf, k0n);
    Jacobian P; scalar_mult_base_comb(P, kf);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    u8 xb[ 32]; fe_to_bytes32(xb, x);
    for (int i=0;i<32;i++) d_out[320+i]=xb[i]; // pubx
    d_done[ 0]=1;
}

__global__ void hd_diag2_kernel(u64 base_nonce, u32 N, u8* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 ent[ 16]; counter_to_entropy(base_nonce, ent);
    char m[ 128]; entropy_to_mnemonic(ent, m, 0);
    for (int i=0;i<128;i++) d_out[i]=(u8)m[i];
    u8 seed[ 64]; device_bip39_seed(m, seed);
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    ckd_priv(k, c, 0x80000000u+44u, 1);
    ckd_priv(k, c, 0x80000000u+60u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    u8 k0n[ 32], c0n[ 32];
    for (int i=0;i<32;i++){ k0n[i]=k[i]; c0n[i]=c[i]; }
    ckd_priv(k0n, c0n, 0u, 0);
    // ===== 完全复用 hd_vanity_kernel 的 search 计算路径 (N=0) =====
    u32 kf0[ 8]; bytes32_to_fe(kf0, k0n);
    Jacobian P0; scalar_mult_base_comb(P0, kf0);
    u32 px[ 8], py[ 8]; jac_to_affine(px, py, P0);
    u32 cb[ 8]; bytes32_to_fe(cb, c0n);
    u8 c8[ 32]; fe_to_bytes32(c8, cb);
    u8 data[ 37];
    data[ 0] = (py[ 0]&1) ? 0x03 : 0x02;
    u8 xb[ 32]; fe_to_bytes32(xb, px);
    for (int i=0;i<32;i++) data[1+i]=xb[i];
    data[33]=(u8)(N>>24); data[34]=(u8)(N>>16); data[35]=(u8)(N>>8); data[36]=(u8)N;
    u8 o2[ 64];
    hmac_sha512_gen(c8, 32, data, 37, o2);
    u32 il[ 8]; bytes32_to_fe(il, o2);
    u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=kf0[i];
    sc_add_mod_n(kf, il, kf);
    u8 kchild[ 32]; fe_to_bytes32(kchild, kf);
    for (int i=0;i<32;i++) d_out[128+i]=kchild[i];   // child priv @ N=0
    Jacobian P; scalar_mult_base_comb(P, kf);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    u8 pub[ 64];
    fe_to_bytes32(pub, x);
    fe_to_bytes32(pub+32, y);
    u8 hh[ 32]; device_keccak256(pub, 64, hh);
    for (int i=0;i<20;i++) d_out[160+i]=hh[12+i];     // addr
    d_done[ 0]=1;
}
// ==========================================
// Kernel: 分阶段性能剖析 (stage: 0=mnemonic 1=+pbkdf2 2=+ckd 3=full)
// 每段结果都累加进 checksum 写入全局，防止 DCE
// ==========================================
__global__ void prof_kernel(u64 start_nonce, int iters, int stage, u64* d_out, u32* d_done) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    u64 acc = 0;
    for (int it=0; it<iters; it++) {
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u8 ent[ 16]; counter_to_entropy(n, ent);
        char m[ 128]; entropy_to_mnemonic(ent, m, 0);
        u64 acc0 = 0;
        for (int i=0;i<128;i++) acc0 = acc0*131 + (u8)m[i];
        if (stage>=1) {
            u8 seed[ 64]; device_bip39_seed(m, seed);
            u64 acc1 = acc0;
            for (int i=0;i<64;i++) acc1 = acc1*131 + seed[i];
            if (stage>=2) {
                u8 privk[ 32]; bip32_derive(privk, seed);
                u64 acc2 = acc1;
                for (int i=0;i<32;i++) acc2 = acc2*131 + privk[i];
                if (stage>=3) {
                    u8 addr[ 20]; derive_address(privk, addr);
                    for (int i=0;i<20;i++) acc2 = acc2*131 + addr[i];
                    acc += acc2;
                } else acc += acc2;
            } else acc += acc1;
        } else acc += acc0;
    }
    if (acc == 0xDEADBEEF) d_out[ tid % 1024 ] = acc;
    d_done[ 0]=1;
}
// 熵 -> 助记词 -> m/44'/60'/0'/0 基点（kbase/cbase/pubx/puby），供多索引派生复用
__device__ void ent_to_hd_base(const u8 ent[ 16], u32 kbase[ 8], u32 cbase[ 8],
                               u32 pubx[ 8], u32 puby[ 8], char* mnem_out) {
    char m[ 128];
    entropy_to_mnemonic(ent, m, 0);
    if (mnem_out) { int l=0; while (m[l]){ mnem_out[l]=m[l]; l++; } mnem_out[l]=0; }
    u8 seed[ 64];
    device_bip39_seed(m, seed);
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    ckd_priv(k, c, 0x80000000u+44u, 1);
    ckd_priv(k, c, 0x80000000u+60u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    u8 k0n[ 32], c0n[ 32];
    for (int i=0;i<32;i++){ k0n[i]=k[i]; c0n[i]=c[i]; }
    ckd_priv(k0n, c0n, 0u, 0);
    bytes32_to_fe(kbase, k0n);
    bytes32_to_fe(cbase, c0n);
    u32 kf[ 8]; bytes32_to_fe(kf, k0n);
    Jacobian P; scalar_mult_base_comb(P, kf);
    jac_to_affine(pubx, puby, P);
}

__global__ __launch_bounds__(256, 2) void vanity_kernel(const u8* d_prefix, int plen, u64 start_nonce, int iters, int nidx,
                              char* d_mnem, u8* d_addr, u8* d_priv, u32* d_index, u32* d_found) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    for (int it=0; it<iters; it++) {
        /* v3: 去掉整批早退，保证每批跑满，速率不虚高 */
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u8 ent[ 16]; counter_to_entropy(n, ent);
        u32 kbase[ 8], cbase[ 8], pubx[ 8], puby[ 8];
        char mnem[ 128];
        ent_to_hd_base(ent, kbase, cbase, pubx, puby, mnem);
        u8 c8[ 32]; fe_to_bytes32(c8, cbase);
        u8 xb0[ 32]; fe_to_bytes32(xb0, pubx);
        u64 cipad[ 8], copad[ 8];
        hmac_sha512_precompute(c8, 32, cipad, copad);  // v2: 同一助记词的 nidx 个子地址共用链码，预计算只做一次
        for (int idx=0; idx<nidx; idx++) {
            u32 N = (u32)idx;
            u8 data[ 37];
            data[ 0] = (puby[ 0]&1) ? 0x03 : 0x02;
            for (int i=0;i<32;i++) data[1+i]=xb0[i];
            data[ 33]=(u8)(N>>24); data[ 34]=(u8)(N>>16); data[ 35]=(u8)(N>>8); data[ 36]=(u8)N;
            u8 out[ 64];
            hmac_sha512_precomp_msg(cipad, copad, data, 37, out);
            u32 il[ 8]; bytes32_to_fe(il, out);
            u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=kbase[i];
            sc_add_mod_n(kf, il, kf);
            Jacobian P; scalar_mult_base_comb(P, kf);
            u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
            u8 pub[ 64];
            fe_to_bytes32(pub, x);
            fe_to_bytes32(pub+32, y);
            u8 hh[ 32];
            device_keccak256(pub, 64, hh);
            u8 addr[ 20]; for (int i=0;i<20;i++) addr[i]=hh[12+i];
            int ok=1;
            for (int i=0;i<plen;i++) {
                u8 nib = (i&1) ? (addr[i>>1]&0xF) : (addr[i>>1]>>4);
                if (nib != d_prefix[i]) { ok=0; break; }
            }
            if (ok) {
                u32 old = atomicCAS(d_found, 0, 1);
                if (old==0) {
                    int l=0; while (mnem[l]){ d_mnem[l]=mnem[l]; l++; } d_mnem[l]=0;
                    u8 privk[ 32]; fe_to_bytes32(privk, kf);
                    for (int i=0;i<20;i++) d_addr[i]=addr[i];
                    for (int i=0;i<32;i++) d_priv[i]=privk[i];
                    d_index[ 0]=N;
                }
                return;
            }
        }
    }
}


// ==========================================
// 文件模式：一个固定种子，同时搜索多个前缀（HD m/44'/60'/0'/0/N）
// 每个前缀独立 found slot + 结果；全部找到则提前退出
// ==========================================
__global__ __launch_bounds__(256, 2) void multi_hd_kernel(
        const u8* d_prefixes, const u8* d_plens, int nprefix,
        const u32* d_kbase, const u32* d_cbase, const u32* d_pubx, const u32* d_puby,
        u64 start_nonce, int iters, const char* d_mnem,
        u8* d_addrs, u8* d_privs, u32* d_idxs, u32* d_found, u32* d_nfound) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    for (int it=0; it<iters; it++) {
        if (*d_nfound >= (u32)nprefix) return;
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u32 N = (u32)(n & 0x7FFFFFFFu);
        // IL = HMAC-SHA512(c0n, 0x02/03||X || N)
        u32 cb[ 8]; for (int i=0;i<8;i++) cb[i]=d_cbase[i];
        u8 c8[ 32]; fe_to_bytes32(c8, cb);
        u8 data[ 37];
        data[ 0] = (d_puby[ 0]&1) ? 0x03 : 0x02;
        u8 xb[ 32]; fe_to_bytes32(xb, d_pubx);
        for (int i=0;i<32;i++) data[1+i]=xb[i];
        data[ 33]=(u8)(N>>24); data[ 34]=(u8)(N>>16); data[ 35]=(u8)(N>>8); data[ 36]=(u8)N;
        u8 out[ 64];
        hmac_sha512_gen(c8, 32, data, 37, out);
        u32 il[ 8]; bytes32_to_fe(il, out);
        u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=d_kbase[i];
        sc_add_mod_n(kf, il, kf);
        Jacobian P; scalar_mult_base_comb(P, kf);
        u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
        u8 pub[ 64];
        fe_to_bytes32(pub, x);
        fe_to_bytes32(pub+32, y);
        u8 hh[ 32];
        device_keccak256(pub, 64, hh);
        u8 addr[ 20]; for (int i=0;i<20;i++) addr[i]=hh[12+i];
        // 遍历所有未完成前缀
        for (int p=0; p<nprefix; p++) {
            if (d_found[p]) continue;
            int pl = d_plens[p];
            int ok=1;
            const u8* pfx = d_prefixes + p*40;
            for (int i=0;i<pl;i++) {
                u8 nib = (i&1) ? (addr[i>>1]&0xF) : (addr[i>>1]>>4);
                if (nib != pfx[i]) { ok=0; break; }
            }
            if (ok) {
                u32 old = atomicCAS(&d_found[p], 0, 1);
                if (old==0) {
                    u8 privk[ 32]; fe_to_bytes32(privk, kf);
                    for (int i=0;i<20;i++) d_addrs[p*20+i]=addr[i];
                    for (int i=0;i<32;i++) d_privs[p*32+i]=privk[i];
                    d_idxs[p]=N;
                    atomicAdd(d_nfound, 1);
                }
            }
        }
    }
}

// ==========================================
// 文件模式（vanity 语义）：随机助记词 + 多前缀并行 + 每助记词派生 nidx 个地址
// ==========================================
__global__ __launch_bounds__(256, 2) void multi_vanity_kernel(
        const u8* d_prefixes, const u8* d_plens, int nprefix, int nidx,
        u64 start_nonce, int iters,
        u8* d_mnems, u8* d_addrs, u8* d_privs, u32* d_idxs, u32* d_found, u32* d_nfound) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    for (int it=0; it<iters; it++) {
        if (*d_nfound >= (u32)nprefix) return;
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u8 ent[ 16]; counter_to_entropy(n, ent);
        u32 kbase[ 8], cbase[ 8], pubx[ 8], puby[ 8];
        char mnem[ 128];
        ent_to_hd_base(ent, kbase, cbase, pubx, puby, mnem);
        u8 c8[ 32]; fe_to_bytes32(c8, cbase);
        u8 xb0[ 32]; fe_to_bytes32(xb0, pubx);
        u64 cipad[ 8], copad[ 8];
        hmac_sha512_precompute(c8, 32, cipad, copad);  // v2: 同一助记词的 nidx 个子地址共用链码，预计算只做一次
        for (int idx=0; idx<nidx; idx++) {
            u32 N = (u32)idx;
            u8 data[ 37];
            data[ 0] = (puby[ 0]&1) ? 0x03 : 0x02;
            for (int i=0;i<32;i++) data[1+i]=xb0[i];
            data[ 33]=(u8)(N>>24); data[ 34]=(u8)(N>>16); data[ 35]=(u8)(N>>8); data[ 36]=(u8)N;
            u8 out[ 64];
            hmac_sha512_precomp_msg(cipad, copad, data, 37, out);
            u32 il[ 8]; bytes32_to_fe(il, out);
            u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=kbase[i];
            sc_add_mod_n(kf, il, kf);
            Jacobian P; scalar_mult_base_comb(P, kf);
            u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
            u8 pub[ 64];
            fe_to_bytes32(pub, x);
            fe_to_bytes32(pub+32, y);
            u8 hh[ 32];
            device_keccak256(pub, 64, hh);
            u8 addr[ 20]; for (int i=0;i<20;i++) addr[i]=hh[12+i];
            for (int p=0; p<nprefix; p++) {
                if (d_found[p]) continue;
                int pl = d_plens[p];
                int ok=1;
                const u8* pfx = d_prefixes + p*40;
                for (int i=0;i<pl;i++) {
                    u8 nib = (i&1) ? (addr[i>>1]&0xF) : (addr[i>>1]>>4);
                    if (nib != pfx[i]) { ok=0; break; }
                }
                if (ok) {
                    u32 old = atomicCAS(&d_found[p], 0, 1);
                    if (old==0) {
                        u8 privk[ 32]; fe_to_bytes32(privk, kf);
                        int l=0; while (mnem[l]){ d_mnems[p*128+l]=(u8)mnem[l]; l++; } d_mnems[p*128+l]=0;
                        for (int i=0;i<20;i++) d_addrs[p*20+i]=addr[i];
                        for (int i=0;i<32;i++) d_privs[p*32+i]=privk[i];
                        d_idxs[p]=N;
                        atomicAdd(d_nfound, 1);
                    }
                }
            }
        }
    }
}

typedef struct { u32 p; u32 idx; u8 mnem[128]; u8 addr[20]; u8 priv[32]; } ResqEntry;
__global__ __launch_bounds__(256, 2) void multi_hash_kernel(
        const u64* d_tbl_key, const u32* d_tbl_p, const u8* d_tbl_occ,
        const u32* d_tbl_off, const u32* d_tbl_mask, const u32* d_tbl_cnt, const u8* d_pfx, u64 uniq,
        int nidx, u64 start_nonce, int iters,
        u32* d_found, u32* d_nfound, ResqEntry* d_resq, u32* d_resq_cnt) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    for (int it=0; it<iters; it++) {
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u8 ent[ 16]; counter_to_entropy(n, ent);
        u32 kbase[ 8], cbase[ 8], pubx[ 8], puby[ 8];
        char mnem[ 128];
        ent_to_hd_base(ent, kbase, cbase, pubx, puby, mnem);
        u8 c8[ 32]; fe_to_bytes32(c8, cbase);
        u8 xb0[ 32]; fe_to_bytes32(xb0, pubx);
        u64 cipad[ 8], copad[ 8];
        hmac_sha512_precompute(c8, 32, cipad, copad);  // v2: 同一助记词的 nidx 个子地址共用链码，预计算只做一次
        for (int idx=0; idx<nidx; idx++) {
            u32 N = (u32)idx;
            u8 data[ 37];
            data[ 0] = (puby[ 0]&1) ? 0x03 : 0x02;
            for (int i=0;i<32;i++) data[1+i]=xb0[i];
            data[ 33]=(u8)(N>>24); data[ 34]=(u8)(N>>16); data[ 35]=(u8)(N>>8); data[ 36]=(u8)N;
            u8 out[ 64];
            hmac_sha512_precomp_msg(cipad, copad, data, 37, out);
            u32 il[ 8]; bytes32_to_fe(il, out);
            u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=kbase[i];
            sc_add_mod_n(kf, il, kf);
            Jacobian P; scalar_mult_base_comb(P, kf);
            u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
            u8 pub[ 64];
            fe_to_bytes32(pub, x);
            fe_to_bytes32(pub+32, y);
            u8 hh[ 32];
            device_keccak256(pub, 64, hh);
            u8 addr[ 20]; for (int i=0;i<20;i++) addr[i]=hh[12+i];
            u8 nib[ 40];
            for (int i=0;i<20;i++) { nib[2*i]=addr[i]>>4; nib[2*i+1]=addr[i]&15; }
            u64 h=0xcbf29ce484222325ull;
            for (int L=1; L<=NL; L++) {
                h = (h ^ nib[L-1]) * 0x100000001b3ull;
                int cnt = d_tbl_cnt[L-1];
                if (!cnt) continue;
                u32 mask = d_tbl_mask[L-1];
                u32 base = d_tbl_off[L-1];
                u32 hh2 = (u32)((h * 0x9E3779B97F4A7C15ull) >> 32);
                int s = (int)(hh2 & mask);
                for (;;) {
                    if (!d_tbl_occ[base+s]) break;
                    if (d_tbl_key[base+s] == h) {
                        u32 p = d_tbl_p[base+s];
                        int ok=1;
                        for (int k=0;k<L;k++) if (d_pfx[p*40+k]!=nib[k]) { ok=0; break; }
                        if (ok && !d_found[p]) {
                            u32 old = atomicCAS(&d_found[p], 0, 1);
                            if (old==0) {
                                u8 privk[ 32]; fe_to_bytes32(privk, kf);
                                u32 q = atomicAdd(d_resq_cnt, 1);
                                ResqEntry* e = &d_resq[q & RESQ_MASK];
                                e->p = p; e->idx = N;
                                int l=0; while (mnem[l] && l<127){ e->mnem[l]=(u8)mnem[l]; l++; } e->mnem[l]=0;
                                for (int i=0;i<20;i++) e->addr[i]=addr[i];
                                for (int i=0;i<32;i++) e->priv[i]=privk[i];
                                atomicAdd(d_nfound, 1);
                            }
                            break;
                        }
                    }
                    s = (s+1) & mask;
                }
            }
        }
    }
}

// ==========================================
// 主机辅助
// ==========================================
static void print_hex(const u8* p, int n) {
    for (int i=0;i<n;i++) printf("%02x", p[i]);
}

// v2: 加载或创建 32B secret 文件（默认 ./.eth_vanity_secret，可用环境变量 ETH_VANITY_SECRET 指定路径）。
// secret 与计数器共同决定助记词序列；secret 绝不打印、不写入日志。
// 注意：丢了 secret 不影响用 eth_hits.txt 里已记录的助记词找回单个钱包，但无法再复现整条搜索序列。
// 首次创建后请离线备份一份，并保持文件权限 0600。
static int load_or_create_secret(u8 sec[ 32]) {
    const char* path = getenv("ETH_VANITY_SECRET");
    if (!path || !path[0]) path = ".eth_vanity_secret";
    FILE* f = fopen(path, "rb");
    if (f) {
        size_t r = fread(sec, 1, 32, f); fclose(f);
        if (r == 32) {
#ifndef _WIN32
            chmod(path, 0600);
#endif
            printf("secret: loaded from %s (keep it private, back it up offline)\n", path); return 1; }
        fprintf(stderr, "secret file %s exists but is not 32 bytes; refusing to overwrite it\n", path);
        return 0;
    }
    FILE* u = fopen("/dev/urandom", "rb");
    if (u) {
        size_t r = fread(sec, 1, 32, u); fclose(u);
        if (r != 32) { fprintf(stderr, "short read from /dev/urandom\n"); return 0; }
    } else {
        std::random_device rd;  // 无 /dev/urandom 的平台（如原生 Windows）走系统 CSPRNG
        for (int i=0;i<32;i++) sec[i]=(u8)(rd() & 0xFF);
    }
    f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot create secret file %s\n", path); return 0; }
    fwrite(sec, 1, 32, f); fclose(f);
#ifndef _WIN32
    chmod(path, 0600);
#endif
    printf("secret: created %s (0600). Back it up offline; never share it.\n", path);
    return 1;
}

static int chain_main(int argc, char** argv) {
    const char* mode = argc>1 ? argv[1] : "bench";
    printf("=== ETH BIP39 Vanity Generator (true secp256k1 + BIP32/44) ===\n");
    u8 secret[ 32];
    if (!load_or_create_secret(secret)) return 1;
    CUDA_CHECK(cudaMemcpyToSymbol(SECRET_K, secret, 32));

    // comb 固定基点预计算表（60 点）
    u32* d_comb;
    CUDA_CHECK(cudaMalloc(&d_comb, 60*16*4));
    precompute_comb_kernel<<<15, 4>>>(d_comb);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyToSymbol(g_comb, &d_comb, sizeof(d_comb)));
    CUDA_CHECK(cudaDeviceSynchronize());  // v2: 确保 comb 表已写完，后续任何流（sK/sP）上的 kernel 都不会读到未就绪的表

    u8 prefix_nib[ 40];
    int plen=0;
    if ((strcmp(mode,"vanity")==0 || strcmp(mode,"hd")==0) && argc>=3) {
        const char* p = argv[2];
        int off = (p[0]=='0' && (p[1]=='x'||p[1]=='X')) ? 2 : 0;
        int i=0;
        while (p[off+i] && i<40) {
            char ch = p[off+i];
            int hv = -1;
            if (ch>='0'&&ch<='9') hv=ch-'0';
            else if (ch>='a'&&ch<='f') hv=ch-'a'+10;
            else if (ch>='A'&&ch<='F') hv=ch-'A'+10;
            if (hv<0) break;
            prefix_nib[i]=(u8)hv;
            i++;
        }
        plen=i;
        if (plen==0) { fprintf(stderr,"bad prefix\n"); return 1; }
        printf("vanity prefix: 0x");
        for (int i=0;i<plen;i++) printf("%x", prefix_nib[i]);
        printf(" (%d hex digits)\n", plen);
    }

    u8* d_prefix;
    char* d_mnem; u8* d_addr; u8* d_priv; u32* d_found; u32* d_index;
    CUDA_CHECK(cudaMalloc(&d_prefix, 40));
    CUDA_CHECK(cudaMalloc(&d_mnem, 128));
    CUDA_CHECK(cudaMalloc(&d_addr, 20));
    CUDA_CHECK(cudaMalloc(&d_priv, 32));
    CUDA_CHECK(cudaMalloc(&d_found, 4));
    CUDA_CHECK(cudaMalloc(&d_index, 4));
    if (plen) CUDA_CHECK(cudaMemcpy(d_prefix, prefix_nib, plen, cudaMemcpyHostToDevice));

    if (strcmp(mode,"selftest")==0) {
        u64* d_out; u32* d_done;
        CUDA_CHECK(cudaMalloc(&d_out, 11*8));
        CUDA_CHECK(cudaMalloc(&d_done, 4));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK));
        CUDA_CHECK(cudaStreamCreate(&sP));
        selftest_kernel<<<1,1,0,sK>>>(d_out, d_done);
        for (int w=0;w<40;w++) {
            u32 p=0;
            cudaMemcpyAsync(&p, d_done, 4, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            if (p==1) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
        u64 h_out[ 11];
        cudaMemcpyAsync(h_out, d_out, 11*8, cudaMemcpyDeviceToHost, sP);
        cudaStreamSynchronize(sP);
        printf("selftest:\n");
        printf("  (p-1)+1        = %016llx %016llx\n", (unsigned long long)(h_out[ 0]>>32), (unsigned long long)(h_out[ 0]&0xFFFFFFFFFFFFFFFFULL));
        printf("  123456789*987..= %016llx %016llx\n", (unsigned long long)(h_out[ 1]>>32), (unsigned long long)h_out[ 1]);
        printf("  2G x           = %016llx %016llx\n", (unsigned long long)(h_out[ 2]>>32), (unsigned long long)h_out[ 2]);
        printf("  2G y           = %016llx %016llx\n", (unsigned long long)(h_out[ 3]>>32), (unsigned long long)h_out[ 3]);
        printf("  1G x           = %016llx %016llx\n", (unsigned long long)(h_out[ 4]>>32), (unsigned long long)h_out[ 4]);
        printf("  1G y           = %016llx %016llx\n", (unsigned long long)(h_out[ 5]>>32), (unsigned long long)h_out[ 5]);
        printf("  (n-1)+1 mod n  = %016llx %016llx\n", (unsigned long long)(h_out[ 6]>>32), (unsigned long long)h_out[ 6]);
        printf("  (n-2)+3 mod n  = %016llx %016llx\n", (unsigned long long)(h_out[ 7]>>32), (unsigned long long)h_out[ 7]);
        printf("  2G+G x (jac)   = %016llx %016llx\n", (unsigned long long)(h_out[ 8]>>32), (unsigned long long)h_out[ 8]);
        printf("  3G x (scalar)  = %016llx %016llx\n", (unsigned long long)(h_out[ 9]>>32), (unsigned long long)h_out[ 9]);
        printf("  a*a^-1         = %016llx %016llx\n", (unsigned long long)(h_out[ 10]>>32), (unsigned long long)h_out[ 10]);
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_done));
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
    }
    else if (strcmp(mode,"fieldtest")==0) {
        // ftprog 已验证模式：自定义流 + 异步轮询，不调用 cudaDeviceSynchronize
        u32* d_out; u32* d_prog;
        CUDA_CHECK(cudaMalloc(&d_out, 8*8*4));
        CUDA_CHECK(cudaMalloc(&d_prog, 4));
        CUDA_CHECK(cudaMemset(d_prog, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK));
        CUDA_CHECK(cudaStreamCreate(&sP));
        ftprog_kernel<<<1,1,0,sK>>>(d_out, d_prog);
        for (int t=0;t<30;t++) {
            u32 p=0;
            cudaMemcpyAsync(&p, d_prog, 4, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            printf("t=%d prog=%u\n", t, p);
            fflush(stdout);
            if (p>=9) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
        u32 h_out[ 64];
        cudaError_t merr = cudaMemcpy(h_out, d_out, 8*8*4, cudaMemcpyDeviceToHost);
        printf("sync-copy err=%s\n", cudaGetErrorString(merr));
        const char* names[ 8] = {
            "(p-1)*(p-1)      ", "(p-2)*(p-1)      ", "2^255 * 2^255     ",
            "aaaa.. * aaaa..  ", "1234567890abcdef^2", "0xffff...+0x100000001",
            "inv(a)*a          ", "(n-1)+2 mod n     "
        };
        for (int t=0;t<8;t++) {
            printf("%s = ", names[t]);
            for (int i=7;i>=0;i--) printf("%08x", h_out[t*8+i]);
            printf("\n");
        }
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_prog));
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
    }
    else if (strcmp(mode,"ftprog")==0) {
        u32* d_out; u32* d_prog;
        CUDA_CHECK(cudaMalloc(&d_out, 8*8*4));
        CUDA_CHECK(cudaMalloc(&d_prog, 4));
        CUDA_CHECK(cudaMemset(d_prog, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK));
        CUDA_CHECK(cudaStreamCreate(&sP));
        ftprog_kernel<<<1,1,0,sK>>>(d_out, d_prog);
        for (int t=0;t<30;t++) {
            u32 p=0;
            cudaMemcpyAsync(&p, d_prog, 4, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            printf("t=%d prog=%u\n", t, p);
            fflush(stdout);
            if (p>=9) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_prog));
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
    }
    else if (strcmp(mode,"bisect")==0) {
        int c = (argc>2)?atoi(argv[ 2]):0;
        u32* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, 8*4));
        bisect_kernel<<<1,1>>>(d_out, c);
        CUDA_CHECK(cudaGetLastError());
        u32 h_out[ 8];
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 8*4, cudaMemcpyDeviceToHost));
        printf("case %d = ", c);
        for (int i=7;i>=0;i--) printf("%08x", h_out[i]);
        printf("\n");
        CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"debug")==0) {
        u32* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, 44*4));
        debug_kernel<<<1,1>>>(d_out);
        CUDA_CHECK(cudaGetLastError());
        u32 h_out[ 44];
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 44*4, cudaMemcpyDeviceToHost));
        printf("debug (p-1)^2 intermediates:\n");
        printf("  raw product t[0..15]:\n    ");
        for (int i=0;i<16;i++) printf("%08x ", h_out[i]);
        printf("\n  v after initial fold:\n    ");
        for (int i=0;i<10;i++) printf("%08x ", h_out[16+i]);
        printf("\n  v after fe_fold_hi:\n    ");
        for (int i=0;i<10;i++) printf("%08x ", h_out[26+i]);
        printf("\n  fe_mul r:\n    ");
        for (int i=0;i<8;i++) printf("%08x ", h_out[36+i]);
        printf("\n");
        CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"jdbl")==0) {
        u32* d_done; CUDA_CHECK(cudaMalloc(&d_done, 4));
        u32* d_out; CUDA_CHECK(cudaMalloc(&d_out, 256));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        jdbl_kernel<<<1,1,0,sK>>>(d_out, d_done);
        for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
        u32 hb[ 64];
        cudaMemcpyAsync(hb, d_out, 256, cudaMemcpyDeviceToHost, sP);
        cudaStreamSynchronize(sP);
        const char* names[ 10] = { "A=X^2", "B=Y^2", "C=B^2", "D=2t", "E=3A", "F0=E^2", "t0=2D", "X3", "Y3", "Z3" };
        int o=0;
        for (int q=0;q<10;q++){ printf("%s: ",names[q]); for(int i=7;i>=0;i--) printf("%08x",hb[o++]); printf("\n"); }
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        CUDA_CHECK(cudaFree(d_done)); CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"sm")==0) {
        u32* d_done; CUDA_CHECK(cudaMalloc(&d_done, 4));
        u32* d_out; CUDA_CHECK(cudaMalloc(&d_out, 128));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        for (int cc=0;cc<6;cc++) {
            CUDA_CHECK(cudaMemset(d_done, 0, 4));
            sm_kernel<<<1,1,0,sK>>>(d_out, d_done, cc);
            for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
            u32 hb[ 24];
            cudaMemcpyAsync(hb, d_out, 96, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            const char* nm[ 6] = { "k0h", "2^255", "k0h_top32", "0xe9", "2^247", "10G_jac" };
            if (cc==1 || cc==5) {
                printf("%s X: ", nm[cc]); for(int i=7;i>=0;i--) printf("%08x",hb[i]); printf("\n");
                printf("%s Y: ", nm[cc]); for(int i=15;i>=8;i--) printf("%08x",hb[i]); printf("\n");
                printf("%s Z: ", nm[cc]); for(int i=23;i>=16;i--) printf("%08x",hb[i]); printf("\n");
            } else {
                printf("%s x: ", nm[cc]); for(int i=7;i>=0;i--) printf("%08x",hb[i]); printf("\n");
                printf("%s y: ", nm[cc]); for(int i=15;i>=8;i--) printf("%08x",hb[i]); printf("\n");
            }
        }
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        CUDA_CHECK(cudaFree(d_done)); CUDA_CHECK(cudaFree(d_out));
    }

    else if (strcmp(mode,"ktest")==0) {
        u32* d_done; CUDA_CHECK(cudaMalloc(&d_done, 4));
        u8* d_out; CUDA_CHECK(cudaMalloc(&d_out, 128));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        ktest_kernel<<<1,1,0,sK>>>(d_out, d_done);
        for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
        u8 hb[ 96];
        cudaMemcpyAsync(hb, d_out, 96, cudaMemcpyDeviceToHost, sP);
        cudaStreamSynchronize(sP);
        printf("keccak(""):  "); for(int i=0;i<32;i++) printf("%02x",hb[i]); printf("\n");
        printf("keccak(abc): "); for(int i=0;i<32;i++) printf("%02x",hb[32+i]); printf("\n");
        printf("keccak(pub): "); for(int i=0;i<32;i++) printf("%02x",hb[64+i]); printf("\n");
        printf("f1600(0) st0: "); for(int i=0;i<8;i++) printf("%02x",hb[96+i]); printf("\n");
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        CUDA_CHECK(cudaFree(d_done)); CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"ptest")==0) {
        u32* d_done; CUDA_CHECK(cudaMalloc(&d_done, 4));
        u32* d_out; CUDA_CHECK(cudaMalloc(&d_out, 256));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        ptest_kernel<<<1,1,0,sK>>>(d_out, d_done);
        for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
        u32 hb[ 64];
        cudaMemcpyAsync(hb, d_out, 256, cudaMemcpyDeviceToHost, sP);
        cudaStreamSynchronize(sP);
        int o=0;
        printf("2G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("2G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("3G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("3G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("4G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("4G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("5G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("5G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("7G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("7G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("8G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("8G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        CUDA_CHECK(cudaFree(d_done)); CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"scdiag")==0) {
        u32* d_done; CUDA_CHECK(cudaMalloc(&d_done, 4));
        u32* d_out; CUDA_CHECK(cudaMalloc(&d_out, 512));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        scdiag_kernel<<<1,1,0,sK>>>(d_out, d_done);
        for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
        u32 hb[ 128];
        cudaMemcpyAsync(hb, d_out, 512, cudaMemcpyDeviceToHost, sP);
        cudaStreamSynchronize(sP);
        int o=0;
        printf("il:  "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("kf:  "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("t0-8:"); for(int i=0;i<9;i++) printf("%08x",hb[o++]); printf("\n");
        printf("r:   "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("fe_ge(t,N):%u  t8:%u\n", hb[o], hb[o+1]);
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        CUDA_CHECK(cudaFree(d_done)); CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"vec")==0) {
        const char* ents[ 1] = { "00000000000000000000000000000000" };
        u32* d_done;
        CUDA_CHECK(cudaMalloc(&d_done, 4));
        u8* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, 600));
        for (int t=0;t<1;t++) {
            u8 ent[ 16];
            for (int i=0;i<16;i++){ int a,b; char ch1=ents[t][2*i], ch2=ents[t][2*i+1];
                a=(ch1<='9')?ch1-'0':ch1-'a'+10; b=(ch2<='9')?ch2-'0':ch2-'a'+10; ent[i]=(u8)((a<<4)|b); }
            u8* d_ent;
            CUDA_CHECK(cudaMalloc(&d_ent, 16));
            CUDA_CHECK(cudaMemcpy(d_ent, ent, 16, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_done, 0, 4));
            cudaStream_t sK, sP;
            CUDA_CHECK(cudaStreamCreate(&sK));
            CUDA_CHECK(cudaStreamCreate(&sP));
            vec_kernel<<<1,1,0,sK>>>(d_ent, d_out, d_done);
            for (int w=0;w<50;w++) {
                u32 p=0;
                cudaMemcpyAsync(&p, d_done, 4, cudaMemcpyDeviceToHost, sP);
                cudaStreamSynchronize(sP);
                if (p==1) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
            }
            u8 hb[ 600];
            cudaMemcpyAsync(hb, d_out, 600, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            int o=0;
            printf("entropy: "); for(int i=0;i<16;i++) printf("%02x",hb[o++]); printf("\n");
            printf("seed:    "); for(int i=0;i<64;i++) printf("%02x",hb[o++]); printf("\n");
            printf("mK:      "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("mC:      "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("k44:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("c44:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("k60:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("c60:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("k0h:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("c0h:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("k0n:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("k0n2:    "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("pubX:    "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("pubY:    "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("addr:    "); for(int i=0;i<20;i++) printf("%02x",hb[o++]); printf("\n");
            printf("mnem:    %s\n", (char*)(hb+o));
            cudaStreamDestroy(sK); cudaStreamDestroy(sP);
            CUDA_CHECK(cudaFree(d_ent));
        }
        CUDA_CHECK(cudaFree(d_done));
        CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"test")==0) {
        const char* ents[ 3] = {
            "00000000000000000000000000000000",
            "7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f",
            "80808080808080808080808080808080"
        };
        u32* d_done;
        CUDA_CHECK(cudaMalloc(&d_done, 4));
        for (int t=0;t<3;t++) {
            u8 ent[ 16];
            for (int i=0;i<16;i++){ int a,b; char ch1=ents[t][2*i], ch2=ents[t][2*i+1];
                a=(ch1<='9')?ch1-'0':ch1-'a'+10; b=(ch2<='9')?ch2-'0':ch2-'a'+10; ent[i]=(u8)((a<<4)|b); }
            u8* d_ent;
            CUDA_CHECK(cudaMalloc(&d_ent, 16));
            CUDA_CHECK(cudaMemcpy(d_ent, ent, 16, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_done, 0, 4));
            cudaStream_t sK, sP;
            CUDA_CHECK(cudaStreamCreate(&sK));
            CUDA_CHECK(cudaStreamCreate(&sP));
            test_kernel<<<1,1,0,sK>>>(d_ent, d_mnem, d_addr, d_priv, d_done);
            for (int w=0;w<40;w++) {
                u32 p=0;
                cudaMemcpyAsync(&p, d_done, 4, cudaMemcpyDeviceToHost, sP);
                cudaStreamSynchronize(sP);
                if (p==1) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
            }
            char mnem[ 128]; u8 addr[ 20], privk[ 32];
            cudaMemcpyAsync(mnem, d_mnem, 128, cudaMemcpyDeviceToHost, sP);
            cudaMemcpyAsync(addr, d_addr, 20, cudaMemcpyDeviceToHost, sP);
            cudaMemcpyAsync(privk, d_priv, 32, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            printf("entropy %s:\n", ents[t]);
            printf("  mnemonic: %s\n", mnem);
            printf("  addr:     0x"); print_hex(addr,20); printf("\n");
            printf("  priv:     0x"); print_hex(privk,32); printf("\n");
            cudaStreamDestroy(sK); cudaStreamDestroy(sP);
            CUDA_CHECK(cudaFree(d_ent));
        }
        CUDA_CHECK(cudaFree(d_done));
    }

    else if (strcmp(mode,"prof")==0) {
        int stage = (argc>=3) ? atoi(argv[2]) : 3;
        int iters = (argc>=4) ? atoi(argv[3]) : 1;
        int blocks = (argc>=5) ? atoi(argv[4]) : 1024;
        u64* d_out; u32* d_done;
        CUDA_CHECK(cudaMalloc(&d_out, 1024*8));
        CUDA_CHECK(cudaMalloc(&d_done, 4));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        cudaEvent_t t0,t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        cudaEventRecord(t0);
        prof_kernel<<<blocks, 256, 0, sK>>>(0, iters, stage, d_out, d_done);
        CUDA_CHECK(cudaGetLastError());
        for (int w=0;w<100;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
        cudaEventRecord(t1, sK);
        cudaEventSynchronize(t1);
        float ms=0; cudaEventElapsedTime(&ms, t0, t1);
        u64 total=(u64)iters*blocks*256;
        printf("prof stage=%d: %llu attempts in %.2f s = %.1f addr/s\n", stage, (unsigned long long)total, ms/1000.0, total/(ms/1000.0));
        CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_done));
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    else if (strcmp(mode,"hd")==0) {
        if (plen==0) { fprintf(stderr,"usage: hd <prefix> [iters] [blocks]\n"); return 1; }
        int iters = (argc>=4) ? atoi(argv[3]) : 1;
        int blocks = (argc>=5) ? atoi(argv[4]) : 1024;
        u32* d_kbase; u32* d_cbase; u32* d_pubx; u32* d_puby;
        CUDA_CHECK(cudaMalloc(&d_kbase, 32)); CUDA_CHECK(cudaMalloc(&d_cbase, 32));
        CUDA_CHECK(cudaMalloc(&d_pubx, 32)); CUDA_CHECK(cudaMalloc(&d_puby, 32));
        CUDA_CHECK(cudaMemset(d_found, 0, 4));
        u32* d_index; CUDA_CHECK(cudaMalloc(&d_index, 4));
        u64 base_nonce = (argc>=6) ? strtoull(argv[5],0,0) : (u64)time(NULL)*2654435761u;
        hd_setup_kernel<<<1,1>>>(base_nonce, d_kbase, d_cbase, d_pubx, d_puby, d_mnem);
        CUDA_CHECK(cudaGetLastError());
        printf("hd mode: mnemonic sequence comes from the secret file + base_nonce (not printed since v2)\n");
        cudaEvent_t t0,t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        printf("searching HD m/44'/60'/0'/0/N for prefix 0x");
        for (int i=0;i<plen;i++) printf("%x", prefix_nib[i]);
        printf(" (%d hex digits), %d iters x %d blocks x 256 threads...\n", plen, iters, blocks);
        CUDA_CHECK(cudaEventRecord(t0));
        hd_vanity_kernel<<<blocks, 256>>>(d_prefix, plen, d_kbase, d_cbase, d_pubx, d_puby, 0, iters, d_mnem, d_addr, d_priv, d_index, d_found);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        u32 found=0; CUDA_CHECK(cudaMemcpy(&found, d_found, 4, cudaMemcpyDeviceToHost));
        u64 total=(u64)iters*blocks*256;
        printf("time: %.2f s, attempts: %llu, rate: %.1f addr/s\n", ms/1000.0, (unsigned long long)total, total/(ms/1000.0));
        char mnem[ 128]; u8 addr[ 20], privk[ 32];
        CUDA_CHECK(cudaMemcpy(mnem, d_mnem, 128, cudaMemcpyDeviceToHost));
        if (found) {
            CUDA_CHECK(cudaMemcpy(addr, d_addr, 20, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(privk, d_priv, 32, cudaMemcpyDeviceToHost));
            u32 fidx=0; CUDA_CHECK(cudaMemcpy(&fidx, d_index, 4, cudaMemcpyDeviceToHost));
            printf("FOUND!\n");
            printf("  mnemonic: %s\n", mnem);
            printf("  path:     m/44'/60'/0'/0/%u\n", fidx);
            printf("  address:  0x"); print_hex(addr,20); printf("\n");
            printf("  private:  0x"); print_hex(privk,32); printf("\n");
        } else {
            printf("not found in this batch (increase iters)\n");
        }
        CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
        CUDA_CHECK(cudaFree(d_kbase)); CUDA_CHECK(cudaFree(d_cbase));
        CUDA_CHECK(cudaFree(d_pubx)); CUDA_CHECK(cudaFree(d_puby));
        CUDA_CHECK(cudaFree(d_index));
    }

    else if (strcmp(mode,"hddiag")==0) {
        u8* d_out; u32* d_done;
        CUDA_CHECK(cudaMalloc(&d_out, 352));
        CUDA_CHECK(cudaMalloc(&d_done, 4));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        u64 base_nonce = (argc>=3) ? strtoull(argv[2],0,0) : (u64)time(NULL)*2654435761u;
        hd_diag_kernel<<<1,1>>>(base_nonce, d_out, d_done);
        CUDA_CHECK(cudaGetLastError());
        for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,0); cudaStreamSynchronize(0); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
        u8 hb[ 352];
        CUDA_CHECK(cudaMemcpy(hb, d_out, 352, cudaMemcpyDeviceToHost));
        printf("mnem: "); for (int i=0;i<128;i++) printf("%c", (char)hb[i]); printf("\n");
        printf("k0h : "); for (int i=0;i<32;i++) printf("%02x", hb[192+i]); printf("\n");
        printf("c0h : "); for (int i=0;i<32;i++) printf("%02x", hb[224+i]); printf("\n");
        printf("k0n : "); for (int i=0;i<32;i++) printf("%02x", hb[256+i]); printf("\n");
        printf("c0n : "); for (int i=0;i<32;i++) printf("%02x", hb[288+i]); printf("\n");
        printf("pubx: "); for (int i=0;i<32;i++) printf("%02x", hb[320+i]); printf("\n");
        CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_done));
    }

    else if (strcmp(mode,"hddiag2")==0) {
        u32 N = (argc>=3) ? (u32)strtoul(argv[2],0,0) : 0;
        u8* d_out; u32* d_done;
        CUDA_CHECK(cudaMalloc(&d_out, 200));
        CUDA_CHECK(cudaMalloc(&d_done, 4));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        u64 base_nonce = 12345u; // 固定，可复现
        hd_diag2_kernel<<<1,1>>>(base_nonce, N, d_out, d_done);
        CUDA_CHECK(cudaGetLastError());
        for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,0); cudaStreamSynchronize(0); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
        u8 hb[ 200];
        CUDA_CHECK(cudaMemcpy(hb, d_out, 200, cudaMemcpyDeviceToHost));
        printf("mnem: "); for (int i=0;i<128;i++) printf("%c", (char)hb[i]); printf("\n");
        printf("kchild: "); for (int i=0;i<32;i++) printf("%02x", hb[128+i]); printf("\n");
        printf("addr:  0x"); for (int i=0;i<20;i++) printf("%02x", hb[160+i]); printf("\n");
        CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_done));
    }
    else if (strcmp(mode,"file")==0) {
        if (argc<3) { fprintf(stderr,"usage: file <prefix_file> [nidx] [blocks]\n"); return 1; }
        const char* fn = argv[2];
        FILE* fp = fopen(fn, "r");
        if (!fp) { fprintf(stderr,"cannot open %s\n", fn); return 1; }
        u64 cap = 1<<16, nprefix=0, nskip=0;
        u8* pfx = (u8*)malloc((size_t)(cap*40));
        u8* plen = (u8*)malloc((size_t)cap);
        char line[ 1024 ];
        while (fgets(line, sizeof(line), fp)) {
            int a=0; while (line[a]==' '||line[a]=='\t'||line[a]=='\r'||line[a]=='\n') a++;
            int b=0; while (line[a+b]!='\0' && line[a+b]!='\n' && line[a+b]!='\r' && line[a+b]!='#' && line[a+b]!=' ' && line[a+b]!='\t') b++;
            if (b==0) continue;
            const char* p = line + a;
            int off = (p[0]=='0' && (p[1]=='x'||p[1]=='X')) ? 2 : 0;
            int len = 0;
            while (len < NL && off+len < b) {
                char ch = p[off+len];
                int hv = -1;
                if (ch>='0'&&ch<='9') hv=ch-'0';
                else if (ch>='a'&&ch<='f') hv=ch-'a'+10;
                else if (ch>='A'&&ch<='F') hv=ch-'A'+10;
                if (hv<0) break;
                pfx[nprefix*40+len]=(u8)hv;
                len++;
            }
            if (len==0) continue;
            int rr=off+len; int rest=0;
            while (rr<b && line[rr]!=' '&&line[rr]!='\t'&&line[rr]!='\n'&&line[rr]!='\r'&&line[rr]!='#') { rest++; rr++; }
            if (len < NL && rest>0) { nskip++; continue; }
            if (len == NL && rest>0) { nskip++; continue; }
            plen[nprefix]=(u8)len;
            nprefix++;
            if (nprefix >= cap) { cap*=2; pfx=(u8*)realloc(pfx,(size_t)(cap*40)); plen=(u8*)realloc(plen,(size_t)cap); }
        }
        fclose(fp);
        if (nprefix==0) { fprintf(stderr,"no valid hex prefixes in file\n"); return 1; }
        printf("file mode: parsed %llu hex prefixes (skipped %llu bad lines)\n",
               (unsigned long long)nprefix, (unsigned long long)nskip);
        int lcnt[ NL+1 ];
        for (int i=0;i<=NL;i++) lcnt[i]=0;
        for (u64 p=0;p<nprefix;p++) lcnt[plen[p]]++;
        printf("length dist: ");
        for (int L=1;L<=NL;L++) if (lcnt[L]) printf("%d-hex:%d ", L, lcnt[L]);
        printf("\n");
        u32 off[ NL ]; u32 mask[ NL ];
        u64 total_cap = 0;
        for (int L=1;L<=NL;L++) {
            int cnt = lcnt[L];
            u64 c2 = 8;
            while (c2 < (u64)cnt*2+1) c2 <<= 1;
            off[L-1]=(u32)total_cap; mask[L-1]=(u32)(c2-1);
            total_cap += c2;
        }
        if (total_cap >= (1ull<<32)) { fprintf(stderr,"hash table too large (>4G slots)\n"); return 1; }
        // 直接在大数组上构建哈希表；去重时把唯一前缀就地压缩到 pfx[0..uniq)
        u64* ck=(u64*)malloc((size_t)(total_cap*8));
        u32* cp=(u32*)malloc((size_t)(total_cap*4));
        u8* co=(u8*)calloc((size_t)total_cap,1);
        u64 uniq=0;
        for (u64 p=0;p<nprefix;p++) {
            int L=plen[p];
            u64 key=0xcbf29ce484222325ull;
            for (int i=0;i<L;i++) key = (key ^ pfx[p*40+i]) * 0x100000001b3ull;
            u32 hh = (u32)((key * 0x9E3779B97F4A7C15ull) >> 32);
            u32 base = off[L-1]; u32 ms = mask[L-1];
            int s = (int)(hh & ms);
            for (;;) {
                if (!co[base+s]) {
                    co[base+s]=1; ck[base+s]=key; cp[base+s]=(u32)uniq;
                    if (uniq < p) { memmove(pfx+uniq*40, pfx+p*40, 40); plen[uniq]=plen[p]; }
                    uniq++; break;
                }
                if (ck[base+s]==key) {
                    u32 q = cp[base+s]; int same=1;
                    for (int i=0;i<L;i++) if (pfx[q*40+i]!=pfx[p*40+i]) { same=0; break; }
                    if (same) break;
                }
                s = (s+1) & ms;
            }
        }
        printf("unique prefixes after dedup: %llu (table %.2f GB, pfx %.2f GB)\n",
               (unsigned long long)uniq, (double)total_cap*13/1073741824.0, (double)uniq*40/1073741824.0);
        if (uniq >= (1ull<<31)) { fprintf(stderr,"too many unique prefixes (>=2^31)\n"); return 1; }
        u64* d_tbl_key; u32* d_tbl_p; u8* d_tbl_occ;
        u32* d_tbl_off; u32* d_tbl_mask; u32* d_tbl_cnt;
        CUDA_CHECK(cudaMalloc(&d_tbl_key, (size_t)(total_cap*8)));
        CUDA_CHECK(cudaMalloc(&d_tbl_p, (size_t)(total_cap*4)));
        CUDA_CHECK(cudaMalloc(&d_tbl_occ, (size_t)total_cap));
        CUDA_CHECK(cudaMalloc(&d_tbl_off, NL*4));
        CUDA_CHECK(cudaMalloc(&d_tbl_mask, NL*4));
        CUDA_CHECK(cudaMalloc(&d_tbl_cnt, NL*4));
        CUDA_CHECK(cudaMemcpy(d_tbl_key, ck, (size_t)(total_cap*8), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tbl_p, cp, (size_t)(total_cap*4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tbl_occ, co, (size_t)total_cap, cudaMemcpyHostToDevice));
        u32 tcnt[ NL ];
        for (int L=1;L<=NL;L++) tcnt[L-1]=0;
        for (u64 pi=0; pi<uniq; pi++) tcnt[plen[pi]-1]++;  // v2: 改用去重后的计数（原版用去重前 lcnt，语义不对；内核只作非零判断故旧版结果未受影响）
        CUDA_CHECK(cudaMemcpy(d_tbl_off, off, NL*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tbl_mask, mask, NL*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tbl_cnt, tcnt, NL*4, cudaMemcpyHostToDevice));
        u8* d_pfx;
        CUDA_CHECK(cudaMalloc(&d_pfx, (size_t)(uniq*40)));
        CUDA_CHECK(cudaMemcpy(d_pfx, pfx, (size_t)(uniq*40), cudaMemcpyHostToDevice));
        free(ck); free(cp); free(co);
        u32* d_found; u32* d_nfound;
        ResqEntry* d_resq; u32* d_resq_cnt;
        CUDA_CHECK(cudaMalloc(&d_found, (size_t)(uniq*4)));
        CUDA_CHECK(cudaMalloc(&d_nfound, 4));
        CUDA_CHECK(cudaMalloc(&d_resq, (size_t)(RESQ_CAP*sizeof(ResqEntry))));
        CUDA_CHECK(cudaMalloc(&d_resq_cnt, 4));
        CUDA_CHECK(cudaMemset(d_found, 0, (size_t)(uniq*4)));
        CUDA_CHECK(cudaMemset(d_nfound, 0, 4));
        CUDA_CHECK(cudaMemset(d_resq_cnt, 0, 4));
        int nidx = (argc>=4) ? atoi(argv[3]) : 1;
        int blocks = (argc>=5) ? atoi(argv[4]) : 1024;
        if (nidx<1) nidx=1;
        if (nidx>1000000) nidx=1000000;
        // 命中即写文件：eth_hits.txt 追加，无缓冲实时落盘（Ctrl+C 强杀不丢）
        FILE* hf = fopen("eth_hits.txt", "a");
        if (hf) { setvbuf(hf, NULL, _IONBF, 0);
#ifndef _WIN32
            chmod("eth_hits.txt", 0600);  // v2: 助记词+私钥明文文件，权限收紧
#endif
        }
        // 每次启动随机起点：两次运行助记词序列不同
        static u64 eth_run_seed = 0;
        u64 start = (u64)std::chrono::high_resolution_clock::now().time_since_epoch().count()
                  ^ (++eth_run_seed * 0x9E3779B97F4A7C15ull);
        int batch_iters=1; u64 total_mnem=0; double total_s=0;
        u64 qprev=0; u64 printed=0; u64 total_hits=0;
        cudaEvent_t t0,t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        printf("searching random mnemonic m/44'/60'/0'/0/0..%d (nidx=%d), %d blocks x 256 threads... Ctrl+C to stop\n",
               nidx-1, nidx, blocks);
        // 无限运行：全部命中后不退出；每轮重置去重表 -> 同一前缀可持续重复命中并持续记录，仅 Ctrl+C 停止
        while (1) {
            int launched = batch_iters;  // v2: 本轮实际启动的批量；统计与区间推进只准用它（batch_iters 会在轮末被自适应调整）
            CUDA_CHECK(cudaMemset(d_found, 0, (size_t)(uniq*4)));
            CUDA_CHECK(cudaEventRecord(t0));
            multi_hash_kernel<<<blocks, 256>>>(d_tbl_key, d_tbl_p, d_tbl_occ,
                                               d_tbl_off, d_tbl_mask, d_tbl_cnt, d_pfx, uniq,
                                               nidx, start, batch_iters,
                                               d_found, d_nfound, d_resq, d_resq_cnt);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(t1));
            CUDA_CHECK(cudaEventSynchronize(t1));
            float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
            total_mnem += (u64)launched*blocks*256;
            total_s += ms/1000.0;
            /* merge v3: 固定 batch=1，速率按实测单批计算 */
            /* merge v3: 固定 batch=1 */
            u32 qcnt=0; CUDA_CHECK(cudaMemcpy(&qcnt, d_resq_cnt, 4, cudaMemcpyDeviceToHost));
            while (qprev < (u64)qcnt) {
                ResqEntry e;
                CUDA_CHECK(cudaMemcpy(&e, d_resq + (qprev & RESQ_MASK), sizeof(ResqEntry), cudaMemcpyDeviceToHost));
                qprev++;
                printf("HIT [%4u] 0x", e.p);
                for (int i=0;i<plen[e.p];i++) printf("%x", pfx[e.p*40+i]);
                printf(" -> m/44'/60'/0'/0/%u\n      mnemonic: %s\n      address:  0x", e.idx, e.mnem);
                for (int i=0;i<20;i++) printf("%02x", e.addr[i]);
                printf("\n      private:  0x");
                for (int i=0;i<32;i++) printf("%02x", e.priv[i]);
                printf("\n");
                if (hf) {
                    fprintf(hf, "HIT [%4u] 0x", e.p);
                    for (int i=0;i<plen[e.p];i++) fprintf(hf, "%x", pfx[e.p*40+i]);
                    fprintf(hf, " -> m/44'/60'/0'/0/%u\n", e.idx);
                    fprintf(hf, "      mnemonic: %s\n", e.mnem);
                    fprintf(hf, "      address:  0x");
                    for (int i=0;i<20;i++) fprintf(hf, "%02x", e.addr[i]);
                    fprintf(hf, "\n      private:  0x");
                    for (int i=0;i<32;i++) fprintf(hf, "%02x", e.priv[i]);
                    fprintf(hf, "\n\n");
                }
                printed++; total_hits++;
            }
            double as = (double)total_mnem*nidx/total_s;
            printf("\r  [%7.1f s] %10.0f mnem/s | %13.0f addr/s | hit %llu  (Ctrl+C stop)", total_s, total_mnem/total_s, as, (unsigned long long)total_hits);
            fflush(stdout);
            start += (u64)launched*blocks*256;
        }
        printf("\n(stopped by Ctrl+C) ran %.1f s, total hits %llu\n", total_s, (unsigned long long)total_hits);
        CUDA_CHECK(cudaFree(d_tbl_key)); CUDA_CHECK(cudaFree(d_tbl_p)); CUDA_CHECK(cudaFree(d_tbl_occ));
        CUDA_CHECK(cudaFree(d_tbl_off)); CUDA_CHECK(cudaFree(d_tbl_mask)); CUDA_CHECK(cudaFree(d_tbl_cnt));
        CUDA_CHECK(cudaFree(d_found)); CUDA_CHECK(cudaFree(d_nfound));
        CUDA_CHECK(cudaFree(d_resq)); CUDA_CHECK(cudaFree(d_resq_cnt));
        CUDA_CHECK(cudaFree(d_pfx));
        CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
        if (hf) fclose(hf);
        free(pfx); free(plen);
    }
    else if (strcmp(mode,"vanity")==0) {
        FILE* hf = fopen("eth_hits.txt", "a");
        if (hf) { setvbuf(hf, NULL, _IONBF, 0);
#ifndef _WIN32
            chmod("eth_hits.txt", 0600);  // v2: 助记词+私钥明文文件，权限收紧
#endif
        }
        static u64 eth_van_seed = 0;
        u64 start = (u64)std::chrono::high_resolution_clock::now().time_since_epoch().count()
                  ^ (++eth_van_seed * 0x9E3779B97F4A7C15ull);
        int nidx = (argc>=4) ? atoi(argv[3]) : 1;
        int blocks = (argc>=5) ? atoi(argv[4]) : 1024;
        if (nidx<1) nidx=1;
        if (nidx>1000000) nidx=1000000;
        int threads = 256;
        CUDA_CHECK(cudaMemset(d_found, 0, 4));
        CUDA_CHECK(cudaMemset(d_index, 0, 4));
        cudaEvent_t t0,t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        printf("searching random mnemonic m/44'/60'/0'/0/0..%d (nidx=%d), %d blocks x %d threads... Ctrl+C to stop\n",
               nidx-1, nidx, blocks, threads);
        u64 total_mnem=0; double total_s=0; u32 found=0; int batch_iters=1; u64 total_hits=0;
        // 无限运行：命中后不退出，持续搜索并持续记录，仅 Ctrl+C 停止
        while (1) {
            int launched = batch_iters;  // v2: 同 file 模式，区间推进只用本轮实际启动值
            CUDA_CHECK(cudaMemset(d_found, 0, 4));
            found = 0;
            CUDA_CHECK(cudaEventRecord(t0));
            vanity_kernel<<<blocks, threads>>>(d_prefix, plen, start, batch_iters, nidx, d_mnem, d_addr, d_priv, d_index, d_found);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(t1));
            CUDA_CHECK(cudaEventSynchronize(t1));
            float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
            CUDA_CHECK(cudaMemcpy(&found, d_found, 4, cudaMemcpyDeviceToHost));
            total_mnem += (u64)launched*blocks*threads;
            total_s += ms/1000.0;
            /* merge v3: 固定 batch=1，速率按实测单批计算 */
            /* merge v3: 固定 batch=1 */
            if (found) {
                char mnem[ 128]; u8 addr[ 20], privk[ 32]; u32 idx;
                CUDA_CHECK(cudaMemcpy(mnem, d_mnem, 128, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(addr, d_addr, 20, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(privk, d_priv, 32, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&idx, d_index, 4, cudaMemcpyDeviceToHost));
                printf("  mnemonic: %s\n", mnem);
                printf("  path:     m/44'/60'/0'/0/%u\n", idx);
                printf("  address:  0x"); print_hex(addr,20); printf("\n");
                printf("  private:  0x"); print_hex(privk,32); printf("\n");
                if (hf) {
                    fprintf(hf, "  mnemonic: %s\n", mnem);
                    fprintf(hf, "  path:     m/44'/60'/0'/0/%u\n", idx);
                    fprintf(hf, "  address:  0x");
                    for (int i=0;i<20;i++) fprintf(hf, "%02x", addr[i]);
                    fprintf(hf, "\n  private:  0x");
                    for (int i=0;i<32;i++) fprintf(hf, "%02x", privk[i]);
                    fprintf(hf, "\n\n");
                }
                total_hits++;
            }
            double as = (double)total_mnem*nidx/total_s;
            printf("\r  [%7.1f s] %10.0f mnem/s | %13.0f addr/s | hit %llu  (Ctrl+C stop)", total_s, total_mnem/total_s, as, (unsigned long long)total_hits);
            fflush(stdout);
            start += (u64)launched*blocks*threads;
        }
        printf("\n(stopped by Ctrl+C) ran %.1f s, total hits %llu\n", total_s, (unsigned long long)total_hits);
        CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
        if (hf) fclose(hf);
    }
    else { // bench
        u8 prefix0[ 40]; for (int i=0;i<40;i++) prefix0[i]=0;
        CUDA_CHECK(cudaMemcpy(d_prefix, prefix0, 40, cudaMemcpyHostToDevice));
        int iters = (argc>=3) ? atoi(argv[2]) : 200000;
        int blocks = (argc>=4) ? atoi(argv[3]) : 1024;
        int threads = 256;
        CUDA_CHECK(cudaMemset(d_found, 0, 4));
        cudaEvent_t t0,t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        int pl=40;
        CUDA_CHECK(cudaEventRecord(t0));
        vanity_kernel<<<blocks, threads>>>(d_prefix, pl, 0, iters, 1, d_mnem, d_addr, d_priv, d_index, d_found);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        u64 total=(u64)iters*blocks*threads;
        printf("bench: %llu attempts in %.2f s = %.1f addr/s\n",
               (unsigned long long)total, ms/1000.0, total/(ms/1000.0));
        CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    }

    CUDA_CHECK(cudaFree(d_prefix)); CUDA_CHECK(cudaFree(d_mnem));
    CUDA_CHECK(cudaFree(d_addr)); CUDA_CHECK(cudaFree(d_priv)); CUDA_CHECK(cudaFree(d_found));
    CUDA_CHECK(cudaFree(d_index));
    return 0;
}

} // namespace eth
#undef CUDA_CHECK
#undef MAXP
#undef NL
#undef RESQ_CAP
#undef RESQ_MASK

// ============================ chain: btc ============================
#define CUDA_CHECK(call) do { cudaError_t err = call; if(err != cudaSuccess) { fprintf(stderr, "CUDA Err: %s\n", cudaGetErrorString(err)); exit(err); } } while(0)
#define MAXP 256
#define PSTR 64
#define NLMAX 62
#define RESQ_CAP (1<<20)
#define RESQ_MASK (RESQ_CAP-1)
namespace btc {
// btc_vanity_bip39.cu  (v2, 2026-11-02 修订版，基于 2_jnqs)
// BIP39 mnemonic -> PBKDF2 -> BIP32/44 m/44'/0'/0'/0/N -> 真 secp256k1 -> BTC 地址
// 支持 P2PKH(1) / P2SH-P2WPKH(3) / P2WPKH(bc1q) / P2TR(bc1p, x-only) 前缀搜索（真链路）
//
// v2 改动清单（详见随附说明）：
//  1) 修复 P2SH 地址生成 bug：type=1 原版只换版本字节（0x05||HASH160(公钥)），不是 P2SH-P2WPKH、无法正常花费；v2 改为 0x05||HASH160(0x0014||HASH160(公钥))
//  2) 修复 run_btc_search 区间推进 bug：start 只按本轮实际启动的 launched 推进，不再用自适应调整后的 batch_iters（原版会跳过或重复搜索区间）
//  3) 修复 selftest/fieldtest 跑完后穿透再跑一遍 bench（两条 if 链合并）；comb 预计算后加 cudaDeviceSynchronize
//  4) 安全：计数器熵改为 SHA256(secret(32B) || counter)，secret 由主机端真随机生成/加载（.btc_vanity_secret，0600，绝不打印）
//  5) 新增 btc_hits.txt 落盘（无缓冲追加 + 0600）：原版命中只打到标准输出，终端关闭/强杀即丢
//  6) 新增不可能前缀预检：违反地址类型约束（如 bc1q/bc1p 开头、base58/bech32 字符集）的前缀永远命中不了，原版会让搜索永远不退出
//  7) 性能：热路径 CKD 的 HMAC 改为预计算 ipad/opad 复用（cbase 固定时只算一次），SHA512 压缩 4->2 次/地址
//  8) 规范/健壮：ckd_priv 处理 IL>=n（置 0 永不命中）；sha256 通用填充修复 len>=56 的情况；tcnt 改用去重后计数
// 未包含（需在目标 GPU 上实测后再动）：批量求逆、fe_mul/fe_sqr 重写、comb 表放共享内存；P2TR 仍为原始 x-only（非 BIP86 tweak），见说明
// 字段算术：8x32 位小端 limb，p = 2^256 - 2^32 - 977


typedef uint32_t u32;
typedef uint64_t u64;
typedef uint8_t u8;
typedef uint16_t u16;
typedef u32 fe[ 8];



// ==========================================
// secp256k1 常数（LE 32 位 limb）
// ==========================================
__constant__ u32 P_FE[ 8] = {0xFFFFFC2Fu,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
__constant__ u32 N_FE[ 8] = {0xD0364141u,0xBFD25E8Cu,0xAF48A03Bu,0xBAAEDCE6u,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
__constant__ u32 C2P256MN[ 8] = {0x2FC9BEBFu,0x402DA173u,0x50B75FC4u,0x45512319u,0x00000001u,0,0,0};
__constant__ u32 GX_FE[ 8] = {0x16F81798u,0x59F2815Bu,0x2DCE28D9u,0x029BFCDBu,0xCE870B07u,0x55A06295u,0xF9DCBBACu,0x79BE667Eu};
__constant__ u32 GY_FE[ 8] = {0xFB10D4B8u,0x9C47D08Fu,0xA6855419u,0xFD17B448u,0x0E1108A8u,0x5DA4FBFCu,0x26A3C465u,0x483ADA77u};
__constant__ u8 SECRET_K[ 32];  // v2: 主机端真随机 secret，由 main() 拷入；与计数器混合决定助记词序列

// ==========================================
// 字段算术（8x32 位，p = 2^256 - 2^32 - 977）
// ==========================================
__device__ __forceinline__ void fe_cpy(u32 r[ 8], const u32 a[ 8]) { for (int i=0;i<8;i++) r[i]=a[i]; }
__device__ __forceinline__ void fe_set0(u32 r[ 8]) { for (int i=0;i<8;i++) r[i]=0; }
__device__ __forceinline__ void fe_set1(u32 r[ 8]) { fe_set0(r); r[ 0]=1; }
__device__ __forceinline__ int fe_iszero(const u32 a[ 8]) { int z=1; for (int i=0;i<8;i++) z &= (a[i]==0); return z; }
__device__ __forceinline__ int fe_isone(const u32 a[ 8]) { int z=(a[ 0]==1); for (int i=1;i<8;i++) z &= (a[i]==0); return z; }
__device__ __forceinline__ int fe_ge(const u32 a[ 8], const u32 b[ 8]) {
    for (int i=7;i>=0;i--) { if (a[i]!=b[i]) return a[i]>b[i]; }
    return 1;
}
__device__ __forceinline__ void fe_sub(u32 r[ 8], const u32 a[ 8], const u32 b[ 8]) {
    u64 bo=0;
    for (int i=0;i<8;i++){ u64 s=(u64)a[i]-b[i]-bo; r[i]=(u32)s; bo=(s>>32)?1:0; }
    if (bo) {
        u64 c=0;
        for (int i=0;i<8;i++){ u64 s=(u64)r[i]+P_FE[i]+c; r[i]=(u32)s; c=s>>32; }
    }
}
// 折叠 10-limb v 的高位 v[8],v[9]（2^256 ≡ 2^32+977），至多 3 次
__device__ __forceinline__ void fe_fold_hi(u32 v[ 10]) {
    for (int it=0; it<3; it++) {
        if ((v[ 8] | v[ 9]) == 0) return;
        u32 h0 = v[ 8], h1 = v[ 9];
        v[ 8] = 0; v[ 9] = 0;
        u64 c = 0;
        for (int i=0;i<10;i++){
            u64 s = c + (i<8 ? (u64)v[i] : 0);
            if (i>=1) {
                if (i-1==0) s += (u64)h0;
                else if (i-1==1) s += (u64)h1;
            }
            if (i==0) s += (u64)h0 * 977u;
            if (i==1) s += (u64)h1 * 977u;
            v[i] = (u32)s;
            c = s >> 32;
        }
        if (c) { u64 s = (u64)v[ 9] + c; v[ 9] = (u32)s; }
    }
}
// r = t mod p，t 为 9 limbs（< 2^257）
__device__ __forceinline__ void fe_reduce9(u32 r[ 8], const u32 t[ 9]) {
    u32 v[ 10];
    u64 c=0;
    for (int i=0;i<10;i++){
        u64 s = c + (i<8 ? (u64)t[i] : 0);
        if (i==1) s += (u64)t[ 8];
        if (i==0) s += (u64)t[ 8]*977u;
        v[i]=(u32)s; c=s>>32;
    }
    if (c){ u64 s=(u64)v[ 9]+c; v[ 9]=(u32)s; }
    fe_fold_hi(v);
    for (int i=0;i<8;i++) r[i]=v[i];
    if (fe_ge(r, P_FE)) fe_sub(r, r, P_FE);
}
__device__ __forceinline__ void fe_add(u32 r[ 8], const u32 a[ 8], const u32 b[ 8]) {
    u64 c=0; u32 t[ 9];
    for (int i=0;i<8;i++){ u64 s=(u64)a[i]+b[i]+c; t[i]=(u32)s; c=s>>32; }
    t[ 8]=(u32)c;
    fe_reduce9(r, t);
}
// r = a * b mod p（32 位 limb 学校式 + 进位链，无 __int128）
__device__ __noinline__ void fe_mul(u32 r[ 8], const u32 a[ 8], const u32 b[ 8]) {
    u32 t[ 16];
    for (int i=0;i<16;i++) t[i]=0;
    for (int i=0;i<8;i++)
        for (int j=0;j<8;j++) {
            u64 p = (u64)a[i]*b[j];
            u64 s = (u64)t[i+j] + (u32)p;
            t[i+j] = (u32)s;
            u64 c = s >> 32;
            int k = i+j+1;
            while (c && k < 16) { u64 s2=(u64)t[k]+c; t[k]=(u32)s2; c=s2>>32; k++; }
            s = (u64)t[i+j+1] + (u32)(p>>32);
            t[i+j+1] = (u32)s;
            c = s >> 32;
            k = i+j+2;
            while (c && k < 16) { u64 s2=(u64)t[k]+c; t[k]=(u32)s2; c=s2>>32; k++; }
        }
    u32 v[ 10];
    u64 c=0;
    for (int i=0;i<10;i++){
        u64 s = c + (i<8 ? (u64)t[i] : 0);
        if (i>=1 && (i-1)<8) s += (u64)t[ 8+(i-1)];
        if (i<8) s += (u64)t[ 8+i]*977u;
        v[i]=(u32)s; c=s>>32;
    }
    if (c){ u64 s=(u64)v[ 9]+c; v[ 9]=(u32)s; }
    fe_fold_hi(v);
    for (int i=0;i<8;i++) r[i]=v[i];
    if (fe_ge(r, P_FE)) fe_sub(r, r, P_FE);
}
__device__ __forceinline__ void fe_sqr(u32 r[ 8], const u32 a[ 8]) { fe_mul(r, a, a); }
// 费马求逆 a^(p-2)
__device__ __forceinline__ void fe_inv(u32 r[ 8], const u32 a[ 8]) {
    u32 e[ 8] = {0xFFFFFC2Du,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
    u32 res[ 8]; fe_set1(res);
    u32 base[ 8]; fe_cpy(base, a);
    for (int i=255;i>=0;i--){
        fe_sqr(res, res);
        if ((e[i>>5] >> (i&31)) & 1) fe_mul(res, res, base);
    }
    fe_cpy(r, res);
}
// 标量加法 mod n（BIP32 CKD：k_new = (IL + k) mod n）
__device__ __forceinline__ void sc_add_mod_n(u32 r[ 8], const u32 a[ 8], const u32 b[ 8]) {
    u64 c=0; u32 t[ 9];
    for (int i=0;i<8;i++){ u64 s=(u64)a[i]+b[i]+c; t[i]=(u32)s; c=s>>32; }
    t[ 8]=(u32)c;
    if (t[ 8]) {
        u64 cc=0;
        for (int i=0;i<8;i++){ u64 s=(u64)t[i]+C2P256MN[i]+cc; r[i]=(u32)s; cc=s>>32; }
    } else if (fe_ge(t, N_FE)) {
        u64 bo=0;
        for (int i=0;i<8;i++){ u64 s=(u64)t[i]-N_FE[i]-bo; r[i]=(u32)s; bo=(s>>32)?1:0; }
    } else {
        for (int i=0;i<8;i++) r[i]=t[i];
    }
}

// ==========================================
// 雅可比坐标点运算
// ==========================================
struct Jacobian { fe X, Y, Z; };
__device__ __forceinline__ void jac_set_inf(Jacobian& P) { fe_set0(P.X); fe_set0(P.Y); fe_set0(P.Z); }
__device__ __forceinline__ int jac_is_inf(const Jacobian& P) { return fe_iszero(P.Z); }

__device__ __forceinline__ void jac_double(Jacobian& R, const Jacobian& P) {
    if (jac_is_inf(P)) { jac_set_inf(R); return; }
    fe A, B, C, D, E, F, t, t2;
    fe_sqr(A, P.X);
    fe_sqr(B, P.Y);
    fe_sqr(C, B);
    fe_add(t, P.X, B); fe_sqr(t, t);
    fe_sub(t, t, A); fe_sub(t, t, C);
    fe_add(D, t, t);
    fe_add(E, A, A); fe_add(E, E, A);
    fe_sqr(F, E);
    fe_add(t, D, D);
    fe_sub(t, F, t);
    fe_sub(t2, D, t);
    fe_mul(t2, t2, E);
    fe_add(F, C, C); fe_add(F, F, F); fe_add(F, F, F);
    fe_sub(t2, t2, F);
    fe_mul(F, P.Y, P.Z); fe_add(F, F, F);
    fe_cpy(R.X, t); fe_cpy(R.Y, t2); fe_cpy(R.Z, F);
}

__device__ __forceinline__ void jac_add_affine(Jacobian& R, const Jacobian& P, const u32 qx[ 8], const u32 qy[ 8]) {
    if (jac_is_inf(P)) { fe_cpy(R.X, qx); fe_cpy(R.Y, qy); fe_set1(R.Z); return; }
    fe Z1Z1, U2, S2, H, Rr, HH, HHH, V, X3, Y3, Z3, t, t2;
    fe_sqr(Z1Z1, P.Z);
    fe_mul(U2, qx, Z1Z1);
    fe_mul(t, P.Z, Z1Z1);
    fe_mul(S2, qy, t);
    fe_sub(H, U2, P.X);
    fe_sub(Rr, S2, P.Y);
    if (fe_iszero(H)) {
        if (fe_iszero(Rr)) { jac_double(R, P); return; }
        else { jac_set_inf(R); return; }
    }
    fe_sqr(HH, H);
    fe_mul(HHH, H, HH);
    fe_mul(V, P.X, HH);
    fe_sqr(X3, Rr);
    fe_sub(X3, X3, HHH);
    fe_add(t, V, V);
    fe_sub(X3, X3, t);
    fe_sub(t, V, X3);
    fe_mul(t, t, Rr);
    fe_mul(t2, P.Y, HHH);
    fe_sub(Y3, t, t2);
    fe_mul(Z3, P.Z, H);
    fe_cpy(R.X, X3); fe_cpy(R.Y, Y3); fe_cpy(R.Z, Z3);
}

// k * G（从 MSB 到 LSB 的 double-and-add）
__device__ void scalar_mult_base(Jacobian& R, const u32 k[ 8]) {
    jac_set_inf(R);
    for (int i=255;i>=0;i--){
        jac_double(R, R);
        if ((k[i>>5] >> (i&31)) & 1) jac_add_affine(R, R, GX_FE, GY_FE);
    }
}


__device__ void jac_to_affine(u32 x[ 8], u32 y[ 8], const Jacobian& P) {
    fe zi, zi2, zi3;
    fe_inv(zi, P.Z);
    fe_sqr(zi2, zi);
    fe_mul(zi3, zi2, zi);
    fe_mul(x, P.X, zi2);
    fe_mul(y, P.Y, zi3);
}
__device__ const u32* g_comb;

// ==========================================
// 4-tooth comb 固定基点标量乘
// T[t][j] = j * 2^(64t) * G, t=0..3, j=1..15 （仿射坐标，16 u32/点）
// 256 次 double -> 64 次 double + ~64 次 affine add
// ==========================================
__global__ void precompute_comb_kernel(u32* d_table) {
    int t = threadIdx.x;      // 0..3
    int j = blockIdx.x + 1;   // 1..15
    if (t > 3) return;
    u32 kf[ 8] = {0,0,0,0,0,0,0,0};
    // k = j * 2^(64t)：j 放入 limb 2t（j<16 不溢出）
    kf[ 2*t] = (u32)j;
    Jacobian P;
    jac_set_inf(P);
    for (int i=255;i>=0;i--){
        jac_double(P, P);
        if ((kf[i>>5]>>(i&31))&1) jac_add_affine(P, P, GX_FE, GY_FE);
    }
    u32 x[ 8], y[ 8];
    jac_to_affine(x, y, P);
    u32* dst = d_table + ((t*15 + (j-1)) * 16);
    for (int i=0;i<8;i++){ dst[i]=x[i]; dst[8+i]=y[i]; }
}

__device__ __forceinline__ void scalar_mult_base_comb(Jacobian& R, const u32 k[ 8]) {
    jac_set_inf(R);
    const u32* T = g_comb;
    for (int r=15; r>=0; r--) {
        jac_double(R,R); jac_double(R,R); jac_double(R,R); jac_double(R,R);
        for (int t=0;t<4;t++) {
            int shift = 64*t + 4*r;
            int d = (k[shift>>5] >> (shift&31)) & 0xF;
            if (d) {
                const u32* p = T + ((t*15 + (d-1)) * 16);
                jac_add_affine(R, R, p, p+8);
            }
        }
    }
}


// ==========================================
// 字节 / limb 转换
// ==========================================
__device__ __forceinline__ void bytes32_to_fe(u32 f[ 8], const u8 b[ 32]) {
    for (int i=0;i<8;i++) f[i] = ((u32)b[28-4*i]<<24)|((u32)b[29-4*i]<<16)|((u32)b[30-4*i]<<8)|b[31-4*i];
}
__device__ __forceinline__ void fe_to_bytes32(u8 b[ 32], const u32 f[ 8]) {
    for (int i=0;i<8;i++){ u32 v=f[i]; b[28-4*i]=(u8)(v>>24); b[29-4*i]=(u8)(v>>16); b[30-4*i]=(u8)(v>>8); b[31-4*i]=(u8)v; }
}
__device__ __forceinline__ int device_strlen(const char* s) { int l=0; while (s[l]) l++; return l; }

// ==========================================
// SHA-256（BIP39 校验和 / vanity 熵）
// ==========================================
__constant__ u32 K256[ 64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
__device__ __forceinline__ u32 rotr32(u32 x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ void sha256_block(u32 h[ 8], const u8 blk[ 64]) {
    u32 w[ 64];
    for (int i=0;i<16;i++) w[i] = ((u32)blk[i*4]<<24)|((u32)blk[i*4+1]<<16)|((u32)blk[i*4+2]<<8)|blk[i*4+3];
    for (int i=16;i<64;i++){
        u32 s0 = rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);
        u32 s1 = rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    u32 a=h[ 0],b=h[ 1],c=h[ 2],d=h[ 3],e=h[ 4],f=h[ 5],g=h[ 6],hh=h[ 7];
    for (int i=0;i<64;i++){
        u32 S1 = rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
        u32 ch = (e&f)^((~e)&g);
        u32 t1 = hh + S1 + ch + K256[i] + w[i];
        u32 S0 = rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
        u32 maj = (a&b)^(a&c)^(b&c);
        u32 t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[ 0]+=a; h[ 1]+=b; h[ 2]+=c; h[ 3]+=d; h[ 4]+=e; h[ 5]+=f; h[ 6]+=g; h[ 7]+=hh;
}
__device__ void sha256(const u8* data, int len, u8 out[ 32]) {
    u32 h[ 8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    u8 blk[ 64];
    int off = 0;
    while (len - off >= 64) { for (int i=0;i<64;i++) blk[i]=data[off+i]; sha256_block(h, blk); off += 64; }
    for (int i=0;i<64;i++) blk[i]=0;
    int rem = len - off;
    for (int i=0;i<rem;i++) blk[i]=data[off+i];
    blk[rem] = 0x80;
    u64 bits = (u64)len * 8;
    if (rem >= 56) {  // v2: 剩余 >=56 字节时长度字段放不进同一块，先冲掉这一块再补长度块（原版此情况算错；旧调用 len<=40 未触发）
        sha256_block(h, blk);
        for (int i=0;i<64;i++) blk[i]=0;
    }
    for (int i=0;i<8;i++) blk[56+i] = (u8)((bits >> (56-8*i)) & 0xFF);
    sha256_block(h, blk);
    for (int i=0;i<8;i++){ out[i*4]=(u8)(h[i]>>24); out[i*4+1]=(u8)(h[i]>>16); out[i*4+2]=(u8)(h[i]>>8); out[i*4+3]=(u8)h[i]; }
}

// v2: 计数器 -> 熵。ent = SHA256(SECRET_K(32B) || counter 小端 8B) 的前 16 字节。
// 没有 secret 的人无法复现/枚举这条助记词序列（原版仅 SHA256(计数器)，等效熵只有计数器本身且顺序可猜）。
__device__ __forceinline__ void counter_to_entropy(u64 n, u8 ent[ 16]) {
    u8 buf[ 40];
    for (int i=0;i<32;i++) buf[i]=SECRET_K[i];
    for (int i=0;i<8;i++) buf[32+i]=(u8)(n>>(8*i));
    u8 h[ 32]; sha256(buf, 40, h);
    for (int i=0;i<16;i++) ent[i]=h[i];
}

// ==========================================
// SHA-512 / HMAC-SHA512（移植自优化版）
// ==========================================
__device__ __constant__ u64 sha512_k[ 80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};
__device__ __forceinline__ u64 rotr64(u64 x, int n) { return (x >> n) | (x << (64 - n)); }
__device__ __forceinline__ u64 Ch(u64 x, u64 y, u64 z) { return (x & y) ^ (~x & z); }
__device__ __forceinline__ u64 Maj(u64 x, u64 y, u64 z) { return (x & y) ^ (x & z) ^ (y & z); }
__device__ __forceinline__ u64 Sigma0(u64 x) { return rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39); }
__device__ __forceinline__ u64 Sigma1(u64 x) { return rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41); }
__device__ __forceinline__ u64 sigma0(u64 x) { return rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7); }
__device__ __forceinline__ u64 sigma1(u64 x) { return rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6); }
__device__ __forceinline__ void sha512_init(u64 st[ 8]) {
    st[ 0]=0x6a09e667f3bcc908ULL; st[ 1]=0xbb67ae8584caa73bULL;
    st[ 2]=0x3c6ef372fe94f82bULL; st[ 3]=0xa54ff53a5f1d36f1ULL;
    st[ 4]=0x510e527fade682d1ULL; st[ 5]=0x9b05688c2b3e6c1fULL;
    st[ 6]=0x1f83d9abfb41bd6bULL; st[ 7]=0x5be0cd19137e2179ULL;
}
__device__ __forceinline__ void sha512_block_words(u64 st[ 8], const u64 blockW[ 16]) {
    u64 W[ 80];
    #pragma unroll
    for (int i = 0; i < 16; i++) W[i] = blockW[i];
    #pragma unroll
    for (int i = 16; i < 80; i++) W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16];
    u64 a=st[ 0], b=st[ 1], c=st[ 2], d=st[ 3], e=st[ 4], f=st[ 5], g=st[ 6], h=st[ 7];
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        u64 T1 = h + Sigma1(e) + Ch(e,f,g) + sha512_k[i] + W[i];
        u64 T2 = Sigma0(a) + Maj(a,b,c);
        h=g; g=f; f=e; e=d + T1; d=c; c=b; b=a; a=T1 + T2;
    }
    st[ 0]+=a; st[ 1]+=b; st[ 2]+=c; st[ 3]+=d; st[ 4]+=e; st[ 5]+=f; st[ 6]+=g; st[ 7]+=h;
}
__device__ __forceinline__ void load_be_words(const u8 blk[ 128], u64 W[ 16]) {
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        u64 v = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) v = (v << 8) | blk[i*8 + j];
        W[i] = v;
    }
}
__device__ __forceinline__ void hmac_sha512_precompute(const u8* key, int key_len,
                                                       u64 ipad_st[ 8], u64 opad_st[ 8]) {
    u8 real_key[ 128] = {0};
    u8 kblk[ 128];
    if (key_len > 128) {
        u64 st[ 8]; sha512_init(st);
        int off = 0;
        while (off + 128 <= key_len) {
            u64 W[ 16]; load_be_words(key + off, W);
            sha512_block_words(st, W); off += 128;
        }
        u8 pad[ 128] = {0};
        int rem = key_len - off;
        for (int i = 0; i < rem; i++) pad[i] = key[off + i];
        pad[rem] = 0x80;
        u64 bits = (u64)key_len * 8;
        for (int i = 0; i < 8; i++) pad[ 120 + i] = (bits >> (56 - 8*i)) & 0xFF;
        u64 W2[ 16]; load_be_words(pad, W2);
        sha512_block_words(st, W2);
        for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) real_key[i*8+j] = (st[i] >> (56 - 8*j)) & 0xFF;
    } else {
        for (int i = 0; i < key_len; i++) real_key[i] = key[i];
    }
    u64 W[ 16];
    for (int i = 0; i < 128; i++) kblk[i] = real_key[i] ^ 0x36;
    load_be_words(kblk, W);
    sha512_init(ipad_st); sha512_block_words(ipad_st, W);
    for (int i = 0; i < 128; i++) kblk[i] = real_key[i] ^ 0x5c;
    load_be_words(kblk, W);
    sha512_init(opad_st); sha512_block_words(opad_st, W);
}
// 通用 HMAC-SHA512（msg_len <= 64，key_len <= 128）
__device__ __forceinline__ void hmac_sha512_gen(const u8* key, int key_len, const u8* msg, int msg_len, u8 out[ 64]) {
    u64 ipad_st[ 8], opad_st[ 8];
    hmac_sha512_precompute(key, key_len, ipad_st, opad_st);
    u8 blk[ 128];
    for (int i=0;i<128;i++) blk[i]=0;
    for (int i=0;i<msg_len && i<128;i++) blk[i]=msg[i];
    blk[msg_len]=0x80;
    u64 bits=(u64)(128+msg_len)*8;
    for (int i=0;i<8;i++) blk[120+i]=(u8)((bits>>(56-8*i))&0xFF);
    u64 W[ 16], st[ 8], ih[ 8];
    load_be_words(blk, W);
    for (int i=0;i<8;i++) st[i]=ipad_st[i];
    sha512_block_words(st, W);
    for (int i=0;i<8;i++) ih[i]=st[i];
    for (int i=0;i<128;i++) blk[i]=0;
    for (int i=0;i<64;i++) blk[i]=(u8)((ih[i>>3]>>(56-8*(i&7)))&0xFF);
    blk[64]=0x80;
    bits=(128+64)*8;
    for (int i=0;i<8;i++) blk[120+i]=(u8)((bits>>(56-8*i))&0xFF);
    load_be_words(blk, W);
    for (int i=0;i<8;i++) st[i]=opad_st[i];
    sha512_block_words(st, W);
    for (int i=0;i<8;i++) for (int j=0;j<8;j++) out[i*8+j]=(u8)((st[i]>>(56-8*j))&0xFF);
}
// v2: 用预计算好的 ipad/opad 状态做 HMAC-SHA512（要求 msg_len<=119），省去每次调用重算密钥填充的 2 次压缩。
// 热路径（CKD 的链码固定）应先 hmac_sha512_precompute 一次，再反复调本函数。
__device__ __forceinline__ void hmac_sha512_precomp_msg(const u64 ipad_st[ 8], const u64 opad_st[ 8], const u8* msg, int msg_len, u8 out[ 64]) {
    u8 blk[ 128];
    for (int i=0;i<128;i++) blk[i]=0;
    for (int i=0;i<msg_len;i++) blk[i]=msg[i];
    blk[msg_len]=0x80;
    u64 bits=(u64)(128+msg_len)*8;
    for (int i=0;i<8;i++) blk[120+i]=(u8)((bits>>(56-8*i))&0xFF);
    u64 W[ 16], st[ 8], ih[ 8];
    load_be_words(blk, W);
    for (int i=0;i<8;i++) st[i]=ipad_st[i];
    sha512_block_words(st, W);
    for (int i=0;i<8;i++) ih[i]=st[i];
    for (int i=0;i<128;i++) blk[i]=0;
    for (int i=0;i<64;i++) blk[i]=(u8)((ih[i>>3]>>(56-8*(i&7)))&0xFF);
    blk[64]=0x80;
    bits=(128+64)*8;
    for (int i=0;i<8;i++) blk[120+i]=(u8)((bits>>(56-8*i))&0xFF);
    load_be_words(blk, W);
    for (int i=0;i<8;i++) st[i]=opad_st[i];
    sha512_block_words(st, W);
    for (int i=0;i<8;i++) for (int j=0;j<8;j++) out[i*8+j]=(u8)((st[i]>>(56-8*j))&0xFF);
}
// PBKDF2 首次迭代：salt = "mnemonic" + 0x00000001
__device__ __forceinline__ void hmac_sha512_salt(const u64 ipad_st[ 8], const u64 opad_st[ 8], u64 out[ 8]) {
    const u64 saltW[ 16] = {
        0x6D6E656D6F6E6963ULL, 0x0000000180000000ULL,
        0,0,0,0,0,0,0,0,0,0,0,0, 0, 0x0000000000000460ULL
    };
    u64 st[ 8], W[ 16], ih[ 8];
    for (int i = 0; i < 8; i++) st[i] = ipad_st[i];
    for (int i = 0; i < 16; i++) W[i] = saltW[i];
    sha512_block_words(st, W);
    for (int i = 0; i < 8; i++) ih[i] = st[i];
    for (int i = 0; i < 8; i++) st[i] = opad_st[i];
    for (int i = 0; i < 8; i++) W[i] = ih[i];
    W[ 8]  = 0x8000000000000000ULL;
    W[ 9]  = 0; W[ 10] = 0; W[ 11] = 0; W[ 12] = 0; W[ 13] = 0; W[ 14] = 0;
    W[ 15] = 0x0000000000000600ULL;
    sha512_block_words(st, W);
    for (int i = 0; i < 8; i++) out[i] = st[i];
}
// HMAC 一步：消息 = 完整 64B 块 msg
__device__ __forceinline__ void hmac_sha512_64(const u64 ipad_st[ 8], const u64 opad_st[ 8],
                                               const u64 msg[ 8], u64 out[ 8]) {
    u64 st[ 8], W[ 16], ih[ 8];
    for (int i = 0; i < 8; i++) st[i] = ipad_st[i];
    for (int i = 0; i < 8; i++) W[i] = msg[i];
    W[ 8]  = 0x8000000000000000ULL;
    W[ 9]  = 0; W[ 10] = 0; W[ 11] = 0; W[ 12] = 0; W[ 13] = 0; W[ 14] = 0;
    W[ 15] = 0x0000000000000600ULL;
    sha512_block_words(st, W);
    for (int i = 0; i < 8; i++) ih[i] = st[i];
    for (int i = 0; i < 8; i++) st[i] = opad_st[i];
    for (int i = 0; i < 8; i++) W[i] = ih[i];
    W[ 8]  = 0x8000000000000000ULL;
    W[ 9]  = 0; W[ 10] = 0; W[ 11] = 0; W[ 12] = 0; W[ 13] = 0; W[ 14] = 0;
    W[ 15] = 0x0000000000000600ULL;
    sha512_block_words(st, W);
    for (int i = 0; i < 8; i++) out[i] = st[i];
}
// 完整 BIP39 Seed 派生（2048 次 PBKDF2-HMAC-SHA512）
__device__ void device_bip39_seed(const char* mnemonic, u8 seed[ 64]) {
    int pass_len = device_strlen(mnemonic);
    u64 ipad_st[ 8], opad_st[ 8];
    hmac_sha512_precompute((const u8*)mnemonic, pass_len, ipad_st, opad_st);
    u64 U[ 8], T[ 8];
    hmac_sha512_salt(ipad_st, opad_st, U);
    for (int i = 0; i < 8; i++) T[i] = U[i];
    for (int i = 1; i < 2048; i++) {
        hmac_sha512_64(ipad_st, opad_st, U, U);
        for (int j = 0; j < 8; j++) T[j] ^= U[j];
    }
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) seed[i*8 + j] = (T[i] >> (56 - 8*j)) & 0xFF;
}

// ==========================================
// Keccak-256（地址）
// ==========================================
__device__ __constant__ u64 keccak_rc[ 24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};
__device__ __forceinline__ u64 rotl64b(u64 x, int n) { n &= 63; return n ? ((x << n) | (x >> (64 - n))) : x; }
__device__ void keccak_f1600(u64 state[ 25]) {
    u64 C[ 5], D[ 5], B[ 25];
    for (int round=0; round<24; round++) {
        for (int i=0;i<5;i++) C[i]=state[i]^state[i+5]^state[i+10]^state[i+15]^state[i+20];
        for (int i=0;i<5;i++) { D[i]=C[(i+4)%5]^rotl64b(C[(i+1)%5],1); for (int j=0;j<25;j+=5) state[j+i]^=D[i]; }
        for (int i=0;i<25;i++) B[i]=state[i];
        int x=1, y=0; u64 cur=B[ 1];
        for (int t=0;t<24;t++) {
            int tmp=x; x=y; y=(2*tmp+3*y)%5;
            u64 tv=B[x+5*y]; B[x+5*y]=rotl64b(cur,(t+1)*(t+2)/2); cur=tv;
        }
        for (int i=1;i<25;i++) state[i]=B[i];
        for (int j=0;j<25;j+=5) {
            for (int i=0;i<5;i++) B[j+i]=state[j+i];
            for (int i=0;i<5;i++) state[j+i]^=(~B[j+(i+1)%5])&B[j+(i+2)%5];
        }
        state[ 0]^=keccak_rc[round];
    }
}
__device__ void device_keccak256(const u8* data, int len, u8 hash[ 32]) {
    u64 state[ 25] = {0}; int rate=136; int off=0;
    while (off+rate<=len) {
        for (int i=0;i<rate/8;i++) {
            u64 val=0;
            for (int j=0;j<8;j++) val |= ((u64)data[off+i*8+j])<<(8*j);
            state[i]^=val;
        }
        keccak_f1600(state); off+=rate;
    }
    u8 blk[ 136] = {0}; int rem=len-off;
    if (rem>0) for (int i=0;i<rem;i++) blk[i]=data[off+i];
    blk[rem]=0x01; blk[rate-1]|=0x80;
    for (int i=0;i<rate/8;i++) {
        u64 val=0;
        for (int j=0;j<8;j++) val |= ((u64)blk[i*8+j])<<(8*j);
        state[i]^=val;
    }
    keccak_f1600(state);
    for (int i=0;i<4;i++) for (int j=0;j<8;j++) hash[i*8+j]=(u8)((state[i]>>(8*j))&0xFF);
}

// ==========================================
// BIP39 编码：128bit 熵 + 4bit 校验和 -> 12 词
// ==========================================
__device__ void entropy_to_mnemonic(const u8 ent[ 16], char* mnem, int* mnem_len) {
    u8 h[ 32];
    sha256(ent, 16, h);
    u16 idx[ 12];
    for (int w=0;w<12;w++) {
        int bp = w*11; u32 val=0;
        for (int b=0;b<11;b++) {
            int p=bp+b; int byte=p>>3, sh=7-(p&7);
            int bit = (byte<16) ? ((ent[byte]>>sh)&1) : ((h[ 0]>>sh)&1);
            val=(val<<1)|bit;
        }
        idx[w]=(u16)val;
    }
    int pos=0;
    for (int w=0;w<12;w++) {
        const char* word = BIP39_WORDS[idx[w]];
        int len=0; while (word[len]) len++;
        for (int i=0;i<len;i++) mnem[pos++]=word[i];
        if (w<11) mnem[pos++]=' ';
    }
    mnem[pos]=0;
    if (mnem_len) *mnem_len=pos;
}

// ==========================================
// BIP32 / BIP44 派生：m/44'/0'/0'/0/0
// ==========================================
__device__ void ckd_priv(u8 k[ 32], u8 c[ 32], u32 index, int hardened) {
    u8 data[ 37], out[ 64];
    if (hardened) {
        data[ 0]=0;
        for (int i=0;i<32;i++) data[1+i]=k[i];
    } else {
        u32 kf[ 8]; bytes32_to_fe(kf, k);
        Jacobian P; scalar_mult_base_comb(P, kf);
        u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
        u8 xb[ 32]; fe_to_bytes32(xb, x);
        data[ 0] = (y[ 0]&1) ? 0x03 : 0x02;
        for (int i=0;i<32;i++) data[1+i]=xb[i];
    }
    data[33]=(u8)(index>>24); data[34]=(u8)(index>>16); data[35]=(u8)(index>>8); data[36]=(u8)index;
    hmac_sha512_gen(c, 32, data, 37, out);
    u32 il[ 8], kf[ 8], kf2[ 8];
    bytes32_to_fe(il, out);
    bytes32_to_fe(kf, k);
    // v2(BIP32 规范): IL>=n 时该索引无效（概率约 2^-127）。置子钥为 0，其点乘为无穷远点、派生地址永不命中前缀，等效丢弃；原版对此未处理、会算出规范外密钥。
    if (fe_ge(il, N_FE)) fe_set0(kf2);
    else sc_add_mod_n(kf2, il, kf);
    fe_to_bytes32(k, kf2);
    for (int i=0;i<32;i++) c[i]=out[32+i];
}
__device__ void bip32_derive(u8 privk[ 32], const u8 seed[ 64]) {
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    ckd_priv(k, c, 0x80000000u+44u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    ckd_priv(k, c, 0u, 0);
    ckd_priv(k, c, 0u, 0);
    for (int i=0;i<32;i++) privk[i]=k[i];
}
__device__ void derive_address(const u8 privk[ 32], u8 addr[ 20]) {
    u32 kf[ 8]; bytes32_to_fe(kf, privk);
    Jacobian P; scalar_mult_base_comb(P, kf);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    u8 pub[ 64];
    fe_to_bytes32(pub, x);
    fe_to_bytes32(pub+32, y);
    u8 h[ 32];
    device_keccak256(pub, 64, h);
    for (int i=0;i<20;i++) addr[i]=h[12+i];
}
__device__ void derive_full(const u8 ent[ 16], char* mnem, u8 addr[ 20], u8 privk[ 32]) {
    char m[ 128];
    entropy_to_mnemonic(ent, m, 0);
    u8 seed[ 64];
    device_bip39_seed(m, seed);
    bip32_derive(privk, seed);
    derive_address(privk, addr);
    if (mnem) { int l=0; while (m[l]){ mnem[l]=m[l]; l++; } mnem[l]=0; }
}

// ==========================================
// Kernel：vec 诊断（逐层输出中间值）
// ==========================================
__global__ void vec_kernel(const u8* d_ent, u8* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 ent[ 16];
    for (int i=0;i<16;i++) ent[i]=d_ent[i];
    char m[ 128];
    entropy_to_mnemonic(ent, m, 0);
    int o=0;
    for (int i=0;i<16;i++) d_out[o++]=ent[i];
    u8 seed[ 64];
    device_bip39_seed(m, seed);
    for (int i=0;i<64;i++) d_out[o++]=seed[i];
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    for (int i=0;i<32;i++) d_out[o++]=c[i];
    ckd_priv(k, c, 0x80000000u+44u, 1);
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    for (int i=0;i<32;i++) d_out[o++]=c[i];
    ckd_priv(k, c, 0x80000000u+0u, 1);
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    for (int i=0;i<32;i++) d_out[o++]=c[i];
    ckd_priv(k, c, 0x80000000u+0u, 1);
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    for (int i=0;i<32;i++) d_out[o++]=c[i];
    ckd_priv(k, c, 0u, 0);
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    ckd_priv(k, c, 0u, 0);
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    u32 kf[ 8]; bytes32_to_fe(kf, k);
    Jacobian P; scalar_mult_base(P, kf);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    u8 xb[ 32], yb[ 32];
    fe_to_bytes32(xb, x); fe_to_bytes32(yb, y);
    for (int i=0;i<32;i++) d_out[o++]=xb[i];
    for (int i=0;i<32;i++) d_out[o++]=yb[i];
    u8 pub[ 64];
    for (int i=0;i<32;i++) pub[i]=xb[i];
    for (int i=0;i<32;i++) pub[32+i]=yb[i];
    u8 h[ 32];
    device_keccak256(pub, 64, h);
    for (int i=0;i<20;i++) d_out[o++]=h[12+i];
    int l=0; while (m[l]){ d_out[o++]=m[l]; l++; }
    d_out[o]=0;
    d_done[ 0]=1;
}

// ==========================================
// Kernel：selftest
// ==========================================
__global__ void selftest_kernel(u64* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu;
    fe b; fe_set1(b);
    fe r; fe_add(r, a, b);
    u64 v0=0; for (int i=7;i>=0;i--) v0=(v0<<32)|r[i];
    d_out[ 0]=v0;
    fe m1; fe_set0(m1); m1[ 0]=123456789u;
    fe m2; fe_set0(m2); m2[ 0]=987654321u;
    fe_mul(r, m1, m2);
    u64 v1=0; for (int i=7;i>=0;i--) v1=(v1<<32)|r[i];
    d_out[ 1]=v1;
    fe two; fe_set0(two); two[ 0]=2u;
    Jacobian P2; scalar_mult_base(P2, two);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P2);
    u64 v2x=0, v2y=0; for (int i=7;i>=0;i--){ v2x=(v2x<<32)|x[i]; v2y=(v2y<<32)|y[i]; }
    d_out[ 2]=v2x; d_out[ 3]=v2y;
    fe one; fe_set1(one);
    Jacobian P1; scalar_mult_base(P1, one);
    u32 x1[ 8], y1[ 8]; jac_to_affine(x1, y1, P1);
    u64 v3x=0, v3y=0; for (int i=7;i>=0;i--){ v3x=(v3x<<32)|x1[i]; v3y=(v3y<<32)|y1[i]; }
    d_out[ 4]=v3x; d_out[ 5]=v3y;
    u32 nm1[ 8]; fe_cpy(nm1, N_FE); nm1[ 0]-=1;
    u32 one32[ 8]; fe_set1(one32);
    u32 sr[ 8]; sc_add_mod_n(sr, nm1, one32);
    u64 v4=0; for (int i=7;i>=0;i--) v4=(v4<<32)|sr[i];
    d_out[ 6]=v4;
    u32 nm2[ 8]; fe_cpy(nm2, N_FE); nm2[ 0]-=2;
    u32 three[ 8]; fe_set0(three); three[ 0]=3;
    sc_add_mod_n(sr, nm2, three);
    u64 v5=0; for (int i=7;i>=0;i--) v5=(v5<<32)|sr[i];
    d_out[ 7]=v5;
    Jacobian P3a; jac_add_affine(P3a, P2, GX_FE, GY_FE);
    u32 x3[ 8], y3[ 8]; jac_to_affine(x3, y3, P3a);
    fe three3; fe_set0(three3); three3[ 0]=3;
    Jacobian P3b; scalar_mult_base(P3b, three3);
    u32 x3b[ 8], y3b[ 8]; jac_to_affine(x3b, y3b, P3b);
    u64 v6=0; for (int i=7;i>=0;i--) v6=(v6<<32)|x3[i];
    u64 v6b=0; for (int i=7;i>=0;i--) v6b=(v6b<<32)|x3b[i];
    d_out[ 8]=v6; d_out[ 9]=v6b;
    fe inv; fe_inv(inv, m1);
    fe_mul(r, m1, inv);
    u64 v7=0; for (int i=7;i>=0;i--) v7=(v7<<32)|r[i];
    d_out[ 10]=v7;
    d_done[ 0]=1;
}

// ==========================================
// Kernel：jac_double(G) 中间量诊断
// ==========================================
__global__ void jdbl_kernel(u32* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    fe A, B, C, D, E, F, t, t2, F0, t0;
    fe_sqr(A, GX_FE);
    fe_sqr(B, GY_FE);
    fe_sqr(C, B);
    fe_add(t, GX_FE, B); fe_sqr(t, t);
    fe_sub(t, t, A); fe_sub(t, t, C);
    fe_add(D, t, t);
    fe_add(E, A, A); fe_add(E, E, A);
    fe_sqr(F, E);
    fe_cpy(F0, F);
    fe_add(t, D, D);
    fe_cpy(t0, t);
    fe_sub(t, F0, t0);
    fe_sub(t2, D, t);
    fe_mul(t2, t2, E);
    fe_add(F, C, C); fe_add(F, F, F); fe_add(F, F, F);
    fe_sub(t2, t2, F);
    fe_mul(F, GY_FE, GX_FE); fe_add(F, F, F);
    int o=0;
    for (int s=0;s<8;s++) d_out[o++]=A[s];
    for (int s=0;s<8;s++) d_out[o++]=B[s];
    for (int s=0;s<8;s++) d_out[o++]=C[s];
    for (int s=0;s<8;s++) d_out[o++]=D[s];
    for (int s=0;s<8;s++) d_out[o++]=E[s];
    for (int s=0;s<8;s++) d_out[o++]=F0[s];  // E^2
    for (int s=0;s<8;s++) d_out[o++]=t0[s];  // 2D
    for (int s=0;s<8;s++) d_out[o++]=t[s];   // X3
    for (int s=0;s<8;s++) d_out[o++]=t2[s];  // Y3
    for (int s=0;s<8;s++) d_out[o++]=F[s];   // Z3
    d_done[ 0]=1;
}

// ==========================================
// Kernel：最小化 scalar_mult_base(k0h) 诊断
// ==========================================

// ==========================================
// Kernel: keccak test
// ==========================================
__global__ void ktest_kernel(u8* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 h[ 32];
    device_keccak256((const u8*)"", 0, h);
    for (int i=0;i<32;i++) d_out[i]=h[i];
    device_keccak256((const u8*)"abc", 3, h);
    for (int i=0;i<32;i++) d_out[32+i]=h[i];
    const u8 pub[ 64] = {
        0x37,0xb0,0xbb,0x7a,0x82,0x88,0xd3,0x8e,0xd4,0x9a,0x52,0x4b,0x5d,0xc9,0x8c,0xff,
        0x3e,0xb5,0xca,0x82,0x4c,0x9f,0x9d,0xc0,0xdf,0xdb,0x3d,0x9c,0xd6,0x00,0xf2,0x99,
        0xa6,0x17,0x99,0x12,0xb7,0x45,0x1c,0x09,0x89,0x6c,0x40,0x98,0xec,0xa7,0xce,0x6b,
        0x2e,0x58,0x33,0x06,0x72,0x79,0x5e,0x84,0x7c,0x4d,0x6a,0xf4,0x4e,0x02,0x42,0x30 };
    device_keccak256(pub, 64, h);
    for (int i=0;i<32;i++) d_out[64+i]=h[i];
    u64 st[ 25] = {0};
    keccak_f1600(st);
    for (int i=0;i<8;i++) d_out[96+i]=(u8)((st[ 0]>>(8*i))&0xFF);
    d_done[ 0]=1;
}

__global__ void sm_kernel(u32* d_out, u32* d_done, int c) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u32 kf[ 8] = {0,0,0,0,0,0,0,0};
    if (c==0) { // k0h
        const u8 kh[ 32] = { 0xe9,0xb9,0x1c,0x42,0x8e,0x67,0x9f,0x85,0x7f,0xbf,0xa0,0x20,0x34,0x40,0xea,0xea,
                             0xab,0xb0,0xca,0x46,0xd8,0x0e,0x70,0xb6,0xaa,0x62,0x50,0x69,0x6a,0xc0,0xdf,0xc7 };
        bytes32_to_fe(kf, kh);
    } else if (c==1) { kf[ 7]=0x80000000u; } // 2^255
    else if (c==2) { kf[ 7]=0xe9b91c42u; }  // top 32 bits of k0h
    else if (c==3) { kf[ 0]=0xe9u; }        // 0xe9
    else if (c==4) { kf[ 7]=0x00000080u; }  // 2^247
    else if (c==5) { kf[ 0]=0x0au; }        // 10G
    Jacobian P; scalar_mult_base(P, kf);
    if (c==1 || c==5) {
        for (int i=0;i<8;i++) d_out[i]=P.X[i];
        for (int i=0;i<8;i++) d_out[8+i]=P.Y[i];
        for (int i=0;i<8;i++) d_out[16+i]=P.Z[i];
        d_done[ 0]=1; return;
    }
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    for (int i=0;i<8;i++) d_out[i]=x[i];
    for (int i=0;i<8;i++) d_out[8+i]=y[i];
    d_done[ 0]=1;
}

// ==========================================
// Kernel：ptest 点运算诊断（2G/3G/pubkey(k0h)）
// ==========================================
__global__ void ptest_kernel(u32* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    int o=0;
    fe two; fe_set0(two); two[ 0]=2u;
    Jacobian P2; scalar_mult_base(P2, two);
    u32 x2[ 8], y2[ 8]; jac_to_affine(x2, y2, P2);
    for (int i=7;i>=0;i--) d_out[o++]=x2[i];
    for (int i=7;i>=0;i--) d_out[o++]=y2[i];
    fe three; fe_set0(three); three[ 0]=3u;
    Jacobian P3; scalar_mult_base(P3, three);
    u32 x3[ 8], y3[ 8]; jac_to_affine(x3, y3, P3);
    for (int i=7;i>=0;i--) d_out[o++]=x3[i];
    for (int i=7;i>=0;i--) d_out[o++]=y3[i];
    fe four; fe_set0(four); four[ 0]=4u;
    Jacobian P4; scalar_mult_base(P4, four);
    u32 x4[ 8], y4[ 8]; jac_to_affine(x4, y4, P4);
    for (int i=7;i>=0;i--) d_out[o++]=x4[i];
    for (int i=7;i>=0;i--) d_out[o++]=y4[i];
    fe five; fe_set0(five); five[ 0]=5u;
    Jacobian P5; scalar_mult_base(P5, five);
    u32 x5[ 8], y5[ 8]; jac_to_affine(x5, y5, P5);
    for (int i=7;i>=0;i--) d_out[o++]=x5[i];
    for (int i=7;i>=0;i--) d_out[o++]=y5[i];
    fe seven; fe_set0(seven); seven[ 0]=7u;
    Jacobian P7; scalar_mult_base(P7, seven);
    u32 x7[ 8], y7[ 8]; jac_to_affine(x7, y7, P7);
    for (int i=7;i>=0;i--) d_out[o++]=x7[i];
    for (int i=7;i>=0;i--) d_out[o++]=y7[i];
    fe eight; fe_set0(eight); eight[ 0]=8u;
    Jacobian P8; scalar_mult_base(P8, eight);
    u32 x8[ 8], y8[ 8]; jac_to_affine(x8, y8, P8);
    for (int i=7;i>=0;i--) d_out[o++]=x8[i];
    for (int i=7;i>=0;i--) d_out[o++]=y8[i];
    d_done[ 0]=1;
}

// ==========================================
// Kernel：sc_add_mod_n 诊断（IL + master_k）
// ==========================================
__global__ void scdiag_kernel(u32* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    // IL (b50e42e12a00b9e0abc8d9a1ced0f570252afcf07ed746e16d9d3a12dcc7d883)
    const u8 ilb[ 32] = { 0xb5,0x0e,0x42,0xe1,0x2a,0x00,0xb9,0xe0,0xab,0xc8,0xd9,0xa1,0xce,0xd0,0xf5,0x70,
                          0x25,0x2a,0xfc,0xf0,0x7e,0xd7,0x46,0xe1,0x6d,0x9d,0x3a,0x12,0xdc,0xc7,0xd8,0x83 };
    // master_k (1837c1be8e2995ec11cda2b066151be2cfb48adf9e47b151d46adab3a21cdf67)
    const u8 kb[ 32] = { 0x18,0x37,0xc1,0xbe,0x8e,0x29,0x95,0xec,0x11,0xcd,0xa2,0xb0,0x66,0x15,0x1b,0xe2,
                         0xcf,0xb4,0x8a,0xdf,0x9e,0x47,0xb1,0x51,0xd4,0x6a,0xda,0xb3,0xa2,0x1c,0xdf,0x67 };
    u32 il[ 8], kf[ 8], r[ 8];
    bytes32_to_fe(il, ilb);
    bytes32_to_fe(kf, kb);
    int o=0;
    for (int i=0;i<8;i++) d_out[o++]=il[i];
    for (int i=0;i<8;i++) d_out[o++]=kf[i];
    u64 c=0; u32 t[ 9];
    for (int i=0;i<8;i++){ u64 s=(u64)il[i]+kf[i]+c; t[i]=(u32)s; c=s>>32; }
    t[ 8]=(u32)c;
    for (int i=0;i<9;i++) d_out[o++]=t[i];
    sc_add_mod_n(r, il, kf);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    // fe_ge check result
    d_out[o++]=fe_ge(t, N_FE);
    d_out[o++]=t[ 8];
    d_done[ 0]=1;
}

// ==========================================
// Kernel：字段测试
// ==========================================
__global__ void fieldtest_kernel(u32* d_out) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    int o=0;
    fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu; a[ 0]=0xFFFFFC2Eu; a[ 1]=0xFFFFFFFEu;
    fe r;
    fe_mul(r, a, a);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe b; for (int i=0;i<8;i++) b[i]=0xFFFFFFFFu; b[ 0]=0xFFFFFC2Du; b[ 1]=0xFFFFFFFEu;
    fe_mul(r, b, a);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe c2; fe_set0(c2); c2[ 7]=0x80000000u;
    fe_mul(r, c2, c2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe d2; for (int i=0;i<8;i++) d2[i]=0xAAAAAAAAu;
    fe_mul(r, d2, d2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe e2; e2[ 0]=0x12345678u; e2[ 1]=0x9abcdef0u; e2[ 2]=0x11223344u; e2[ 3]=0x55667788u;
    e2[ 4]=0x99aabbccu; e2[ 5]=0xddeeff00u; e2[ 6]=0x10203040u; e2[ 7]=0x50607080u;
    fe_mul(r, e2, e2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe f1; for (int i=0;i<8;i++) f1[i]=0xFFFFFFFFu;
    fe f2; fe_set0(f2); f2[ 0]=0x00000001u; f2[ 1]=0x00000001u;
    fe_add(r, f1, f2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe g2; g2[ 0]=0x12345678u; g2[ 1]=0x9abcdef0u; g2[ 2]=0x11223344u; g2[ 3]=0x55667788u;
    g2[ 4]=0x99aabbccu; g2[ 5]=0xddeeff00u; g2[ 6]=0x10203040u; g2[ 7]=0x50607080u;
    fe_inv(r, g2);
    fe_mul(r, r, g2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    u32 nm1[ 8]; fe_cpy(nm1, N_FE); nm1[ 0]-=1;
    u32 two[ 8]; fe_set0(two); two[ 0]=2u;
    u32 sr[ 8]; sc_add_mod_n(sr, nm1, two);
    for (int i=0;i<8;i++) d_out[o++]=sr[i];
}

// ==========================================
// Kernel：fieldtest 进度标记版（定位挂起用例）
// ==========================================
__global__ void ftprog_kernel(u32* d_out, u32* d_prog) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    int o=0;
    fe r;
    d_prog[ 0]=1;
    fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu; a[ 0]=0xFFFFFC2Eu; a[ 1]=0xFFFFFFFEu;
    fe_mul(r, a, a);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=2;
    fe b; for (int i=0;i<8;i++) b[i]=0xFFFFFFFFu; b[ 0]=0xFFFFFC2Du; b[ 1]=0xFFFFFFFEu;
    fe_mul(r, b, a);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=3;
    fe c2; fe_set0(c2); c2[ 7]=0x80000000u;
    fe_mul(r, c2, c2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=4;
    fe d2; for (int i=0;i<8;i++) d2[i]=0xAAAAAAAAu;
    fe_mul(r, d2, d2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=5;
    fe e2; e2[ 0]=0x12345678u; e2[ 1]=0x9abcdef0u; e2[ 2]=0x11223344u; e2[ 3]=0x55667788u;
    e2[ 4]=0x99aabbccu; e2[ 5]=0xddeeff00u; e2[ 6]=0x10203040u; e2[ 7]=0x50607080u;
    fe_mul(r, e2, e2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=6;
    fe f1; for (int i=0;i<8;i++) f1[i]=0xFFFFFFFFu;
    fe f2; fe_set0(f2); f2[ 0]=0x00000001u; f2[ 1]=0x00000001u;
    fe_add(r, f1, f2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=7;
    fe g2; g2[ 0]=0x12345678u; g2[ 1]=0x9abcdef0u; g2[ 2]=0x11223344u; g2[ 3]=0x55667788u;
    g2[ 4]=0x99aabbccu; g2[ 5]=0xddeeff00u; g2[ 6]=0x10203040u; g2[ 7]=0x50607080u;
    fe_inv(r, g2);
    fe_mul(r, r, g2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=8;
    u32 nm1[ 8]; fe_cpy(nm1, N_FE); nm1[ 0]-=1;
    u32 two[ 8]; fe_set0(two); two[ 0]=2u;
    u32 sr[ 8]; sc_add_mod_n(sr, nm1, two);
    for (int i=0;i<8;i++) d_out[o++]=sr[i];
    d_prog[ 0]=9;
}

// ==========================================
// Kernel：单用例 bisect（定位 GPU 挂起用例）
// ==========================================
__global__ void bisect_kernel(u32* d_out, int c) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    fe r;
    if (c==0) { fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu; a[ 0]=0xFFFFFC2Eu; a[ 1]=0xFFFFFFFEu; fe_mul(r, a, a); }
    else if (c==1) { fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu; a[ 0]=0xFFFFFC2Eu; a[ 1]=0xFFFFFFFEu;
                     fe b; for (int i=0;i<8;i++) b[i]=0xFFFFFFFFu; b[ 0]=0xFFFFFC2Du; b[ 1]=0xFFFFFFFEu; fe_mul(r, b, a); }
    else if (c==2) { fe c2; fe_set0(c2); c2[ 7]=0x80000000u; fe_mul(r, c2, c2); }
    else if (c==3) { fe d2; for (int i=0;i<8;i++) d2[i]=0xAAAAAAAAu; fe_mul(r, d2, d2); }
    else if (c==4) { fe e2; e2[ 0]=0x12345678u; e2[ 1]=0x9abcdef0u; e2[ 2]=0x11223344u; e2[ 3]=0x55667788u;
                     e2[ 4]=0x99aabbccu; e2[ 5]=0xddeeff00u; e2[ 6]=0x10203040u; e2[ 7]=0x50607080u; fe_mul(r, e2, e2); }
    else if (c==5) { fe f1; for (int i=0;i<8;i++) f1[i]=0xFFFFFFFFu;
                     fe f2; fe_set0(f2); f2[ 0]=0x00000001u; f2[ 1]=0x00000001u; fe_add(r, f1, f2); }
    else if (c==6) { fe g2; g2[ 0]=0x12345678u; g2[ 1]=0x9abcdef0u; g2[ 2]=0x11223344u; g2[ 3]=0x55667788u;
                     g2[ 4]=0x99aabbccu; g2[ 5]=0xddeeff00u; g2[ 6]=0x10203040u; g2[ 7]=0x50607080u;
                     fe_inv(r, g2); fe_mul(r, r, g2); }
    else if (c==7) { u32 nm1[ 8]; fe_cpy(nm1, N_FE); nm1[ 0]-=1;
           u32 two[ 8]; fe_set0(two); two[ 0]=2u;
           sc_add_mod_n(r, nm1, two); }
    else if (c==8) { // fe_add(D,D); D=cbc1950e... -> expect 97832a1c...
           u32 D[ 8] = { 0x69cfc671u, 0xd2076736u, 0xf147c112u, 0x6bcb0bbcu, 0xcb96483du, 0xd0271c6bu, 0x6fec4514u, 0xcbc1950eu };
           fe_add(r, D, D); }
    else if (c==9) { // fe_sub(F, T); F=E^2 mod p, T=2D mod p -> expect X3
           u32 F[ 8] = { 0x87054446u, 0x35e2d2b3u, 0xd92564edu, 0x8e25a078u, 0xb6d66bd0u, 0xc1dfbd14u, 0xfe816c05u, 0x14985620u };
           u32 T[ 8] = { 0xd39f90b3u, 0xa40ece6du, 0xe28f8225u, 0xd7961779u, 0x972c907au, 0xa04e38d7u, 0xdfd88a29u, 0x97832a1cu };
           fe_sub(r, F, T); }
    else { u32 nm1[ 8]; fe_cpy(nm1, N_FE); nm1[ 0]-=1;
           u32 two[ 8]; fe_set0(two); two[ 0]=2u;
           sc_add_mod_n(r, nm1, two); }
    for (int i=0;i<8;i++) d_out[i]=r[i];
}

// ==========================================
// Kernel：debug（(p-1)^2 中间量转储）
// ==========================================
__global__ void debug_kernel(u32* d_out) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    int o=0;
    fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu; a[ 0]=0xFFFFFC2Eu; a[ 1]=0xFFFFFFFEu;
    u32 t[ 16];
    for (int i=0;i<16;i++) t[i]=0;
    for (int i=0;i<8;i++)
        for (int j=0;j<8;j++) {
            u64 p = (u64)a[i]*a[j];
            u64 s = (u64)t[i+j] + (u32)p;
            t[i+j] = (u32)s;
            u64 c = s >> 32;
            int k = i+j+1;
            while (c && k < 16) { u64 s2=(u64)t[k]+c; t[k]=(u32)s2; c=s2>>32; k++; }
            s = (u64)t[i+j+1] + (u32)(p>>32);
            t[i+j+1] = (u32)s;
            c = s >> 32;
            k = i+j+2;
            while (c && k < 16) { u64 s2=(u64)t[k]+c; t[k]=(u32)s2; c=s2>>32; k++; }
        }
    for (int i=0;i<16;i++) d_out[o++]=t[i];       // 0-15: 原始乘积
    u32 v[ 10];
    u64 c=0;
    for (int i=0;i<10;i++){
        u64 s = c + (i<8 ? (u64)t[i] : 0);
        if (i>=1 && (i-1)<8) s += (u64)t[ 8+(i-1)];
        if (i<8) s += (u64)t[ 8+i]*977u;
        v[i]=(u32)s; c=s>>32;
    }
    if (c){ u64 s=(u64)v[ 9]+c; v[ 9]=(u32)s; }
    for (int i=0;i<10;i++) d_out[o++]=v[i];       // 16-25: 初始折叠 v
    fe_fold_hi(v);
    for (int i=0;i<10;i++) d_out[o++]=v[i];       // 26-35: fe_fold_hi 后 v
    fe r; fe_mul(r, a, a);
    for (int i=0;i<8;i++) d_out[o++]=r[i];        // 36-43: fe_mul 结果
}

// ==========================================
// Kernel：固定熵测试（BIP39 标准向量）
// ==========================================
__global__ void test_kernel(const u8* d_ent, char* d_mnem, u8* d_addr, u8* d_priv, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 ent[ 16];
    for (int i=0;i<16;i++) ent[i]=d_ent[i];
    char mnem[ 128];
    u8 addr[ 20], privk[ 32];
    derive_full(ent, mnem, addr, privk);
    int l=0; while (mnem[l]){ d_mnem[l]=mnem[l]; l++; } d_mnem[l]=0;
    for (int i=0;i<20;i++) d_addr[i]=addr[i];
    for (int i=0;i<32;i++) d_priv[i]=privk[i];
    d_done[ 0]=1;
}

// ==========================================
// Kernel：vanity 搜索
// ==========================================


// ==========================================
// HD 快速模式：固定种子，变索引 m/44'/0'/0'/0/N 搜索
// setup: 熵->助记词->seed->派生到 m/44'/0'/0'/0 (k0n,c0n) + pubkey(k0n)
// search: N 变化 -> k_N = IL + k0n (mod n) -> pubkey -> keccak -> 前缀比对
// ==========================================
__global__ void hd_setup_kernel(u64 base_nonce, u32* d_kbase, u32* d_cbase, u32* d_pubx, u32* d_puby, char* d_mnem) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 ent[ 16]; counter_to_entropy(base_nonce, ent);  // v2: secret 混合熵
    char m[ 128]; entropy_to_mnemonic(ent, m, 0);
    int l=0; while(m[l]){ d_mnem[l]=m[l]; l++; } d_mnem[l]=0;
    u8 seed[ 64]; device_bip39_seed(m, seed);
    // master
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    // 44' 0' 0'
    ckd_priv(k, c, 0x80000000u+44u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    // 0 (non-hardened) -> k0n
    u8 k0n[ 32], c0n[ 32];
    for (int i=0;i<32;i++){ k0n[i]=k[i]; c0n[i]=c[i]; }
    ckd_priv(k0n, c0n, 0u, 0);
    // 先写 kbase/cbase，再算 pubkey（避免 comb 潜在污染）
    u32 kb[ 8], cb[ 8];
    bytes32_to_fe(kb, k0n); bytes32_to_fe(cb, c0n);
    for (int i=0;i<8;i++){ d_kbase[i]=kb[i]; d_cbase[i]=cb[i]; }
    // pubkey(k0n)
    u32 kf[ 8]; bytes32_to_fe(kf, k0n);
    Jacobian P; scalar_mult_base_comb(P, kf);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    for (int i=0;i<8;i++){ d_pubx[i]=x[i]; d_puby[i]=y[i]; }
}

__global__ __launch_bounds__(256, 2) void hd_vanity_kernel(const u8* d_prefix, int plen,
        const u32* d_kbase, const u32* d_cbase, const u32* d_pubx, const u32* d_puby,
        u64 start_nonce, int iters, char* d_mnem, u8* d_addr, u8* d_priv, u32* d_index, u32* d_found) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    // v2: 链码与父公钥在整个 kernel 生命周期内固定，HMAC 预计算和字节转换提到循环外只做一次
    u32 cb0[ 8]; for (int i=0;i<8;i++) cb0[i]=d_cbase[i];
    u8 c8[ 32]; fe_to_bytes32(c8, cb0);
    u64 cipad[ 8], copad[ 8];
    hmac_sha512_precompute(c8, 32, cipad, copad);
    u8 data0[ 33];
    data0[ 0] = (d_puby[ 0]&1) ? 0x03 : 0x02;
    u8 xb[ 32]; fe_to_bytes32(xb, d_pubx);
    for (int i=0;i<32;i++) data0[1+i]=xb[i];
    for (int it=0; it<iters; it++) {
        /* v3: 去掉整批早退，保证每批跑满，速率不虚高 */
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u32 N = (u32)(n & 0x7FFFFFFFu);
        // IL = HMAC-SHA512(c0n, 0x02/03||X || N)
        u8 data[ 37];
        for (int i=0;i<33;i++) data[i]=data0[i];
        data[ 33]=(u8)(N>>24); data[ 34]=(u8)(N>>16); data[ 35]=(u8)(N>>8); data[ 36]=(u8)N;
        u8 out[ 64];
        hmac_sha512_precomp_msg(cipad, copad, data, 37, out);
        u32 il[ 8]; bytes32_to_fe(il, out);
        u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=d_kbase[i];
        sc_add_mod_n(kf, il, kf);
        Jacobian P; scalar_mult_base_comb(P, kf);
        u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
        u8 pub[ 64];
        fe_to_bytes32(pub, x);
        fe_to_bytes32(pub+32, y);
        u8 hh[ 32];
        device_keccak256(pub, 64, hh);
        u8 addr[ 20]; for (int i=0;i<20;i++) addr[i]=hh[12+i];
        int ok=1;
        for (int i=0;i<plen;i++) {
            u8 nib = (i&1) ? (addr[i>>1]&0xF) : (addr[i>>1]>>4);
            if (nib != d_prefix[i]) { ok=0; break; }
        }
        if (ok) {
            u32 old = atomicCAS(d_found, 0, 1);
            if (old==0) {
                u8 privk[ 32]; fe_to_bytes32(privk, kf);
                for (int i=0;i<20;i++) d_addr[i]=addr[i];
                for (int i=0;i<32;i++) d_priv[i]=privk[i];
                d_index[ 0]=N;
            }
            return;
        }
    }
}

__global__ void hd_diag_kernel(u64 base_nonce, u8* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 ent[ 16]; counter_to_entropy(base_nonce, ent);  // v2: secret 混合熵
    char m[ 128]; entropy_to_mnemonic(ent, m, 0);
    for (int i=0;i<128;i++) d_out[i]=(u8)m[i]; // mnemonic bytes 0..127
    u8 seed[ 64]; device_bip39_seed(m, seed);
    for (int i=0;i<64;i++) d_out[128+i]=seed[i];
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    ckd_priv(k, c, 0x80000000u+44u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    for (int i=0;i<32;i++) d_out[192+i]=k[i];  // k0h
    for (int i=0;i<32;i++) d_out[224+i]=c[i];  // c0h
    u8 k0n[ 32], c0n[ 32];
    for (int i=0;i<32;i++){ k0n[i]=k[i]; c0n[i]=c[i]; }
    ckd_priv(k0n, c0n, 0u, 0);
    for (int i=0;i<32;i++) d_out[256+i]=k0n[i]; // k0n
    for (int i=0;i<32;i++) d_out[288+i]=c0n[i]; // c0n
    u32 kf[ 8]; bytes32_to_fe(kf, k0n);
    Jacobian P; scalar_mult_base_comb(P, kf);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    u8 xb[ 32]; fe_to_bytes32(xb, x);
    for (int i=0;i<32;i++) d_out[320+i]=xb[i]; // pubx
    d_done[ 0]=1;
}

__global__ void hd_diag2_kernel(u64 base_nonce, u32 N, u8* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 ent[ 16]; counter_to_entropy(base_nonce, ent);  // v2: secret 混合熵
    char m[ 128]; entropy_to_mnemonic(ent, m, 0);
    for (int i=0;i<128;i++) d_out[i]=(u8)m[i];
    u8 seed[ 64]; device_bip39_seed(m, seed);
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    ckd_priv(k, c, 0x80000000u+44u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    u8 k0n[ 32], c0n[ 32];
    for (int i=0;i<32;i++){ k0n[i]=k[i]; c0n[i]=c[i]; }
    ckd_priv(k0n, c0n, 0u, 0);
    // ===== 完全复用 hd_vanity_kernel 的 search 计算路径 (N=0) =====
    u32 kf0[ 8]; bytes32_to_fe(kf0, k0n);
    Jacobian P0; scalar_mult_base_comb(P0, kf0);
    u32 px[ 8], py[ 8]; jac_to_affine(px, py, P0);
    u32 cb[ 8]; bytes32_to_fe(cb, c0n);
    u8 c8[ 32]; fe_to_bytes32(c8, cb);
    u8 data[ 37];
    data[ 0] = (py[ 0]&1) ? 0x03 : 0x02;
    u8 xb[ 32]; fe_to_bytes32(xb, px);
    for (int i=0;i<32;i++) data[1+i]=xb[i];
    data[33]=(u8)(N>>24); data[34]=(u8)(N>>16); data[35]=(u8)(N>>8); data[36]=(u8)N;
    u8 o2[ 64];
    hmac_sha512_gen(c8, 32, data, 37, o2);
    u32 il[ 8]; bytes32_to_fe(il, o2);
    u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=kf0[i];
    sc_add_mod_n(kf, il, kf);
    u8 kchild[ 32]; fe_to_bytes32(kchild, kf);
    for (int i=0;i<32;i++) d_out[128+i]=kchild[i];   // child priv @ N=0
    Jacobian P; scalar_mult_base_comb(P, kf);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    u8 pub[ 64];
    fe_to_bytes32(pub, x);
    fe_to_bytes32(pub+32, y);
    u8 hh[ 32]; device_keccak256(pub, 64, hh);
    for (int i=0;i<20;i++) d_out[160+i]=hh[12+i];     // addr
    d_done[ 0]=1;
}
// ==========================================
// Kernel: 分阶段性能剖析 (stage: 0=mnemonic 1=+pbkdf2 2=+ckd 3=full)
// 每段结果都累加进 checksum 写入全局，防止 DCE
// ==========================================
__global__ void prof_kernel(u64 start_nonce, int iters, int stage, u64* d_out, u32* d_done) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    u64 acc = 0;
    for (int it=0; it<iters; it++) {
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u8 ent[ 16]; counter_to_entropy(n, ent);  // v2: secret 混合熵
        char m[ 128]; entropy_to_mnemonic(ent, m, 0);
        u64 acc0 = 0;
        for (int i=0;i<128;i++) acc0 = acc0*131 + (u8)m[i];
        if (stage>=1) {
            u8 seed[ 64]; device_bip39_seed(m, seed);
            u64 acc1 = acc0;
            for (int i=0;i<64;i++) acc1 = acc1*131 + seed[i];
            if (stage>=2) {
                u8 privk[ 32]; bip32_derive(privk, seed);
                u64 acc2 = acc1;
                for (int i=0;i<32;i++) acc2 = acc2*131 + privk[i];
                if (stage>=3) {
                    u8 addr[ 20]; derive_address(privk, addr);
                    for (int i=0;i<20;i++) acc2 = acc2*131 + addr[i];
                    acc += acc2;
                } else acc += acc2;
            } else acc += acc1;
        } else acc += acc0;
    }
    if (acc == 0xDEADBEEF) d_out[ tid % 1024 ] = acc;
    d_done[ 0]=1;
}

__device__ void ent_to_hd_base(const u8 ent[ 16], u32 kbase[ 8], u32 cbase[ 8],
                               u32 pubx[ 8], u32 puby[ 8], char* mnem_out);

// ==========================================
// BTC 地址原语：RIPEMD160 / Base58Check / bech32(bech32m) / 地址生成
// ==========================================
__device__ __forceinline__ u32 btc_rol32(u32 x, int n) { return (x << n) | (x >> (32 - n)); }
__device__ __forceinline__ u32 rp_f(int j, u32 x, u32 y, u32 z) {
    if (j < 16) return x ^ y ^ z;
    if (j < 32) return (x & y) | (~x & z);
    if (j < 48) return (x | ~y) ^ z;
    if (j < 64) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}
__device__ __forceinline__ u32 rp_f2(int j, u32 x, u32 y, u32 z) {
    if (j < 16) return x ^ (y | ~z);
    if (j < 32) return (x & z) | (y & ~z);
    if (j < 48) return (x | ~y) ^ z;
    if (j < 64) return (x & y) | (~x & z);
    return x ^ y ^ z;
}
__device__ __forceinline__ void ripemd160(const u8* msg, int len, u8 out[ 20]) {
    static const u32 rp_r[ 80] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8, 3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2, 4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
    static const u32 rp_r2[ 80] = {5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2, 15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14, 12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
    static const u32 rp_s[ 80] = {11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12, 11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12, 9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
    static const u32 rp_s2[ 80] = {8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11, 9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8, 8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
    static const u32 rp_k[ 5] = {0x00000000, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e};
    static const u32 rp_k2[ 5] = {0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0x00000000};
    u8 blk[ 64];
    for (int i = 0; i < 64; i++) blk[i] = 0;
    for (int i = 0; i < len && i < 64; i++) blk[i] = msg[i];
    blk[len] = 0x80;
    u64 bits = (u64)len * 8;
    for (int i = 0; i < 8; i++) blk[56 + i] = (u8)((bits >> (8 * i)) & 0xFF);
    u32 M[ 16];
    for (int i = 0; i < 16; i++) { u32 v = 0; for (int j = 0; j < 4; j++) v |= ((u32)blk[i * 4 + j]) << (8 * j); M[i] = v; }
    u32 A = 0x67452301, B = 0xEFCDAB89, C = 0x98BADCFE, D = 0x10325476, E = 0xC3D2E1F0;
    u32 A2 = A, B2 = B, C2 = C, D2 = D, E2 = E;
    for (int j = 0; j < 80; j++) {
        u32 T = btc_rol32(A + rp_f(j, B, C, D) + M[rp_r[j]] + rp_k[j / 16], rp_s[j]) + E;
        A = E; E = D; D = btc_rol32(C, 10); C = B; B = T;
        u32 T2 = btc_rol32(A2 + rp_f2(j, B2, C2, D2) + M[rp_r2[j]] + rp_k2[j / 16], rp_s2[j]) + E2;
        A2 = E2; E2 = D2; D2 = btc_rol32(C2, 10); C2 = B2; B2 = T2;
    }
    u32 h0n = 0xEFCDAB89u + C + D2;
    u32 h1n = 0x98BADCFEu + D + E2;
    u32 h2n = 0x10325476u + E + A2;
    u32 h3n = 0xC3D2E1F0u + A + B2;
    u32 h4n = 0x67452301u + B + C2;
    u32 h[ 5] = {h0n, h1n, h2n, h3n, h4n};
    for (int i = 0; i < 20; i++) out[i] = (u8)((h[i / 4] >> (8 * (i % 4))) & 0xFF);
}
__device__ __forceinline__ void sha256d(const u8* msg, int len, u8 out[ 32]) {
    u8 h1[ 32]; sha256(msg, len, h1);
    sha256(h1, 32, out);
}
__device__ int base58check_encode(const u8* payload, int plen, char* out) {
    static const char* B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    u8 full[ 40];
    for (int i = 0; i < plen; i++) full[i] = payload[i];
    u8 chk[ 32]; sha256d(payload, plen, chk);
    for (int i = 0; i < 4; i++) full[plen + i] = chk[i];
    int n = plen + 4;
    int zeros = 0; while (zeros < n && full[zeros] == 0) zeros++;
    u8 buf[ 40]; for (int i = 0; i < n; i++) buf[i] = full[i];
    char b58[ 40]; int bl = 0;
    int start = 0;
    while (start < n) {
        int rem = 0;
        for (int i = start; i < n; i++) {
            int acc = rem * 256 + buf[i];
            buf[i] = (u8)(acc / 58);
            rem = acc % 58;
        }
        b58[bl++] = B58[rem];
        while (start < n && buf[start] == 0) start++;
    }
    int idx = 0;
    for (int i = 0; i < zeros; i++) out[idx++] = '1';
    for (int i = bl - 1; i >= 0; i--) out[idx++] = b58[i];
    return idx;
}
__device__ int bech32_encode(int witver, const u8* payload, int plen, char* out) {
    static const char* CH = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    u8 conv[ 80]; int nc = 0;
    int acc = 0, bits = 0;
    for (int i = 0; i < plen; i++) {
        acc = (acc << 8) | payload[i]; bits += 8;
        while (bits >= 5) { bits -= 5; conv[nc++] = (u8)((acc >> bits) & 31); }
    }
    if (bits > 0) conv[nc++] = (u8)((acc << (5 - bits)) & 31);
    u8 d[ 90]; int m = 0; d[m++] = (u8)witver;
    for (int i = 0; i < nc; i++) d[m++] = conv[i];
    u32 v[ 100]; int nv = 0;
    v[nv++] = 'b' >> 5; v[nv++] = 'c' >> 5; v[nv++] = 0; v[nv++] = 'b' & 31; v[nv++] = 'c' & 31;
    for (int i = 0; i < m; i++) v[nv++] = d[i];
    u32 chk = 1;
    static const u32 gen[ 5] = {0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3};
    for (int i = 0; i < nv + 6; i++) {
        u32 x = (i < nv) ? v[i] : 0;
        u32 b = chk >> 25;
        chk = ((chk & 0x1ffffff) << 5) ^ x;
        for (int j = 0; j < 5; j++) chk ^= ((b >> j) & 1) ? gen[j] : 0;
    }
    u32 constv = (witver == 0) ? 1u : 0x2bc830a3u;
    chk ^= constv;
    out[0] = 'b'; out[1] = 'c'; out[2] = '1';
    int oi = 3;
    for (int i = 0; i < m; i++) out[oi++] = CH[d[i]];
    for (int i = 0; i < 6; i++) out[oi++] = CH[(chk >> (5 * (5 - i))) & 31];
    return oi;
}
// type: 0=P2PKH 1=P2SH-P2WPKH 2=P2WPKH(bech32 v0) 3=P2TR(bech32m v1, x-only；原始 X、非 BIP86 tweak，见 v2 说明)
// pub = 64 字节 (x||y) 非压缩
__device__ int btc_address_from_pub(const u8 pub[ 64], int type, char* out) {
    u8 comp[ 33];
    comp[ 0] = (pub[ 63] & 1) ? 0x03 : 0x02;
    for (int i = 0; i < 32; i++) comp[ 1 + i] = pub[i];
    u8 h1[ 32]; sha256(comp, 33, h1);
    u8 h160[ 20]; ripemd160(h1, 32, h160);
    if (type == 0) { u8 pl[ 21]; pl[ 0] = 0x00; for (int i = 0; i < 20; i++) pl[ 1 + i] = h160[i]; return base58check_encode(pl, 21, out); }
    if (type == 1) { // v2 修复: P2SH-P2WPKH。兑换脚本 = 0x00 0x14||h160，地址 = Base58Check(0x05||HASH160(兑换脚本))。原版 0x05||h160(公钥) 不对应任何可花费脚本，打进去的币等于锁死。
        u8 rs[ 22]; rs[ 0] = 0x00; rs[ 1] = 0x14; for (int i = 0; i < 20; i++) rs[ 2 + i] = h160[i];
        u8 rh1[ 32]; sha256(rs, 22, rh1);
        u8 sh[ 20]; ripemd160(rh1, 32, sh);
        u8 pl[ 21]; pl[ 0] = 0x05; for (int i = 0; i < 20; i++) pl[ 1 + i] = sh[i]; return base58check_encode(pl, 21, out); }
    if (type == 2) { return bech32_encode(0, h160, 20, out); }
    return bech32_encode(1, pub, 32, out);
}
// 自检：固定向量 -> BTC m/44'/0'/0'/0/N (N=0,3) 各类型地址。熵 = SHA256(8 字节全 0) 的前 16 字节（不是全零熵）；v2 有意不混 secret，保证 v1/v2 输出可逐字对照
__global__ void btc_test_kernel(u8* d_out, u32* d_done) {
    u8 nb[ 8]; for (int i = 0; i < 8; i++) nb[i] = 0;
    u8 h[ 32]; sha256(nb, 8, h);
    u8 ent[ 16]; for (int i = 0; i < 16; i++) ent[i] = h[i];
    u32 kbase[ 8], cbase[ 8], pubx[ 8], puby[ 8];
    char mnem[ 128];
    ent_to_hd_base(ent, kbase, cbase, pubx, puby, mnem);
    u8 c8[ 32]; fe_to_bytes32(c8, cbase);
    u8 xb0[ 32]; fe_to_bytes32(xb0, pubx);
    u32 NN[ 2] = {0, 3};
    for (int q = 0; q < 2; q++) {
        u32 N = NN[ q];
        u8 data[ 37];
        data[ 0] = (puby[ 0] & 1) ? 0x03 : 0x02;
        for (int i = 0; i < 32; i++) data[ 1 + i] = xb0[i];
        data[ 33] = (u8)(N >> 24); data[ 34] = (u8)(N >> 16); data[ 35] = (u8)(N >> 8); data[ 36] = (u8)N;
        u8 o2[ 64]; hmac_sha512_gen(c8, 32, data, 37, o2);
        u32 il[ 8]; bytes32_to_fe(il, o2);
        u32 kf[ 8]; for (int i = 0; i < 8; i++) kf[i] = kbase[i];
        sc_add_mod_n(kf, il, kf);
        u8 privk[ 32]; fe_to_bytes32(privk, kf);
        Jacobian P; scalar_mult_base_comb(P, kf);
        u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
        u8 pub[ 64]; fe_to_bytes32(pub, x); fe_to_bytes32(pub + 32, y);
        u8 comp[ 33]; comp[ 0] = (pub[ 63] & 1) ? 0x03 : 0x02;
        for (int i = 0; i < 32; i++) comp[ 1 + i] = pub[i];
        u8 h1[ 32]; sha256(comp, 33, h1);
        u8 h160[ 20]; ripemd160(h1, 32, h160);
        int o = q * 320;
        for (int i = 0; i < 32; i++) d_out[o + i] = privk[i];
        for (int i = 0; i < 33; i++) d_out[o + 32 + i] = comp[i];
        for (int i = 0; i < 20; i++) d_out[o + 65 + i] = h160[i];
        u8 pl[ 21]; pl[ 0] = 0x00; for (int i = 0; i < 20; i++) pl[ 1 + i] = h160[i];
        int l = base58check_encode(pl, 21, (char*)(d_out + o + 85));
        d_out[o + 85 + l] = 0;
        l = bech32_encode(0, h160, 20, (char*)(d_out + o + 149));
        d_out[o + 149 + l] = 0;
        l = bech32_encode(1, pub, 32, (char*)(d_out + o + 229));
        d_out[o + 229 + l] = 0;
    }
    for (int i = 0; i < 128; i++) d_out[640 + i] = (u8)mnem[i];
    d_done[ 0] = 1;
}

// 熵 -> 助记词 -> m/44'/0'/0'/0 基点（kbase/cbase/pubx/puby），供多索引派生复用
__device__ void ent_to_hd_base(const u8 ent[ 16], u32 kbase[ 8], u32 cbase[ 8],
                               u32 pubx[ 8], u32 puby[ 8], char* mnem_out) {
    char m[ 128];
    entropy_to_mnemonic(ent, m, 0);
    if (mnem_out) { int l=0; while (m[l]){ mnem_out[l]=m[l]; l++; } mnem_out[l]=0; }
    u8 seed[ 64];
    device_bip39_seed(m, seed);
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    ckd_priv(k, c, 0x80000000u+44u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    u8 k0n[ 32], c0n[ 32];
    for (int i=0;i<32;i++){ k0n[i]=k[i]; c0n[i]=c[i]; }
    ckd_priv(k0n, c0n, 0u, 0);
    bytes32_to_fe(kbase, k0n);
    bytes32_to_fe(cbase, c0n);
    u32 kf[ 8]; bytes32_to_fe(kf, k0n);
    Jacobian P; scalar_mult_base_comb(P, kf);
    jac_to_affine(pubx, puby, P);
}

__global__ __launch_bounds__(256, 2) void vanity_kernel(const u8* d_prefix, int plen, u64 start_nonce, int iters, int nidx,
                              char* d_mnem, u8* d_addr, u8* d_priv, u32* d_index, u32* d_found) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    for (int it=0; it<iters; it++) {
        /* v3: 去掉整批早退，保证每批跑满，速率不虚高 */
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u8 ent[ 16]; counter_to_entropy(n, ent);  // v2: secret 混合熵
        u32 kbase[ 8], cbase[ 8], pubx[ 8], puby[ 8];
        char mnem[ 128];
        ent_to_hd_base(ent, kbase, cbase, pubx, puby, mnem);
        u8 c8[ 32]; fe_to_bytes32(c8, cbase);
        u8 xb0[ 32]; fe_to_bytes32(xb0, pubx);
        u64 cipad[ 8], copad[ 8];
        hmac_sha512_precompute(c8, 32, cipad, copad);  // v2: 同一助记词的 nidx 个子地址共用链码，预计算只做一次
        for (int idx=0; idx<nidx; idx++) {
            u32 N = (u32)idx;
            u8 data[ 37];
            data[ 0] = (puby[ 0]&1) ? 0x03 : 0x02;
            for (int i=0;i<32;i++) data[1+i]=xb0[i];
            data[ 33]=(u8)(N>>24); data[ 34]=(u8)(N>>16); data[ 35]=(u8)(N>>8); data[ 36]=(u8)N;
            u8 out[ 64];
            hmac_sha512_precomp_msg(cipad, copad, data, 37, out);
            u32 il[ 8]; bytes32_to_fe(il, out);
            u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=kbase[i];
            sc_add_mod_n(kf, il, kf);
            Jacobian P; scalar_mult_base_comb(P, kf);
            u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
            u8 pub[ 64];
            fe_to_bytes32(pub, x);
            fe_to_bytes32(pub+32, y);
            u8 hh[ 32];
            device_keccak256(pub, 64, hh);
            u8 addr[ 20]; for (int i=0;i<20;i++) addr[i]=hh[12+i];
            int ok=1;
            for (int i=0;i<plen;i++) {
                u8 nib = (i&1) ? (addr[i>>1]&0xF) : (addr[i>>1]>>4);
                if (nib != d_prefix[i]) { ok=0; break; }
            }
            if (ok) {
                u32 old = atomicCAS(d_found, 0, 1);
                if (old==0) {
                    int l=0; while (mnem[l]){ d_mnem[l]=mnem[l]; l++; } d_mnem[l]=0;
                    u8 privk[ 32]; fe_to_bytes32(privk, kf);
                    for (int i=0;i<20;i++) d_addr[i]=addr[i];
                    for (int i=0;i<32;i++) d_priv[i]=privk[i];
                    d_index[ 0]=N;
                }
                return;
            }
        }
    }
}


// ==========================================
// 文件模式：一个固定种子，同时搜索多个前缀（HD m/44'/0'/0'/0/N）
// 每个前缀独立 found slot + 结果；全部找到则提前退出
// ==========================================
__global__ __launch_bounds__(256, 2) void multi_hd_kernel(
        const u8* d_prefixes, const u8* d_plens, int nprefix,
        const u32* d_kbase, const u32* d_cbase, const u32* d_pubx, const u32* d_puby,
        u64 start_nonce, int iters, const char* d_mnem,
        u8* d_addrs, u8* d_privs, u32* d_idxs, u32* d_found, u32* d_nfound) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    for (int it=0; it<iters; it++) {
        if (*d_nfound >= (u32)nprefix) return;
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u32 N = (u32)(n & 0x7FFFFFFFu);
        // IL = HMAC-SHA512(c0n, 0x02/03||X || N)
        u32 cb[ 8]; for (int i=0;i<8;i++) cb[i]=d_cbase[i];
        u8 c8[ 32]; fe_to_bytes32(c8, cb);
        u8 data[ 37];
        data[ 0] = (d_puby[ 0]&1) ? 0x03 : 0x02;
        u8 xb[ 32]; fe_to_bytes32(xb, d_pubx);
        for (int i=0;i<32;i++) data[1+i]=xb[i];
        data[ 33]=(u8)(N>>24); data[ 34]=(u8)(N>>16); data[ 35]=(u8)(N>>8); data[ 36]=(u8)N;
        u8 out[ 64];
        hmac_sha512_gen(c8, 32, data, 37, out);
        u32 il[ 8]; bytes32_to_fe(il, out);
        u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=d_kbase[i];
        sc_add_mod_n(kf, il, kf);
        Jacobian P; scalar_mult_base_comb(P, kf);
        u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
        u8 pub[ 64];
        fe_to_bytes32(pub, x);
        fe_to_bytes32(pub+32, y);
        u8 hh[ 32];
        device_keccak256(pub, 64, hh);
        u8 addr[ 20]; for (int i=0;i<20;i++) addr[i]=hh[12+i];
        // 遍历所有未完成前缀
        for (int p=0; p<nprefix; p++) {
            if (d_found[p]) continue;
            int pl = d_plens[p];
            int ok=1;
            const u8* pfx = d_prefixes + p*40;
            for (int i=0;i<pl;i++) {
                u8 nib = (i&1) ? (addr[i>>1]&0xF) : (addr[i>>1]>>4);
                if (nib != pfx[i]) { ok=0; break; }
            }
            if (ok) {
                u32 old = atomicCAS(&d_found[p], 0, 1);
                if (old==0) {
                    u8 privk[ 32]; fe_to_bytes32(privk, kf);
                    for (int i=0;i<20;i++) d_addrs[p*20+i]=addr[i];
                    for (int i=0;i<32;i++) d_privs[p*32+i]=privk[i];
                    d_idxs[p]=N;
                    atomicAdd(d_nfound, 1);
                }
            }
        }
    }
}

// ==========================================
// 文件模式（vanity 语义）：随机助记词 + 多前缀并行 + 每助记词派生 nidx 个地址
// ==========================================
__global__ __launch_bounds__(256, 2) void multi_vanity_kernel(
        const u8* d_prefixes, const u8* d_plens, int nprefix, int nidx,
        u64 start_nonce, int iters,
        u8* d_mnems, u8* d_addrs, u8* d_privs, u32* d_idxs, u32* d_found, u32* d_nfound) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    for (int it=0; it<iters; it++) {
        if (*d_nfound >= (u32)nprefix) return;
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u8 ent[ 16]; counter_to_entropy(n, ent);  // v2: secret 混合熵
        u32 kbase[ 8], cbase[ 8], pubx[ 8], puby[ 8];
        char mnem[ 128];
        ent_to_hd_base(ent, kbase, cbase, pubx, puby, mnem);
        u8 c8[ 32]; fe_to_bytes32(c8, cbase);
        u8 xb0[ 32]; fe_to_bytes32(xb0, pubx);
        u64 cipad[ 8], copad[ 8];
        hmac_sha512_precompute(c8, 32, cipad, copad);  // v2: 同一助记词的 nidx 个子地址共用链码，预计算只做一次
        for (int idx=0; idx<nidx; idx++) {
            u32 N = (u32)idx;
            u8 data[ 37];
            data[ 0] = (puby[ 0]&1) ? 0x03 : 0x02;
            for (int i=0;i<32;i++) data[1+i]=xb0[i];
            data[ 33]=(u8)(N>>24); data[ 34]=(u8)(N>>16); data[ 35]=(u8)(N>>8); data[ 36]=(u8)N;
            u8 out[ 64];
            hmac_sha512_precomp_msg(cipad, copad, data, 37, out);
            u32 il[ 8]; bytes32_to_fe(il, out);
            u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=kbase[i];
            sc_add_mod_n(kf, il, kf);
            Jacobian P; scalar_mult_base_comb(P, kf);
            u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
            u8 pub[ 64];
            fe_to_bytes32(pub, x);
            fe_to_bytes32(pub+32, y);
            u8 hh[ 32];
            device_keccak256(pub, 64, hh);
            u8 addr[ 20]; for (int i=0;i<20;i++) addr[i]=hh[12+i];
            for (int p=0; p<nprefix; p++) {
                if (d_found[p]) continue;
                int pl = d_plens[p];
                int ok=1;
                const u8* pfx = d_prefixes + p*40;
                for (int i=0;i<pl;i++) {
                    u8 nib = (i&1) ? (addr[i>>1]&0xF) : (addr[i>>1]>>4);
                    if (nib != pfx[i]) { ok=0; break; }
                }
                if (ok) {
                    u32 old = atomicCAS(&d_found[p], 0, 1);
                    if (old==0) {
                        u8 privk[ 32]; fe_to_bytes32(privk, kf);
                        int l=0; while (mnem[l]){ d_mnems[p*128+l]=(u8)mnem[l]; l++; } d_mnems[p*128+l]=0;
                        for (int i=0;i<20;i++) d_addrs[p*20+i]=addr[i];
                        for (int i=0;i<32;i++) d_privs[p*32+i]=privk[i];
                        d_idxs[p]=N;
                        atomicAdd(d_nfound, 1);
                    }
                }
            }
        }
    }
}

typedef struct { u32 p; u32 idx; u8 mnem[128]; u8 addr[PSTR]; u8 priv[32]; } ResqEntry;
__global__ __launch_bounds__(256, 2) void multi_hash_kernel(
        const u64* d_tbl_key, const u32* d_tbl_p, const u8* d_tbl_occ,
        const u32* d_tbl_off, const u32* d_tbl_mask, const u32* d_tbl_cnt, const u8* d_pfx, u64 uniq,
        int nidx, int addrtype, int nl_max, u64 start_nonce, int iters,
        u32* d_found, u32* d_nfound, ResqEntry* d_resq, u32* d_resq_cnt) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    for (int it=0; it<iters; it++) {
        if (uniq>0 && (u64)*d_nfound >= uniq) return;
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u8 ent[ 16]; counter_to_entropy(n, ent);  // v2: secret 混合熵
        u32 kbase[ 8], cbase[ 8], pubx[ 8], puby[ 8];
        char mnem[ 128];
        ent_to_hd_base(ent, kbase, cbase, pubx, puby, mnem);
        u8 c8[ 32]; fe_to_bytes32(c8, cbase);
        u8 xb0[ 32]; fe_to_bytes32(xb0, pubx);
        u64 cipad[ 8], copad[ 8];
        hmac_sha512_precompute(c8, 32, cipad, copad);  // v2: 同一助记词的 nidx 个子地址共用链码，预计算只做一次
        for (int idx=0; idx<nidx; idx++) {
            u32 N = (u32)idx;
            u8 data[ 37];
            data[ 0] = (puby[ 0]&1) ? 0x03 : 0x02;
            for (int i=0;i<32;i++) data[1+i]=xb0[i];
            data[ 33]=(u8)(N>>24); data[ 34]=(u8)(N>>16); data[ 35]=(u8)(N>>8); data[ 36]=(u8)N;
            u8 out[ 64];
            hmac_sha512_precomp_msg(cipad, copad, data, 37, out);
            u32 il[ 8]; bytes32_to_fe(il, out);
            u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=kbase[i];
            sc_add_mod_n(kf, il, kf);
            Jacobian P; scalar_mult_base_comb(P, kf);
            u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
            u8 pub[ 64];
            fe_to_bytes32(pub, x);
            fe_to_bytes32(pub+32, y);
            char ast[ PSTR ];
            int laddr = btc_address_from_pub(pub, addrtype, ast);
            if (laddr > PSTR) laddr = PSTR;
            u64 h=0xcbf29ce484222325ull;
            for (int L=1; L<=nl_max; L++) {
                if (L > laddr) break;
                h = (h ^ (u8)ast[L-1]) * 0x100000001b3ull;
                int cnt = d_tbl_cnt[L-1];
                if (!cnt) continue;
                u32 mask = d_tbl_mask[L-1];
                u32 base = d_tbl_off[L-1];
                u32 hh2 = (u32)((h * 0x9E3779B97F4A7C15ull) >> 32);
                int s = (int)(hh2 & mask);
                for (;;) {
                    if (!d_tbl_occ[base+s]) break;
                    if (d_tbl_key[base+s] == h) {
                        u32 p = d_tbl_p[base+s];
                        int ok=1;
                        for (int k=0;k<L;k++) if (d_pfx[(u64)p*PSTR+k]!=(u8)ast[k]) { ok=0; break; }
                        if (ok && !d_found[p]) {
                            u32 old = atomicCAS(&d_found[p], 0, 1);
                            if (old==0) {
                                u8 privk[ 32]; fe_to_bytes32(privk, kf);
                                u32 q = atomicAdd(d_resq_cnt, 1);
                                ResqEntry* e = &d_resq[q & RESQ_MASK];
                                e->p = p; e->idx = N;
                                int l=0; while (mnem[l] && l<127){ e->mnem[l]=(u8)mnem[l]; l++; } e->mnem[l]=0;
                                for (int i=0;i<laddr;i++) e->addr[i]=(u8)ast[i];
                                e->addr[laddr]=0;
                                for (int i=0;i<32;i++) e->priv[i]=privk[i];
                                atomicAdd(d_nfound, 1);
                            }
                            break;
                        }
                    }
                    s = (s+1) & mask;
                }
            }
        }
    }
}

// ==========================================
// 主机辅助
// ==========================================
static void print_hex(const u8* p, int n) {
    for (int i=0;i<n;i++) printf("%02x", p[i]);
}

static int btc_nlmax(int addrtype) {
    return (addrtype==0||addrtype==1) ? 34 : (addrtype==2 ? 42 : 62);
}
static const char* btc_aname(int addrtype) {
    return (addrtype==0) ? "P2PKH(1)" : (addrtype==1) ? "P2SH-P2WPKH(3)" : (addrtype==2) ? "P2WPKH(bc1q)" : "P2TR(bc1p)";
}
// pfx 行宽 PSTR, plen 每前缀长度; 统一搜索入口（file/vanity 共用）
static void run_btc_search(u8* pfx, u8* plen, u64 nprefix, int addrtype, int nidx, int blocks) {
    if (nprefix==0) { fprintf(stderr,"no valid prefixes\n"); return; }
    int nl_max = btc_nlmax(addrtype);
    // 校验超长（调用方已处理则这里再兜底）
    u64 nskip=0;
    for (u64 p=0;p<nprefix;p++) if (plen[p] > nl_max) nskip++;
    if (nskip) printf("note: %llu prefix(es) longer than %d chars skipped\n", (unsigned long long)nskip, nl_max);
    printf("BTC vanity mode | addr type %s | max prefix len %d | file prefixes %llu\n", btc_aname(addrtype), nl_max, (unsigned long long)nprefix);
    // v2 预检：违反地址类型约束的前缀在数学上永远命中不了（搜索条件是全部命中才退出，原版会永远跑下去）
    {
        u64 impossible=0;
        const char* B58C = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
        const char* BECH = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
        for (u64 p=0;p<nprefix;p++) {
            int L=plen[p];
            if (L>nl_max) continue;
            int bad=0;
            if (addrtype==0 || addrtype==1) {
                char want = (addrtype==0) ? '1' : '3';
                if (L>=1 && pfx[p*PSTR]!=want) bad=1;
                for (int i=0;i<L && !bad;i++) if (!strchr(B58C, pfx[p*PSTR+i])) bad=1;
            } else {
                const char* head = (addrtype==2) ? "bc1q" : "bc1p";
                for (int i=0;i<L && i<4 && !bad;i++) if (pfx[p*PSTR+i]!=head[i]) bad=1;
                for (int i=4;i<L && !bad;i++) if (!strchr(BECH, pfx[p*PSTR+i])) bad=1;
            }
            if (bad) impossible++;
        }
        if (impossible)
            printf("WARNING: %llu prefix(es) can never match a %s address - they will never hit and the search will never finish\n", (unsigned long long)impossible, btc_aname(addrtype));
    }
    int lcnt[ NLMAX+1 ];
    for (int i=0;i<=NLMAX;i++) lcnt[i]=0;
    for (u64 p=0;p<nprefix;p++) if (plen[p]<=nl_max) lcnt[plen[p]]++;
    printf("length dist: ");
    for (int L=1;L<=nl_max;L++) if (lcnt[L]) printf("%d-char:%d ", L, lcnt[L]);
    printf("\n");
    int hcap[ NLMAX ]; u32 off[ NLMAX ]; u32 mask[ NLMAX ];
    u64* hk[ NLMAX ]; u32* hp[ NLMAX ]; u8* hoc[ NLMAX ];
    u64 total_cap = 0;
    for (int L=1;L<=NLMAX;L++) {
        int cnt = (L<=nl_max) ? lcnt[L] : 0;
        u64 c2 = 8;
        while (c2 < (u64)cnt*2+1) c2 <<= 1;
        hcap[L-1]=(int)c2; off[L-1]=(u32)total_cap; mask[L-1]=(u32)(c2-1);
        hk[L-1]=(u64*)malloc((size_t)(c2*8));
        hp[L-1]=(u32*)malloc((size_t)(c2*4));
        hoc[L-1]=(u8*)malloc((size_t)c2);
        memset(hoc[L-1],0,(size_t)c2);
        total_cap += c2;
    }
    if (total_cap >= (1ull<<32)) { for (int L=1;L<=NLMAX;L++){free(hk[L-1]);free(hp[L-1]);free(hoc[L-1]);} fprintf(stderr,"hash table too large (>4G slots)\n"); return; }
    u64 uniq=0;
    for (u64 p=0;p<nprefix;p++) {
        int L=plen[p];
        if (L>nl_max) continue;
        u64 key=0xcbf29ce484222325ull;
        for (int i=0;i<L;i++) key = (key ^ pfx[p*PSTR+i]) * 0x100000001b3ull;
        u32 hh = (u32)((key * 0x9E3779B97F4A7C15ull) >> 32);
        int s = (int)(hh & mask[L-1]);
        u8* oc = hoc[L-1]; u64* kk = hk[L-1]; u32* pp = hp[L-1];
        for (;;) {
            if (!oc[s]) {
                oc[s]=1; kk[s]=key; pp[s]=(u32)uniq;
                if (uniq < p) { memmove(pfx+uniq*PSTR, pfx+p*PSTR, PSTR); plen[uniq]=plen[p]; }
                uniq++; break;
            }
            if (kk[s]==key) {
                u32 q = pp[s]; int same=1;
                for (int i=0;i<L;i++) if (pfx[(u64)q*PSTR+i]!=pfx[p*PSTR+i]) { same=0; break; }
                if (same) break;
            }
            s = (s+1) & mask[L-1];
        }
    }
    printf("unique prefixes after dedup: %llu\n", (unsigned long long)uniq);
    if (uniq >= (1ull<<31)) { for (int L=1;L<=NLMAX;L++){free(hk[L-1]);free(hp[L-1]);free(hoc[L-1]);} fprintf(stderr,"too many unique prefixes (>=2^31)\n"); return; }
    if (uniq==0) { for (int L=1;L<=NLMAX;L++){free(hk[L-1]);free(hp[L-1]);free(hoc[L-1]);} return; }
    u64* ck=(u64*)malloc((size_t)(total_cap*8));
    u32* cp=(u32*)malloc((size_t)(total_cap*4));
    u8* co=(u8*)malloc((size_t)total_cap);
    for (int L=1;L<=NLMAX;L++) {
        memcpy(ck+off[L-1], hk[L-1], hcap[L-1]*8);
        memcpy(cp+off[L-1], hp[L-1], hcap[L-1]*4);
        memcpy(co+off[L-1], hoc[L-1], hcap[L-1]);
        free(hk[L-1]); free(hp[L-1]); free(hoc[L-1]);
    }
    u64* d_tbl_key; u32* d_tbl_p; u8* d_tbl_occ;
    u32* d_tbl_off; u32* d_tbl_mask; u32* d_tbl_cnt;
    CUDA_CHECK(cudaMalloc(&d_tbl_key, (size_t)(total_cap*8)));
    CUDA_CHECK(cudaMalloc(&d_tbl_p, (size_t)(total_cap*4)));
    CUDA_CHECK(cudaMalloc(&d_tbl_occ, (size_t)total_cap));
    CUDA_CHECK(cudaMalloc(&d_tbl_off, NLMAX*4));
    CUDA_CHECK(cudaMalloc(&d_tbl_mask, NLMAX*4));
    CUDA_CHECK(cudaMalloc(&d_tbl_cnt, NLMAX*4));
    CUDA_CHECK(cudaMemcpy(d_tbl_key, ck, (size_t)(total_cap*8), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl_p, cp, (size_t)(total_cap*4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl_occ, co, (size_t)total_cap, cudaMemcpyHostToDevice));
    u32 tcnt[ NLMAX ];
    for (int L=1;L<=NLMAX;L++) tcnt[L-1]=0;
    for (u64 pi=0; pi<uniq; pi++) tcnt[plen[pi]-1]++;  // v2: 改用去重后的计数（原版用去重前 lcnt；内核只作非零判断故旧版结果未受影响）
    CUDA_CHECK(cudaMemcpy(d_tbl_off, off, NLMAX*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl_mask, mask, NLMAX*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl_cnt, tcnt, NLMAX*4, cudaMemcpyHostToDevice));
    u8* d_pfx;
    CUDA_CHECK(cudaMalloc(&d_pfx, (size_t)(uniq*PSTR)));
    CUDA_CHECK(cudaMemcpy(d_pfx, pfx, (size_t)(uniq*PSTR), cudaMemcpyHostToDevice));
    free(ck); free(cp); free(co);
    u32* d_found; u32* d_nfound;
    ResqEntry* d_resq; u32* d_resq_cnt;
    CUDA_CHECK(cudaMalloc(&d_found, (size_t)(uniq*4)));
    CUDA_CHECK(cudaMalloc(&d_nfound, 4));
    CUDA_CHECK(cudaMalloc(&d_resq, (size_t)(RESQ_CAP*sizeof(ResqEntry))));
    CUDA_CHECK(cudaMalloc(&d_resq_cnt, 4));
    CUDA_CHECK(cudaMemset(d_found, 0, (size_t)(uniq*4)));
    CUDA_CHECK(cudaMemset(d_nfound, 0, 4));
    CUDA_CHECK(cudaMemset(d_resq_cnt, 0, 4));
    if (nidx<1) nidx=1;
    if (nidx>1000000) nidx=1000000;
    u64 start=0; int batch_iters=1; u64 total_mnem=0; double total_s=0; u32 nfound=0;
    u64 qprev=0; u64 printed=0;
    cudaEvent_t t0,t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    printf("searching random mnemonic m/44'/0'/0'/0/0..%d (nidx=%d), %d blocks x 256 threads... Ctrl+C to stop\n",
           nidx-1, nidx, blocks);
    // v2: 命中即写文件 btc_hits.txt（追加、无缓冲实时落盘，Ctrl+C 强杀不丢）；原版只打标准输出
    FILE* hf = fopen("btc_hits.txt", "a");
    if (hf) { setvbuf(hf, NULL, _IONBF, 0);
#ifndef _WIN32
        chmod("btc_hits.txt", 0600);  // 助记词+私钥明文文件，权限收紧
#endif
    }
    while (1) {
        int launched = batch_iters;  // v2: 本轮实际启动的批量；统计与区间推进只准用它
        CUDA_CHECK(cudaMemset(d_found, 0, (size_t)((u64)uniq*4)));
        CUDA_CHECK(cudaMemset(d_nfound, 0, 4));
        nfound = 0;                  // 每轮重置，同一前缀可持续出新命中
        CUDA_CHECK(cudaEventRecord(t0));
        multi_hash_kernel<<<blocks, 256>>>(d_tbl_key, d_tbl_p, d_tbl_occ,
                                           d_tbl_off, d_tbl_mask, d_tbl_cnt, d_pfx, uniq,
                                           nidx, addrtype, nl_max, start, batch_iters,
                                           d_found, d_nfound, d_resq, d_resq_cnt);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        CUDA_CHECK(cudaMemcpy(&nfound, d_nfound, 4, cudaMemcpyDeviceToHost));
        total_mnem += (u64)launched*blocks*256;
        total_s += ms/1000.0;
        if (ms > 1000.0f) { int nb=batch_iters/2; if (nb<1) nb=1; batch_iters=nb; }
        else if (ms < 200.0f) { int nb=batch_iters*3; if (nb<1) nb=1; if (nb>1000000) nb=1000000; batch_iters=nb; }
        u32 qcnt=0; CUDA_CHECK(cudaMemcpy(&qcnt, d_resq_cnt, 4, cudaMemcpyDeviceToHost));
        while (qprev < (u64)qcnt) {
            ResqEntry e;
            CUDA_CHECK(cudaMemcpy(&e, d_resq + (qprev & RESQ_MASK), sizeof(ResqEntry), cudaMemcpyDeviceToHost));
            qprev++;
            printf("HIT [%4u] \"%s\" -> m/44'/0'/0'/0/%u\n", e.p, e.addr, e.idx);
            printf("      mnemonic: %s\n", e.mnem);
            printf("      address:  %s\n", e.addr);
            printf("      private:  "); for (int i=0;i<32;i++) printf("%02x", e.priv[i]); printf("\n");
            if (hf) {
                fprintf(hf, "HIT [%4u] \"%s\" -> m/44'/0'/0'/0/%u\n", e.p, e.addr, e.idx);
                fprintf(hf, "      mnemonic: %s\n", e.mnem);
                fprintf(hf, "      address:  %s\n", e.addr);
                fprintf(hf, "      private:  ");
                for (int i=0;i<32;i++) fprintf(hf, "%02x", e.priv[i]);
                fprintf(hf, "\n\n");
            }
            printed++;
        }
        double as = (double)total_mnem*nidx/total_s;
        printf("\r  [%7.1f s] %10.0f mnem/s | %13.0f addr/s | hit %u/%llu  (Ctrl+C stop)", total_s, total_mnem/total_s, as, nfound, (unsigned long long)uniq);
        fflush(stdout);
        start += (u64)launched*blocks*256;
    }
    printf("\nall %llu unique prefixes found in %.1f s, printed %llu hits\n", (unsigned long long)uniq, total_s, (unsigned long long)printed);
    if (hf) fclose(hf);
    CUDA_CHECK(cudaFree(d_tbl_key)); CUDA_CHECK(cudaFree(d_tbl_p)); CUDA_CHECK(cudaFree(d_tbl_occ));
    CUDA_CHECK(cudaFree(d_tbl_off)); CUDA_CHECK(cudaFree(d_tbl_mask)); CUDA_CHECK(cudaFree(d_tbl_cnt));
    CUDA_CHECK(cudaFree(d_found)); CUDA_CHECK(cudaFree(d_nfound));
    CUDA_CHECK(cudaFree(d_resq)); CUDA_CHECK(cudaFree(d_resq_cnt));
    CUDA_CHECK(cudaFree(d_pfx));
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
}


// v2: 加载或创建 32B secret 文件（默认 ./.btc_vanity_secret，可用环境变量 BTC_VANITY_SECRET 指定路径）。
// secret 与计数器共同决定助记词序列；secret 绝不打印、不写入日志。
// 注意：丢了 secret 不影响用 btc_hits.txt 里已记录的助记词找回单个钱包，但无法再复现整条搜索序列。
// 首次创建后请离线备份一份，并保持文件权限 0600。
static int load_or_create_secret(u8 sec[ 32]) {
    const char* path = getenv("BTC_VANITY_SECRET");
    if (!path || !path[0]) path = ".btc_vanity_secret";
    FILE* f = fopen(path, "rb");
    if (f) {
        size_t r = fread(sec, 1, 32, f); fclose(f);
        if (r == 32) {
#ifndef _WIN32
            chmod(path, 0600);
#endif
            printf("secret: loaded from %s (keep it private, back it up offline)\n", path); return 1; }
        fprintf(stderr, "secret file %s exists but is not 32 bytes; refusing to overwrite it\n", path);
        return 0;
    }
    FILE* u = fopen("/dev/urandom", "rb");
    if (u) {
        size_t r = fread(sec, 1, 32, u); fclose(u);
        if (r != 32) { fprintf(stderr, "short read from /dev/urandom\n"); return 0; }
    } else {
        std::random_device rd;  // 无 /dev/urandom 的平台（如原生 Windows）走系统 CSPRNG
        for (int i=0;i<32;i++) sec[i]=(u8)(rd() & 0xFF);
    }
    f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot create secret file %s\n", path); return 0; }
    fwrite(sec, 1, 32, f); fclose(f);
#ifndef _WIN32
    chmod(path, 0600);
#endif
    printf("secret: created %s (0600). Back it up offline; never share it.\n", path);
    return 1;
}

static int chain_main(int argc, char** argv) {
    const char* mode = argc>1 ? argv[1] : "bench";
    printf("=== BTC BIP39 Vanity Generator (true secp256k1 + BIP32/44) ===\n");
    u8 secret[ 32];
    if (!load_or_create_secret(secret)) return 1;
    CUDA_CHECK(cudaMemcpyToSymbol(SECRET_K, secret, 32));

    // comb 固定基点预计算表（60 点）
    u32* d_comb;
    CUDA_CHECK(cudaMalloc(&d_comb, 60*16*4));
    precompute_comb_kernel<<<15, 4>>>(d_comb);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyToSymbol(g_comb, &d_comb, sizeof(d_comb)));
    CUDA_CHECK(cudaDeviceSynchronize());  // v2: 确保 comb 表已写完，后续任何流（sK/sP）上的 kernel 都不会读到未就绪的表

    u8 prefix_nib[ 40];
    int plen=0;

    u8* d_prefix;
    char* d_mnem; u8* d_addr; u8* d_priv; u32* d_found; u32* d_index;
    CUDA_CHECK(cudaMalloc(&d_prefix, 40));
    CUDA_CHECK(cudaMalloc(&d_mnem, 128));
    CUDA_CHECK(cudaMalloc(&d_addr, 20));
    CUDA_CHECK(cudaMalloc(&d_priv, 32));
    CUDA_CHECK(cudaMalloc(&d_found, 4));
    CUDA_CHECK(cudaMalloc(&d_index, 4));
    if (plen) CUDA_CHECK(cudaMemcpy(d_prefix, prefix_nib, plen, cudaMemcpyHostToDevice));

    if (strcmp(mode,"selftest")==0) {
        u64* d_out; u32* d_done;
        CUDA_CHECK(cudaMalloc(&d_out, 11*8));
        CUDA_CHECK(cudaMalloc(&d_done, 4));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK));
        CUDA_CHECK(cudaStreamCreate(&sP));
        selftest_kernel<<<1,1,0,sK>>>(d_out, d_done);
        for (int w=0;w<40;w++) {
            u32 p=0;
            cudaMemcpyAsync(&p, d_done, 4, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            if (p==1) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
        u64 h_out[ 11];
        cudaMemcpyAsync(h_out, d_out, 11*8, cudaMemcpyDeviceToHost, sP);
        cudaStreamSynchronize(sP);
        printf("selftest:\n");
        printf("  (p-1)+1        = %016llx %016llx\n", (unsigned long long)(h_out[ 0]>>32), (unsigned long long)(h_out[ 0]&0xFFFFFFFFFFFFFFFFULL));
        printf("  123456789*987..= %016llx %016llx\n", (unsigned long long)(h_out[ 1]>>32), (unsigned long long)h_out[ 1]);
        printf("  2G x           = %016llx %016llx\n", (unsigned long long)(h_out[ 2]>>32), (unsigned long long)h_out[ 2]);
        printf("  2G y           = %016llx %016llx\n", (unsigned long long)(h_out[ 3]>>32), (unsigned long long)h_out[ 3]);
        printf("  1G x           = %016llx %016llx\n", (unsigned long long)(h_out[ 4]>>32), (unsigned long long)h_out[ 4]);
        printf("  1G y           = %016llx %016llx\n", (unsigned long long)(h_out[ 5]>>32), (unsigned long long)h_out[ 5]);
        printf("  (n-1)+1 mod n  = %016llx %016llx\n", (unsigned long long)(h_out[ 6]>>32), (unsigned long long)h_out[ 6]);
        printf("  (n-2)+3 mod n  = %016llx %016llx\n", (unsigned long long)(h_out[ 7]>>32), (unsigned long long)h_out[ 7]);
        printf("  2G+G x (jac)   = %016llx %016llx\n", (unsigned long long)(h_out[ 8]>>32), (unsigned long long)h_out[ 8]);
        printf("  3G x (scalar)  = %016llx %016llx\n", (unsigned long long)(h_out[ 9]>>32), (unsigned long long)h_out[ 9]);
        printf("  a*a^-1         = %016llx %016llx\n", (unsigned long long)(h_out[ 10]>>32), (unsigned long long)h_out[ 10]);
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_done));
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
    }
    else if (strcmp(mode,"fieldtest")==0) {
        // ftprog 已验证模式：自定义流 + 异步轮询，不调用 cudaDeviceSynchronize
        u32* d_out; u32* d_prog;
        CUDA_CHECK(cudaMalloc(&d_out, 8*8*4));
        CUDA_CHECK(cudaMalloc(&d_prog, 4));
        CUDA_CHECK(cudaMemset(d_prog, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK));
        CUDA_CHECK(cudaStreamCreate(&sP));
        ftprog_kernel<<<1,1,0,sK>>>(d_out, d_prog);
        for (int t=0;t<30;t++) {
            u32 p=0;
            cudaMemcpyAsync(&p, d_prog, 4, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            printf("t=%d prog=%u\n", t, p);
            fflush(stdout);
            if (p>=9) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
        u32 h_out[ 64];
        cudaError_t merr = cudaMemcpy(h_out, d_out, 8*8*4, cudaMemcpyDeviceToHost);
        printf("sync-copy err=%s\n", cudaGetErrorString(merr));
        const char* names[ 8] = {
            "(p-1)*(p-1)      ", "(p-2)*(p-1)      ", "2^255 * 2^255     ",
            "aaaa.. * aaaa..  ", "1234567890abcdef^2", "0xffff...+0x100000001",
            "inv(a)*a          ", "(n-1)+2 mod n     "
        };
        for (int t=0;t<8;t++) {
            printf("%s = ", names[t]);
            for (int i=7;i>=0;i--) printf("%08x", h_out[t*8+i]);
            printf("\n");
        }
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_prog));
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
    }
    else if (strcmp(mode,"ftprog")==0) {
        u32* d_out; u32* d_prog;
        CUDA_CHECK(cudaMalloc(&d_out, 8*8*4));
        CUDA_CHECK(cudaMalloc(&d_prog, 4));
        CUDA_CHECK(cudaMemset(d_prog, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK));
        CUDA_CHECK(cudaStreamCreate(&sP));
        ftprog_kernel<<<1,1,0,sK>>>(d_out, d_prog);
        for (int t=0;t<30;t++) {
            u32 p=0;
            cudaMemcpyAsync(&p, d_prog, 4, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            printf("t=%d prog=%u\n", t, p);
            fflush(stdout);
            if (p>=9) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_prog));
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
    }
    else if (strcmp(mode,"bisect")==0) {
        int c = (argc>2)?atoi(argv[ 2]):0;
        u32* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, 8*4));
        bisect_kernel<<<1,1>>>(d_out, c);
        CUDA_CHECK(cudaGetLastError());
        u32 h_out[ 8];
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 8*4, cudaMemcpyDeviceToHost));
        printf("case %d = ", c);
        for (int i=7;i>=0;i--) printf("%08x", h_out[i]);
        printf("\n");
        CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"debug")==0) {
        u32* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, 44*4));
        debug_kernel<<<1,1>>>(d_out);
        CUDA_CHECK(cudaGetLastError());
        u32 h_out[ 44];
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 44*4, cudaMemcpyDeviceToHost));
        printf("debug (p-1)^2 intermediates:\n");
        printf("  raw product t[0..15]:\n    ");
        for (int i=0;i<16;i++) printf("%08x ", h_out[i]);
        printf("\n  v after initial fold:\n    ");
        for (int i=0;i<10;i++) printf("%08x ", h_out[16+i]);
        printf("\n  v after fe_fold_hi:\n    ");
        for (int i=0;i<10;i++) printf("%08x ", h_out[26+i]);
        printf("\n  fe_mul r:\n    ");
        for (int i=0;i<8;i++) printf("%08x ", h_out[36+i]);
        printf("\n");
        CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"jdbl")==0) {
        u32* d_done; CUDA_CHECK(cudaMalloc(&d_done, 4));
        u32* d_out; CUDA_CHECK(cudaMalloc(&d_out, 256));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        jdbl_kernel<<<1,1,0,sK>>>(d_out, d_done);
        for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
        u32 hb[ 64];
        cudaMemcpyAsync(hb, d_out, 256, cudaMemcpyDeviceToHost, sP);
        cudaStreamSynchronize(sP);
        const char* names[ 10] = { "A=X^2", "B=Y^2", "C=B^2", "D=2t", "E=3A", "F0=E^2", "t0=2D", "X3", "Y3", "Z3" };
        int o=0;
        for (int q=0;q<10;q++){ printf("%s: ",names[q]); for(int i=7;i>=0;i--) printf("%08x",hb[o++]); printf("\n"); }
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        CUDA_CHECK(cudaFree(d_done)); CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"sm")==0) {
        u32* d_done; CUDA_CHECK(cudaMalloc(&d_done, 4));
        u32* d_out; CUDA_CHECK(cudaMalloc(&d_out, 128));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        for (int cc=0;cc<6;cc++) {
            CUDA_CHECK(cudaMemset(d_done, 0, 4));
            sm_kernel<<<1,1,0,sK>>>(d_out, d_done, cc);
            for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
            u32 hb[ 24];
            cudaMemcpyAsync(hb, d_out, 96, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            const char* nm[ 6] = { "k0h", "2^255", "k0h_top32", "0xe9", "2^247", "10G_jac" };
            if (cc==1 || cc==5) {
                printf("%s X: ", nm[cc]); for(int i=7;i>=0;i--) printf("%08x",hb[i]); printf("\n");
                printf("%s Y: ", nm[cc]); for(int i=15;i>=8;i--) printf("%08x",hb[i]); printf("\n");
                printf("%s Z: ", nm[cc]); for(int i=23;i>=16;i--) printf("%08x",hb[i]); printf("\n");
            } else {
                printf("%s x: ", nm[cc]); for(int i=7;i>=0;i--) printf("%08x",hb[i]); printf("\n");
                printf("%s y: ", nm[cc]); for(int i=15;i>=8;i--) printf("%08x",hb[i]); printf("\n");
            }
        }
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        CUDA_CHECK(cudaFree(d_done)); CUDA_CHECK(cudaFree(d_out));
    }

    else if (strcmp(mode,"ktest")==0) {
        u32* d_done; CUDA_CHECK(cudaMalloc(&d_done, 4));
        u8* d_out; CUDA_CHECK(cudaMalloc(&d_out, 128));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        ktest_kernel<<<1,1,0,sK>>>(d_out, d_done);
        for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
        u8 hb[ 96];
        cudaMemcpyAsync(hb, d_out, 96, cudaMemcpyDeviceToHost, sP);
        cudaStreamSynchronize(sP);
        printf("keccak(""):  "); for(int i=0;i<32;i++) printf("%02x",hb[i]); printf("\n");
        printf("keccak(abc): "); for(int i=0;i<32;i++) printf("%02x",hb[32+i]); printf("\n");
        printf("keccak(pub): "); for(int i=0;i<32;i++) printf("%02x",hb[64+i]); printf("\n");
        printf("f1600(0) st0: "); for(int i=0;i<8;i++) printf("%02x",hb[96+i]); printf("\n");
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        CUDA_CHECK(cudaFree(d_done)); CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"ptest")==0) {
        u32* d_done; CUDA_CHECK(cudaMalloc(&d_done, 4));
        u32* d_out; CUDA_CHECK(cudaMalloc(&d_out, 256));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        ptest_kernel<<<1,1,0,sK>>>(d_out, d_done);
        for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
        u32 hb[ 64];
        cudaMemcpyAsync(hb, d_out, 256, cudaMemcpyDeviceToHost, sP);
        cudaStreamSynchronize(sP);
        int o=0;
        printf("2G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("2G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("3G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("3G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("4G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("4G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("5G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("5G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("7G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("7G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("8G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("8G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        CUDA_CHECK(cudaFree(d_done)); CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"scdiag")==0) {
        u32* d_done; CUDA_CHECK(cudaMalloc(&d_done, 4));
        u32* d_out; CUDA_CHECK(cudaMalloc(&d_out, 512));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        scdiag_kernel<<<1,1,0,sK>>>(d_out, d_done);
        for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
        u32 hb[ 128];
        cudaMemcpyAsync(hb, d_out, 512, cudaMemcpyDeviceToHost, sP);
        cudaStreamSynchronize(sP);
        int o=0;
        printf("il:  "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("kf:  "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("t0-8:"); for(int i=0;i<9;i++) printf("%08x",hb[o++]); printf("\n");
        printf("r:   "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("fe_ge(t,N):%u  t8:%u\n", hb[o], hb[o+1]);
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        CUDA_CHECK(cudaFree(d_done)); CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"vec")==0) {
        const char* ents[ 1] = { "00000000000000000000000000000000" };
        u32* d_done;
        CUDA_CHECK(cudaMalloc(&d_done, 4));
        u8* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, 600));
        for (int t=0;t<1;t++) {
            u8 ent[ 16];
            for (int i=0;i<16;i++){ int a,b; char ch1=ents[t][2*i], ch2=ents[t][2*i+1];
                a=(ch1<='9')?ch1-'0':ch1-'a'+10; b=(ch2<='9')?ch2-'0':ch2-'a'+10; ent[i]=(u8)((a<<4)|b); }
            u8* d_ent;
            CUDA_CHECK(cudaMalloc(&d_ent, 16));
            CUDA_CHECK(cudaMemcpy(d_ent, ent, 16, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_done, 0, 4));
            cudaStream_t sK, sP;
            CUDA_CHECK(cudaStreamCreate(&sK));
            CUDA_CHECK(cudaStreamCreate(&sP));
            vec_kernel<<<1,1,0,sK>>>(d_ent, d_out, d_done);
            for (int w=0;w<50;w++) {
                u32 p=0;
                cudaMemcpyAsync(&p, d_done, 4, cudaMemcpyDeviceToHost, sP);
                cudaStreamSynchronize(sP);
                if (p==1) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
            }
            u8 hb[ 600];
            cudaMemcpyAsync(hb, d_out, 600, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            int o=0;
            printf("entropy: "); for(int i=0;i<16;i++) printf("%02x",hb[o++]); printf("\n");
            printf("seed:    "); for(int i=0;i<64;i++) printf("%02x",hb[o++]); printf("\n");
            printf("mK:      "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("mC:      "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("k44:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("c44:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("kCoin:   "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("cCoin:   "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("k0h:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("c0h:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("k0n:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("k0n2:    "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("pubX:    "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("pubY:    "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("addr:    "); for(int i=0;i<20;i++) printf("%02x",hb[o++]); printf("\n");
            printf("mnem:    %s\n", (char*)(hb+o));
            cudaStreamDestroy(sK); cudaStreamDestroy(sP);
            CUDA_CHECK(cudaFree(d_ent));
        }
        CUDA_CHECK(cudaFree(d_done));
        CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"test")==0) {
        const char* ents[ 3] = {
            "00000000000000000000000000000000",
            "7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f",
            "80808080808080808080808080808080"
        };
        u32* d_done;
        CUDA_CHECK(cudaMalloc(&d_done, 4));
        for (int t=0;t<3;t++) {
            u8 ent[ 16];
            for (int i=0;i<16;i++){ int a,b; char ch1=ents[t][2*i], ch2=ents[t][2*i+1];
                a=(ch1<='9')?ch1-'0':ch1-'a'+10; b=(ch2<='9')?ch2-'0':ch2-'a'+10; ent[i]=(u8)((a<<4)|b); }
            u8* d_ent;
            CUDA_CHECK(cudaMalloc(&d_ent, 16));
            CUDA_CHECK(cudaMemcpy(d_ent, ent, 16, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_done, 0, 4));
            cudaStream_t sK, sP;
            CUDA_CHECK(cudaStreamCreate(&sK));
            CUDA_CHECK(cudaStreamCreate(&sP));
            test_kernel<<<1,1,0,sK>>>(d_ent, d_mnem, d_addr, d_priv, d_done);
            for (int w=0;w<40;w++) {
                u32 p=0;
                cudaMemcpyAsync(&p, d_done, 4, cudaMemcpyDeviceToHost, sP);
                cudaStreamSynchronize(sP);
                if (p==1) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
            }
            char mnem[ 128]; u8 addr[ 20], privk[ 32];
            cudaMemcpyAsync(mnem, d_mnem, 128, cudaMemcpyDeviceToHost, sP);
            cudaMemcpyAsync(addr, d_addr, 20, cudaMemcpyDeviceToHost, sP);
            cudaMemcpyAsync(privk, d_priv, 32, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            printf("entropy %s:\n", ents[t]);
            printf("  mnemonic: %s\n", mnem);
            printf("  addr:     0x"); print_hex(addr,20); printf("\n");
            printf("  priv:     0x"); print_hex(privk,32); printf("\n");
            cudaStreamDestroy(sK); cudaStreamDestroy(sP);
            CUDA_CHECK(cudaFree(d_ent));
        }
        CUDA_CHECK(cudaFree(d_done));
    }

    else if (strcmp(mode,"prof")==0) {
        int stage = (argc>=3) ? atoi(argv[2]) : 3;
        int iters = (argc>=4) ? atoi(argv[3]) : 1;
        int blocks = (argc>=5) ? atoi(argv[4]) : 1024;
        u64* d_out; u32* d_done;
        CUDA_CHECK(cudaMalloc(&d_out, 1024*8));
        CUDA_CHECK(cudaMalloc(&d_done, 4));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        cudaEvent_t t0,t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        cudaEventRecord(t0);
        prof_kernel<<<blocks, 256, 0, sK>>>(0, iters, stage, d_out, d_done);
        CUDA_CHECK(cudaGetLastError());
        for (int w=0;w<100;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
        cudaEventRecord(t1, sK);
        cudaEventSynchronize(t1);
        float ms=0; cudaEventElapsedTime(&ms, t0, t1);
        u64 total=(u64)iters*blocks*256;
        printf("prof stage=%d: %llu attempts in %.2f s = %.1f addr/s\n", stage, (unsigned long long)total, ms/1000.0, total/(ms/1000.0));
        CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_done));
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    else if (strcmp(mode,"hd")==0 || strcmp(mode,"hddiag")==0 || strcmp(mode,"hddiag2")==0) {
        fprintf(stderr, "HD mode is not available in the BTC build (vanity/file modes only)\n");
        return 1;
    }
    else if (strcmp(mode,"btctest")==0) {
        u8* d_out; u32* d_done;
        CUDA_CHECK(cudaMalloc(&d_out, 768));
        CUDA_CHECK(cudaMalloc(&d_done, 4));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        btc_test_kernel<<<1,1>>>(d_out, d_done);
        CUDA_CHECK(cudaGetLastError());
        u32 done=0; while (!done) { CUDA_CHECK(cudaMemcpy(&done, d_done, 4, cudaMemcpyDeviceToHost)); }
        u8 hb[ 768]; CUDA_CHECK(cudaMemcpy(hb, d_out, 768, cudaMemcpyDeviceToHost));
        for (int q=0;q<2;q++) {
            int o=q*320;
            printf("N=%d\n", q==0?0:3);
            printf("  priv:      "); print_hex(hb+o, 32); printf("\n");
            printf("  pub_comp:  "); print_hex(hb+o+32, 33); printf("\n");
            printf("  hash160:   "); print_hex(hb+o+65, 20); printf("\n");
            printf("  P2PKH:     %s\n", hb+o+85);
            printf("  P2WPKH:    %s\n", hb+o+149);
            printf("  P2TR:      %s\n", hb+o+229);
        }
        printf("  mnemonic:  %s\n", hb+640);
        CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_done));
    }

    else if (strcmp(mode,"file")==0) {
        if (argc<3) { fprintf(stderr,"usage: file <prefix_file> [addrtype] [nidx] [blocks]\n"); return 1; }
        const char* fn = argv[2];
        int addrtype = (argc>=4) ? atoi(argv[3]) : 3;
        if (addrtype<0||addrtype>3) addrtype=3;
        int nidx = (argc>=5) ? atoi(argv[4]) : 1;
        int blocks = (argc>=6) ? atoi(argv[5]) : 1024;
        int nl_max = btc_nlmax(addrtype);
        FILE* fp = fopen(fn, "r");
        if (!fp) { fprintf(stderr,"cannot open %s\n", fn); return 1; }
        u64 cap = 1<<16, nprefix=0, nskip=0;
        u8* pfx = (u8*)malloc((size_t)(cap*PSTR));
        u8* plen = (u8*)malloc((size_t)cap);
        char line[ 1024 ];
        while (fgets(line, sizeof(line), fp)) {
            int a=0; while (line[a]==' '||line[a]=='\t'||line[a]=='\r'||line[a]=='\n') a++;
            int b=0; while (line[a+b]!='\0' && line[a+b]!='\n' && line[a+b]!='\r' && line[a+b]!='#' && line[a+b]!=' ' && line[a+b]!='\t') b++;
            if (b==0) continue;
            if (b > nl_max) { nskip++; continue; }
            const char* p = line + a;
            for (int i=0;i<b;i++) pfx[nprefix*PSTR+i]=(u8)p[i];
            plen[nprefix]=(u8)b;
            nprefix++;
            if (nprefix >= cap) { cap*=2; pfx=(u8*)realloc(pfx,cap*PSTR); plen=(u8*)realloc(plen,cap); }
        }
        fclose(fp);
        if (nprefix==0) { fprintf(stderr,"no valid prefixes in file\n"); return 1; }
        printf("file mode: read %llu string prefixes (skipped %llu with len>%d)\n", (unsigned long long)nprefix, (unsigned long long)nskip, nl_max);
        run_btc_search(pfx, plen, nprefix, addrtype, nidx, blocks);
        free(pfx); free(plen);
    }
    else if (strcmp(mode,"vanity")==0) {
        if (argc<3) { fprintf(stderr,"usage: vanity <prefix> [addrtype] [nidx] [blocks]\n"); return 1; }
        const char* p = argv[2];
        int addrtype = (argc>=4) ? atoi(argv[3]) : 3;
        if (addrtype<0||addrtype>3) addrtype=3;
        int nidx = (argc>=5) ? atoi(argv[4]) : 1;
        int blocks = (argc>=6) ? atoi(argv[5]) : 1024;
        int nl_max = btc_nlmax(addrtype);
        int len = (int)strlen(p);
        if (len==0) { fprintf(stderr,"bad prefix\n"); return 1; }
        if (len > nl_max) { fprintf(stderr,"prefix too long for %s (max %d chars)\n", btc_aname(addrtype), nl_max); return 1; }
        u8 pfx[ PSTR ]; u8 plen[ 1 ];
        for (int i=0;i<len;i++) pfx[i]=(u8)p[i];
        plen[0]=(u8)len;
        printf("vanity prefix: \"%s\" (%d chars), addr type %s\n", p, len, btc_aname(addrtype));
        run_btc_search(pfx, plen, 1, addrtype, nidx, blocks);
    }
    else { // bench: BTC 地址生成吞吐（不查表）
        int addrtype = (argc>=3) ? atoi(argv[2]) : 3;
        if (addrtype<0||addrtype>3) addrtype=3;
        int iters = (argc>=4) ? atoi(argv[3]) : 5000;
        int blocks = (argc>=5) ? atoi(argv[4]) : 1024;
        u64* d_tbl_key; u32* d_tbl_p; u8* d_tbl_occ;
        u32* d_tbl_off; u32* d_tbl_mask; u32* d_tbl_cnt;
        CUDA_CHECK(cudaMalloc(&d_tbl_key, 8*8));
        CUDA_CHECK(cudaMalloc(&d_tbl_p, 8*4));
        CUDA_CHECK(cudaMalloc(&d_tbl_occ, 8));
        CUDA_CHECK(cudaMalloc(&d_tbl_off, NLMAX*4));
        CUDA_CHECK(cudaMalloc(&d_tbl_mask, NLMAX*4));
        CUDA_CHECK(cudaMalloc(&d_tbl_cnt, NLMAX*4));
        CUDA_CHECK(cudaMemset(d_tbl_occ, 0, 8));
        u8* d_pfx2; CUDA_CHECK(cudaMalloc(&d_pfx2, 1));
        u32* d_found2; u32* d_nfound; ResqEntry* d_resq; u32* d_resq_cnt;
        CUDA_CHECK(cudaMalloc(&d_found2, 4));
        CUDA_CHECK(cudaMalloc(&d_nfound, 4));
        CUDA_CHECK(cudaMalloc(&d_resq, (size_t)(RESQ_CAP*sizeof(ResqEntry))));
        CUDA_CHECK(cudaMalloc(&d_resq_cnt, 4));
        CUDA_CHECK(cudaMemset(d_found2, 0, 4)); CUDA_CHECK(cudaMemset(d_nfound, 0, 4)); CUDA_CHECK(cudaMemset(d_resq_cnt, 0, 4));
        cudaEvent_t t0,t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        multi_hash_kernel<<<blocks, 256>>>(d_tbl_key, d_tbl_p, d_tbl_occ,
                                           d_tbl_off, d_tbl_mask, d_tbl_cnt, d_pfx2, 0,
                                           1, addrtype, 0, 0, iters,
                                           d_found2, d_nfound, d_resq, d_resq_cnt);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        u64 total=(u64)iters*blocks*256;
        printf("bench BTC %s: %llu addresses in %.2f s = %.1f addr/s\n", btc_aname(addrtype),
               (unsigned long long)total, ms/1000.0, total/(ms/1000.0));
        CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
        CUDA_CHECK(cudaFree(d_tbl_key)); CUDA_CHECK(cudaFree(d_tbl_p)); CUDA_CHECK(cudaFree(d_tbl_occ));
        CUDA_CHECK(cudaFree(d_tbl_off)); CUDA_CHECK(cudaFree(d_tbl_mask)); CUDA_CHECK(cudaFree(d_tbl_cnt));
        CUDA_CHECK(cudaFree(d_pfx2));
        CUDA_CHECK(cudaFree(d_found2)); CUDA_CHECK(cudaFree(d_nfound)); CUDA_CHECK(cudaFree(d_resq)); CUDA_CHECK(cudaFree(d_resq_cnt));
    }

    CUDA_CHECK(cudaFree(d_prefix)); CUDA_CHECK(cudaFree(d_mnem));
    CUDA_CHECK(cudaFree(d_addr)); CUDA_CHECK(cudaFree(d_priv)); CUDA_CHECK(cudaFree(d_found));
    CUDA_CHECK(cudaFree(d_index));
    return 0;
}

} // namespace btc
#undef CUDA_CHECK
#undef MAXP
#undef PSTR
#undef NLMAX
#undef RESQ_CAP
#undef RESQ_MASK

// ============================ chain: trx ============================
#define CUDA_CHECK(call) do { cudaError_t err = call; if(err != cudaSuccess) { fprintf(stderr, "CUDA Err: %s\n", cudaGetErrorString(err)); exit(err); } } while(0)
#define MAXP 256
#define PSTR 64
#define NLMAX 62
#define RESQ_N (1<<20)
#define RESQ_MASK (RESQ_N-1)
#define TRX_NLMAX 34
namespace trx {
// trx_vanity_bip39.cu  (v2, 2026-11-02 修订版，基于 1_pi7f)
// BIP39 mnemonic -> PBKDF2 -> BIP32/44 m/44'/195'/0'/0/0 -> 真 secp256k1 -> keccak256 后 20 字节 + 0x41 + Base58Check -> TRON 地址（T 开头 34 字符）
// vanity 前缀比对生成器（真链路，非 BitCrack）
// 字段算术：8x32 位小端 limb，p = 2^256 - 2^32 - 977
//
// v2 改动清单（详见随附说明）：
//  1) 修复 run_trx_search 无限循环的区间推进 bug：start 只按本轮实际启动的 launched 推进（原版用自适应调整后的 batch_iters，会跳过或重复搜索区间）
//  2) 修复命中记录里的派生路径印错：原版 HIT/横幅印 m/44'/0'/0'/0/N，实际派生是 m/44'/195'/0'/0/N（照印错的路径去钱包恢复会找不到币）
//  3) 修复 selftest/fieldtest 跑完后穿透再跑一遍 bench（两条 if 链合并）；comb 预计算后加 cudaDeviceSynchronize
//  4) 安全：计数器熵改为 SHA256(secret(32B) || counter)，secret 由主机端真随机生成/加载（.trx_vanity_secret，0600，绝不打印）
//  5) 性能：热路径 CKD 的 HMAC 改为预计算 ipad/opad 复用，SHA512 压缩 4->2 次/地址
//  6) 规范/健壮：ckd_priv 处理 IL>=n；sha256 通用填充修复 len>=56；tcnt 改用去重后计数；trx_hits.txt chmod 0600；清理多处仍写着 60'/ETH 的旧注释
// 未包含（需在目标 GPU 上实测后再动）：批量求逆（Montgomery）、fe_mul/fe_sqr 重写、comb 表放共享内存


typedef uint32_t u32;
typedef uint64_t u64;
typedef uint8_t u8;
typedef uint16_t u16;
typedef u32 fe[ 8];



// ==========================================
// secp256k1 常数（LE 32 位 limb）
// ==========================================
__constant__ u32 P_FE[ 8] = {0xFFFFFC2Fu,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
__constant__ u32 N_FE[ 8] = {0xD0364141u,0xBFD25E8Cu,0xAF48A03Bu,0xBAAEDCE6u,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
__constant__ u32 C2P256MN[ 8] = {0x2FC9BEBFu,0x402DA173u,0x50B75FC4u,0x45512319u,0x00000001u,0,0,0};
__constant__ u32 GX_FE[ 8] = {0x16F81798u,0x59F2815Bu,0x2DCE28D9u,0x029BFCDBu,0xCE870B07u,0x55A06295u,0xF9DCBBACu,0x79BE667Eu};
__constant__ u32 GY_FE[ 8] = {0xFB10D4B8u,0x9C47D08Fu,0xA6855419u,0xFD17B448u,0x0E1108A8u,0x5DA4FBFCu,0x26A3C465u,0x483ADA77u};
__constant__ u8 SECRET_K[ 32];  // v2: 主机端真随机 secret，由 main() 拷入；与计数器混合决定助记词序列

// ==========================================
// 字段算术（8x32 位，p = 2^256 - 2^32 - 977）
// ==========================================
__device__ __forceinline__ void fe_cpy(u32 r[ 8], const u32 a[ 8]) { for (int i=0;i<8;i++) r[i]=a[i]; }
__device__ __forceinline__ void fe_set0(u32 r[ 8]) { for (int i=0;i<8;i++) r[i]=0; }
__device__ __forceinline__ void fe_set1(u32 r[ 8]) { fe_set0(r); r[ 0]=1; }
__device__ __forceinline__ int fe_iszero(const u32 a[ 8]) { int z=1; for (int i=0;i<8;i++) z &= (a[i]==0); return z; }
__device__ __forceinline__ int fe_isone(const u32 a[ 8]) { int z=(a[ 0]==1); for (int i=1;i<8;i++) z &= (a[i]==0); return z; }
__device__ __forceinline__ int fe_ge(const u32 a[ 8], const u32 b[ 8]) {
    for (int i=7;i>=0;i--) { if (a[i]!=b[i]) return a[i]>b[i]; }
    return 1;
}
__device__ __forceinline__ void fe_sub(u32 r[ 8], const u32 a[ 8], const u32 b[ 8]) {
    u64 bo=0;
    for (int i=0;i<8;i++){ u64 s=(u64)a[i]-b[i]-bo; r[i]=(u32)s; bo=(s>>32)?1:0; }
    if (bo) {
        u64 c=0;
        for (int i=0;i<8;i++){ u64 s=(u64)r[i]+P_FE[i]+c; r[i]=(u32)s; c=s>>32; }
    }
}
// 折叠 10-limb v 的高位 v[8],v[9]（2^256 ≡ 2^32+977），至多 3 次
__device__ __forceinline__ void fe_fold_hi(u32 v[ 10]) {
    for (int it=0; it<3; it++) {
        if ((v[ 8] | v[ 9]) == 0) return;
        u32 h0 = v[ 8], h1 = v[ 9];
        v[ 8] = 0; v[ 9] = 0;
        u64 c = 0;
        for (int i=0;i<10;i++){
            u64 s = c + (i<8 ? (u64)v[i] : 0);
            if (i>=1) {
                if (i-1==0) s += (u64)h0;
                else if (i-1==1) s += (u64)h1;
            }
            if (i==0) s += (u64)h0 * 977u;
            if (i==1) s += (u64)h1 * 977u;
            v[i] = (u32)s;
            c = s >> 32;
        }
        if (c) { u64 s = (u64)v[ 9] + c; v[ 9] = (u32)s; }
    }
}
// r = t mod p，t 为 9 limbs（< 2^257）
__device__ __forceinline__ void fe_reduce9(u32 r[ 8], const u32 t[ 9]) {
    u32 v[ 10];
    u64 c=0;
    for (int i=0;i<10;i++){
        u64 s = c + (i<8 ? (u64)t[i] : 0);
        if (i==1) s += (u64)t[ 8];
        if (i==0) s += (u64)t[ 8]*977u;
        v[i]=(u32)s; c=s>>32;
    }
    if (c){ u64 s=(u64)v[ 9]+c; v[ 9]=(u32)s; }
    fe_fold_hi(v);
    for (int i=0;i<8;i++) r[i]=v[i];
    if (fe_ge(r, P_FE)) fe_sub(r, r, P_FE);
}
__device__ __forceinline__ void fe_add(u32 r[ 8], const u32 a[ 8], const u32 b[ 8]) {
    u64 c=0; u32 t[ 9];
    for (int i=0;i<8;i++){ u64 s=(u64)a[i]+b[i]+c; t[i]=(u32)s; c=s>>32; }
    t[ 8]=(u32)c;
    fe_reduce9(r, t);
}
// r = a * b mod p（32 位 limb 学校式 + 进位链，无 __int128）
__device__ __noinline__ void fe_mul(u32 r[ 8], const u32 a[ 8], const u32 b[ 8]) {
    u32 t[ 16];
    for (int i=0;i<16;i++) t[i]=0;
    for (int i=0;i<8;i++)
        for (int j=0;j<8;j++) {
            u64 p = (u64)a[i]*b[j];
            u64 s = (u64)t[i+j] + (u32)p;
            t[i+j] = (u32)s;
            u64 c = s >> 32;
            int k = i+j+1;
            while (c && k < 16) { u64 s2=(u64)t[k]+c; t[k]=(u32)s2; c=s2>>32; k++; }
            s = (u64)t[i+j+1] + (u32)(p>>32);
            t[i+j+1] = (u32)s;
            c = s >> 32;
            k = i+j+2;
            while (c && k < 16) { u64 s2=(u64)t[k]+c; t[k]=(u32)s2; c=s2>>32; k++; }
        }
    u32 v[ 10];
    u64 c=0;
    for (int i=0;i<10;i++){
        u64 s = c + (i<8 ? (u64)t[i] : 0);
        if (i>=1 && (i-1)<8) s += (u64)t[ 8+(i-1)];
        if (i<8) s += (u64)t[ 8+i]*977u;
        v[i]=(u32)s; c=s>>32;
    }
    if (c){ u64 s=(u64)v[ 9]+c; v[ 9]=(u32)s; }
    fe_fold_hi(v);
    for (int i=0;i<8;i++) r[i]=v[i];
    if (fe_ge(r, P_FE)) fe_sub(r, r, P_FE);
}
__device__ __forceinline__ void fe_sqr(u32 r[ 8], const u32 a[ 8]) { fe_mul(r, a, a); }
// 费马求逆 a^(p-2)
__device__ __forceinline__ void fe_inv(u32 r[ 8], const u32 a[ 8]) {
    u32 e[ 8] = {0xFFFFFC2Du,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
    u32 res[ 8]; fe_set1(res);
    u32 base[ 8]; fe_cpy(base, a);
    for (int i=255;i>=0;i--){
        fe_sqr(res, res);
        if ((e[i>>5] >> (i&31)) & 1) fe_mul(res, res, base);
    }
    fe_cpy(r, res);
}
// 标量加法 mod n（BIP32 CKD：k_new = (IL + k) mod n）
__device__ __forceinline__ void sc_add_mod_n(u32 r[ 8], const u32 a[ 8], const u32 b[ 8]) {
    u64 c=0; u32 t[ 9];
    for (int i=0;i<8;i++){ u64 s=(u64)a[i]+b[i]+c; t[i]=(u32)s; c=s>>32; }
    t[ 8]=(u32)c;
    if (t[ 8]) {
        u64 cc=0;
        for (int i=0;i<8;i++){ u64 s=(u64)t[i]+C2P256MN[i]+cc; r[i]=(u32)s; cc=s>>32; }
    } else if (fe_ge(t, N_FE)) {
        u64 bo=0;
        for (int i=0;i<8;i++){ u64 s=(u64)t[i]-N_FE[i]-bo; r[i]=(u32)s; bo=(s>>32)?1:0; }
    } else {
        for (int i=0;i<8;i++) r[i]=t[i];
    }
}

// ==========================================
// 雅可比坐标点运算
// ==========================================
struct Jacobian { fe X, Y, Z; };
__device__ __forceinline__ void jac_set_inf(Jacobian& P) { fe_set0(P.X); fe_set0(P.Y); fe_set0(P.Z); }
__device__ __forceinline__ int jac_is_inf(const Jacobian& P) { return fe_iszero(P.Z); }

__device__ __forceinline__ void jac_double(Jacobian& R, const Jacobian& P) {
    if (jac_is_inf(P)) { jac_set_inf(R); return; }
    fe A, B, C, D, E, F, t, t2;
    fe_sqr(A, P.X);
    fe_sqr(B, P.Y);
    fe_sqr(C, B);
    fe_add(t, P.X, B); fe_sqr(t, t);
    fe_sub(t, t, A); fe_sub(t, t, C);
    fe_add(D, t, t);
    fe_add(E, A, A); fe_add(E, E, A);
    fe_sqr(F, E);
    fe_add(t, D, D);
    fe_sub(t, F, t);
    fe_sub(t2, D, t);
    fe_mul(t2, t2, E);
    fe_add(F, C, C); fe_add(F, F, F); fe_add(F, F, F);
    fe_sub(t2, t2, F);
    fe_mul(F, P.Y, P.Z); fe_add(F, F, F);
    fe_cpy(R.X, t); fe_cpy(R.Y, t2); fe_cpy(R.Z, F);
}

__device__ __forceinline__ void jac_add_affine(Jacobian& R, const Jacobian& P, const u32 qx[ 8], const u32 qy[ 8]) {
    if (jac_is_inf(P)) { fe_cpy(R.X, qx); fe_cpy(R.Y, qy); fe_set1(R.Z); return; }
    fe Z1Z1, U2, S2, H, Rr, HH, HHH, V, X3, Y3, Z3, t, t2;
    fe_sqr(Z1Z1, P.Z);
    fe_mul(U2, qx, Z1Z1);
    fe_mul(t, P.Z, Z1Z1);
    fe_mul(S2, qy, t);
    fe_sub(H, U2, P.X);
    fe_sub(Rr, S2, P.Y);
    if (fe_iszero(H)) {
        if (fe_iszero(Rr)) { jac_double(R, P); return; }
        else { jac_set_inf(R); return; }
    }
    fe_sqr(HH, H);
    fe_mul(HHH, H, HH);
    fe_mul(V, P.X, HH);
    fe_sqr(X3, Rr);
    fe_sub(X3, X3, HHH);
    fe_add(t, V, V);
    fe_sub(X3, X3, t);
    fe_sub(t, V, X3);
    fe_mul(t, t, Rr);
    fe_mul(t2, P.Y, HHH);
    fe_sub(Y3, t, t2);
    fe_mul(Z3, P.Z, H);
    fe_cpy(R.X, X3); fe_cpy(R.Y, Y3); fe_cpy(R.Z, Z3);
}

// k * G（从 MSB 到 LSB 的 double-and-add）
__device__ void scalar_mult_base(Jacobian& R, const u32 k[ 8]) {
    jac_set_inf(R);
    for (int i=255;i>=0;i--){
        jac_double(R, R);
        if ((k[i>>5] >> (i&31)) & 1) jac_add_affine(R, R, GX_FE, GY_FE);
    }
}


__device__ void jac_to_affine(u32 x[ 8], u32 y[ 8], const Jacobian& P) {
    fe zi, zi2, zi3;
    fe_inv(zi, P.Z);
    fe_sqr(zi2, zi);
    fe_mul(zi3, zi2, zi);
    fe_mul(x, P.X, zi2);
    fe_mul(y, P.Y, zi3);
}
__device__ const u32* g_comb;

// ==========================================
// 4-tooth comb 固定基点标量乘
// T[t][j] = j * 2^(64t) * G, t=0..3, j=1..15 （仿射坐标，16 u32/点）
// 256 次 double -> 64 次 double + ~64 次 affine add
// ==========================================
__global__ void precompute_comb_kernel(u32* d_table) {
    int t = threadIdx.x;      // 0..3
    int j = blockIdx.x + 1;   // 1..15
    if (t > 3) return;
    u32 kf[ 8] = {0,0,0,0,0,0,0,0};
    // k = j * 2^(64t)：j 放入 limb 2t（j<16 不溢出）
    kf[ 2*t] = (u32)j;
    Jacobian P;
    jac_set_inf(P);
    for (int i=255;i>=0;i--){
        jac_double(P, P);
        if ((kf[i>>5]>>(i&31))&1) jac_add_affine(P, P, GX_FE, GY_FE);
    }
    u32 x[ 8], y[ 8];
    jac_to_affine(x, y, P);
    u32* dst = d_table + ((t*15 + (j-1)) * 16);
    for (int i=0;i<8;i++){ dst[i]=x[i]; dst[8+i]=y[i]; }
}

__device__ __forceinline__ void scalar_mult_base_comb(Jacobian& R, const u32 k[ 8]) {
    jac_set_inf(R);
    const u32* T = g_comb;
    for (int r=15; r>=0; r--) {
        jac_double(R,R); jac_double(R,R); jac_double(R,R); jac_double(R,R);
        for (int t=0;t<4;t++) {
            int shift = 64*t + 4*r;
            int d = (k[shift>>5] >> (shift&31)) & 0xF;
            if (d) {
                const u32* p = T + ((t*15 + (d-1)) * 16);
                jac_add_affine(R, R, p, p+8);
            }
        }
    }
}


// ==========================================
// 字节 / limb 转换
// ==========================================
__device__ __forceinline__ void bytes32_to_fe(u32 f[ 8], const u8 b[ 32]) {
    for (int i=0;i<8;i++) f[i] = ((u32)b[28-4*i]<<24)|((u32)b[29-4*i]<<16)|((u32)b[30-4*i]<<8)|b[31-4*i];
}
__device__ __forceinline__ void fe_to_bytes32(u8 b[ 32], const u32 f[ 8]) {
    for (int i=0;i<8;i++){ u32 v=f[i]; b[28-4*i]=(u8)(v>>24); b[29-4*i]=(u8)(v>>16); b[30-4*i]=(u8)(v>>8); b[31-4*i]=(u8)v; }
}
__device__ __forceinline__ int device_strlen(const char* s) { int l=0; while (s[l]) l++; return l; }

// ==========================================
// SHA-256（BIP39 校验和 / vanity 熵）
// ==========================================
__constant__ u32 K256[ 64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
__device__ __forceinline__ u32 rotr32(u32 x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ void sha256_block(u32 h[ 8], const u8 blk[ 64]) {
    u32 w[ 64];
    for (int i=0;i<16;i++) w[i] = ((u32)blk[i*4]<<24)|((u32)blk[i*4+1]<<16)|((u32)blk[i*4+2]<<8)|blk[i*4+3];
    for (int i=16;i<64;i++){
        u32 s0 = rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);
        u32 s1 = rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    u32 a=h[ 0],b=h[ 1],c=h[ 2],d=h[ 3],e=h[ 4],f=h[ 5],g=h[ 6],hh=h[ 7];
    for (int i=0;i<64;i++){
        u32 S1 = rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
        u32 ch = (e&f)^((~e)&g);
        u32 t1 = hh + S1 + ch + K256[i] + w[i];
        u32 S0 = rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
        u32 maj = (a&b)^(a&c)^(b&c);
        u32 t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[ 0]+=a; h[ 1]+=b; h[ 2]+=c; h[ 3]+=d; h[ 4]+=e; h[ 5]+=f; h[ 6]+=g; h[ 7]+=hh;
}
__device__ void sha256(const u8* data, int len, u8 out[ 32]) {
    u32 h[ 8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    u8 blk[ 64];
    int off = 0;
    while (len - off >= 64) { for (int i=0;i<64;i++) blk[i]=data[off+i]; sha256_block(h, blk); off += 64; }
    for (int i=0;i<64;i++) blk[i]=0;
    int rem = len - off;
    for (int i=0;i<rem;i++) blk[i]=data[off+i];
    blk[rem] = 0x80;
    u64 bits = (u64)len * 8;
    if (rem >= 56) {  // v2: 剩余 >=56 字节时长度字段放不进同一块，先冲掉这一块再补长度块（原版此情况算错；旧调用 len<=40 未触发；RIPEMD160 的 sha256d 走本函数 len=32/21 亦未触发）
        sha256_block(h, blk);
        for (int i=0;i<64;i++) blk[i]=0;
    }
    for (int i=0;i<8;i++) blk[56+i] = (u8)((bits >> (56-8*i)) & 0xFF);
    sha256_block(h, blk);
    for (int i=0;i<8;i++){ out[i*4]=(u8)(h[i]>>24); out[i*4+1]=(u8)(h[i]>>16); out[i*4+2]=(u8)(h[i]>>8); out[i*4+3]=(u8)h[i]; }
}

// v2: 计数器 -> 熵。ent = SHA256(SECRET_K(32B) || counter 小端 8B) 的前 16 字节。
// 没有 secret 的人无法复现/枚举这条助记词序列（原版仅 SHA256(计数器)，等效熵只有计数器本身且顺序可猜）。
__device__ __forceinline__ void counter_to_entropy(u64 n, u8 ent[ 16]) {
    u8 buf[ 40];
    for (int i=0;i<32;i++) buf[i]=SECRET_K[i];
    for (int i=0;i<8;i++) buf[32+i]=(u8)(n>>(8*i));
    u8 h[ 32]; sha256(buf, 40, h);
    for (int i=0;i<16;i++) ent[i]=h[i];
}

// ==========================================
// SHA-512 / HMAC-SHA512（移植自优化版）
// ==========================================
__device__ __constant__ u64 sha512_k[ 80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};
__device__ __forceinline__ u64 rotr64(u64 x, int n) { return (x >> n) | (x << (64 - n)); }
__device__ __forceinline__ u64 Ch(u64 x, u64 y, u64 z) { return (x & y) ^ (~x & z); }
__device__ __forceinline__ u64 Maj(u64 x, u64 y, u64 z) { return (x & y) ^ (x & z) ^ (y & z); }
__device__ __forceinline__ u64 Sigma0(u64 x) { return rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39); }
__device__ __forceinline__ u64 Sigma1(u64 x) { return rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41); }
__device__ __forceinline__ u64 sigma0(u64 x) { return rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7); }
__device__ __forceinline__ u64 sigma1(u64 x) { return rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6); }
__device__ __forceinline__ void sha512_init(u64 st[ 8]) {
    st[ 0]=0x6a09e667f3bcc908ULL; st[ 1]=0xbb67ae8584caa73bULL;
    st[ 2]=0x3c6ef372fe94f82bULL; st[ 3]=0xa54ff53a5f1d36f1ULL;
    st[ 4]=0x510e527fade682d1ULL; st[ 5]=0x9b05688c2b3e6c1fULL;
    st[ 6]=0x1f83d9abfb41bd6bULL; st[ 7]=0x5be0cd19137e2179ULL;
}
__device__ __forceinline__ void sha512_block_words(u64 st[ 8], const u64 blockW[ 16]) {
    u64 W[ 80];
    #pragma unroll
    for (int i = 0; i < 16; i++) W[i] = blockW[i];
    #pragma unroll
    for (int i = 16; i < 80; i++) W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16];
    u64 a=st[ 0], b=st[ 1], c=st[ 2], d=st[ 3], e=st[ 4], f=st[ 5], g=st[ 6], h=st[ 7];
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        u64 T1 = h + Sigma1(e) + Ch(e,f,g) + sha512_k[i] + W[i];
        u64 T2 = Sigma0(a) + Maj(a,b,c);
        h=g; g=f; f=e; e=d + T1; d=c; c=b; b=a; a=T1 + T2;
    }
    st[ 0]+=a; st[ 1]+=b; st[ 2]+=c; st[ 3]+=d; st[ 4]+=e; st[ 5]+=f; st[ 6]+=g; st[ 7]+=h;
}
__device__ __forceinline__ void load_be_words(const u8 blk[ 128], u64 W[ 16]) {
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        u64 v = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) v = (v << 8) | blk[i*8 + j];
        W[i] = v;
    }
}
__device__ __forceinline__ void hmac_sha512_precompute(const u8* key, int key_len,
                                                       u64 ipad_st[ 8], u64 opad_st[ 8]) {
    u8 real_key[ 128] = {0};
    u8 kblk[ 128];
    if (key_len > 128) {
        u64 st[ 8]; sha512_init(st);
        int off = 0;
        while (off + 128 <= key_len) {
            u64 W[ 16]; load_be_words(key + off, W);
            sha512_block_words(st, W); off += 128;
        }
        u8 pad[ 128] = {0};
        int rem = key_len - off;
        for (int i = 0; i < rem; i++) pad[i] = key[off + i];
        pad[rem] = 0x80;
        u64 bits = (u64)key_len * 8;
        for (int i = 0; i < 8; i++) pad[ 120 + i] = (bits >> (56 - 8*i)) & 0xFF;
        u64 W2[ 16]; load_be_words(pad, W2);
        sha512_block_words(st, W2);
        for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) real_key[i*8+j] = (st[i] >> (56 - 8*j)) & 0xFF;
    } else {
        for (int i = 0; i < key_len; i++) real_key[i] = key[i];
    }
    u64 W[ 16];
    for (int i = 0; i < 128; i++) kblk[i] = real_key[i] ^ 0x36;
    load_be_words(kblk, W);
    sha512_init(ipad_st); sha512_block_words(ipad_st, W);
    for (int i = 0; i < 128; i++) kblk[i] = real_key[i] ^ 0x5c;
    load_be_words(kblk, W);
    sha512_init(opad_st); sha512_block_words(opad_st, W);
}
// 通用 HMAC-SHA512（msg_len <= 64，key_len <= 128）
__device__ __forceinline__ void hmac_sha512_gen(const u8* key, int key_len, const u8* msg, int msg_len, u8 out[ 64]) {
    u64 ipad_st[ 8], opad_st[ 8];
    hmac_sha512_precompute(key, key_len, ipad_st, opad_st);
    u8 blk[ 128];
    for (int i=0;i<128;i++) blk[i]=0;
    for (int i=0;i<msg_len && i<128;i++) blk[i]=msg[i];
    blk[msg_len]=0x80;
    u64 bits=(u64)(128+msg_len)*8;
    for (int i=0;i<8;i++) blk[120+i]=(u8)((bits>>(56-8*i))&0xFF);
    u64 W[ 16], st[ 8], ih[ 8];
    load_be_words(blk, W);
    for (int i=0;i<8;i++) st[i]=ipad_st[i];
    sha512_block_words(st, W);
    for (int i=0;i<8;i++) ih[i]=st[i];
    for (int i=0;i<128;i++) blk[i]=0;
    for (int i=0;i<64;i++) blk[i]=(u8)((ih[i>>3]>>(56-8*(i&7)))&0xFF);
    blk[64]=0x80;
    bits=(128+64)*8;
    for (int i=0;i<8;i++) blk[120+i]=(u8)((bits>>(56-8*i))&0xFF);
    load_be_words(blk, W);
    for (int i=0;i<8;i++) st[i]=opad_st[i];
    sha512_block_words(st, W);
    for (int i=0;i<8;i++) for (int j=0;j<8;j++) out[i*8+j]=(u8)((st[i]>>(56-8*j))&0xFF);
}
// v2: 用预计算好的 ipad/opad 状态做 HMAC-SHA512（要求 msg_len<=119），省去每次调用重算密钥填充的 2 次压缩。
// 热路径（CKD 的链码固定）应先 hmac_sha512_precompute 一次，再反复调本函数。
__device__ __forceinline__ void hmac_sha512_precomp_msg(const u64 ipad_st[ 8], const u64 opad_st[ 8], const u8* msg, int msg_len, u8 out[ 64]) {
    u8 blk[ 128];
    for (int i=0;i<128;i++) blk[i]=0;
    for (int i=0;i<msg_len;i++) blk[i]=msg[i];
    blk[msg_len]=0x80;
    u64 bits=(u64)(128+msg_len)*8;
    for (int i=0;i<8;i++) blk[120+i]=(u8)((bits>>(56-8*i))&0xFF);
    u64 W[ 16], st[ 8], ih[ 8];
    load_be_words(blk, W);
    for (int i=0;i<8;i++) st[i]=ipad_st[i];
    sha512_block_words(st, W);
    for (int i=0;i<8;i++) ih[i]=st[i];
    for (int i=0;i<128;i++) blk[i]=0;
    for (int i=0;i<64;i++) blk[i]=(u8)((ih[i>>3]>>(56-8*(i&7)))&0xFF);
    blk[64]=0x80;
    bits=(128+64)*8;
    for (int i=0;i<8;i++) blk[120+i]=(u8)((bits>>(56-8*i))&0xFF);
    load_be_words(blk, W);
    for (int i=0;i<8;i++) st[i]=opad_st[i];
    sha512_block_words(st, W);
    for (int i=0;i<8;i++) for (int j=0;j<8;j++) out[i*8+j]=(u8)((st[i]>>(56-8*j))&0xFF);
}
// PBKDF2 首次迭代：salt = "mnemonic" + 0x00000001
__device__ __forceinline__ void hmac_sha512_salt(const u64 ipad_st[ 8], const u64 opad_st[ 8], u64 out[ 8]) {
    const u64 saltW[ 16] = {
        0x6D6E656D6F6E6963ULL, 0x0000000180000000ULL,
        0,0,0,0,0,0,0,0,0,0,0,0, 0, 0x0000000000000460ULL
    };
    u64 st[ 8], W[ 16], ih[ 8];
    for (int i = 0; i < 8; i++) st[i] = ipad_st[i];
    for (int i = 0; i < 16; i++) W[i] = saltW[i];
    sha512_block_words(st, W);
    for (int i = 0; i < 8; i++) ih[i] = st[i];
    for (int i = 0; i < 8; i++) st[i] = opad_st[i];
    for (int i = 0; i < 8; i++) W[i] = ih[i];
    W[ 8]  = 0x8000000000000000ULL;
    W[ 9]  = 0; W[ 10] = 0; W[ 11] = 0; W[ 12] = 0; W[ 13] = 0; W[ 14] = 0;
    W[ 15] = 0x0000000000000600ULL;
    sha512_block_words(st, W);
    for (int i = 0; i < 8; i++) out[i] = st[i];
}
// HMAC 一步：消息 = 完整 64B 块 msg
__device__ __forceinline__ void hmac_sha512_64(const u64 ipad_st[ 8], const u64 opad_st[ 8],
                                               const u64 msg[ 8], u64 out[ 8]) {
    u64 st[ 8], W[ 16], ih[ 8];
    for (int i = 0; i < 8; i++) st[i] = ipad_st[i];
    for (int i = 0; i < 8; i++) W[i] = msg[i];
    W[ 8]  = 0x8000000000000000ULL;
    W[ 9]  = 0; W[ 10] = 0; W[ 11] = 0; W[ 12] = 0; W[ 13] = 0; W[ 14] = 0;
    W[ 15] = 0x0000000000000600ULL;
    sha512_block_words(st, W);
    for (int i = 0; i < 8; i++) ih[i] = st[i];
    for (int i = 0; i < 8; i++) st[i] = opad_st[i];
    for (int i = 0; i < 8; i++) W[i] = ih[i];
    W[ 8]  = 0x8000000000000000ULL;
    W[ 9]  = 0; W[ 10] = 0; W[ 11] = 0; W[ 12] = 0; W[ 13] = 0; W[ 14] = 0;
    W[ 15] = 0x0000000000000600ULL;
    sha512_block_words(st, W);
    for (int i = 0; i < 8; i++) out[i] = st[i];
}
// 完整 BIP39 Seed 派生（2048 次 PBKDF2-HMAC-SHA512）
__device__ void device_bip39_seed(const char* mnemonic, u8 seed[ 64]) {
    int pass_len = device_strlen(mnemonic);
    u64 ipad_st[ 8], opad_st[ 8];
    hmac_sha512_precompute((const u8*)mnemonic, pass_len, ipad_st, opad_st);
    u64 U[ 8], T[ 8];
    hmac_sha512_salt(ipad_st, opad_st, U);
    for (int i = 0; i < 8; i++) T[i] = U[i];
    for (int i = 1; i < 2048; i++) {
        hmac_sha512_64(ipad_st, opad_st, U, U);
        for (int j = 0; j < 8; j++) T[j] ^= U[j];
    }
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) seed[i*8 + j] = (T[i] >> (56 - 8*j)) & 0xFF;
}

// ==========================================
// Keccak-256（地址）
// ==========================================
__device__ __constant__ u64 keccak_rc[ 24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};
__device__ __forceinline__ u64 rotl64b(u64 x, int n) { n &= 63; return n ? ((x << n) | (x >> (64 - n))) : x; }
__device__ void keccak_f1600(u64 state[ 25]) {
    u64 C[ 5], D[ 5], B[ 25];
    for (int round=0; round<24; round++) {
        for (int i=0;i<5;i++) C[i]=state[i]^state[i+5]^state[i+10]^state[i+15]^state[i+20];
        for (int i=0;i<5;i++) { D[i]=C[(i+4)%5]^rotl64b(C[(i+1)%5],1); for (int j=0;j<25;j+=5) state[j+i]^=D[i]; }
        for (int i=0;i<25;i++) B[i]=state[i];
        int x=1, y=0; u64 cur=B[ 1];
        for (int t=0;t<24;t++) {
            int tmp=x; x=y; y=(2*tmp+3*y)%5;
            u64 tv=B[x+5*y]; B[x+5*y]=rotl64b(cur,(t+1)*(t+2)/2); cur=tv;
        }
        for (int i=1;i<25;i++) state[i]=B[i];
        for (int j=0;j<25;j+=5) {
            for (int i=0;i<5;i++) B[j+i]=state[j+i];
            for (int i=0;i<5;i++) state[j+i]^=(~B[j+(i+1)%5])&B[j+(i+2)%5];
        }
        state[ 0]^=keccak_rc[round];
    }
}
__device__ void device_keccak256(const u8* data, int len, u8 hash[ 32]) {
    u64 state[ 25] = {0}; int rate=136; int off=0;
    while (off+rate<=len) {
        for (int i=0;i<rate/8;i++) {
            u64 val=0;
            for (int j=0;j<8;j++) val |= ((u64)data[off+i*8+j])<<(8*j);
            state[i]^=val;
        }
        keccak_f1600(state); off+=rate;
    }
    u8 blk[ 136] = {0}; int rem=len-off;
    if (rem>0) for (int i=0;i<rem;i++) blk[i]=data[off+i];
    blk[rem]=0x01; blk[rate-1]|=0x80;
    for (int i=0;i<rate/8;i++) {
        u64 val=0;
        for (int j=0;j<8;j++) val |= ((u64)blk[i*8+j])<<(8*j);
        state[i]^=val;
    }
    keccak_f1600(state);
    for (int i=0;i<4;i++) for (int j=0;j<8;j++) hash[i*8+j]=(u8)((state[i]>>(8*j))&0xFF);
}

// ==========================================
// BIP39 编码：128bit 熵 + 4bit 校验和 -> 12 词
// ==========================================
__device__ void entropy_to_mnemonic(const u8 ent[ 16], char* mnem, int* mnem_len) {
    u8 h[ 32];
    sha256(ent, 16, h);
    u16 idx[ 12];
    for (int w=0;w<12;w++) {
        int bp = w*11; u32 val=0;
        for (int b=0;b<11;b++) {
            int p=bp+b; int byte=p>>3, sh=7-(p&7);
            int bit = (byte<16) ? ((ent[byte]>>sh)&1) : ((h[ 0]>>sh)&1);
            val=(val<<1)|bit;
        }
        idx[w]=(u16)val;
    }
    int pos=0;
    for (int w=0;w<12;w++) {
        const char* word = BIP39_WORDS[idx[w]];
        int len=0; while (word[len]) len++;
        for (int i=0;i<len;i++) mnem[pos++]=word[i];
        if (w<11) mnem[pos++]=' ';
    }
    mnem[pos]=0;
    if (mnem_len) *mnem_len=pos;
}

// ==========================================
// BIP32 / BIP44 派生：m/44'/195'/0'/0/0
// ==========================================
__device__ void ckd_priv(u8 k[ 32], u8 c[ 32], u32 index, int hardened) {
    u8 data[ 37], out[ 64];
    if (hardened) {
        data[ 0]=0;
        for (int i=0;i<32;i++) data[1+i]=k[i];
    } else {
        u32 kf[ 8]; bytes32_to_fe(kf, k);
        Jacobian P; scalar_mult_base_comb(P, kf);
        u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
        u8 xb[ 32]; fe_to_bytes32(xb, x);
        data[ 0] = (y[ 0]&1) ? 0x03 : 0x02;
        for (int i=0;i<32;i++) data[1+i]=xb[i];
    }
    data[33]=(u8)(index>>24); data[34]=(u8)(index>>16); data[35]=(u8)(index>>8); data[36]=(u8)index;
    hmac_sha512_gen(c, 32, data, 37, out);
    u32 il[ 8], kf[ 8], kf2[ 8];
    bytes32_to_fe(il, out);
    bytes32_to_fe(kf, k);
    // v2(BIP32 规范): IL>=n 时该索引无效（概率约 2^-127）。置子钥为 0 使其永不命中，等效丢弃。
    if (fe_ge(il, N_FE)) fe_set0(kf2);
    else sc_add_mod_n(kf2, il, kf);
    fe_to_bytes32(k, kf2);
    for (int i=0;i<32;i++) c[i]=out[32+i];
}
__device__ void bip32_derive(u8 privk[ 32], const u8 seed[ 64]) {
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    ckd_priv(k, c, 0x80000000u+44u, 1);
    ckd_priv(k, c, 0x80000000u+195u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    ckd_priv(k, c, 0u, 0);
    ckd_priv(k, c, 0u, 0);
    for (int i=0;i<32;i++) privk[i]=k[i];
}
__device__ void derive_address(const u8 privk[ 32], u8 addr[ 20]) {
    u32 kf[ 8]; bytes32_to_fe(kf, privk);
    Jacobian P; scalar_mult_base_comb(P, kf);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    u8 pub[ 64];
    fe_to_bytes32(pub, x);
    fe_to_bytes32(pub+32, y);
    u8 h[ 32];
    device_keccak256(pub, 64, h);
    for (int i=0;i<20;i++) addr[i]=h[12+i];
}
__device__ void derive_full(const u8 ent[ 16], char* mnem, u8 addr[ 20], u8 privk[ 32]) {
    char m[ 128];
    entropy_to_mnemonic(ent, m, 0);
    u8 seed[ 64];
    device_bip39_seed(m, seed);
    bip32_derive(privk, seed);
    derive_address(privk, addr);
    if (mnem) { int l=0; while (m[l]){ mnem[l]=m[l]; l++; } mnem[l]=0; }
}

// ==========================================
// Kernel：vec 诊断（逐层输出中间值）
// ==========================================
__global__ void vec_kernel(const u8* d_ent, u8* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 ent[ 16];
    for (int i=0;i<16;i++) ent[i]=d_ent[i];
    char m[ 128];
    entropy_to_mnemonic(ent, m, 0);
    int o=0;
    for (int i=0;i<16;i++) d_out[o++]=ent[i];
    u8 seed[ 64];
    device_bip39_seed(m, seed);
    for (int i=0;i<64;i++) d_out[o++]=seed[i];
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    for (int i=0;i<32;i++) d_out[o++]=c[i];
    ckd_priv(k, c, 0x80000000u+44u, 1);
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    for (int i=0;i<32;i++) d_out[o++]=c[i];
    ckd_priv(k, c, 0x80000000u+195u, 1);
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    for (int i=0;i<32;i++) d_out[o++]=c[i];
    ckd_priv(k, c, 0x80000000u+0u, 1);
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    for (int i=0;i<32;i++) d_out[o++]=c[i];
    ckd_priv(k, c, 0u, 0);
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    ckd_priv(k, c, 0u, 0);
    for (int i=0;i<32;i++) d_out[o++]=k[i];
    u32 kf[ 8]; bytes32_to_fe(kf, k);
    Jacobian P; scalar_mult_base(P, kf);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    u8 xb[ 32], yb[ 32];
    fe_to_bytes32(xb, x); fe_to_bytes32(yb, y);
    for (int i=0;i<32;i++) d_out[o++]=xb[i];
    for (int i=0;i<32;i++) d_out[o++]=yb[i];
    u8 pub[ 64];
    for (int i=0;i<32;i++) pub[i]=xb[i];
    for (int i=0;i<32;i++) pub[32+i]=yb[i];
    u8 h[ 32];
    device_keccak256(pub, 64, h);
    for (int i=0;i<20;i++) d_out[o++]=h[12+i];
    int l=0; while (m[l]){ d_out[o++]=m[l]; l++; }
    d_out[o]=0;
    d_done[ 0]=1;
}

// ==========================================
// Kernel：selftest
// ==========================================
__global__ void selftest_kernel(u64* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu;
    fe b; fe_set1(b);
    fe r; fe_add(r, a, b);
    u64 v0=0; for (int i=7;i>=0;i--) v0=(v0<<32)|r[i];
    d_out[ 0]=v0;
    fe m1; fe_set0(m1); m1[ 0]=123456789u;
    fe m2; fe_set0(m2); m2[ 0]=987654321u;
    fe_mul(r, m1, m2);
    u64 v1=0; for (int i=7;i>=0;i--) v1=(v1<<32)|r[i];
    d_out[ 1]=v1;
    fe two; fe_set0(two); two[ 0]=2u;
    Jacobian P2; scalar_mult_base(P2, two);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P2);
    u64 v2x=0, v2y=0; for (int i=7;i>=0;i--){ v2x=(v2x<<32)|x[i]; v2y=(v2y<<32)|y[i]; }
    d_out[ 2]=v2x; d_out[ 3]=v2y;
    fe one; fe_set1(one);
    Jacobian P1; scalar_mult_base(P1, one);
    u32 x1[ 8], y1[ 8]; jac_to_affine(x1, y1, P1);
    u64 v3x=0, v3y=0; for (int i=7;i>=0;i--){ v3x=(v3x<<32)|x1[i]; v3y=(v3y<<32)|y1[i]; }
    d_out[ 4]=v3x; d_out[ 5]=v3y;
    u32 nm1[ 8]; fe_cpy(nm1, N_FE); nm1[ 0]-=1;
    u32 one32[ 8]; fe_set1(one32);
    u32 sr[ 8]; sc_add_mod_n(sr, nm1, one32);
    u64 v4=0; for (int i=7;i>=0;i--) v4=(v4<<32)|sr[i];
    d_out[ 6]=v4;
    u32 nm2[ 8]; fe_cpy(nm2, N_FE); nm2[ 0]-=2;
    u32 three[ 8]; fe_set0(three); three[ 0]=3;
    sc_add_mod_n(sr, nm2, three);
    u64 v5=0; for (int i=7;i>=0;i--) v5=(v5<<32)|sr[i];
    d_out[ 7]=v5;
    Jacobian P3a; jac_add_affine(P3a, P2, GX_FE, GY_FE);
    u32 x3[ 8], y3[ 8]; jac_to_affine(x3, y3, P3a);
    fe three3; fe_set0(three3); three3[ 0]=3;
    Jacobian P3b; scalar_mult_base(P3b, three3);
    u32 x3b[ 8], y3b[ 8]; jac_to_affine(x3b, y3b, P3b);
    u64 v6=0; for (int i=7;i>=0;i--) v6=(v6<<32)|x3[i];
    u64 v6b=0; for (int i=7;i>=0;i--) v6b=(v6b<<32)|x3b[i];
    d_out[ 8]=v6; d_out[ 9]=v6b;
    fe inv; fe_inv(inv, m1);
    fe_mul(r, m1, inv);
    u64 v7=0; for (int i=7;i>=0;i--) v7=(v7<<32)|r[i];
    d_out[ 10]=v7;
    d_done[ 0]=1;
}

// ==========================================
// Kernel：jac_double(G) 中间量诊断
// ==========================================
__global__ void jdbl_kernel(u32* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    fe A, B, C, D, E, F, t, t2, F0, t0;
    fe_sqr(A, GX_FE);
    fe_sqr(B, GY_FE);
    fe_sqr(C, B);
    fe_add(t, GX_FE, B); fe_sqr(t, t);
    fe_sub(t, t, A); fe_sub(t, t, C);
    fe_add(D, t, t);
    fe_add(E, A, A); fe_add(E, E, A);
    fe_sqr(F, E);
    fe_cpy(F0, F);
    fe_add(t, D, D);
    fe_cpy(t0, t);
    fe_sub(t, F0, t0);
    fe_sub(t2, D, t);
    fe_mul(t2, t2, E);
    fe_add(F, C, C); fe_add(F, F, F); fe_add(F, F, F);
    fe_sub(t2, t2, F);
    fe_mul(F, GY_FE, GX_FE); fe_add(F, F, F);
    int o=0;
    for (int s=0;s<8;s++) d_out[o++]=A[s];
    for (int s=0;s<8;s++) d_out[o++]=B[s];
    for (int s=0;s<8;s++) d_out[o++]=C[s];
    for (int s=0;s<8;s++) d_out[o++]=D[s];
    for (int s=0;s<8;s++) d_out[o++]=E[s];
    for (int s=0;s<8;s++) d_out[o++]=F0[s];  // E^2
    for (int s=0;s<8;s++) d_out[o++]=t0[s];  // 2D
    for (int s=0;s<8;s++) d_out[o++]=t[s];   // X3
    for (int s=0;s<8;s++) d_out[o++]=t2[s];  // Y3
    for (int s=0;s<8;s++) d_out[o++]=F[s];   // Z3
    d_done[ 0]=1;
}

// ==========================================
// Kernel：最小化 scalar_mult_base(k0h) 诊断
// ==========================================

// ==========================================
// Kernel: keccak test
// ==========================================
__global__ void ktest_kernel(u8* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 h[ 32];
    device_keccak256((const u8*)"", 0, h);
    for (int i=0;i<32;i++) d_out[i]=h[i];
    device_keccak256((const u8*)"abc", 3, h);
    for (int i=0;i<32;i++) d_out[32+i]=h[i];
    const u8 pub[ 64] = {
        0x37,0xb0,0xbb,0x7a,0x82,0x88,0xd3,0x8e,0xd4,0x9a,0x52,0x4b,0x5d,0xc9,0x8c,0xff,
        0x3e,0xb5,0xca,0x82,0x4c,0x9f,0x9d,0xc0,0xdf,0xdb,0x3d,0x9c,0xd6,0x00,0xf2,0x99,
        0xa6,0x17,0x99,0x12,0xb7,0x45,0x1c,0x09,0x89,0x6c,0x40,0x98,0xec,0xa7,0xce,0x6b,
        0x2e,0x58,0x33,0x06,0x72,0x79,0x5e,0x84,0x7c,0x4d,0x6a,0xf4,0x4e,0x02,0x42,0x30 };
    device_keccak256(pub, 64, h);
    for (int i=0;i<32;i++) d_out[64+i]=h[i];
    u64 st[ 25] = {0};
    keccak_f1600(st);
    for (int i=0;i<8;i++) d_out[96+i]=(u8)((st[ 0]>>(8*i))&0xFF);
    d_done[ 0]=1;
}

__global__ void sm_kernel(u32* d_out, u32* d_done, int c) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u32 kf[ 8] = {0,0,0,0,0,0,0,0};
    if (c==0) { // k0h
        const u8 kh[ 32] = { 0xe9,0xb9,0x1c,0x42,0x8e,0x67,0x9f,0x85,0x7f,0xbf,0xa0,0x20,0x34,0x40,0xea,0xea,
                             0xab,0xb0,0xca,0x46,0xd8,0x0e,0x70,0xb6,0xaa,0x62,0x50,0x69,0x6a,0xc0,0xdf,0xc7 };
        bytes32_to_fe(kf, kh);
    } else if (c==1) { kf[ 7]=0x80000000u; } // 2^255
    else if (c==2) { kf[ 7]=0xe9b91c42u; }  // top 32 bits of k0h
    else if (c==3) { kf[ 0]=0xe9u; }        // 0xe9
    else if (c==4) { kf[ 7]=0x00000080u; }  // 2^247
    else if (c==5) { kf[ 0]=0x0au; }        // 10G
    Jacobian P; scalar_mult_base(P, kf);
    if (c==1 || c==5) {
        for (int i=0;i<8;i++) d_out[i]=P.X[i];
        for (int i=0;i<8;i++) d_out[8+i]=P.Y[i];
        for (int i=0;i<8;i++) d_out[16+i]=P.Z[i];
        d_done[ 0]=1; return;
    }
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    for (int i=0;i<8;i++) d_out[i]=x[i];
    for (int i=0;i<8;i++) d_out[8+i]=y[i];
    d_done[ 0]=1;
}

// ==========================================
// Kernel：ptest 点运算诊断（2G/3G/pubkey(k0h)）
// ==========================================
__global__ void ptest_kernel(u32* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    int o=0;
    fe two; fe_set0(two); two[ 0]=2u;
    Jacobian P2; scalar_mult_base(P2, two);
    u32 x2[ 8], y2[ 8]; jac_to_affine(x2, y2, P2);
    for (int i=7;i>=0;i--) d_out[o++]=x2[i];
    for (int i=7;i>=0;i--) d_out[o++]=y2[i];
    fe three; fe_set0(three); three[ 0]=3u;
    Jacobian P3; scalar_mult_base(P3, three);
    u32 x3[ 8], y3[ 8]; jac_to_affine(x3, y3, P3);
    for (int i=7;i>=0;i--) d_out[o++]=x3[i];
    for (int i=7;i>=0;i--) d_out[o++]=y3[i];
    fe four; fe_set0(four); four[ 0]=4u;
    Jacobian P4; scalar_mult_base(P4, four);
    u32 x4[ 8], y4[ 8]; jac_to_affine(x4, y4, P4);
    for (int i=7;i>=0;i--) d_out[o++]=x4[i];
    for (int i=7;i>=0;i--) d_out[o++]=y4[i];
    fe five; fe_set0(five); five[ 0]=5u;
    Jacobian P5; scalar_mult_base(P5, five);
    u32 x5[ 8], y5[ 8]; jac_to_affine(x5, y5, P5);
    for (int i=7;i>=0;i--) d_out[o++]=x5[i];
    for (int i=7;i>=0;i--) d_out[o++]=y5[i];
    fe seven; fe_set0(seven); seven[ 0]=7u;
    Jacobian P7; scalar_mult_base(P7, seven);
    u32 x7[ 8], y7[ 8]; jac_to_affine(x7, y7, P7);
    for (int i=7;i>=0;i--) d_out[o++]=x7[i];
    for (int i=7;i>=0;i--) d_out[o++]=y7[i];
    fe eight; fe_set0(eight); eight[ 0]=8u;
    Jacobian P8; scalar_mult_base(P8, eight);
    u32 x8[ 8], y8[ 8]; jac_to_affine(x8, y8, P8);
    for (int i=7;i>=0;i--) d_out[o++]=x8[i];
    for (int i=7;i>=0;i--) d_out[o++]=y8[i];
    d_done[ 0]=1;
}

// ==========================================
// Kernel：sc_add_mod_n 诊断（IL + master_k）
// ==========================================
__global__ void scdiag_kernel(u32* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    // IL (b50e42e12a00b9e0abc8d9a1ced0f570252afcf07ed746e16d9d3a12dcc7d883)
    const u8 ilb[ 32] = { 0xb5,0x0e,0x42,0xe1,0x2a,0x00,0xb9,0xe0,0xab,0xc8,0xd9,0xa1,0xce,0xd0,0xf5,0x70,
                          0x25,0x2a,0xfc,0xf0,0x7e,0xd7,0x46,0xe1,0x6d,0x9d,0x3a,0x12,0xdc,0xc7,0xd8,0x83 };
    // master_k (1837c1be8e2995ec11cda2b066151be2cfb48adf9e47b151d46adab3a21cdf67)
    const u8 kb[ 32] = { 0x18,0x37,0xc1,0xbe,0x8e,0x29,0x95,0xec,0x11,0xcd,0xa2,0xb0,0x66,0x15,0x1b,0xe2,
                         0xcf,0xb4,0x8a,0xdf,0x9e,0x47,0xb1,0x51,0xd4,0x6a,0xda,0xb3,0xa2,0x1c,0xdf,0x67 };
    u32 il[ 8], kf[ 8], r[ 8];
    bytes32_to_fe(il, ilb);
    bytes32_to_fe(kf, kb);
    int o=0;
    for (int i=0;i<8;i++) d_out[o++]=il[i];
    for (int i=0;i<8;i++) d_out[o++]=kf[i];
    u64 c=0; u32 t[ 9];
    for (int i=0;i<8;i++){ u64 s=(u64)il[i]+kf[i]+c; t[i]=(u32)s; c=s>>32; }
    t[ 8]=(u32)c;
    for (int i=0;i<9;i++) d_out[o++]=t[i];
    sc_add_mod_n(r, il, kf);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    // fe_ge check result
    d_out[o++]=fe_ge(t, N_FE);
    d_out[o++]=t[ 8];
    d_done[ 0]=1;
}

// ==========================================
// Kernel：字段测试
// ==========================================
__global__ void fieldtest_kernel(u32* d_out) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    int o=0;
    fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu; a[ 0]=0xFFFFFC2Eu; a[ 1]=0xFFFFFFFEu;
    fe r;
    fe_mul(r, a, a);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe b; for (int i=0;i<8;i++) b[i]=0xFFFFFFFFu; b[ 0]=0xFFFFFC2Du; b[ 1]=0xFFFFFFFEu;
    fe_mul(r, b, a);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe c2; fe_set0(c2); c2[ 7]=0x80000000u;
    fe_mul(r, c2, c2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe d2; for (int i=0;i<8;i++) d2[i]=0xAAAAAAAAu;
    fe_mul(r, d2, d2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe e2; e2[ 0]=0x12345678u; e2[ 1]=0x9abcdef0u; e2[ 2]=0x11223344u; e2[ 3]=0x55667788u;
    e2[ 4]=0x99aabbccu; e2[ 5]=0xddeeff00u; e2[ 6]=0x10203040u; e2[ 7]=0x50607080u;
    fe_mul(r, e2, e2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe f1; for (int i=0;i<8;i++) f1[i]=0xFFFFFFFFu;
    fe f2; fe_set0(f2); f2[ 0]=0x00000001u; f2[ 1]=0x00000001u;
    fe_add(r, f1, f2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    fe g2; g2[ 0]=0x12345678u; g2[ 1]=0x9abcdef0u; g2[ 2]=0x11223344u; g2[ 3]=0x55667788u;
    g2[ 4]=0x99aabbccu; g2[ 5]=0xddeeff00u; g2[ 6]=0x10203040u; g2[ 7]=0x50607080u;
    fe_inv(r, g2);
    fe_mul(r, r, g2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    u32 nm1[ 8]; fe_cpy(nm1, N_FE); nm1[ 0]-=1;
    u32 two[ 8]; fe_set0(two); two[ 0]=2u;
    u32 sr[ 8]; sc_add_mod_n(sr, nm1, two);
    for (int i=0;i<8;i++) d_out[o++]=sr[i];
}

// ==========================================
// Kernel：fieldtest 进度标记版（定位挂起用例）
// ==========================================
__global__ void ftprog_kernel(u32* d_out, u32* d_prog) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    int o=0;
    fe r;
    d_prog[ 0]=1;
    fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu; a[ 0]=0xFFFFFC2Eu; a[ 1]=0xFFFFFFFEu;
    fe_mul(r, a, a);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=2;
    fe b; for (int i=0;i<8;i++) b[i]=0xFFFFFFFFu; b[ 0]=0xFFFFFC2Du; b[ 1]=0xFFFFFFFEu;
    fe_mul(r, b, a);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=3;
    fe c2; fe_set0(c2); c2[ 7]=0x80000000u;
    fe_mul(r, c2, c2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=4;
    fe d2; for (int i=0;i<8;i++) d2[i]=0xAAAAAAAAu;
    fe_mul(r, d2, d2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=5;
    fe e2; e2[ 0]=0x12345678u; e2[ 1]=0x9abcdef0u; e2[ 2]=0x11223344u; e2[ 3]=0x55667788u;
    e2[ 4]=0x99aabbccu; e2[ 5]=0xddeeff00u; e2[ 6]=0x10203040u; e2[ 7]=0x50607080u;
    fe_mul(r, e2, e2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=6;
    fe f1; for (int i=0;i<8;i++) f1[i]=0xFFFFFFFFu;
    fe f2; fe_set0(f2); f2[ 0]=0x00000001u; f2[ 1]=0x00000001u;
    fe_add(r, f1, f2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=7;
    fe g2; g2[ 0]=0x12345678u; g2[ 1]=0x9abcdef0u; g2[ 2]=0x11223344u; g2[ 3]=0x55667788u;
    g2[ 4]=0x99aabbccu; g2[ 5]=0xddeeff00u; g2[ 6]=0x10203040u; g2[ 7]=0x50607080u;
    fe_inv(r, g2);
    fe_mul(r, r, g2);
    for (int i=0;i<8;i++) d_out[o++]=r[i];
    d_prog[ 0]=8;
    u32 nm1[ 8]; fe_cpy(nm1, N_FE); nm1[ 0]-=1;
    u32 two[ 8]; fe_set0(two); two[ 0]=2u;
    u32 sr[ 8]; sc_add_mod_n(sr, nm1, two);
    for (int i=0;i<8;i++) d_out[o++]=sr[i];
    d_prog[ 0]=9;
}

// ==========================================
// Kernel：单用例 bisect（定位 GPU 挂起用例）
// ==========================================
__global__ void bisect_kernel(u32* d_out, int c) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    fe r;
    if (c==0) { fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu; a[ 0]=0xFFFFFC2Eu; a[ 1]=0xFFFFFFFEu; fe_mul(r, a, a); }
    else if (c==1) { fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu; a[ 0]=0xFFFFFC2Eu; a[ 1]=0xFFFFFFFEu;
                     fe b; for (int i=0;i<8;i++) b[i]=0xFFFFFFFFu; b[ 0]=0xFFFFFC2Du; b[ 1]=0xFFFFFFFEu; fe_mul(r, b, a); }
    else if (c==2) { fe c2; fe_set0(c2); c2[ 7]=0x80000000u; fe_mul(r, c2, c2); }
    else if (c==3) { fe d2; for (int i=0;i<8;i++) d2[i]=0xAAAAAAAAu; fe_mul(r, d2, d2); }
    else if (c==4) { fe e2; e2[ 0]=0x12345678u; e2[ 1]=0x9abcdef0u; e2[ 2]=0x11223344u; e2[ 3]=0x55667788u;
                     e2[ 4]=0x99aabbccu; e2[ 5]=0xddeeff00u; e2[ 6]=0x10203040u; e2[ 7]=0x50607080u; fe_mul(r, e2, e2); }
    else if (c==5) { fe f1; for (int i=0;i<8;i++) f1[i]=0xFFFFFFFFu;
                     fe f2; fe_set0(f2); f2[ 0]=0x00000001u; f2[ 1]=0x00000001u; fe_add(r, f1, f2); }
    else if (c==6) { fe g2; g2[ 0]=0x12345678u; g2[ 1]=0x9abcdef0u; g2[ 2]=0x11223344u; g2[ 3]=0x55667788u;
                     g2[ 4]=0x99aabbccu; g2[ 5]=0xddeeff00u; g2[ 6]=0x10203040u; g2[ 7]=0x50607080u;
                     fe_inv(r, g2); fe_mul(r, r, g2); }
    else if (c==7) { u32 nm1[ 8]; fe_cpy(nm1, N_FE); nm1[ 0]-=1;
           u32 two[ 8]; fe_set0(two); two[ 0]=2u;
           sc_add_mod_n(r, nm1, two); }
    else if (c==8) { // fe_add(D,D); D=cbc1950e... -> expect 97832a1c...
           u32 D[ 8] = { 0x69cfc671u, 0xd2076736u, 0xf147c112u, 0x6bcb0bbcu, 0xcb96483du, 0xd0271c6bu, 0x6fec4514u, 0xcbc1950eu };
           fe_add(r, D, D); }
    else if (c==9) { // fe_sub(F, T); F=E^2 mod p, T=2D mod p -> expect X3
           u32 F[ 8] = { 0x87054446u, 0x35e2d2b3u, 0xd92564edu, 0x8e25a078u, 0xb6d66bd0u, 0xc1dfbd14u, 0xfe816c05u, 0x14985620u };
           u32 T[ 8] = { 0xd39f90b3u, 0xa40ece6du, 0xe28f8225u, 0xd7961779u, 0x972c907au, 0xa04e38d7u, 0xdfd88a29u, 0x97832a1cu };
           fe_sub(r, F, T); }
    else { u32 nm1[ 8]; fe_cpy(nm1, N_FE); nm1[ 0]-=1;
           u32 two[ 8]; fe_set0(two); two[ 0]=2u;
           sc_add_mod_n(r, nm1, two); }
    for (int i=0;i<8;i++) d_out[i]=r[i];
}

// ==========================================
// Kernel：debug（(p-1)^2 中间量转储）
// ==========================================
__global__ void debug_kernel(u32* d_out) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    int o=0;
    fe a; for (int i=0;i<8;i++) a[i]=0xFFFFFFFFu; a[ 0]=0xFFFFFC2Eu; a[ 1]=0xFFFFFFFEu;
    u32 t[ 16];
    for (int i=0;i<16;i++) t[i]=0;
    for (int i=0;i<8;i++)
        for (int j=0;j<8;j++) {
            u64 p = (u64)a[i]*a[j];
            u64 s = (u64)t[i+j] + (u32)p;
            t[i+j] = (u32)s;
            u64 c = s >> 32;
            int k = i+j+1;
            while (c && k < 16) { u64 s2=(u64)t[k]+c; t[k]=(u32)s2; c=s2>>32; k++; }
            s = (u64)t[i+j+1] + (u32)(p>>32);
            t[i+j+1] = (u32)s;
            c = s >> 32;
            k = i+j+2;
            while (c && k < 16) { u64 s2=(u64)t[k]+c; t[k]=(u32)s2; c=s2>>32; k++; }
        }
    for (int i=0;i<16;i++) d_out[o++]=t[i];       // 0-15: 原始乘积
    u32 v[ 10];
    u64 c=0;
    for (int i=0;i<10;i++){
        u64 s = c + (i<8 ? (u64)t[i] : 0);
        if (i>=1 && (i-1)<8) s += (u64)t[ 8+(i-1)];
        if (i<8) s += (u64)t[ 8+i]*977u;
        v[i]=(u32)s; c=s>>32;
    }
    if (c){ u64 s=(u64)v[ 9]+c; v[ 9]=(u32)s; }
    for (int i=0;i<10;i++) d_out[o++]=v[i];       // 16-25: 初始折叠 v
    fe_fold_hi(v);
    for (int i=0;i<10;i++) d_out[o++]=v[i];       // 26-35: fe_fold_hi 后 v
    fe r; fe_mul(r, a, a);
    for (int i=0;i<8;i++) d_out[o++]=r[i];        // 36-43: fe_mul 结果
}

// ==========================================
// Kernel：固定熵测试（BIP39 标准向量）
// ==========================================
__global__ void test_kernel(const u8* d_ent, char* d_mnem, u8* d_addr, u8* d_priv, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 ent[ 16];
    for (int i=0;i<16;i++) ent[i]=d_ent[i];
    char mnem[ 128];
    u8 addr[ 20], privk[ 32];
    derive_full(ent, mnem, addr, privk);
    int l=0; while (mnem[l]){ d_mnem[l]=mnem[l]; l++; } d_mnem[l]=0;
    for (int i=0;i<20;i++) d_addr[i]=addr[i];
    for (int i=0;i<32;i++) d_priv[i]=privk[i];
    d_done[ 0]=1;
}

// ==========================================
// Kernel：vanity 搜索
// ==========================================


// ==========================================
// HD 快速模式：固定种子，变索引 m/44'/195'/0'/0/N 搜索
// setup: 熵->助记词->seed->派生到 m/44'/195'/0'/0 (k0n,c0n) + pubkey(k0n)
// search: N 变化 -> k_N = IL + k0n (mod n) -> pubkey -> keccak -> 前缀比对
// ==========================================
__global__ void hd_setup_kernel(u64 base_nonce, u32* d_kbase, u32* d_cbase, u32* d_pubx, u32* d_puby, char* d_mnem) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 ent[ 16]; counter_to_entropy(base_nonce, ent);
    char m[ 128]; entropy_to_mnemonic(ent, m, 0);
    int l=0; while(m[l]){ d_mnem[l]=m[l]; l++; } d_mnem[l]=0;
    u8 seed[ 64]; device_bip39_seed(m, seed);
    // master
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    // 44' 60' 0'
    ckd_priv(k, c, 0x80000000u+44u, 1);
    ckd_priv(k, c, 0x80000000u+195u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    // 0 (non-hardened) -> k0n
    u8 k0n[ 32], c0n[ 32];
    for (int i=0;i<32;i++){ k0n[i]=k[i]; c0n[i]=c[i]; }
    ckd_priv(k0n, c0n, 0u, 0);
    // 先写 kbase/cbase，再算 pubkey（避免 comb 潜在污染）
    u32 kb[ 8], cb[ 8];
    bytes32_to_fe(kb, k0n); bytes32_to_fe(cb, c0n);
    for (int i=0;i<8;i++){ d_kbase[i]=kb[i]; d_cbase[i]=cb[i]; }
    // pubkey(k0n)
    u32 kf[ 8]; bytes32_to_fe(kf, k0n);
    Jacobian P; scalar_mult_base_comb(P, kf);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    for (int i=0;i<8;i++){ d_pubx[i]=x[i]; d_puby[i]=y[i]; }
}

__global__ __launch_bounds__(256, 2) void hd_vanity_kernel(const u8* d_prefix, int plen,
        const u32* d_kbase, const u32* d_cbase, const u32* d_pubx, const u32* d_puby,
        u64 start_nonce, int iters, char* d_mnem, u8* d_addr, u8* d_priv, u32* d_index, u32* d_found) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    // v2: 链码与父公钥在整个 kernel 生命周期内固定，HMAC 预计算和字节转换提到循环外只做一次
    u32 cb0[ 8]; for (int i=0;i<8;i++) cb0[i]=d_cbase[i];
    u8 c8[ 32]; fe_to_bytes32(c8, cb0);
    u64 cipad[ 8], copad[ 8];
    hmac_sha512_precompute(c8, 32, cipad, copad);
    u8 data0[ 33];
    data0[ 0] = (d_puby[ 0]&1) ? 0x03 : 0x02;
    u8 xb[ 32]; fe_to_bytes32(xb, d_pubx);
    for (int i=0;i<32;i++) data0[1+i]=xb[i];
    for (int it=0; it<iters; it++) {
        /* v3: 去掉整批早退，保证每批跑满，速率不虚高 */
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u32 N = (u32)(n & 0x7FFFFFFFu);
        // IL = HMAC-SHA512(c0n, 0x02/03||X || N)
        u8 data[ 37];
        for (int i=0;i<33;i++) data[i]=data0[i];
        data[ 33]=(u8)(N>>24); data[ 34]=(u8)(N>>16); data[ 35]=(u8)(N>>8); data[ 36]=(u8)N;
        u8 out[ 64];
        hmac_sha512_precomp_msg(cipad, copad, data, 37, out);
        u32 il[ 8]; bytes32_to_fe(il, out);
        u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=d_kbase[i];
        sc_add_mod_n(kf, il, kf);
        Jacobian P; scalar_mult_base_comb(P, kf);
        u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
        u8 pub[ 64];
        fe_to_bytes32(pub, x);
        fe_to_bytes32(pub+32, y);
        u8 hh[ 32];
        device_keccak256(pub, 64, hh);
        u8 addr[ 20]; for (int i=0;i<20;i++) addr[i]=hh[12+i];
        int ok=1;
        for (int i=0;i<plen;i++) {
            u8 nib = (i&1) ? (addr[i>>1]&0xF) : (addr[i>>1]>>4);
            if (nib != d_prefix[i]) { ok=0; break; }
        }
        if (ok) {
            u32 old = atomicCAS(d_found, 0, 1);
            if (old==0) {
                u8 privk[ 32]; fe_to_bytes32(privk, kf);
                for (int i=0;i<20;i++) d_addr[i]=addr[i];
                for (int i=0;i<32;i++) d_priv[i]=privk[i];
                d_index[ 0]=N;
            }
            return;
        }
    }
}

__global__ void hd_diag_kernel(u64 base_nonce, u8* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 ent[ 16]; counter_to_entropy(base_nonce, ent);
    char m[ 128]; entropy_to_mnemonic(ent, m, 0);
    for (int i=0;i<128;i++) d_out[i]=(u8)m[i]; // mnemonic bytes 0..127
    u8 seed[ 64]; device_bip39_seed(m, seed);
    for (int i=0;i<64;i++) d_out[128+i]=seed[i];
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    ckd_priv(k, c, 0x80000000u+44u, 1);
    ckd_priv(k, c, 0x80000000u+195u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    for (int i=0;i<32;i++) d_out[192+i]=k[i];  // k0h
    for (int i=0;i<32;i++) d_out[224+i]=c[i];  // c0h
    u8 k0n[ 32], c0n[ 32];
    for (int i=0;i<32;i++){ k0n[i]=k[i]; c0n[i]=c[i]; }
    ckd_priv(k0n, c0n, 0u, 0);
    for (int i=0;i<32;i++) d_out[256+i]=k0n[i]; // k0n
    for (int i=0;i<32;i++) d_out[288+i]=c0n[i]; // c0n
    u32 kf[ 8]; bytes32_to_fe(kf, k0n);
    Jacobian P; scalar_mult_base_comb(P, kf);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    u8 xb[ 32]; fe_to_bytes32(xb, x);
    for (int i=0;i<32;i++) d_out[320+i]=xb[i]; // pubx
    d_done[ 0]=1;
}

__global__ void hd_diag2_kernel(u64 base_nonce, u32 N, u8* d_out, u32* d_done) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    u8 ent[ 16]; counter_to_entropy(base_nonce, ent);
    char m[ 128]; entropy_to_mnemonic(ent, m, 0);
    for (int i=0;i<128;i++) d_out[i]=(u8)m[i];
    u8 seed[ 64]; device_bip39_seed(m, seed);
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    ckd_priv(k, c, 0x80000000u+44u, 1);
    ckd_priv(k, c, 0x80000000u+195u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    u8 k0n[ 32], c0n[ 32];
    for (int i=0;i<32;i++){ k0n[i]=k[i]; c0n[i]=c[i]; }
    ckd_priv(k0n, c0n, 0u, 0);
    // ===== 完全复用 hd_vanity_kernel 的 search 计算路径 (N=0) =====
    u32 kf0[ 8]; bytes32_to_fe(kf0, k0n);
    Jacobian P0; scalar_mult_base_comb(P0, kf0);
    u32 px[ 8], py[ 8]; jac_to_affine(px, py, P0);
    u32 cb[ 8]; bytes32_to_fe(cb, c0n);
    u8 c8[ 32]; fe_to_bytes32(c8, cb);
    u8 data[ 37];
    data[ 0] = (py[ 0]&1) ? 0x03 : 0x02;
    u8 xb[ 32]; fe_to_bytes32(xb, px);
    for (int i=0;i<32;i++) data[1+i]=xb[i];
    data[33]=(u8)(N>>24); data[34]=(u8)(N>>16); data[35]=(u8)(N>>8); data[36]=(u8)N;
    u8 o2[ 64];
    hmac_sha512_gen(c8, 32, data, 37, o2);
    u32 il[ 8]; bytes32_to_fe(il, o2);
    u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=kf0[i];
    sc_add_mod_n(kf, il, kf);
    u8 kchild[ 32]; fe_to_bytes32(kchild, kf);
    for (int i=0;i<32;i++) d_out[128+i]=kchild[i];   // child priv @ N=0
    Jacobian P; scalar_mult_base_comb(P, kf);
    u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
    u8 pub[ 64];
    fe_to_bytes32(pub, x);
    fe_to_bytes32(pub+32, y);
    u8 hh[ 32]; device_keccak256(pub, 64, hh);
    for (int i=0;i<20;i++) d_out[160+i]=hh[12+i];     // addr
    d_done[ 0]=1;
}
// ==========================================
// Kernel: 分阶段性能剖析 (stage: 0=mnemonic 1=+pbkdf2 2=+ckd 3=full)
// 每段结果都累加进 checksum 写入全局，防止 DCE
// ==========================================
__global__ void prof_kernel(u64 start_nonce, int iters, int stage, u64* d_out, u32* d_done) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    u64 acc = 0;
    for (int it=0; it<iters; it++) {
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u8 ent[ 16]; counter_to_entropy(n, ent);
        char m[ 128]; entropy_to_mnemonic(ent, m, 0);
        u64 acc0 = 0;
        for (int i=0;i<128;i++) acc0 = acc0*131 + (u8)m[i];
        if (stage>=1) {
            u8 seed[ 64]; device_bip39_seed(m, seed);
            u64 acc1 = acc0;
            for (int i=0;i<64;i++) acc1 = acc1*131 + seed[i];
            if (stage>=2) {
                u8 privk[ 32]; bip32_derive(privk, seed);
                u64 acc2 = acc1;
                for (int i=0;i<32;i++) acc2 = acc2*131 + privk[i];
                if (stage>=3) {
                    u8 addr[ 20]; derive_address(privk, addr);
                    for (int i=0;i<20;i++) acc2 = acc2*131 + addr[i];
                    acc += acc2;
                } else acc += acc2;
            } else acc += acc1;
        } else acc += acc0;
    }
    if (acc == 0xDEADBEEF) d_out[ tid % 1024 ] = acc;
    d_done[ 0]=1;
}

__device__ void ent_to_hd_base(const u8 ent[ 16], u32 kbase[ 8], u32 cbase[ 8],
                               u32 pubx[ 8], u32 puby[ 8], char* mnem_out);

// ==========================================
// BTC 地址原语：RIPEMD160 / Base58Check / bech32(bech32m) / 地址生成
// ==========================================
__device__ __forceinline__ u32 btc_rol32(u32 x, int n) { return (x << n) | (x >> (32 - n)); }
__device__ __forceinline__ u32 rp_f(int j, u32 x, u32 y, u32 z) {
    if (j < 16) return x ^ y ^ z;
    if (j < 32) return (x & y) | (~x & z);
    if (j < 48) return (x | ~y) ^ z;
    if (j < 64) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}
__device__ __forceinline__ u32 rp_f2(int j, u32 x, u32 y, u32 z) {
    if (j < 16) return x ^ (y | ~z);
    if (j < 32) return (x & z) | (y & ~z);
    if (j < 48) return (x | ~y) ^ z;
    if (j < 64) return (x & y) | (~x & z);
    return x ^ y ^ z;
}
__device__ __forceinline__ void ripemd160(const u8* msg, int len, u8 out[ 20]) {
    static const u32 rp_r[ 80] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8, 3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2, 4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
    static const u32 rp_r2[ 80] = {5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2, 15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14, 12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
    static const u32 rp_s[ 80] = {11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12, 11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12, 9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
    static const u32 rp_s2[ 80] = {8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11, 9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8, 8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
    static const u32 rp_k[ 5] = {0x00000000, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e};
    static const u32 rp_k2[ 5] = {0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0x00000000};
    u8 blk[ 64];
    for (int i = 0; i < 64; i++) blk[i] = 0;
    for (int i = 0; i < len && i < 64; i++) blk[i] = msg[i];
    blk[len] = 0x80;
    u64 bits = (u64)len * 8;
    for (int i = 0; i < 8; i++) blk[56 + i] = (u8)((bits >> (8 * i)) & 0xFF);
    u32 M[ 16];
    for (int i = 0; i < 16; i++) { u32 v = 0; for (int j = 0; j < 4; j++) v |= ((u32)blk[i * 4 + j]) << (8 * j); M[i] = v; }
    u32 A = 0x67452301, B = 0xEFCDAB89, C = 0x98BADCFE, D = 0x10325476, E = 0xC3D2E1F0;
    u32 A2 = A, B2 = B, C2 = C, D2 = D, E2 = E;
    for (int j = 0; j < 80; j++) {
        u32 T = btc_rol32(A + rp_f(j, B, C, D) + M[rp_r[j]] + rp_k[j / 16], rp_s[j]) + E;
        A = E; E = D; D = btc_rol32(C, 10); C = B; B = T;
        u32 T2 = btc_rol32(A2 + rp_f2(j, B2, C2, D2) + M[rp_r2[j]] + rp_k2[j / 16], rp_s2[j]) + E2;
        A2 = E2; E2 = D2; D2 = btc_rol32(C2, 10); C2 = B2; B2 = T2;
    }
    u32 h0n = 0xEFCDAB89u + C + D2;
    u32 h1n = 0x98BADCFEu + D + E2;
    u32 h2n = 0x10325476u + E + A2;
    u32 h3n = 0xC3D2E1F0u + A + B2;
    u32 h4n = 0x67452301u + B + C2;
    u32 h[ 5] = {h0n, h1n, h2n, h3n, h4n};
    for (int i = 0; i < 20; i++) out[i] = (u8)((h[i / 4] >> (8 * (i % 4))) & 0xFF);
}
__device__ __forceinline__ void sha256d(const u8* msg, int len, u8 out[ 32]) {
    u8 h1[ 32]; sha256(msg, len, h1);
    sha256(h1, 32, out);
}
__device__ int base58check_encode(const u8* payload, int plen, char* out) {
    static const char* B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    u8 full[ 40];
    for (int i = 0; i < plen; i++) full[i] = payload[i];
    u8 chk[ 32]; sha256d(payload, plen, chk);
    for (int i = 0; i < 4; i++) full[plen + i] = chk[i];
    int n = plen + 4;
    int zeros = 0; while (zeros < n && full[zeros] == 0) zeros++;
    u8 buf[ 40]; for (int i = 0; i < n; i++) buf[i] = full[i];
    char b58[ 40]; int bl = 0;
    int start = 0;
    while (start < n) {
        int rem = 0;
        for (int i = start; i < n; i++) {
            int acc = rem * 256 + buf[i];
            buf[i] = (u8)(acc / 58);
            rem = acc % 58;
        }
        b58[bl++] = B58[rem];
        while (start < n && buf[start] == 0) start++;
    }
    int idx = 0;
    for (int i = 0; i < zeros; i++) out[idx++] = '1';
    for (int i = bl - 1; i >= 0; i--) out[idx++] = b58[i];
    return idx;
}
__device__ int bech32_encode(int witver, const u8* payload, int plen, char* out) {
    static const char* CH = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    u8 conv[ 80]; int nc = 0;
    int acc = 0, bits = 0;
    for (int i = 0; i < plen; i++) {
        acc = (acc << 8) | payload[i]; bits += 8;
        while (bits >= 5) { bits -= 5; conv[nc++] = (u8)((acc >> bits) & 31); }
    }
    if (bits > 0) conv[nc++] = (u8)((acc << (5 - bits)) & 31);
    u8 d[ 90]; int m = 0; d[m++] = (u8)witver;
    for (int i = 0; i < nc; i++) d[m++] = conv[i];
    u32 v[ 100]; int nv = 0;
    v[nv++] = 'b' >> 5; v[nv++] = 'c' >> 5; v[nv++] = 0; v[nv++] = 'b' & 31; v[nv++] = 'c' & 31;
    for (int i = 0; i < m; i++) v[nv++] = d[i];
    u32 chk = 1;
    static const u32 gen[ 5] = {0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3};
    for (int i = 0; i < nv + 6; i++) {
        u32 x = (i < nv) ? v[i] : 0;
        u32 b = chk >> 25;
        chk = ((chk & 0x1ffffff) << 5) ^ x;
        for (int j = 0; j < 5; j++) chk ^= ((b >> j) & 1) ? gen[j] : 0;
    }
    u32 constv = (witver == 0) ? 1u : 0x2bc830a3u;
    chk ^= constv;
    out[0] = 'b'; out[1] = 'c'; out[2] = '1';
    int oi = 3;
    for (int i = 0; i < m; i++) out[oi++] = CH[d[i]];
    for (int i = 0; i < 6; i++) out[oi++] = CH[(chk >> (5 * (5 - i))) & 31];
    return oi;
}
// TRX 地址：keccak256(非压缩公钥 64B) 后 20 字节 + 0x41 + Base58Check -> T 开头 34 字符
// pub = 64 字节 (x||y) 非压缩
__device__ int trx_address_from_pub(const u8 pub[ 64], char* out) {
    u8 h[ 32]; device_keccak256(pub, 64, h);
    u8 pl[ 21]; pl[ 0] = 0x41;
    for (int i = 0; i < 20; i++) pl[ 1 + i] = h[ 12 + i];
    return base58check_encode(pl, 21, out);
}
// 自检：固定向量（ent = SHA256(8 字节全 0) 的前 16 字节，并非全零熵）-> TRX m/44'/195'/0'/0/N (N=0,3) T 地址；v2 有意不混 secret，保证输出可跨版本对照
__global__ void trx_test_kernel(u8* d_out, u32* d_done) {
    u8 nb[ 8]; for (int i = 0; i < 8; i++) nb[i] = 0;
    u8 h[ 32]; sha256(nb, 8, h);
    u8 ent[ 16]; for (int i = 0; i < 16; i++) ent[i] = h[i];
    u32 kbase[ 8], cbase[ 8], pubx[ 8], puby[ 8];
    char mnem[ 128];
    ent_to_hd_base(ent, kbase, cbase, pubx, puby, mnem);
    u8 c8[ 32]; fe_to_bytes32(c8, cbase);
    u8 xb0[ 32]; fe_to_bytes32(xb0, pubx);
    u32 NN[ 2] = {0, 3};
    for (int q = 0; q < 2; q++) {
        u32 N = NN[ q];
        u8 data[ 37];
        data[ 0] = (puby[ 0] & 1) ? 0x03 : 0x02;
        for (int i = 0; i < 32; i++) data[ 1 + i] = xb0[i];
        data[ 33] = (u8)(N >> 24); data[ 34] = (u8)(N >> 16); data[ 35] = (u8)(N >> 8); data[ 36] = (u8)N;
        u8 o2[ 64]; hmac_sha512_gen(c8, 32, data, 37, o2);
        u32 il[ 8]; bytes32_to_fe(il, o2);
        u32 kf[ 8]; for (int i = 0; i < 8; i++) kf[i] = kbase[i];
        sc_add_mod_n(kf, il, kf);
        u8 privk[ 32]; fe_to_bytes32(privk, kf);
        Jacobian P; scalar_mult_base_comb(P, kf);
        u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
        u8 pub[ 64]; fe_to_bytes32(pub, x); fe_to_bytes32(pub + 32, y);
        int o = q * 320;
        for (int i = 0; i < 32; i++) d_out[o + i] = privk[i];
        d_out[o + 32] = 0x04;
        for (int i = 0; i < 64; i++) d_out[o + 33 + i] = pub[i];
        char* aout = (char*)(d_out + o + 97);
        int l = trx_address_from_pub(pub, aout);
        aout[l] = 0;
    }
    for (int i = 0; i < 128; i++) d_out[640 + i] = (u8)mnem[i];
    d_done[ 0] = 1;
}

// 熵 -> 助记词 -> m/44'/195'/0'/0 基点（kbase/cbase/pubx/puby），供多索引派生复用
__device__ void ent_to_hd_base(const u8 ent[ 16], u32 kbase[ 8], u32 cbase[ 8],
                               u32 pubx[ 8], u32 puby[ 8], char* mnem_out) {
    char m[ 128];
    entropy_to_mnemonic(ent, m, 0);
    if (mnem_out) { int l=0; while (m[l]){ mnem_out[l]=m[l]; l++; } mnem_out[l]=0; }
    u8 seed[ 64];
    device_bip39_seed(m, seed);
    u8 k[ 32], c[ 32], out[ 64];
    hmac_sha512_gen((const u8*)"Bitcoin seed", 12, seed, 64, out);
    for (int i=0;i<32;i++){ k[i]=out[i]; c[i]=out[32+i]; }
    ckd_priv(k, c, 0x80000000u+44u, 1);
    ckd_priv(k, c, 0x80000000u+195u, 1);
    ckd_priv(k, c, 0x80000000u+0u, 1);
    u8 k0n[ 32], c0n[ 32];
    for (int i=0;i<32;i++){ k0n[i]=k[i]; c0n[i]=c[i]; }
    ckd_priv(k0n, c0n, 0u, 0);
    bytes32_to_fe(kbase, k0n);
    bytes32_to_fe(cbase, c0n);
    u32 kf[ 8]; bytes32_to_fe(kf, k0n);
    Jacobian P; scalar_mult_base_comb(P, kf);
    jac_to_affine(pubx, puby, P);
}

__global__ __launch_bounds__(256, 2) void vanity_kernel(const u8* d_prefix, int plen, u64 start_nonce, int iters, int nidx,
                              char* d_mnem, u8* d_addr, u8* d_priv, u32* d_index, u32* d_found) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    for (int it=0; it<iters; it++) {
        /* v3: 去掉整批早退，保证每批跑满，速率不虚高 */
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u8 ent[ 16]; counter_to_entropy(n, ent);
        u32 kbase[ 8], cbase[ 8], pubx[ 8], puby[ 8];
        char mnem[ 128];
        ent_to_hd_base(ent, kbase, cbase, pubx, puby, mnem);
        u8 c8[ 32]; fe_to_bytes32(c8, cbase);
        u8 xb0[ 32]; fe_to_bytes32(xb0, pubx);
        u64 cipad[ 8], copad[ 8];
        hmac_sha512_precompute(c8, 32, cipad, copad);  // v2: 同一助记词的 nidx 个子地址共用链码，预计算只做一次
        for (int idx=0; idx<nidx; idx++) {
            u32 N = (u32)idx;
            u8 data[ 37];
            data[ 0] = (puby[ 0]&1) ? 0x03 : 0x02;
            for (int i=0;i<32;i++) data[1+i]=xb0[i];
            data[ 33]=(u8)(N>>24); data[ 34]=(u8)(N>>16); data[ 35]=(u8)(N>>8); data[ 36]=(u8)N;
            u8 out[ 64];
            hmac_sha512_precomp_msg(cipad, copad, data, 37, out);
            u32 il[ 8]; bytes32_to_fe(il, out);
            u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=kbase[i];
            sc_add_mod_n(kf, il, kf);
            Jacobian P; scalar_mult_base_comb(P, kf);
            u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
            u8 pub[ 64];
            fe_to_bytes32(pub, x);
            fe_to_bytes32(pub+32, y);
            u8 hh[ 32];
            device_keccak256(pub, 64, hh);
            u8 addr[ 20]; for (int i=0;i<20;i++) addr[i]=hh[12+i];
            int ok=1;
            for (int i=0;i<plen;i++) {
                u8 nib = (i&1) ? (addr[i>>1]&0xF) : (addr[i>>1]>>4);
                if (nib != d_prefix[i]) { ok=0; break; }
            }
            if (ok) {
                u32 old = atomicCAS(d_found, 0, 1);
                if (old==0) {
                    int l=0; while (mnem[l]){ d_mnem[l]=mnem[l]; l++; } d_mnem[l]=0;
                    u8 privk[ 32]; fe_to_bytes32(privk, kf);
                    for (int i=0;i<20;i++) d_addr[i]=addr[i];
                    for (int i=0;i<32;i++) d_priv[i]=privk[i];
                    d_index[ 0]=N;
                }
                return;
            }
        }
    }
}


// ==========================================
// 文件模式：一个固定种子，同时搜索多个前缀（HD m/44'/195'/0'/0/N）
// 每个前缀独立 found slot + 结果；全部找到则提前退出
// ==========================================
__global__ __launch_bounds__(256, 2) void multi_hd_kernel(
        const u8* d_prefixes, const u8* d_plens, int nprefix,
        const u32* d_kbase, const u32* d_cbase, const u32* d_pubx, const u32* d_puby,
        u64 start_nonce, int iters, const char* d_mnem,
        u8* d_addrs, u8* d_privs, u32* d_idxs, u32* d_found, u32* d_nfound) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    for (int it=0; it<iters; it++) {
        if (*d_nfound >= (u32)nprefix) return;
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u32 N = (u32)(n & 0x7FFFFFFFu);
        // IL = HMAC-SHA512(c0n, 0x02/03||X || N)
        u32 cb[ 8]; for (int i=0;i<8;i++) cb[i]=d_cbase[i];
        u8 c8[ 32]; fe_to_bytes32(c8, cb);
        u8 data[ 37];
        data[ 0] = (d_puby[ 0]&1) ? 0x03 : 0x02;
        u8 xb[ 32]; fe_to_bytes32(xb, d_pubx);
        for (int i=0;i<32;i++) data[1+i]=xb[i];
        data[ 33]=(u8)(N>>24); data[ 34]=(u8)(N>>16); data[ 35]=(u8)(N>>8); data[ 36]=(u8)N;
        u8 out[ 64];
        hmac_sha512_gen(c8, 32, data, 37, out);
        u32 il[ 8]; bytes32_to_fe(il, out);
        u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=d_kbase[i];
        sc_add_mod_n(kf, il, kf);
        Jacobian P; scalar_mult_base_comb(P, kf);
        u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
        u8 pub[ 64];
        fe_to_bytes32(pub, x);
        fe_to_bytes32(pub+32, y);
        u8 hh[ 32];
        device_keccak256(pub, 64, hh);
        u8 addr[ 20]; for (int i=0;i<20;i++) addr[i]=hh[12+i];
        // 遍历所有未完成前缀
        for (int p=0; p<nprefix; p++) {
            if (d_found[p]) continue;
            int pl = d_plens[p];
            int ok=1;
            const u8* pfx = d_prefixes + p*40;
            for (int i=0;i<pl;i++) {
                u8 nib = (i&1) ? (addr[i>>1]&0xF) : (addr[i>>1]>>4);
                if (nib != pfx[i]) { ok=0; break; }
            }
            if (ok) {
                u32 old = atomicCAS(&d_found[p], 0, 1);
                if (old==0) {
                    u8 privk[ 32]; fe_to_bytes32(privk, kf);
                    for (int i=0;i<20;i++) d_addrs[p*20+i]=addr[i];
                    for (int i=0;i<32;i++) d_privs[p*32+i]=privk[i];
                    d_idxs[p]=N;
                    atomicAdd(d_nfound, 1);
                }
            }
        }
    }
}

// ==========================================
// 文件模式（vanity 语义）：随机助记词 + 多前缀并行 + 每助记词派生 nidx 个地址
// ==========================================
__global__ __launch_bounds__(256, 2) void multi_vanity_kernel(
        const u8* d_prefixes, const u8* d_plens, int nprefix, int nidx,
        u64 start_nonce, int iters,
        u8* d_mnems, u8* d_addrs, u8* d_privs, u32* d_idxs, u32* d_found, u32* d_nfound) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    for (int it=0; it<iters; it++) {
        if (*d_nfound >= (u32)nprefix) return;
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u8 ent[ 16]; counter_to_entropy(n, ent);
        u32 kbase[ 8], cbase[ 8], pubx[ 8], puby[ 8];
        char mnem[ 128];
        ent_to_hd_base(ent, kbase, cbase, pubx, puby, mnem);
        u8 c8[ 32]; fe_to_bytes32(c8, cbase);
        u8 xb0[ 32]; fe_to_bytes32(xb0, pubx);
        u64 cipad[ 8], copad[ 8];
        hmac_sha512_precompute(c8, 32, cipad, copad);  // v2: 同一助记词的 nidx 个子地址共用链码，预计算只做一次
        for (int idx=0; idx<nidx; idx++) {
            u32 N = (u32)idx;
            u8 data[ 37];
            data[ 0] = (puby[ 0]&1) ? 0x03 : 0x02;
            for (int i=0;i<32;i++) data[1+i]=xb0[i];
            data[ 33]=(u8)(N>>24); data[ 34]=(u8)(N>>16); data[ 35]=(u8)(N>>8); data[ 36]=(u8)N;
            u8 out[ 64];
            hmac_sha512_precomp_msg(cipad, copad, data, 37, out);
            u32 il[ 8]; bytes32_to_fe(il, out);
            u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=kbase[i];
            sc_add_mod_n(kf, il, kf);
            Jacobian P; scalar_mult_base_comb(P, kf);
            u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
            u8 pub[ 64];
            fe_to_bytes32(pub, x);
            fe_to_bytes32(pub+32, y);
            u8 hh[ 32];
            device_keccak256(pub, 64, hh);
            u8 addr[ 20]; for (int i=0;i<20;i++) addr[i]=hh[12+i];
            for (int p=0; p<nprefix; p++) {
                if (d_found[p]) continue;
                int pl = d_plens[p];
                int ok=1;
                const u8* pfx = d_prefixes + p*40;
                for (int i=0;i<pl;i++) {
                    u8 nib = (i&1) ? (addr[i>>1]&0xF) : (addr[i>>1]>>4);
                    if (nib != pfx[i]) { ok=0; break; }
                }
                if (ok) {
                    u32 old = atomicCAS(&d_found[p], 0, 1);
                    if (old==0) {
                        u8 privk[ 32]; fe_to_bytes32(privk, kf);
                        int l=0; while (mnem[l]){ d_mnems[p*128+l]=(u8)mnem[l]; l++; } d_mnems[p*128+l]=0;
                        for (int i=0;i<20;i++) d_addrs[p*20+i]=addr[i];
                        for (int i=0;i<32;i++) d_privs[p*32+i]=privk[i];
                        d_idxs[p]=N;
                        atomicAdd(d_nfound, 1);
                    }
                }
            }
        }
    }
}

// 命中结果环形队列：固定容量（约 235MB），不随前缀数增长，支持超大前缀文件
typedef struct {
    u32 p; u32 idx;
    u8 mnem[ 128];
    u8 addr[ PSTR];
    u8 priv[ 32];
} ResqEntry;
__global__ __launch_bounds__(256, 2) void multi_hash_kernel(
        const u64* d_tbl_key, const u32* d_tbl_p, const u8* d_tbl_occ,
        const u32* d_tbl_off, const u32* d_tbl_mask, const u32* d_tbl_cnt, const u8* d_pfx, int uniq,
        int nidx, int nl_max, u64 start_nonce, int iters,
        u32* d_found, u32* d_nfound, ResqEntry* d_resq, u32* d_resq_cnt) {
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    // 无限运行内核：不设"全部命中即退出"短路，每轮跑满 batch_iters，持续搜索重复命中
    for (int it=0; it<iters; it++) {
        u64 n = start_nonce + tid + (u64)it*gridDim.x*blockDim.x;
        u8 ent[ 16]; counter_to_entropy(n, ent);
        u32 kbase[ 8], cbase[ 8], pubx[ 8], puby[ 8];
        char mnem[ 128];
        ent_to_hd_base(ent, kbase, cbase, pubx, puby, mnem);
        u8 c8[ 32]; fe_to_bytes32(c8, cbase);
        u8 xb0[ 32]; fe_to_bytes32(xb0, pubx);
        u64 cipad[ 8], copad[ 8];
        hmac_sha512_precompute(c8, 32, cipad, copad);  // v2: 同一助记词的 nidx 个子地址共用链码，预计算只做一次
        for (int idx=0; idx<nidx; idx++) {
            u32 N = (u32)idx;
            u8 data[ 37];
            data[ 0] = (puby[ 0]&1) ? 0x03 : 0x02;
            for (int i=0;i<32;i++) data[1+i]=xb0[i];
            data[ 33]=(u8)(N>>24); data[ 34]=(u8)(N>>16); data[ 35]=(u8)(N>>8); data[ 36]=(u8)N;
            u8 out[ 64];
            hmac_sha512_precomp_msg(cipad, copad, data, 37, out);
            u32 il[ 8]; bytes32_to_fe(il, out);
            u32 kf[ 8]; for (int i=0;i<8;i++) kf[i]=kbase[i];
            sc_add_mod_n(kf, il, kf);
            Jacobian P; scalar_mult_base_comb(P, kf);
            u32 x[ 8], y[ 8]; jac_to_affine(x, y, P);
            u8 pub[ 64];
            fe_to_bytes32(pub, x);
            fe_to_bytes32(pub+32, y);
            char ast[ PSTR ];
            int laddr = trx_address_from_pub(pub, ast);
            if (laddr > PSTR) laddr = PSTR;
            u64 h=0xcbf29ce484222325ull;
            for (int L=1; L<=nl_max; L++) {
                if (L > laddr) break;
                h = (h ^ (u8)ast[L-1]) * 0x100000001b3ull;
                int cnt = d_tbl_cnt[L-1];
                if (!cnt) continue;
                u32 mask = d_tbl_mask[L-1];
                u32 base = d_tbl_off[L-1];
                u32 hh2 = (u32)((h * 0x9E3779B97F4A7C15ull) >> 32);
                int s = (int)(hh2 & mask);
                for (;;) {
                    if (!d_tbl_occ[base+s]) break;
                    if (d_tbl_key[base+s] == h) {
                        u32 p = d_tbl_p[base+s];
                        size_t po = (size_t)p;
                        int ok=1;
                        for (int k=0;k<L;k++) if (d_pfx[po*PSTR+k]!=(u8)ast[k]) { ok=0; break; }
                        if (ok && !d_found[p]) {
                            u32 old = atomicCAS(&d_found[p], 0, 1);
                            if (old==0) {
                                u8 privk[ 32]; fe_to_bytes32(privk, kf);
                                u32 q = atomicAdd(d_resq_cnt, 1);
                                ResqEntry* e = &d_resq[q & RESQ_MASK];
                                e->p = p; e->idx = N;
                                int l=0; while (mnem[l] && l<127){ e->mnem[l]=(u8)mnem[l]; l++; } e->mnem[l]=0;
                                for (int i=0;i<laddr;i++) e->addr[i]=(u8)ast[i];
                                e->addr[laddr]=0;
                                for (int i=0;i<32;i++) e->priv[i]=privk[i];
                                atomicAdd(d_nfound, 1);
                            }
                            break;
                        }
                    }
                    s = (s+1) & mask;
                }
            }
        }
    }
}

// ==========================================
// 主机辅助
// ==========================================
static void print_hex(const u8* p, int n) {
    for (int i=0;i<n;i++) printf("%02x", p[i]);
}

// TRX 地址固定 T 开头 34 字符
// pfx 行宽 PSTR, plen 每前缀长度; 统一搜索入口（file/vanity 共用）
static void run_trx_search(const u8* pfx, const u8* plen, u64 nprefix, int nidx, int blocks) {
    if (nprefix==0) { fprintf(stderr,"no valid prefixes\n"); return; }
    int nl_max = TRX_NLMAX;
    // 校验超长（调用方已处理则这里再兜底）
    u64 nskip=0;
    for (u64 p=0;p<nprefix;p++) if (plen[p] > nl_max) nskip++;
    if (nskip) printf("note: %llu prefix(es) longer than %d chars skipped\n", (unsigned long long)nskip, nl_max);
    printf("TRON vanity mode | addr T-prefix | max prefix len %d | file prefixes %llu\n", nl_max, (unsigned long long)nprefix);
    // 预检：TRX 地址是 base58check(0x41+20B)，首字符固定 'T'，第 2 字符只可能是 9ABCDEFGHJKLMNPQRSTUVWXYZ
    // 违反这两个约束的前缀在数学上永远不可能命中（小写字母/数字 1-8 开头的第 2 字符不存在）
    u64 impossible=0;
    for (u64 p=0;p<nprefix;p++) {
        int L=plen[p];
        if (L>nl_max) continue;
        if (pfx[p*PSTR] != 'T') { impossible++; continue; }
        if (L>=2) {
            char c=(char)pfx[p*PSTR+1];
            if (!strchr("9ABCDEFGHJKLMNPQRSTUVWXYZ", c)) impossible++;
        }
    }
    if (impossible)
        printf("WARNING: %llu prefix(es) can never match a TRON address (1st char must be T, 2nd char must be 9A-Z) - they will never hit\n", (unsigned long long)impossible);

    int lcnt[ NLMAX+1 ];
    for (int i=0;i<=NLMAX;i++) lcnt[i]=0;
    for (u64 p=0;p<nprefix;p++) if (plen[p]<=nl_max) lcnt[plen[p]]++;
    printf("length dist: ");
    for (int L=1;L<=nl_max;L++) if (lcnt[L]) printf("%d-char:%d ", L, lcnt[L]);
    printf("\n");
    int hcap[ NLMAX ]; u32 off[ NLMAX ]; u32 mask[ NLMAX ];
    u64* hk[ NLMAX ]; u32* hp[ NLMAX ]; u8* hoc[ NLMAX ];
    u64 total_cap = 0;
    for (int L=1;L<=NLMAX;L++) {
        int cnt = (L<=nl_max) ? lcnt[L] : 0;
        u64 c2 = 8;
        while (c2 < (u64)cnt*2+1) c2 <<= 1;
        hcap[L-1]=(int)c2; off[L-1]=(u32)total_cap; mask[L-1]=(u32)(c2-1);
        hk[L-1]=(u64*)malloc((size_t)(c2*8));
        hp[L-1]=(u32*)malloc((size_t)(c2*4));
        hoc[L-1]=(u8*)malloc((size_t)c2);
        memset(hoc[L-1],0,(size_t)c2);
        total_cap += c2;
    }
    int uniq=0;
    u32 uq_cnt[ NLMAX ]; for (int i=0;i<NLMAX;i++) uq_cnt[i]=0;  // v2: 去重后按长度计数，供 tcnt 使用
    for (u64 p=0;p<nprefix;p++) {
        int L=plen[p];
        if (L>nl_max) continue;
        u64 key=0xcbf29ce484222325ull;
        for (int i=0;i<L;i++) key = (key ^ pfx[p*PSTR+i]) * 0x100000001b3ull;
        u32 hh = (u32)((key * 0x9E3779B97F4A7C15ull) >> 32);
        int s = (int)(hh & mask[L-1]);
        u8* oc = hoc[L-1]; u64* kk = hk[L-1]; u32* pp = hp[L-1];
        for (;;) {
            if (!oc[s]) { oc[s]=1; kk[s]=key; pp[s]=(u32)p; uniq++; uq_cnt[L-1]++; break; }
            if (kk[s]==key) {
                size_t q = pp[s]; int same=1;
                for (int i=0;i<L;i++) if (pfx[q*PSTR+i]!=pfx[p*PSTR+i]) { same=0; break; }
                if (same) break;
            }
            s = (s+1) & mask[L-1];
        }
    }
    printf("unique prefixes after dedup: %d\n", uniq);
    if (uniq==0) { for (int L=1;L<=NLMAX;L++){free(hk[L-1]);free(hp[L-1]);free(hoc[L-1]);} return; }
    u64* ck=(u64*)malloc((size_t)(total_cap*8));
    u32* cp=(u32*)malloc((size_t)(total_cap*4));
    u8* co=(u8*)malloc((size_t)total_cap);
    for (int L=1;L<=NLMAX;L++) {
        memcpy(ck+off[L-1], hk[L-1], (size_t)hcap[L-1]*8);
        memcpy(cp+off[L-1], hp[L-1], (size_t)hcap[L-1]*4);
        memcpy(co+off[L-1], hoc[L-1], (size_t)hcap[L-1]);
        free(hk[L-1]); free(hp[L-1]); free(hoc[L-1]);
    }
    u64* d_tbl_key; u32* d_tbl_p; u8* d_tbl_occ;
    u32* d_tbl_off; u32* d_tbl_mask; u32* d_tbl_cnt;
    CUDA_CHECK(cudaMalloc(&d_tbl_key, (size_t)(total_cap*8)));
    CUDA_CHECK(cudaMalloc(&d_tbl_p, (size_t)(total_cap*4)));
    CUDA_CHECK(cudaMalloc(&d_tbl_occ, (size_t)total_cap));
    CUDA_CHECK(cudaMalloc(&d_tbl_off, NLMAX*4));
    CUDA_CHECK(cudaMalloc(&d_tbl_mask, NLMAX*4));
    CUDA_CHECK(cudaMalloc(&d_tbl_cnt, NLMAX*4));
    CUDA_CHECK(cudaMemcpy(d_tbl_key, ck, (size_t)(total_cap*8), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl_p, cp, (size_t)(total_cap*4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl_occ, co, (size_t)total_cap, cudaMemcpyHostToDevice));
    u32 tcnt[ NLMAX ];
    for (int L=1;L<=NLMAX;L++) tcnt[L-1]=uq_cnt[L-1];  // v2: 去重后计数（原版用去重前 lcnt；内核只作非零判断故旧版结果未受影响）
    CUDA_CHECK(cudaMemcpy(d_tbl_off, off, NLMAX*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl_mask, mask, NLMAX*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl_cnt, tcnt, NLMAX*4, cudaMemcpyHostToDevice));
    u8* d_pfx;
    CUDA_CHECK(cudaMalloc(&d_pfx, (size_t)(nprefix*PSTR)));
    CUDA_CHECK(cudaMemcpy(d_pfx, pfx, (size_t)(nprefix*PSTR), cudaMemcpyHostToDevice));
    free(ck); free(cp); free(co);
    u32* d_found; u32* d_nfound;
    CUDA_CHECK(cudaMalloc(&d_found, (size_t)(nprefix*4)));
    CUDA_CHECK(cudaMalloc(&d_nfound, 4));
    ResqEntry* d_resq; u32* d_resq_cnt;
    CUDA_CHECK(cudaMalloc(&d_resq, (size_t)(RESQ_N*sizeof(ResqEntry))));
    CUDA_CHECK(cudaMalloc(&d_resq_cnt, 4));
    CUDA_CHECK(cudaMemset(d_found, 0, (size_t)(nprefix*4)));
    CUDA_CHECK(cudaMemset(d_nfound, 0, 4));
    CUDA_CHECK(cudaMemset(d_resq_cnt, 0, 4));
    if (nidx<1) nidx=1;
    if (nidx>1000000) nidx=1000000;
    // 命中即写文件：trx_hits.txt 追加，无缓冲实时落盘（Ctrl+C 强杀不丢）
    FILE* hf = fopen("trx_hits.txt", "a");
    if (hf) { setvbuf(hf, NULL, _IONBF, 0);
#ifndef _WIN32
        chmod("trx_hits.txt", 0600);  // v2: 助记词+私钥明文文件，权限收紧
#endif
    }
    // 每次启动随机起点：两次运行助记词序列不同
    static u64 trx_run_seed = 0;
    u64 start = (u64)std::chrono::high_resolution_clock::now().time_since_epoch().count()
              ^ (++trx_run_seed * 0x9E3779B97F4A7C15ull);
    int batch_iters=1; u64 total_mnem=0; double total_s=0;
    u32 qprev=0; int printed=0; u64 total_hits=0;
    cudaEvent_t t0,t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    printf("searching random mnemonic m/44'/195'/0'/0/0..%d (nidx=%d), %d blocks x 256 threads... Ctrl+C to stop\n",
           nidx-1, nidx, blocks);
    // 无限运行：全部命中后不退出；每轮重置去重表 -> 同一前缀可持续重复命中并持续记录，仅 Ctrl+C 停止
    while (1) {
        int launched = batch_iters;  // v2: 本轮实际启动的批量；统计与区间推进只准用它（batch_iters 会在轮末被自适应调整）
        CUDA_CHECK(cudaMemset(d_found, 0, (size_t)((u64)nprefix*4)));
        CUDA_CHECK(cudaEventRecord(t0));
        multi_hash_kernel<<<blocks, 256>>>(d_tbl_key, d_tbl_p, d_tbl_occ,
                                           d_tbl_off, d_tbl_mask, d_tbl_cnt, d_pfx, uniq,
                                           nidx, nl_max, start, batch_iters,
                                           d_found, d_nfound, d_resq, d_resq_cnt);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_mnem += (u64)launched*blocks*256;
        total_s += ms/1000.0;
        if (ms > 1000.0f) { int nb=batch_iters/2; if (nb<1) nb=1; batch_iters=nb; }
        else if (ms < 200.0f) { int nb=batch_iters*3; if (nb<1) nb=1; if (nb>1000000) nb=1000000; batch_iters=nb; }
        u32 qcnt=0; CUDA_CHECK(cudaMemcpy(&qcnt, d_resq_cnt, 4, cudaMemcpyDeviceToHost));
        while (qprev < qcnt) {
            ResqEntry e;
            CUDA_CHECK(cudaMemcpy(&e, d_resq+(qprev & RESQ_MASK), sizeof(ResqEntry), cudaMemcpyDeviceToHost));
            qprev++;
            printf("HIT [%4d] \"%s\" -> m/44'/195'/0'/0/%u\n", (int)e.p, (const char*)e.addr, e.idx);
            printf("      mnemonic: %s\n", (const char*)e.mnem);
            printf("      address:  %s\n", (const char*)e.addr);
            printf("      private:  "); for (int i=0;i<32;i++) printf("%02x", e.priv[i]); printf("\n");
            if (hf) {
                fprintf(hf, "HIT [%4d] \"%s\" -> m/44'/195'/0'/0/%u\n", (int)e.p, (const char*)e.addr, e.idx);
                fprintf(hf, "      mnemonic: %s\n", (const char*)e.mnem);
                fprintf(hf, "      address:  %s\n", (const char*)e.addr);
                fprintf(hf, "      private:  ");
                for (int i=0;i<32;i++) fprintf(hf, "%02x", e.priv[i]);
                fprintf(hf, "\n\n");
            }
            printed++; total_hits++;
        }
        // 每轮重置命中队列计数：host 已同步读走本轮全部命中，下轮从 0 写（环形容量 > 单轮最大命中数）
        qprev = 0;
        CUDA_CHECK(cudaMemset(d_resq_cnt, 0, 4));
        double as = (double)total_mnem*nidx/total_s;
        printf("\r  [%7.1f s] %10.0f mnem/s | %13.0f addr/s | hit %llu  (Ctrl+C stop)", total_s, total_mnem/total_s, as, (unsigned long long)total_hits);
        fflush(stdout);
        start += (u64)launched*blocks*256;
    }
    printf("\n(stopped by Ctrl+C) ran %.1f s, total hits %llu\n", total_s, (unsigned long long)total_hits);
    CUDA_CHECK(cudaFree(d_tbl_key)); CUDA_CHECK(cudaFree(d_tbl_p)); CUDA_CHECK(cudaFree(d_tbl_occ));
    CUDA_CHECK(cudaFree(d_tbl_off)); CUDA_CHECK(cudaFree(d_tbl_mask)); CUDA_CHECK(cudaFree(d_tbl_cnt));
    CUDA_CHECK(cudaFree(d_found)); CUDA_CHECK(cudaFree(d_nfound));
    CUDA_CHECK(cudaFree(d_resq)); CUDA_CHECK(cudaFree(d_resq_cnt));
    CUDA_CHECK(cudaFree(d_pfx));
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
}


// v2: 加载或创建 32B secret 文件（默认 ./.trx_vanity_secret，可用环境变量 TRX_VANITY_SECRET 指定路径；与 ETH 版分开存放、互不影响）。
// secret 与计数器共同决定助记词序列；secret 绝不打印、不写入日志。
// 注意：丢了 secret 不影响用 trx_hits.txt 里已记录的助记词找回单个钱包，但无法再复现整条搜索序列。
// 首次创建后请离线备份一份，并保持文件权限 0600。
static int load_or_create_secret(u8 sec[ 32]) {
    const char* path = getenv("TRX_VANITY_SECRET");
    if (!path || !path[0]) path = ".trx_vanity_secret";
    FILE* f = fopen(path, "rb");
    if (f) {
        size_t r = fread(sec, 1, 32, f); fclose(f);
        if (r == 32) {
#ifndef _WIN32
            chmod(path, 0600);
#endif
            printf("secret: loaded from %s (keep it private, back it up offline)\n", path); return 1; }
        fprintf(stderr, "secret file %s exists but is not 32 bytes; refusing to overwrite it\n", path);
        return 0;
    }
    FILE* u = fopen("/dev/urandom", "rb");
    if (u) {
        size_t r = fread(sec, 1, 32, u); fclose(u);
        if (r != 32) { fprintf(stderr, "short read from /dev/urandom\n"); return 0; }
    } else {
        std::random_device rd;  // 无 /dev/urandom 的平台（如原生 Windows）走系统 CSPRNG
        for (int i=0;i<32;i++) sec[i]=(u8)(rd() & 0xFF);
    }
    f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot create secret file %s\n", path); return 0; }
    fwrite(sec, 1, 32, f); fclose(f);
#ifndef _WIN32
    chmod(path, 0600);
#endif
    printf("secret: created %s (0600). Back it up offline; never share it.\n", path);
    return 1;
}

static int chain_main(int argc, char** argv) {
    const char* mode = argc>1 ? argv[1] : "bench";
    printf("=== TRX BIP39 Vanity Generator (true secp256k1 + BIP32/44) ===\n");
    u8 secret[ 32];
    if (!load_or_create_secret(secret)) return 1;
    CUDA_CHECK(cudaMemcpyToSymbol(SECRET_K, secret, 32));

    // comb 固定基点预计算表（60 点）
    u32* d_comb;
    CUDA_CHECK(cudaMalloc(&d_comb, 60*16*4));
    precompute_comb_kernel<<<15, 4>>>(d_comb);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyToSymbol(g_comb, &d_comb, sizeof(d_comb)));
    CUDA_CHECK(cudaDeviceSynchronize());  // v2: 确保 comb 表已写完，后续任何流上的 kernel 都不会读到未就绪的表

    u8 prefix_nib[ 40];
    int plen=0;

    u8* d_prefix;
    char* d_mnem; u8* d_addr; u8* d_priv; u32* d_found; u32* d_index;
    CUDA_CHECK(cudaMalloc(&d_prefix, 40));
    CUDA_CHECK(cudaMalloc(&d_mnem, 128));
    CUDA_CHECK(cudaMalloc(&d_addr, 20));
    CUDA_CHECK(cudaMalloc(&d_priv, 32));
    CUDA_CHECK(cudaMalloc(&d_found, 4));
    CUDA_CHECK(cudaMalloc(&d_index, 4));
    if (plen) CUDA_CHECK(cudaMemcpy(d_prefix, prefix_nib, plen, cudaMemcpyHostToDevice));

    if (strcmp(mode,"selftest")==0) {
        u64* d_out; u32* d_done;
        CUDA_CHECK(cudaMalloc(&d_out, 11*8));
        CUDA_CHECK(cudaMalloc(&d_done, 4));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK));
        CUDA_CHECK(cudaStreamCreate(&sP));
        selftest_kernel<<<1,1,0,sK>>>(d_out, d_done);
        for (int w=0;w<40;w++) {
            u32 p=0;
            cudaMemcpyAsync(&p, d_done, 4, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            if (p==1) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
        u64 h_out[ 11];
        cudaMemcpyAsync(h_out, d_out, 11*8, cudaMemcpyDeviceToHost, sP);
        cudaStreamSynchronize(sP);
        printf("selftest:\n");
        printf("  (p-1)+1        = %016llx %016llx\n", (unsigned long long)(h_out[ 0]>>32), (unsigned long long)(h_out[ 0]&0xFFFFFFFFFFFFFFFFULL));
        printf("  123456789*987..= %016llx %016llx\n", (unsigned long long)(h_out[ 1]>>32), (unsigned long long)h_out[ 1]);
        printf("  2G x           = %016llx %016llx\n", (unsigned long long)(h_out[ 2]>>32), (unsigned long long)h_out[ 2]);
        printf("  2G y           = %016llx %016llx\n", (unsigned long long)(h_out[ 3]>>32), (unsigned long long)h_out[ 3]);
        printf("  1G x           = %016llx %016llx\n", (unsigned long long)(h_out[ 4]>>32), (unsigned long long)h_out[ 4]);
        printf("  1G y           = %016llx %016llx\n", (unsigned long long)(h_out[ 5]>>32), (unsigned long long)h_out[ 5]);
        printf("  (n-1)+1 mod n  = %016llx %016llx\n", (unsigned long long)(h_out[ 6]>>32), (unsigned long long)h_out[ 6]);
        printf("  (n-2)+3 mod n  = %016llx %016llx\n", (unsigned long long)(h_out[ 7]>>32), (unsigned long long)h_out[ 7]);
        printf("  2G+G x (jac)   = %016llx %016llx\n", (unsigned long long)(h_out[ 8]>>32), (unsigned long long)h_out[ 8]);
        printf("  3G x (scalar)  = %016llx %016llx\n", (unsigned long long)(h_out[ 9]>>32), (unsigned long long)h_out[ 9]);
        printf("  a*a^-1         = %016llx %016llx\n", (unsigned long long)(h_out[ 10]>>32), (unsigned long long)h_out[ 10]);
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_done));
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
    }
    else if (strcmp(mode,"fieldtest")==0) {
        // ftprog 已验证模式：自定义流 + 异步轮询，不调用 cudaDeviceSynchronize
        u32* d_out; u32* d_prog;
        CUDA_CHECK(cudaMalloc(&d_out, 8*8*4));
        CUDA_CHECK(cudaMalloc(&d_prog, 4));
        CUDA_CHECK(cudaMemset(d_prog, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK));
        CUDA_CHECK(cudaStreamCreate(&sP));
        ftprog_kernel<<<1,1,0,sK>>>(d_out, d_prog);
        for (int t=0;t<30;t++) {
            u32 p=0;
            cudaMemcpyAsync(&p, d_prog, 4, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            printf("t=%d prog=%u\n", t, p);
            fflush(stdout);
            if (p>=9) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
        u32 h_out[ 64];
        cudaError_t merr = cudaMemcpy(h_out, d_out, 8*8*4, cudaMemcpyDeviceToHost);
        printf("sync-copy err=%s\n", cudaGetErrorString(merr));
        const char* names[ 8] = {
            "(p-1)*(p-1)      ", "(p-2)*(p-1)      ", "2^255 * 2^255     ",
            "aaaa.. * aaaa..  ", "1234567890abcdef^2", "0xffff...+0x100000001",
            "inv(a)*a          ", "(n-1)+2 mod n     "
        };
        for (int t=0;t<8;t++) {
            printf("%s = ", names[t]);
            for (int i=7;i>=0;i--) printf("%08x", h_out[t*8+i]);
            printf("\n");
        }
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_prog));
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
    }
    else if (strcmp(mode,"ftprog")==0) {
        u32* d_out; u32* d_prog;
        CUDA_CHECK(cudaMalloc(&d_out, 8*8*4));
        CUDA_CHECK(cudaMalloc(&d_prog, 4));
        CUDA_CHECK(cudaMemset(d_prog, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK));
        CUDA_CHECK(cudaStreamCreate(&sP));
        ftprog_kernel<<<1,1,0,sK>>>(d_out, d_prog);
        for (int t=0;t<30;t++) {
            u32 p=0;
            cudaMemcpyAsync(&p, d_prog, 4, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            printf("t=%d prog=%u\n", t, p);
            fflush(stdout);
            if (p>=9) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_prog));
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
    }
    else if (strcmp(mode,"bisect")==0) {
        int c = (argc>2)?atoi(argv[ 2]):0;
        u32* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, 8*4));
        bisect_kernel<<<1,1>>>(d_out, c);
        CUDA_CHECK(cudaGetLastError());
        u32 h_out[ 8];
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 8*4, cudaMemcpyDeviceToHost));
        printf("case %d = ", c);
        for (int i=7;i>=0;i--) printf("%08x", h_out[i]);
        printf("\n");
        CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"debug")==0) {
        u32* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, 44*4));
        debug_kernel<<<1,1>>>(d_out);
        CUDA_CHECK(cudaGetLastError());
        u32 h_out[ 44];
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 44*4, cudaMemcpyDeviceToHost));
        printf("debug (p-1)^2 intermediates:\n");
        printf("  raw product t[0..15]:\n    ");
        for (int i=0;i<16;i++) printf("%08x ", h_out[i]);
        printf("\n  v after initial fold:\n    ");
        for (int i=0;i<10;i++) printf("%08x ", h_out[16+i]);
        printf("\n  v after fe_fold_hi:\n    ");
        for (int i=0;i<10;i++) printf("%08x ", h_out[26+i]);
        printf("\n  fe_mul r:\n    ");
        for (int i=0;i<8;i++) printf("%08x ", h_out[36+i]);
        printf("\n");
        CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"jdbl")==0) {
        u32* d_done; CUDA_CHECK(cudaMalloc(&d_done, 4));
        u32* d_out; CUDA_CHECK(cudaMalloc(&d_out, 256));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        jdbl_kernel<<<1,1,0,sK>>>(d_out, d_done);
        for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
        u32 hb[ 64];
        cudaMemcpyAsync(hb, d_out, 256, cudaMemcpyDeviceToHost, sP);
        cudaStreamSynchronize(sP);
        const char* names[ 10] = { "A=X^2", "B=Y^2", "C=B^2", "D=2t", "E=3A", "F0=E^2", "t0=2D", "X3", "Y3", "Z3" };
        int o=0;
        for (int q=0;q<10;q++){ printf("%s: ",names[q]); for(int i=7;i>=0;i--) printf("%08x",hb[o++]); printf("\n"); }
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        CUDA_CHECK(cudaFree(d_done)); CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"sm")==0) {
        u32* d_done; CUDA_CHECK(cudaMalloc(&d_done, 4));
        u32* d_out; CUDA_CHECK(cudaMalloc(&d_out, 128));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        for (int cc=0;cc<6;cc++) {
            CUDA_CHECK(cudaMemset(d_done, 0, 4));
            sm_kernel<<<1,1,0,sK>>>(d_out, d_done, cc);
            for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
            u32 hb[ 24];
            cudaMemcpyAsync(hb, d_out, 96, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            const char* nm[ 6] = { "k0h", "2^255", "k0h_top32", "0xe9", "2^247", "10G_jac" };
            if (cc==1 || cc==5) {
                printf("%s X: ", nm[cc]); for(int i=7;i>=0;i--) printf("%08x",hb[i]); printf("\n");
                printf("%s Y: ", nm[cc]); for(int i=15;i>=8;i--) printf("%08x",hb[i]); printf("\n");
                printf("%s Z: ", nm[cc]); for(int i=23;i>=16;i--) printf("%08x",hb[i]); printf("\n");
            } else {
                printf("%s x: ", nm[cc]); for(int i=7;i>=0;i--) printf("%08x",hb[i]); printf("\n");
                printf("%s y: ", nm[cc]); for(int i=15;i>=8;i--) printf("%08x",hb[i]); printf("\n");
            }
        }
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        CUDA_CHECK(cudaFree(d_done)); CUDA_CHECK(cudaFree(d_out));
    }

    else if (strcmp(mode,"ktest")==0) {
        u32* d_done; CUDA_CHECK(cudaMalloc(&d_done, 4));
        u8* d_out; CUDA_CHECK(cudaMalloc(&d_out, 128));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        ktest_kernel<<<1,1,0,sK>>>(d_out, d_done);
        for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
        u8 hb[ 96];
        cudaMemcpyAsync(hb, d_out, 96, cudaMemcpyDeviceToHost, sP);
        cudaStreamSynchronize(sP);
        printf("keccak(""):  "); for(int i=0;i<32;i++) printf("%02x",hb[i]); printf("\n");
        printf("keccak(abc): "); for(int i=0;i<32;i++) printf("%02x",hb[32+i]); printf("\n");
        printf("keccak(pub): "); for(int i=0;i<32;i++) printf("%02x",hb[64+i]); printf("\n");
        printf("f1600(0) st0: "); for(int i=0;i<8;i++) printf("%02x",hb[96+i]); printf("\n");
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        CUDA_CHECK(cudaFree(d_done)); CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"ptest")==0) {
        u32* d_done; CUDA_CHECK(cudaMalloc(&d_done, 4));
        u32* d_out; CUDA_CHECK(cudaMalloc(&d_out, 256));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        ptest_kernel<<<1,1,0,sK>>>(d_out, d_done);
        for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
        u32 hb[ 64];
        cudaMemcpyAsync(hb, d_out, 256, cudaMemcpyDeviceToHost, sP);
        cudaStreamSynchronize(sP);
        int o=0;
        printf("2G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("2G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("3G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("3G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("4G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("4G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("5G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("5G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("7G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("7G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("8G x: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("8G y: "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        CUDA_CHECK(cudaFree(d_done)); CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"scdiag")==0) {
        u32* d_done; CUDA_CHECK(cudaMalloc(&d_done, 4));
        u32* d_out; CUDA_CHECK(cudaMalloc(&d_out, 512));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        scdiag_kernel<<<1,1,0,sK>>>(d_out, d_done);
        for (int w=0;w<50;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
        u32 hb[ 128];
        cudaMemcpyAsync(hb, d_out, 512, cudaMemcpyDeviceToHost, sP);
        cudaStreamSynchronize(sP);
        int o=0;
        printf("il:  "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("kf:  "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("t0-8:"); for(int i=0;i<9;i++) printf("%08x",hb[o++]); printf("\n");
        printf("r:   "); for(int i=0;i<8;i++) printf("%08x",hb[o++]); printf("\n");
        printf("fe_ge(t,N):%u  t8:%u\n", hb[o], hb[o+1]);
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        CUDA_CHECK(cudaFree(d_done)); CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"vec")==0) {
        const char* ents[ 1] = { "00000000000000000000000000000000" };
        u32* d_done;
        CUDA_CHECK(cudaMalloc(&d_done, 4));
        u8* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, 600));
        for (int t=0;t<1;t++) {
            u8 ent[ 16];
            for (int i=0;i<16;i++){ int a,b; char ch1=ents[t][2*i], ch2=ents[t][2*i+1];
                a=(ch1<='9')?ch1-'0':ch1-'a'+10; b=(ch2<='9')?ch2-'0':ch2-'a'+10; ent[i]=(u8)((a<<4)|b); }
            u8* d_ent;
            CUDA_CHECK(cudaMalloc(&d_ent, 16));
            CUDA_CHECK(cudaMemcpy(d_ent, ent, 16, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_done, 0, 4));
            cudaStream_t sK, sP;
            CUDA_CHECK(cudaStreamCreate(&sK));
            CUDA_CHECK(cudaStreamCreate(&sP));
            vec_kernel<<<1,1,0,sK>>>(d_ent, d_out, d_done);
            for (int w=0;w<50;w++) {
                u32 p=0;
                cudaMemcpyAsync(&p, d_done, 4, cudaMemcpyDeviceToHost, sP);
                cudaStreamSynchronize(sP);
                if (p==1) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
            }
            u8 hb[ 600];
            cudaMemcpyAsync(hb, d_out, 600, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            int o=0;
            printf("entropy: "); for(int i=0;i<16;i++) printf("%02x",hb[o++]); printf("\n");
            printf("seed:    "); for(int i=0;i<64;i++) printf("%02x",hb[o++]); printf("\n");
            printf("mK:      "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("mC:      "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("k44:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("c44:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("k60:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("c60:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("k0h:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("c0h:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("k0n:     "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("k0n2:    "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("pubX:    "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("pubY:    "); for(int i=0;i<32;i++) printf("%02x",hb[o++]); printf("\n");
            printf("addr:    "); for(int i=0;i<20;i++) printf("%02x",hb[o++]); printf("\n");
            printf("mnem:    %s\n", (char*)(hb+o));
            cudaStreamDestroy(sK); cudaStreamDestroy(sP);
            CUDA_CHECK(cudaFree(d_ent));
        }
        CUDA_CHECK(cudaFree(d_done));
        CUDA_CHECK(cudaFree(d_out));
    }
    else if (strcmp(mode,"test")==0) {
        const char* ents[ 3] = {
            "00000000000000000000000000000000",
            "7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f",
            "80808080808080808080808080808080"
        };
        u32* d_done;
        CUDA_CHECK(cudaMalloc(&d_done, 4));
        for (int t=0;t<3;t++) {
            u8 ent[ 16];
            for (int i=0;i<16;i++){ int a,b; char ch1=ents[t][2*i], ch2=ents[t][2*i+1];
                a=(ch1<='9')?ch1-'0':ch1-'a'+10; b=(ch2<='9')?ch2-'0':ch2-'a'+10; ent[i]=(u8)((a<<4)|b); }
            u8* d_ent;
            CUDA_CHECK(cudaMalloc(&d_ent, 16));
            CUDA_CHECK(cudaMemcpy(d_ent, ent, 16, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_done, 0, 4));
            cudaStream_t sK, sP;
            CUDA_CHECK(cudaStreamCreate(&sK));
            CUDA_CHECK(cudaStreamCreate(&sP));
            test_kernel<<<1,1,0,sK>>>(d_ent, d_mnem, d_addr, d_priv, d_done);
            for (int w=0;w<40;w++) {
                u32 p=0;
                cudaMemcpyAsync(&p, d_done, 4, cudaMemcpyDeviceToHost, sP);
                cudaStreamSynchronize(sP);
                if (p==1) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
            }
            char mnem[ 128]; u8 addr[ 20], privk[ 32];
            cudaMemcpyAsync(mnem, d_mnem, 128, cudaMemcpyDeviceToHost, sP);
            cudaMemcpyAsync(addr, d_addr, 20, cudaMemcpyDeviceToHost, sP);
            cudaMemcpyAsync(privk, d_priv, 32, cudaMemcpyDeviceToHost, sP);
            cudaStreamSynchronize(sP);
            printf("entropy %s:\n", ents[t]);
            printf("  mnemonic: %s\n", mnem);
            printf("  addr:     0x"); print_hex(addr,20); printf("\n");
            printf("  priv:     0x"); print_hex(privk,32); printf("\n");
            cudaStreamDestroy(sK); cudaStreamDestroy(sP);
            CUDA_CHECK(cudaFree(d_ent));
        }
        CUDA_CHECK(cudaFree(d_done));
    }

    else if (strcmp(mode,"prof")==0) {
        int stage = (argc>=3) ? atoi(argv[2]) : 3;
        int iters = (argc>=4) ? atoi(argv[3]) : 1;
        int blocks = (argc>=5) ? atoi(argv[4]) : 1024;
        u64* d_out; u32* d_done;
        CUDA_CHECK(cudaMalloc(&d_out, 1024*8));
        CUDA_CHECK(cudaMalloc(&d_done, 4));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        cudaStream_t sK, sP;
        CUDA_CHECK(cudaStreamCreate(&sK)); CUDA_CHECK(cudaStreamCreate(&sP));
        cudaEvent_t t0,t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        cudaEventRecord(t0);
        prof_kernel<<<blocks, 256, 0, sK>>>(0, iters, stage, d_out, d_done);
        CUDA_CHECK(cudaGetLastError());
        for (int w=0;w<100;w++){ u32 p=0; cudaMemcpyAsync(&p,d_done,4,cudaMemcpyDeviceToHost,sP); cudaStreamSynchronize(sP); if(p==1)break; std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
        cudaEventRecord(t1, sK);
        cudaEventSynchronize(t1);
        float ms=0; cudaEventElapsedTime(&ms, t0, t1);
        u64 total=(u64)iters*blocks*256;
        printf("prof stage=%d: %llu attempts in %.2f s = %.1f addr/s\n", stage, (unsigned long long)total, ms/1000.0, total/(ms/1000.0));
        CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_done));
        cudaStreamDestroy(sK); cudaStreamDestroy(sP);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    else if (strcmp(mode,"hd")==0 || strcmp(mode,"hddiag")==0 || strcmp(mode,"hddiag2")==0) {
        fprintf(stderr, "HD mode is not available in the BTC build (vanity/file modes only)\n");
        return 1;
    }
    else if (strcmp(mode,"trxtest")==0) {
        u8* d_out; u32* d_done;
        CUDA_CHECK(cudaMalloc(&d_out, 768));
        CUDA_CHECK(cudaMalloc(&d_done, 4));
        CUDA_CHECK(cudaMemset(d_done, 0, 4));
        trx_test_kernel<<<1,1>>>(d_out, d_done);
        CUDA_CHECK(cudaGetLastError());
        u32 done=0; while (!done) { CUDA_CHECK(cudaMemcpy(&done, d_done, 4, cudaMemcpyDeviceToHost)); }
        u8 hb[ 768]; CUDA_CHECK(cudaMemcpy(hb, d_out, 768, cudaMemcpyDeviceToHost));
        for (int q=0;q<2;q++) {
            int o=q*320;
            printf("N=%d\n", q==0?0:3);
            printf("  priv:      "); print_hex(hb+o, 32); printf("\n");
            printf("  pub_un:    "); print_hex(hb+o+32, 65); printf("\n");
            printf("  address:   %s\n", hb+o+97);
        }
        printf("  mnemonic:  %s\n", hb+640);
        CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_done));
    }

    else if (strcmp(mode,"file")==0) {
        if (argc<3) { fprintf(stderr,"usage: file <prefix_file> [nidx] [blocks]\n"); return 1; }
        const char* fn = argv[2];
        int nidx = (argc>=4) ? atoi(argv[3]) : 1;
        int blocks = (argc>=5) ? atoi(argv[4]) : 1024;
        int nl_max = TRX_NLMAX;
        FILE* fp = fopen(fn, "r");
        if (!fp) { fprintf(stderr,"cannot open %s\n", fn); return 1; }
        u64 cap = 1<<16, nprefix=0, nskip=0;
        u8* pfx = (u8*)malloc((size_t)(cap*PSTR));
        u8* plen = (u8*)malloc((size_t)cap);
        char line[ 512 ];
        while (fgets(line, sizeof(line), fp)) {
            int a=0; while (line[a]==' '||line[a]=='\t'||line[a]=='\r'||line[a]=='\n') a++;
            int b=0; while (line[a+b]!='\0' && line[a+b]!='\n' && line[a+b]!='\r' && line[a+b]!='#' && line[a+b]!=' ' && line[a+b]!='\t') b++;
            if (b==0) continue;
            if (b > nl_max) { nskip++; continue; }
            const char* p = line + a;
            for (int i=0;i<b;i++) pfx[(size_t)(nprefix*PSTR+i)]=(u8)p[i];
            plen[(size_t)nprefix]=(u8)b;
            nprefix++;
            if (nprefix >= cap) { cap*=2; pfx=(u8*)realloc(pfx,(size_t)(cap*PSTR)); plen=(u8*)realloc(plen,(size_t)cap); }
        }
        fclose(fp);
        if (nprefix==0) { fprintf(stderr,"no valid prefixes in file\n"); return 1; }
        printf("file mode: read %llu string prefixes (skipped %llu with len>%d)\n", (unsigned long long)nprefix, (unsigned long long)nskip, nl_max);
        run_trx_search(pfx, plen, nprefix, nidx, blocks);
        free(pfx); free(plen);
    }
    else if (strcmp(mode,"vanity")==0) {
        if (argc<3) { fprintf(stderr,"usage: vanity <prefix> [nidx] [blocks]\n"); return 1; }
        const char* p = argv[2];
        int nidx = (argc>=4) ? atoi(argv[3]) : 1;
        int blocks = (argc>=5) ? atoi(argv[4]) : 1024;
        int nl_max = TRX_NLMAX;
        int len = (int)strlen(p);
        if (len==0) { fprintf(stderr,"bad prefix\n"); return 1; }
        if (len > nl_max) { fprintf(stderr,"prefix too long (max %d chars)\n", nl_max); return 1; }
        u8 pfx[ PSTR ]; u8 plen[ 1 ];
        for (int i=0;i<len;i++) pfx[i]=(u8)p[i];
        plen[0]=(u8)len;
        printf("vanity prefix: \"%s\" (%d chars), TRON T-address\n", p, len);
        run_trx_search(pfx, plen, 1, nidx, blocks);
    }
    else { // bench: TRX 地址生成吞吐（不查表）
        int iters = (argc>=3) ? atoi(argv[2]) : 100;
        int blocks = (argc>=4) ? atoi(argv[3]) : 1024;
        u64* d_tbl_key; u32* d_tbl_p; u8* d_tbl_occ;
        u32* d_tbl_off; u32* d_tbl_mask; u32* d_tbl_cnt;
        CUDA_CHECK(cudaMalloc(&d_tbl_key, 8*8));
        CUDA_CHECK(cudaMalloc(&d_tbl_p, 8*4));
        CUDA_CHECK(cudaMalloc(&d_tbl_occ, 8));
        CUDA_CHECK(cudaMalloc(&d_tbl_off, NLMAX*4));
        CUDA_CHECK(cudaMalloc(&d_tbl_mask, NLMAX*4));
        CUDA_CHECK(cudaMalloc(&d_tbl_cnt, NLMAX*4));
        CUDA_CHECK(cudaMemset(d_tbl_occ, 0, 8));
        u8* d_pfx2; CUDA_CHECK(cudaMalloc(&d_pfx2, 1));
        u32* d_found2; u32* d_nfound; ResqEntry* d_resq; u32* d_resq_cnt;
        CUDA_CHECK(cudaMalloc(&d_found2, 4));
        CUDA_CHECK(cudaMalloc(&d_nfound, 4));
        CUDA_CHECK(cudaMalloc(&d_resq, (size_t)(RESQ_N*sizeof(ResqEntry))));
        CUDA_CHECK(cudaMalloc(&d_resq_cnt, 4));
        CUDA_CHECK(cudaMemset(d_found2, 0, 4)); CUDA_CHECK(cudaMemset(d_nfound, 0, 4)); CUDA_CHECK(cudaMemset(d_resq_cnt, 0, 4));
        cudaEvent_t t0,t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        multi_hash_kernel<<<blocks, 256>>>(d_tbl_key, d_tbl_p, d_tbl_occ,
                                           d_tbl_off, d_tbl_mask, d_tbl_cnt, d_pfx2, 0,
                                           1, TRX_NLMAX, 0, iters,
                                           d_found2, d_nfound, d_resq, d_resq_cnt);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        u64 total=(u64)iters*blocks*256;
        printf("bench TRX: %llu addresses in %.2f s = %.1f addr/s\n",
               (unsigned long long)total, ms/1000.0, total/(ms/1000.0));
        CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
        CUDA_CHECK(cudaFree(d_tbl_key)); CUDA_CHECK(cudaFree(d_tbl_p)); CUDA_CHECK(cudaFree(d_tbl_occ));
        CUDA_CHECK(cudaFree(d_tbl_off)); CUDA_CHECK(cudaFree(d_tbl_mask)); CUDA_CHECK(cudaFree(d_tbl_cnt));
        CUDA_CHECK(cudaFree(d_pfx2));
        CUDA_CHECK(cudaFree(d_found2)); CUDA_CHECK(cudaFree(d_nfound)); CUDA_CHECK(cudaFree(d_resq)); CUDA_CHECK(cudaFree(d_resq_cnt));
    }

    CUDA_CHECK(cudaFree(d_prefix)); CUDA_CHECK(cudaFree(d_mnem));
    CUDA_CHECK(cudaFree(d_addr)); CUDA_CHECK(cudaFree(d_priv)); CUDA_CHECK(cudaFree(d_found));
    CUDA_CHECK(cudaFree(d_index));
    return 0;
}

} // namespace trx
#undef CUDA_CHECK
#undef MAXP
#undef PSTR
#undef NLMAX
#undef RESQ_N
#undef RESQ_MASK
#undef TRX_NLMAX

// ============================ chain: sol ============================
#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Err: %s (line %d)\n", cudaGetErrorString(err), __LINE__); exit(1); } } while (0)
#define PSTR 48          // 每行前缀存储宽度（Solana 地址最长 44）
#define NLMAX 44         // 最大前缀长度
#define NW 51            // 5-bit 窗口数 (51*5 = 255 bit)
#define TBLN 32          // 每窗口 32 项
#define RESQ_CAP (1<<19)
#define RESQ_MASK (RESQ_CAP-1)
namespace sol {
// ============================================================================
//  sol_vanity_v2.cu  -  Solana vanity address generator (v2)
//  rand seed(32B) -> SHA512 -> clamp -> edwards25519 a*B -> base58 地址
//
//  v2 相对旧脚本化的增强：
//    * 大前缀文件（2GB+）流式解析，u64 计数
//    * 命中实时落盘 sol_hits.txt（无缓冲追加，Ctrl+C 强杀不丢）
//    * 每次运行随机 salt：同样的前缀列表不会每次都命中同一批地址
//    * 命中不退出：无限跑到全部唯一前缀命中或 Ctrl+C
//    * 不可能前缀预检（长度>44 / 非 base58 字符集）
//    * 输出可导入钱包的 secret key（base58 64B = seed||pubkey，等同 Phantom 私钥）
// ============================================================================

typedef uint8_t  u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t  i32;
typedef int64_t  i64;



// ============================================================================
//  fe: GF(2^255-19)，radix 2^25.5 (10 limbs)，ref10 风格
// ============================================================================
struct fe { i32 v[10]; };

__device__ __constant__ fe C_D2  = {{ 45281625,27714825,36363642,13898781,229458,15978800,54557047,27058993,29715967,9444199 }};
__device__ __constant__ fe C_BX  = {{ 52811034,25909283,16144682,17082669,27570973,30858332,40966398,8378388,20764389,8758491 }};
__device__ __constant__ fe C_BY  = {{ 40265304,26843545,13421772,20132659,26843545,6710886,53687091,13421772,40265318,26843545 }};

__device__ __forceinline__ void fe_zero(fe& h) {
    #pragma unroll
    for (int i=0;i<10;i++) h.v[i]=0; }
__device__ __forceinline__ void fe_one(fe& h)  {
    #pragma unroll
    for (int i=0;i<10;i++) h.v[i]=0; h.v[0]=1; }
__device__ __forceinline__ void fe_copy(fe& h, const fe& f) {
    #pragma unroll
    for (int i=0;i<10;i++) h.v[i]=f.v[i]; }
__device__ __forceinline__ int fe_iszero(const fe& f) {
    int z=1;
    #pragma unroll
    for (int i=0;i<10;i++) z &= (f.v[i]==0); return z; }

__device__ __forceinline__ void fe_add(fe& h, const fe& f, const fe& g) {
    i32 t0=f.v[0]+g.v[0], t1=f.v[1]+g.v[1], t2=f.v[2]+g.v[2], t3=f.v[3]+g.v[3], t4=f.v[4]+g.v[4];
    i32 t5=f.v[5]+g.v[5], t6=f.v[6]+g.v[6], t7=f.v[7]+g.v[7], t8=f.v[8]+g.v[8], t9=f.v[9]+g.v[9];
    i32 c;
    c=(t0+(1<<25))>>26; t1+=c; t0-=c<<26;
    c=(t1+(1<<24))>>25; t2+=c; t1-=c<<25;
    c=(t2+(1<<25))>>26; t3+=c; t2-=c<<26;
    c=(t3+(1<<24))>>25; t4+=c; t3-=c<<25;
    c=(t4+(1<<25))>>26; t5+=c; t4-=c<<26;
    c=(t5+(1<<24))>>25; t6+=c; t5-=c<<25;
    c=(t6+(1<<25))>>26; t7+=c; t6-=c<<26;
    c=(t7+(1<<24))>>25; t8+=c; t7-=c<<25;
    c=(t8+(1<<25))>>26; t9+=c; t8-=c<<26;
    c=(t9+(1<<24))>>25; t0+=c*19; t9-=c<<25;
    h.v[0]=t0; h.v[1]=t1; h.v[2]=t2; h.v[3]=t3; h.v[4]=t4;
    h.v[5]=t5; h.v[6]=t6; h.v[7]=t7; h.v[8]=t8; h.v[9]=t9;
}

__device__ __forceinline__ void fe_sub(fe& h, const fe& f, const fe& g) {
    i32 t0=f.v[0]-g.v[0], t1=f.v[1]-g.v[1], t2=f.v[2]-g.v[2], t3=f.v[3]-g.v[3], t4=f.v[4]-g.v[4];
    i32 t5=f.v[5]-g.v[5], t6=f.v[6]-g.v[6], t7=f.v[7]-g.v[7], t8=f.v[8]-g.v[8], t9=f.v[9]-g.v[9];
    i32 c;
    c=(t0+(1<<25))>>26; t1+=c; t0-=c<<26;
    c=(t1+(1<<24))>>25; t2+=c; t1-=c<<25;
    c=(t2+(1<<25))>>26; t3+=c; t2-=c<<26;
    c=(t3+(1<<24))>>25; t4+=c; t3-=c<<25;
    c=(t4+(1<<25))>>26; t5+=c; t4-=c<<26;
    c=(t5+(1<<24))>>25; t6+=c; t5-=c<<25;
    c=(t6+(1<<25))>>26; t7+=c; t6-=c<<26;
    c=(t7+(1<<24))>>25; t8+=c; t7-=c<<25;
    c=(t8+(1<<25))>>26; t9+=c; t8-=c<<26;
    c=(t9+(1<<24))>>25; t0+=c*19; t9-=c<<25;
    h.v[0]=t0; h.v[1]=t1; h.v[2]=t2; h.v[3]=t3; h.v[4]=t4;
    h.v[5]=t5; h.v[6]=t6; h.v[7]=t7; h.v[8]=t8; h.v[9]=t9;
}

__device__ __forceinline__ void fe_neg(fe& h, const fe& f) { fe z; fe_zero(z); fe_sub(h, z, f); }

// 10x10 卷积 + 19 折叠。d = weight(i)+weight(j)-weight(i+j) = 1 仅当 i,j 均奇。
__device__ __noinline__ void fe_mul(fe& h, const fe& f, const fe& g) {
    i64 t[19];
    #pragma unroll
    for (int k=0;k<19;k++) t[k]=0;
    #pragma unroll
    for (int i=0;i<10;i++) {
        #pragma unroll
        for (int j=0;j<10;j++) {
            i64 p = (i64)f.v[i] * (i64)g.v[j];
            if ((i&1) && (j&1)) p += p;
            t[i+j] += p;
        }
    }
    #pragma unroll
    for (int k=10;k<19;k++) t[k-10] += 19*t[k];
    i64 c;
    c=(t[0]+(1LL<<25))>>26; t[1]+=c; t[0]-=c*(1LL<<26);
    c=(t[1]+(1LL<<24))>>25; t[2]+=c; t[1]-=c*(1LL<<25);
    c=(t[2]+(1LL<<25))>>26; t[3]+=c; t[2]-=c*(1LL<<26);
    c=(t[3]+(1LL<<24))>>25; t[4]+=c; t[3]-=c*(1LL<<25);
    c=(t[4]+(1LL<<25))>>26; t[5]+=c; t[4]-=c*(1LL<<26);
    c=(t[5]+(1LL<<24))>>25; t[6]+=c; t[5]-=c*(1LL<<25);
    c=(t[6]+(1LL<<25))>>26; t[7]+=c; t[6]-=c*(1LL<<26);
    c=(t[7]+(1LL<<24))>>25; t[8]+=c; t[7]-=c*(1LL<<25);
    c=(t[8]+(1LL<<25))>>26; t[9]+=c; t[8]-=c*(1LL<<26);
    c=(t[9]+(1LL<<24))>>25; t[0]+=19*c; t[9]-=c*(1LL<<25);
    c=(t[0]+(1LL<<25))>>26; t[1]+=c; t[0]-=c*(1LL<<26);
    #pragma unroll
    for (int i=0;i<10;i++) h.v[i]=(i32)t[i];
}

__device__ __forceinline__ void fe_sq(fe& h, const fe& f) { fe_mul(h, f, f); }

// a^(p-2)  (ref10 pow225521 链：255 平方 + 11 乘)
__device__ __noinline__ void fe_invert(fe& out, const fe& z) {
    fe t0,t1,t2,t3;
    fe_sq(t0,z);                    // 2
    fe_sq(t1,t0);                   // 4
    fe_sq(t1,t1);                   // 8
    fe_mul(t1,z,t1);                // 9
    fe_mul(t0,t0,t1);               // 11
    fe_sq(t2,t0);                   // 22
    fe_mul(t1,t1,t2);               // 31 = 2^5-1
    fe_sq(t2,t1);
    #pragma unroll
    for (int i=1;i<5;i++) fe_sq(t2,t2);
    fe_mul(t1,t2,t1);               // 2^10-1
    fe_sq(t2,t1);
    #pragma unroll
    for (int i=1;i<10;i++) fe_sq(t2,t2);
    fe_mul(t2,t2,t1);               // 2^20-1
    fe_sq(t3,t2);
    #pragma unroll
    for (int i=1;i<20;i++) fe_sq(t3,t3);
    fe_mul(t2,t3,t2);               // 2^40-1
    fe_sq(t2,t2);
    #pragma unroll
    for (int i=1;i<10;i++) fe_sq(t2,t2);
    fe_mul(t1,t2,t1);               // 2^50-1
    fe_sq(t2,t1);
    #pragma unroll
    for (int i=1;i<50;i++) fe_sq(t2,t2);
    fe_mul(t2,t2,t1);               // 2^100-1
    fe_sq(t3,t2);
    #pragma unroll
    for (int i=1;i<100;i++) fe_sq(t3,t3);
    fe_mul(t2,t3,t2);               // 2^200-1
    fe_sq(t2,t2);
    #pragma unroll
    for (int i=1;i<50;i++) fe_sq(t2,t2);
    fe_mul(t1,t2,t1);               // 2^250-1
    fe_sq(t1,t1);
    #pragma unroll
    for (int i=1;i<5;i++) fe_sq(t1,t1);
    fe_mul(out,t1,t0);              // 2^255-21 = p-2
}

__device__ __noinline__ void fe_tobytes(u8 s[32], const fe& hin) {
    i32 t0=hin.v[0],t1=hin.v[1],t2=hin.v[2],t3=hin.v[3],t4=hin.v[4];
    i32 t5=hin.v[5],t6=hin.v[6],t7=hin.v[7],t8=hin.v[8],t9=hin.v[9];
    i32 q;
    q = (19*t9 + (1<<24)) >> 25;
    q = (t0 + q) >> 26;
    q = (t1 + q) >> 25;
    q = (t2 + q) >> 26;
    q = (t3 + q) >> 25;
    q = (t4 + q) >> 26;
    q = (t5 + q) >> 25;
    q = (t6 + q) >> 26;
    q = (t7 + q) >> 25;
    q = (t8 + q) >> 26;
    q = (t9 + q) >> 25;
    t0 += 19*q;
    i32 c;
    // 这里必须用截断式进位（不是 ref10 fe_add/fe_mul 的四舍五入式），
    // 否则 limb 会变负、打包时的 << 会带出符号位。
    c=(t0)>>26; t1+=c; t0-=c<<26;
    c=(t1)>>25; t2+=c; t1-=c<<25;
    c=(t2)>>26; t3+=c; t2-=c<<26;
    c=(t3)>>25; t4+=c; t3-=c<<25;
    c=(t4)>>26; t5+=c; t4-=c<<26;
    c=(t5)>>25; t6+=c; t5-=c<<25;
    c=(t6)>>26; t7+=c; t6-=c<<26;
    c=(t7)>>25; t8+=c; t7-=c<<25;
    c=(t8)>>26; t9+=c; t8-=c<<26;
    c=(t9)>>25; t9-=c<<25;
    s[ 0]=(u8)(t0>> 0); s[ 1]=(u8)(t0>> 8); s[ 2]=(u8)(t0>>16);
    s[ 3]=(u8)((t0>>24)|(t1<<2));
    s[ 4]=(u8)(t1>> 6); s[ 5]=(u8)(t1>>14);
    s[ 6]=(u8)((t1>>22)|(t2<<3));
    s[ 7]=(u8)(t2>> 5); s[ 8]=(u8)(t2>>13);
    s[ 9]=(u8)((t2>>21)|(t3<<5));
    s[10]=(u8)(t3>> 3); s[11]=(u8)(t3>>11);
    s[12]=(u8)((t3>>19)|(t4<<6));
    s[13]=(u8)(t4>> 2); s[14]=(u8)(t4>>10); s[15]=(u8)(t4>>18);
    s[16]=(u8)(t5>> 0); s[17]=(u8)(t5>> 8); s[18]=(u8)(t5>>16);
    s[19]=(u8)((t5>>24)|(t6<<1));
    s[20]=(u8)(t6>> 7); s[21]=(u8)(t6>>15);
    s[22]=(u8)((t6>>23)|(t7<<3));
    s[23]=(u8)(t7>> 5); s[24]=(u8)(t7>>13);
    s[25]=(u8)((t7>>21)|(t8<<4));
    s[26]=(u8)(t8>> 4); s[27]=(u8)(t8>>12);
    s[28]=(u8)((t8>>20)|(t9<<6));
    s[29]=(u8)(t9>> 2); s[30]=(u8)(t9>>10); s[31]=(u8)(t9>>18);
}

// ============================================================================
//  Edwards 点运算（extended coords: x=X/Z, y=Y/Z, T=XY/Z）
// ============================================================================
struct ge_p3     { fe X, Y, Z, T; };
struct ge_cached { fe YpX, YmX, Z, T2d; };   // Z 存 2*Z2
typedef struct { ge_p3 p; ge_cached c; } TblEntry;

__device__ __forceinline__ void ge_set_base(ge_p3& p) {
    fe_copy(p.X, C_BX); fe_copy(p.Y, C_BY); fe_one(p.Z); fe_mul(p.T, C_BX, C_BY);
}

__device__ __forceinline__ void ge_p3_to_cached(ge_cached& q, const ge_p3& p) {
    fe_add(q.YpX, p.Y, p.X);
    fe_sub(q.YmX, p.Y, p.X);
    fe_add(q.Z, p.Z, p.Z);          // 2Z
    fe_mul(q.T2d, p.T, C_D2);
}

// add-2008-hwcd-3: 8M
__device__ __forceinline__ void ge_add(ge_p3& r, const ge_p3& p, const ge_cached& q) {
    fe A,B,C,D,E,F,G,H,t0,t1;
    fe_sub(t0, p.Y, p.X); fe_mul(A, t0, q.YmX);
    fe_add(t1, p.Y, p.X); fe_mul(B, t1, q.YpX);
    fe_mul(C, p.T, q.T2d);
    fe_mul(D, p.Z, q.Z);
    fe_add(H, B, A);                 // H = B+A
    fe_sub(E, B, A);                 // E = B-A  (A 复用)
    fe_sub(F, D, C);                 // F = D-C  (C 复用)
    fe_add(G, D, C);                 // G = D+C  (D 复用)
    fe_mul(r.X, E, F);
    fe_mul(r.Y, G, H);
    fe_mul(r.T, E, H);
    fe_mul(r.Z, F, G);
}

// dbl-2008-hwcd (a=-1): 3M+4S
__device__ __forceinline__ void ge_dbl(ge_p3& r, const ge_p3& p) {
    fe A,B,C,E,F,G,H;
    fe_sq(A, p.X);
    fe_sq(B, p.Y);
    fe_sq(C, p.Z); fe_add(C, C, C);      // C = 2Z^2
    fe_add(E, p.X, p.Y); fe_sq(E, E); fe_sub(E, E, A); fe_sub(E, E, B);   // E = (X+Y)^2-A-B
    fe_sub(G, B, A);                     // G = B-A   (D = -A)
    fe_add(H, B, A); fe_neg(H, H);       // H = -A-B
    fe_sub(F, G, C);                     // F = G-C
    fe_mul(r.X, E, F);
    fe_mul(r.Y, G, H);
    fe_mul(r.T, E, H);
    fe_mul(r.Z, F, G);
}

__device__ __forceinline__ void ge_to_affine(fe& x, fe& y, const ge_p3& p) {
    fe zi; fe_invert(zi, p.Z);
    fe_mul(x, p.X, zi);
    fe_mul(y, p.Y, zi);
}

__device__ __forceinline__ int ge_encode(u8 out[32], const ge_p3& p) {
    fe x,y; ge_to_affine(x,y,p);
    u8 yb[32], xb[32];
    fe_tobytes(yb, y);
    fe_tobytes(xb, x);
    #pragma unroll
    for (int i=0;i<32;i++) out[i]=yb[i];
    out[31] |= (u8)((xb[0]&1) << 7);
    return 1;
}

// 固定基点预计算表：tbl[w*TBLN+d] = d * 2^(5w) * B
__global__ void build_table_kernel(TblEntry* tbl) {
    if (blockIdx.x || threadIdx.x) return;
    ge_p3 P; ge_set_base(P);
    ge_p3 Q;
    for (int w=0; w<NW; w++) {
        ge_cached cp; ge_p3_to_cached(cp, P);
        fe_zero(Q.X); fe_zero(Q.Y); fe_one(Q.Z); fe_zero(Q.T);
        for (int dg=1; dg<TBLN; dg++) {
            if (dg==1) Q = P; else ge_add(Q, Q, cp);
            tbl[w*TBLN+dg].p = Q;
            ge_p3_to_cached(tbl[w*TBLN+dg].c, Q);
        }
        // P = 32 * P
        for (int s=0;s<5;s++) ge_dbl(P, P);
    }
}

// ============================================================================
//  SHA-256 / SHA-512
// ============================================================================
__device__ __constant__ u32 K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };

__device__ __forceinline__ u32 rotr32(u32 x, int n) { return (x>>n)|(x<<(32-n)); }

__device__ __forceinline__ void sha256_block(u32 h[8], const u8 blk[64]) {
    u32 w[64];
    #pragma unroll
    for (int i=0;i<16;i++) w[i]=((u32)blk[i*4]<<24)|((u32)blk[i*4+1]<<16)|((u32)blk[i*4+2]<<8)|blk[i*4+3];
    #pragma unroll
    for (int i=16;i<64;i++) {
        u32 s0 = rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);
        u32 s1 = rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    u32 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    #pragma unroll
    for (int i=0;i<64;i++) {
        u32 S1 = rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
        u32 ch = (e&f)^((~e)&g);
        u32 t1 = hh + S1 + ch + K256[i] + w[i];
        u32 S0 = rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
        u32 maj = (a&b)^(a&c)^(b&c);
        u32 t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

__device__ __forceinline__ void sha256(const u8* data, int len, u8 out[32]) {
    u32 h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    u8 blk[64]; int off=0;
    while (len-off >= 64) {
        #pragma unroll
        for (int i=0;i<64;i++) blk[i]=data[off+i];
        sha256_block(h, blk); off += 64;
    }
    #pragma unroll
    for (int i=0;i<64;i++) blk[i]=0;
    int rem = len-off;
    for (int i=0;i<rem;i++) blk[i]=data[off+i];
    blk[rem]=0x80;
    u64 bits = (u64)len*8;
    if (rem >= 56) { sha256_block(h, blk);
        #pragma unroll
        for (int i=0;i<64;i++) blk[i]=0; }
    #pragma unroll
    for (int i=0;i<8;i++) blk[56+i]=(u8)((bits>>(56-8*i))&0xFF);
    sha256_block(h, blk);
    #pragma unroll
    for (int i=0;i<8;i++) { out[i*4]=(u8)(h[i]>>24); out[i*4+1]=(u8)(h[i]>>16); out[i*4+2]=(u8)(h[i]>>8); out[i*4+3]=(u8)h[i]; }
}

__device__ __constant__ u64 K512[80] = {
    0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL,0x12835b0145706fbeULL,0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL,0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL,0xd5a79147930aa725ULL,0x06ca6351e003826fULL,0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,0x53380d139d95b3dfULL,
    0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL,0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL,
    0xca273eceea26619cULL,0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,0x113f9804bef90daeULL,0x1b710b35131c471bULL,
    0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL };

__device__ __forceinline__ u64 rotr64(u64 x, int n) { return (x>>n)|(x<<(64-n)); }
__device__ __forceinline__ u64 S0f(u64 x) { return rotr64(x,28)^rotr64(x,34)^rotr64(x,39); }
__device__ __forceinline__ u64 S1f(u64 x) { return rotr64(x,14)^rotr64(x,18)^rotr64(x,41); }
__device__ __forceinline__ u64 s0f(u64 x) { return rotr64(x,1)^rotr64(x,8)^(x>>7); }
__device__ __forceinline__ u64 s1f(u64 x) { return rotr64(x,19)^rotr64(x,61)^(x>>6); }

__device__ __forceinline__ void sha512_init(u64 st[8]) {
    st[0]=0x6a09e667f3bcc908ULL; st[1]=0xbb67ae8584caa73bULL;
    st[2]=0x3c6ef372fe94f82bULL; st[3]=0xa54ff53a5f1d36f1ULL;
    st[4]=0x510e527fade682d1ULL; st[5]=0x9b05688c2b3e6c1fULL;
    st[6]=0x1f83d9abfb41bd6bULL; st[7]=0x5be0cd19137e2179ULL;
}

__device__ __noinline__ void sha512_compress(u64 st[8], u64 w[16]) {
    u64 a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
    for (int i=0;i<80;i++) {
        u64 wi;
        if (i < 16) wi = w[i];
        else {
            w[i&15] += s1f(w[(i-2)&15]) + w[(i-7)&15] + s0f(w[(i-15)&15]);
            wi = w[i&15];
        }
        u64 T1 = h + S1f(e) + ((e&f)^((~e)&g)) + K512[i] + wi;
        u64 T2 = S0f(a) + ((a&b)^(a&c)^(b&c));
        h=g; g=f; f=e; e=d+T1; d=c; c=b; b=a; a=T1+T2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d; st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}

// SHA-512(32 字节消息)：单块
__device__ __noinline__ void sha512_of32(const u8 seed[32], u8 out[64]) {
    u64 st[8]; sha512_init(st);
    u8 blk[128];
    #pragma unroll
    for (int i=0;i<128;i++) blk[i]=0;
    #pragma unroll
    for (int i=0;i<32;i++) blk[i]=seed[i];
    blk[32]=0x80;
    blk[126]=0x01; blk[127]=0x00;          // 256 bit 长度
    u64 w[16];
    #pragma unroll
    for (int i=0;i<16;i++) {
        u64 v=0;
        #pragma unroll
        for (int j=0;j<8;j++) v = (v<<8) | blk[i*8+j];
        w[i]=v;
    }
    sha512_compress(st, w);
    #pragma unroll
    for (int i=0;i<8;i++) {
        out[i*8+0]=(u8)(st[i]>>56); out[i*8+1]=(u8)(st[i]>>48); out[i*8+2]=(u8)(st[i]>>40); out[i*8+3]=(u8)(st[i]>>32);
        out[i*8+4]=(u8)(st[i]>>24); out[i*8+5]=(u8)(st[i]>>16); out[i*8+6]=(u8)(st[i]>>8);  out[i*8+7]=(u8)st[i];
    }
}

// ============================================================================
//  Base58
// ============================================================================
__device__ __constant__ char C_B58[59] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

__device__ __noinline__ int b58_enc32(const u8 in[32], char* out) {
    int zeros=0;
    while (zeros<32 && in[zeros]==0) zeros++;
    u8 num[32];
    #pragma unroll
    for (int i=0;i<32;i++) num[i]=in[i];
    char digs[48]; int nd=0;
    for (;;) {
        int allzero=1;
        for (int i=zeros;i<32;i++) if (num[i]) { allzero=0; break; }
        if (allzero) break;
        int rem=0;
        for (int i=zeros;i<32;i++) {
            int v = rem*256 + num[i];
            num[i] = (u8)(v/58);
            rem = v - (v/58)*58;
        }
        digs[nd++] = C_B58[rem];
        if (nd>=47) break;
    }
    int l=0;
    for (int i=0;i<zeros;i++) out[l++]='1';
    for (int i=nd-1;i>=0;i--) out[l++]=digs[i];
    if (l==0) out[l++]='1';
    out[l]=0;
    return l;
}

// ============================================================================
//  主搜索内核
// ============================================================================
typedef struct { u32 p; u32 pad; u8 addr[PSTR]; u8 seed[32]; u8 pk[32]; } ResqEntry;

__device__ __forceinline__ u32 get_bits5(const u32 w[8], int bit) {
    int idx = bit>>5, sh = bit&31;
    u64 v = w[idx];
    if (sh > 27 && idx < 7) v |= ((u64)w[idx+1])<<32;
    return (u32)((v>>sh) & 31u);
}

__global__ __launch_bounds__(256, 2) void sol_search_kernel(
        const u64* d_tbl_key, const u32* d_tbl_p, const u8* d_tbl_occ,
        const u32* d_tbl_off, const u32* d_tbl_mask, const u32* d_tbl_cnt,
        const u8* d_pfx, u64 uniq, int nl_max,
        const u8* d_salt, u64 start_nonce, int iters,
        const TblEntry* tbl,
        u32* d_found, u32* d_nfound, ResqEntry* d_resq, u32* d_resq_cnt) {

    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x*blockDim.x;

    for (int it=0; it<iters; it++) {
        // 不再“全部命中就 return”：提前返回会让空转迭代也被计入 addr/s，出现天文数字假速度。
        // 已命中的前缀只是不再上报，标量乘照常做满。
        u64 n = start_nonce + tid + (u64)it*stride;

        u8 buf[40];
        #pragma unroll
        for (int i=0;i<32;i++) buf[i]=d_salt[i];
        #pragma unroll
        for (int i=0;i<8;i++) buf[32+i]=(u8)(n>>(8*i));

        u8 seed[32]; sha256(buf, 40, seed);
        u8 hsh[64];  sha512_of32(seed, hsh);

        u8 a[32];
        #pragma unroll
        for (int i=0;i<32;i++) a[i]=hsh[i];
        a[0] &= 248; a[31] &= 127; a[31] |= 64;

        u32 w8[8];
        #pragma unroll
        for (int i=0;i<8;i++) w8[i] = (u32)a[i*4] | ((u32)a[i*4+1]<<8) | ((u32)a[i*4+2]<<16) | ((u32)a[i*4+3]<<24);

        ge_p3 R; int started=0;
        // v2: 每线程每个候选做一次完整的固定基点乘（钱包可导入的 seed 决定标量，无法做增量点加）
        for (int w=NW-1; w>=0; w--) {
            u32 dg = get_bits5(w8, w*5);
            if (!dg) continue;
            const TblEntry& e = tbl[w*TBLN + dg];
            if (!started) { R = e.p; started = 1; }
            else ge_add(R, R, e.c);
        }
        if (!started) { continue; }

        u8 pk[32]; ge_encode(pk, R);

        char ast[PSTR];
        int laddr = b58_enc32(pk, ast);
        if (laddr > PSTR-1) laddr = PSTR-1;

        u64 hv = 0xcbf29ce484222325ull;
        for (int L=1; L<=nl_max; L++) {
            if (L > laddr) break;
            hv = (hv ^ (u8)ast[L-1]) * 0x100000001b3ull;
            int cnt = d_tbl_cnt[L-1];
            if (!cnt) continue;
            u32 mask = d_tbl_mask[L-1];
            u32 base = d_tbl_off[L-1];
            u32 hh2 = (u32)((hv * 0x9E3779B97F4A7C15ull) >> 32);
            int s = (int)(hh2 & mask);
            for (;;) {
                if (!d_tbl_occ[base+s]) break;
                if (d_tbl_key[base+s] == hv) {
                    u32 p = d_tbl_p[base+s];
                    int ok=1;
                    for (int k=0;k<L;k++) if (d_pfx[(u64)p*PSTR+k]!=(u8)ast[k]) { ok=0; break; }
                    if (ok && !d_found[p]) {
                        u32 old = atomicCAS(&d_found[p], 0u, 1u);
                        if (old==0) {
                            u32 q = atomicAdd(d_resq_cnt, 1);
                            ResqEntry* re = &d_resq[q & RESQ_MASK];
                            re->p = p; re->pad = 0;
                            for (int i=0;i<laddr;i++) re->addr[i]=(u8)ast[i];
                            re->addr[laddr]=0;
                            #pragma unroll
                            for (int i=0;i<32;i++) { re->seed[i]=seed[i]; re->pk[i]=pk[i]; }
                            atomicAdd(d_nfound, 1);
                        }
                        break;
                    }
                }
                s = (s+1) & mask;
            }
        }
    }
}

// ============================================================================
//  自检内核
// ============================================================================
typedef struct { u8 mode; u8 seed[32]; u8 salt[32]; u64 ctr; } TestCase;   // mode 0=直接 seed, 1=salt+ctr

__global__ void selftest_kernel(const TestCase* cases, int ncase, u8* d_pub, u8* d_addr, int* d_dbl, u32* d_invok, const TblEntry* tbl) {
    if (blockIdx.x || threadIdx.x) return;
    for (int t=0;t<ncase;t++) {
        u8 seed[32];
        if (cases[t].mode == 0) {
            #pragma unroll
            for (int i=0;i<32;i++) seed[i]=cases[t].seed[i];
        } else {
            u8 buf[40];
            #pragma unroll
            for (int i=0;i<32;i++) buf[i]=cases[t].salt[i];
            #pragma unroll
            for (int i=0;i<8;i++) buf[32+i]=(u8)(cases[t].ctr>>(8*i));
            sha256(buf, 40, seed);
        }
        u8 hsh[64]; sha512_of32(seed, hsh);
        u8 a[32];
        #pragma unroll
        for (int i=0;i<32;i++) a[i]=hsh[i];
        a[0]&=248; a[31]&=127; a[31]|=64;
        u32 w8[8];
        #pragma unroll
        for (int i=0;i<8;i++) w8[i] = (u32)a[i*4] | ((u32)a[i*4+1]<<8) | ((u32)a[i*4+2]<<16) | ((u32)a[i*4+3]<<24);
        ge_p3 R; int started=0;
        for (int w=NW-1; w>=0; w--) {
            u32 dg = get_bits5(w8, w*5);
            if (!dg) continue;
            const TblEntry& e = tbl[w*TBLN + dg];
            if (!started) { R = e.p; started = 1; }
            else ge_add(R, R, e.c);
        }
        u8 pk[32]; ge_encode(pk, R);
        #pragma unroll
        for (int i=0;i<32;i++) d_pub[t*32+i]=pk[i];
        char ast[PSTR]; int la = b58_enc32(pk, ast);
        for (int i=0;i<PSTR;i++) d_addr[t*PSTR+i]=(u8)(i<la?ast[i]:0);
    }
    // 2B: dbl vs add 一致性
    ge_p3 B, R1, R2;
    ge_set_base(B);
    ge_cached cb; ge_p3_to_cached(cb, B);
    ge_add(R1, B, cb);
    ge_dbl(R2, B);
    u8 e1[32], e2[32];
    ge_encode(e1, R1); ge_encode(e2, R2);
    int same=1;
    #pragma unroll
    for (int i=0;i<32;i++) if (e1[i]!=e2[i]) same=0;
    d_dbl[0] = same;

    // (x^-1)*x == 1
    fe x, xi, y;
    fe z; fe_zero(z);
    x.v[0]=12345678; x.v[1]=-98765; x.v[2]=777; x.v[3]=-4242424; x.v[4]=99;
    x.v[5]=-1234; x.v[6]=5678901; x.v[7]=-77; x.v[8]=3333; x.v[9]=-55555;
    fe_invert(xi, x);
    fe_mul(y, xi, x);
    u8 yb[32]; fe_tobytes(yb, y);
    u8 one[32]; fe_one(z); fe_tobytes(one, z);
    same=1;
    #pragma unroll
    for (int i=0;i<32;i++) if (yb[i]!=one[i]) same=0;
    d_invok[0] = same;
}


// ============================================================================
//  主机侧
// ============================================================================
static const char B58C[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

static int b58_enc(const u8* in, int len, char* out) {
    int zeros=0; while (zeros<len && in[zeros]==0) zeros++;
    u8 num[96]; memcpy(num, in, len);
    char digs[160]; int nd=0;
    for (;;) {
        int allzero=1;
        for (int i=zeros;i<len;i++) if (num[i]) { allzero=0; break; }
        if (allzero) break;
        int rem=0;
        for (int i=zeros;i<len;i++) { int v=rem*256+num[i]; num[i]=(u8)(v/58); rem=v-(v/58)*58; }
        digs[nd++]=B58C[rem];
        if (nd>=159) break;
    }
    int l=0;
    for (int i=0;i<zeros;i++) out[l++]='1';
    for (int i=nd-1;i>=0;i--) out[l++]=digs[i];
    if (l==0) out[l++]='1';
    out[l]=0;
    return l;
}

static const char* B58_ALPHA = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// 流式读前缀文件（支持 2GB+）；返回条数，pfx 行宽 PSTR
static u64 load_prefix_file(const char* fn, u8** pfx_out, u8** plen_out, u64* skip_out) {
    FILE* fp = fopen(fn, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", fn); return 0; }
    u64 cap = 1ull<<20, n = 0, skipped = 0;
    u8* pfx = (u8*)malloc((size_t)(cap*PSTR));
    u8* plen = (u8*)malloc((size_t)cap);
    if (!pfx || !plen) { fprintf(stderr, "out of memory\n"); fclose(fp); return 0; }
    const size_t CH = 32ull<<20;
    u8* buf = (u8*)malloc(CH + 8192);
    if (!buf) { fprintf(stderr, "out of memory\n"); fclose(fp); return 0; }
    size_t carry = 0;
    while (1) {
        size_t r = fread(buf + carry, 1, CH, fp);
        size_t total = carry + r;
        if (total == 0) break;
        size_t start = 0;
        for (size_t i=0;i<total;i++) {
            if (buf[i] != '\n' && buf[i] != '\r') continue;
            size_t s0 = start, e0 = i;
            while (s0 < e0 && (buf[s0]==' '||buf[s0]=='\t')) s0++;
            size_t e1 = s0;
            while (e1 < e0 && buf[e1]!=' ' && buf[e1]!='\t' && buf[e1]!='#') e1++;
            size_t L = e1 - s0;
            if (L > 0) {
                if (L > NLMAX) { skipped++; }
                else {
                    if (n >= cap) { cap <<= 1;
                        pfx = (u8*)realloc(pfx, (size_t)(cap*PSTR));
                        plen = (u8*)realloc(plen, (size_t)cap);
                        if (!pfx || !plen) { fprintf(stderr,"out of memory\n"); fclose(fp); return 0; } }
                    memcpy(pfx + n*PSTR, buf + s0, L);
                    plen[n] = (u8)L;
                    n++;
                }
            }
            start = i + 1;
        }
        carry = total - start;
        if (r == 0) {
            // 文件最后一行可能没有换行符，必须补处理，否则会丢掉最后一条前缀
            size_t s0 = start;
            while (s0 < total && (buf[s0]==' '||buf[s0]=='\t')) s0++;
            size_t e1 = s0;
            while (e1 < total && buf[e1]!=' ' && buf[e1]!='\t' && buf[e1]!='#' &&
                   buf[e1]!='\n' && buf[e1]!='\r') e1++;
            size_t L = e1 - s0;
            if (L > 0) {
                if (L > NLMAX) { skipped++; }
                else {
                    if (n >= cap) { cap <<= 1;
                        u8* np = (u8*)realloc(pfx, (size_t)(cap*PSTR));
                        u8* nl = (u8*)realloc(plen, (size_t)cap);
                        if (!np || !nl) { fprintf(stderr,"out of memory\n"); fclose(fp); return 0; }
                        pfx = np; plen = nl; }
                    memcpy(pfx + n*PSTR, buf + s0, L);
                    plen[n] = (u8)L;
                    n++;
                }
            }
            break;
        }
        if (carry > 4096) { skipped++; carry = 0; }        // 超长行直接丢弃
        else if (carry) memcpy(buf, buf + start, carry);
    }
    free(buf); fclose(fp);
    *pfx_out = pfx; *plen_out = plen; *skip_out = skipped;
    return n;
}

static u64 fnv1a(const u8* p, int L) {
    u64 h = 0xcbf29ce484222325ull;
    for (int i=0;i<L;i++) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

static void make_secret_key_b58(const u8* seed, const u8* pk, char* out) {
    u8 sk[64]; memcpy(sk, seed, 32); memcpy(sk+32, pk, 32);
    b58_enc(sk, 64, out);
}

static void print_json_array(FILE* f, const u8* seed, const u8* pk) {
    u8 sk[64]; memcpy(sk, seed, 32); memcpy(sk+32, pk, 32);
    fprintf(f, "[");
    for (int i=0;i<64;i++) fprintf(f, "%u%s", sk[i], i==63?"":",");
    fprintf(f, "]");
}

static void run_sol_search(u8* pfx, u8* plen, u64 nprefix, int blocks) {
    if (nprefix == 0) { fprintf(stderr, "no valid prefixes\n"); return; }

    // 预检
    u64 impossible = 0;
    for (u64 p=0;p<nprefix;p++) {
        int L = plen[p]; int bad = 0;
        for (int i=0;i<L;i++) if (!strchr(B58_ALPHA, pfx[p*PSTR+i])) { bad=1; break; }
        // 44 字符地址的首位至少是 '2'：以 '1' 开头的 44 字符编码会退化成 43 字符
        if (!bad && L == 44 && pfx[p*PSTR] == '1') bad = 1;
        if (bad) impossible++;
    }
    if (impossible)
        printf("WARNING: %llu prefix(es) contain characters outside the base58 alphabet and can never match\n",
               (unsigned long long)impossible);

    int lcnt[NLMAX+1]; for (int i=0;i<=NLMAX;i++) lcnt[i]=0;
    for (u64 p=0;p<nprefix;p++) if (plen[p]<=NLMAX) lcnt[plen[p]]++;
    printf("length dist: ");
    for (int L=1;L<=NLMAX;L++) if (lcnt[L]) printf("%d-char:%d ", L, lcnt[L]);
    printf("\n");

    // 每长度一条哈希表
    int hcap[NLMAX]; u32 off[NLMAX]; u32 mask[NLMAX];
    u64* hk[NLMAX]; u32* hp[NLMAX]; u8* hoc[NLMAX];
    u64 total_cap = 0;
    for (int L=1;L<=NLMAX;L++) {
        u64 cnt = (u64)lcnt[L];
        u64 c2 = 8; while (c2 < cnt*2+1) c2 <<= 1;
        hcap[L-1] = (int)c2; off[L-1] = (u32)total_cap; mask[L-1] = (u32)(c2-1);
        hk[L-1]  = (u64*)malloc((size_t)(c2*8));
        hp[L-1]  = (u32*)malloc((size_t)(c2*4));
        hoc[L-1] = (u8*)malloc((size_t)c2);
        if (!hk[L-1]||!hp[L-1]||!hoc[L-1]) { fprintf(stderr,"out of memory building table\n"); return; }
        memset(hoc[L-1], 0, (size_t)c2);
        total_cap += c2;
    }
    if (total_cap >= (1ull<<32)) { fprintf(stderr,"hash table too large (>4G slots)\n"); return; }

    u64 uniq = 0;
    for (u64 p=0;p<nprefix;p++) {
        int L = plen[p];
        if (L > NLMAX) continue;
        u64 key = fnv1a(pfx + p*PSTR, L);
        u32 hh = (u32)((key * 0x9E3779B97F4A7C15ull) >> 32);
        int s = (int)(hh & mask[L-1]);
        u8* oc = hoc[L-1]; u64* kk = hk[L-1]; u32* pp = hp[L-1];
        for (;;) {
            if (!oc[s]) {
                oc[s]=1; kk[s]=key; pp[s]=(u32)uniq;
                if (uniq < p) { memmove(pfx + uniq*PSTR, pfx + p*PSTR, PSTR); plen[uniq]=plen[p]; }
                uniq++; break;
            }
            if (kk[s]==key) {
                u32 q = pp[s]; int same=1;
                for (int i=0;i<L;i++) if (pfx[(u64)q*PSTR+i]!=pfx[p*PSTR+i]) { same=0; break; }
                if (same) break;
            }
            s = (s+1) & mask[L-1];
        }
    }
    printf("unique prefixes after dedup: %llu\n", (unsigned long long)uniq);
    if (uniq == 0) return;
    if (uniq >= (1ull<<31)) { fprintf(stderr,"too many unique prefixes (>=2^31)\n"); return; }

    u64* ck = (u64*)malloc((size_t)(total_cap*8));
    u32* cp = (u32*)malloc((size_t)(total_cap*4));
    u8*  co = (u8*)malloc((size_t)total_cap);
    if (!ck||!cp||!co) { fprintf(stderr,"out of memory\n"); return; }
    for (int L=1;L<=NLMAX;L++) {
        memcpy(ck + off[L-1], hk[L-1], (size_t)(hcap[L-1]*8));
        memcpy(cp + off[L-1], hp[L-1], (size_t)(hcap[L-1]*4));
        memcpy(co + off[L-1], hoc[L-1], (size_t)hcap[L-1]);
        free(hk[L-1]); free(hp[L-1]); free(hoc[L-1]);
    }

    u64* d_tbl_key; u32* d_tbl_p; u8* d_tbl_occ;
    u32* d_tbl_off; u32* d_tbl_mask; u32* d_tbl_cnt;
    CUDA_CHECK(cudaMalloc(&d_tbl_key, (size_t)(total_cap*8)));
    CUDA_CHECK(cudaMalloc(&d_tbl_p, (size_t)(total_cap*4)));
    CUDA_CHECK(cudaMalloc(&d_tbl_occ, (size_t)total_cap));
    CUDA_CHECK(cudaMalloc(&d_tbl_off, NLMAX*4));
    CUDA_CHECK(cudaMalloc(&d_tbl_mask, NLMAX*4));
    CUDA_CHECK(cudaMalloc(&d_tbl_cnt, NLMAX*4));
    CUDA_CHECK(cudaMemcpy(d_tbl_key, ck, (size_t)(total_cap*8), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl_p, cp, (size_t)(total_cap*4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl_occ, co, (size_t)total_cap, cudaMemcpyHostToDevice));

    u32 tcnt[NLMAX];
    for (int L=1;L<=NLMAX;L++) tcnt[L-1]=0;
    for (u64 pi=0;pi<uniq;pi++) tcnt[plen[pi]-1]++;
    CUDA_CHECK(cudaMemcpy(d_tbl_off, off, NLMAX*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl_mask, mask, NLMAX*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl_cnt, tcnt, NLMAX*4, cudaMemcpyHostToDevice));
    free(ck); free(cp); free(co);

    u8* d_pfx; CUDA_CHECK(cudaMalloc(&d_pfx, (size_t)(uniq*PSTR)));
    CUDA_CHECK(cudaMemcpy(d_pfx, pfx, (size_t)(uniq*PSTR), cudaMemcpyHostToDevice));

    u32* d_found; u32* d_nfound;
    ResqEntry* d_resq; u32* d_resq_cnt;
    CUDA_CHECK(cudaMalloc(&d_found, (size_t)(uniq*4)));
    CUDA_CHECK(cudaMalloc(&d_nfound, 4));
    CUDA_CHECK(cudaMalloc(&d_resq, (size_t)(RESQ_CAP*sizeof(ResqEntry))));
    CUDA_CHECK(cudaMalloc(&d_resq_cnt, 4));
    CUDA_CHECK(cudaMemset(d_found, 0, (size_t)(uniq*4)));
    CUDA_CHECK(cudaMemset(d_nfound, 0, 4));
    CUDA_CHECK(cudaMemset(d_resq_cnt, 0, 4));

    u8* d_salt; CUDA_CHECK(cudaMalloc(&d_salt, 32));
    u8 salt[32];
    FILE* ur = fopen("/dev/urandom", "rb");
    size_t got = 0;
    if (ur) { got = fread(salt, 1, 32, ur); fclose(ur); }
    if (got != 32) { std::random_device rd;
        for (int i=0;i<32;i++) salt[i]=(u8)(rd()&0xFF); }
    CUDA_CHECK(cudaMemcpy(d_salt, salt, 32, cudaMemcpyHostToDevice));

    TblEntry* d_tbl; CUDA_CHECK(cudaMalloc(&d_tbl, (size_t)(NW*TBLN*sizeof(TblEntry))));
    build_table_kernel<<<1,1>>>(d_tbl);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("run salt (32B hex): ");
    for (int i=0;i<32;i++) printf("%02x", salt[i]);
    printf("\n  (fresh salt each run: seed = SHA256(salt||counter), counter from 0; sequences differ per run)\n");

    printf("searching random ed25519 keypairs (seed=SHA256(run-salt||ctr)): %d blocks x 256 threads\n", blocks);
    printf("each key costs 1 full scalar mult; runs forever until Ctrl+C (hits are appended to sol_hits.txt immediately)\n");

    FILE* hf = fopen("sol_hits.txt", "a");
    if (hf) setvbuf(hf, NULL, _IONBF, 0);

    u64 start = 0; int batch = 1; double total_s = 0; u32 nfound = 0;
    u64 qprev = 0, printed = 0;
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    char skb58[128];

    for (;;) {
        int launched = batch;
        CUDA_CHECK(cudaEventRecord(t0));
        sol_search_kernel<<<blocks, 256>>>(d_tbl_key, d_tbl_p, d_tbl_occ,
            d_tbl_off, d_tbl_mask, d_tbl_cnt, d_pfx, uniq, NLMAX,
            d_salt, start, batch, d_tbl,
            d_found, d_nfound, d_resq, d_resq_cnt);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        CUDA_CHECK(cudaMemcpy(&nfound, d_nfound, 4, cudaMemcpyDeviceToHost));
        total_s += ms/1000.0;
        u64 done = (u64)launched*blocks*256;
        if (ms > 1000.0f) { int nb = batch/2; if (nb<1) nb=1; batch = nb; }
        else if (ms < 200.0f) { int nb = batch*2; if (nb>1<<20) nb=1<<20; batch = nb; }

        u32 qcnt = 0; CUDA_CHECK(cudaMemcpy(&qcnt, d_resq_cnt, 4, cudaMemcpyDeviceToHost));
        while (qprev < (u64)qcnt) {
            ResqEntry e;
            CUDA_CHECK(cudaMemcpy(&e, d_resq + (qprev & RESQ_MASK), sizeof(ResqEntry), cudaMemcpyDeviceToHost));
            qprev++;
            char pf[PSTR+1]; int L = plen[e.p]; memcpy(pf, pfx + (u64)e.p*PSTR, L); pf[L]=0;
            make_secret_key_b58(e.seed, e.pk, skb58);
            printf("\nHIT [%5u] prefix \"%s\"\n", e.p, pf);
            printf("      Address:                 %s\n", e.addr);
            printf("      Secret key (base58 64B): %s\n", skb58);
            printf("      Seed (base58 32B):       ");
            { char sd[64]; b58_enc(e.seed, 32, sd); printf("%s\n", sd); }
            printf("      Keypair JSON:            ");
            print_json_array(stdout, e.seed, e.pk); printf("\n");
            fflush(stdout);
            if (hf) {
                fprintf(hf, "HIT [%5u] prefix \"%s\"\n", e.p, pf);
                fprintf(hf, "      Address:                 %s\n", e.addr);
                fprintf(hf, "      Secret key (base58 64B): %s\n", skb58);
                fprintf(hf, "      Seed (base58 32B):       ");
                { char sd[64]; b58_enc(e.seed, 32, sd); fprintf(hf, "%s\n", sd); }
                fprintf(hf, "      Keypair JSON:            ");
                print_json_array(hf, e.seed, e.pk); fprintf(hf, "\n\n");
            }
            printed++;
        }
        u64 total_keys = start + done;                       // 累计已算过的 key 数（含本批）
        double as = (double)total_keys / (total_s>0?total_s:1.0);
        double ks = as / 1000.0;
        printf("\r  [%7.1f s] %10.0f addr/s (%.0f k/s) | hit %llu | covered %u/%llu  (Ctrl+C stop)",
               total_s, as, ks, (unsigned long long)printed, nfound, (unsigned long long)uniq);
        fflush(stdout);
        start += done;

        // 所有前缀都命中过之后不清表、不停机：清空命中标记后继续无限跑，
        // 这样单个前缀（或小文件）也不会出现“命中一次就结束”的情况。
        if (nfound >= (u64)uniq) {
            CUDA_CHECK(cudaMemset(d_found, 0, (size_t)(uniq*4)));
            CUDA_CHECK(cudaMemset(d_nfound, 0, 4));
            nfound = 0;
        }
    }
    printf("\nall %llu unique prefixes found in %.1f s, %llu hits\n", (unsigned long long)uniq, total_s, (unsigned long long)printed);
    if (hf) fclose(hf);
    CUDA_CHECK(cudaFree(d_tbl_key)); CUDA_CHECK(cudaFree(d_tbl_p)); CUDA_CHECK(cudaFree(d_tbl_occ));
    CUDA_CHECK(cudaFree(d_tbl_off)); CUDA_CHECK(cudaFree(d_tbl_mask)); CUDA_CHECK(cudaFree(d_tbl_cnt));
    CUDA_CHECK(cudaFree(d_pfx)); CUDA_CHECK(cudaFree(d_found)); CUDA_CHECK(cudaFree(d_nfound));
    CUDA_CHECK(cudaFree(d_resq)); CUDA_CHECK(cudaFree(d_resq_cnt)); CUDA_CHECK(cudaFree(d_salt));
    CUDA_CHECK(cudaFree(d_tbl));
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
}

// ============================================================================
static int chain_main(int argc, char** argv) {
    const char* mode = argc>1 ? argv[1] : "bench";
    printf("=== Solana Vanity Generator v2 (ed25519, seed -> base58 address) ===\n");

    if (strcmp(mode,"selftest")==0) {
        TestCase* hc = (TestCase*)calloc(9, sizeof(TestCase));
        const char* rfc[3] = {
            "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
            "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
            "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7" };
        const char* exp_pub[9] = {
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
            "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
            "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
            "fa6fffb4b5eb1d429c0ffcb1a9b7f8594ad7eef7a7c993d7785e4d082070f417",
            "92b25a878a0963dc1152611d4cbe0ee096bd8c67f512b010694d248c565a4bb8",
            "426406b8f41f1513ffd37d4cbd3d646f280f9e7c77f02191eec028e0133087f5",
            "0050ba1c0b5f42fb067644af04c65d1f8db80eaf3a50722fc9ec5f6fa4d8c94a",
            "000f45a82245cd56235f6283466dab50d60eee29ab6d18107254f72549c9d56a",
            "" };
        const char* exp_addr[9] = {
            "FVen3X669xLzsi6N2V91DoiyzHzg1uAgqiT8jZ9nS96Z",
            "586Z7H2vpX9qNhN2T4e9Utugie3ogjbxzGaMtM3E6HR5",
            "Hyx62wPQGyvXCoihZq1BrbUjBRh2LuNxWiiqMkfAuSZr",
            "HrbyevrPyc5P7AAjpn6xJLfUJVV3mZ7KBUZKSwGGpF34",
            "AseG86D8WSo6VRHsxJ7TqvQFojB1E1JEfzwJPMVtUiA3",
            "5UANZgPx5jeL4FtpMQDu7uL2hc21yQR9qiBVRQM2sLPn",
            "12EPvzPxQ27Gdc8DGZihWWZjYsh4sFxvEpz1wLtsp2Y9",
            "1EWPc7d7HwH25LWP6xTYTzbGvf6fMUDLKbhgrzfsC2M",
            "" };
        for (int i=0;i<3;i++) {
            hc[i].mode = 0;
            for (int j=0;j<32;j++) { int a,b; char c1=rfc[i][2*j], c2=rfc[i][2*j+1];
                a=(c1<='9')?c1-'0':c1-'a'+10; b=(c2<='9')?c2-'0':c2-'a'+10;
                hc[i].seed[j]=(u8)((a<<4)|b); }
        }
        for (int i=0;i<4;i++) { hc[3+i].mode = 1; for (int j=0;j<32;j++) hc[3+i].salt[j]=(u8)j; hc[3+i].ctr=(u64)i; }
        hc[7].mode = 1; for (int j=0;j<32;j++) hc[7].salt[j]=(u8)(32+j); hc[7].ctr = 274;

        TestCase* d_c; CUDA_CHECK(cudaMalloc(&d_c, 9*sizeof(TestCase)));
        CUDA_CHECK(cudaMemcpy(d_c, hc, 9*sizeof(TestCase), cudaMemcpyHostToDevice));
        u8* d_pub; u8* d_addr; int* d_dbl; u32* d_inv;
        CUDA_CHECK(cudaMalloc(&d_pub, 9*32));
        CUDA_CHECK(cudaMalloc(&d_addr, 9*PSTR));
        CUDA_CHECK(cudaMalloc(&d_dbl, 4));
        CUDA_CHECK(cudaMalloc(&d_inv, 4));
        TblEntry* d_tbl; CUDA_CHECK(cudaMalloc(&d_tbl, (size_t)(NW*TBLN*sizeof(TblEntry))));
        build_table_kernel<<<1,1>>>(d_tbl);
        CUDA_CHECK(cudaDeviceSynchronize());
        selftest_kernel<<<1,1>>>(d_c, 8, d_pub, d_addr, d_dbl, d_inv, d_tbl);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        u8 pub[9*32]; u8 addr[9*PSTR]; int dbl=0; u32 inv=0;
        CUDA_CHECK(cudaMemcpy(pub, d_pub, 9*32, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(addr, d_addr, 9*PSTR, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&dbl, d_dbl, 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&inv, d_inv, 4, cudaMemcpyDeviceToHost));

        int pass=0, fail=0;
        printf("case  pubkey                                                            address\n");
        for (int i=0;i<8;i++) {
            char ph[65]; for (int j=0;j<32;j++) sprintf(ph+2*j, "%02x", pub[i*32+j]);
            char ad[PSTR+1]; memcpy(ad, addr+i*PSTR, PSTR); ad[PSTR]=0;
            int okp = (strcmp(ph, exp_pub[i])==0);
            int oka = (strcmp(ad, exp_addr[i])==0);
            printf("[%d] %s %-50s %-44s\n", i, (okp&&oka)?"PASS":"FAIL", ph, ad);
            if (okp&&oka) pass++; else { fail++; if(!okp) printf("    expect pub %s\n", exp_pub[i]); if(!oka) printf("    expect addr %s\n", exp_addr[i]); }
        }
        printf("2B via dbl == via add : %s\n", dbl?"PASS":"FAIL"); if(dbl)pass++; else fail++;
        printf("(x^-1)*x == 1         : %s\n", inv?"PASS":"FAIL"); if(inv)pass++; else fail++;
        printf("\nselftest: %d pass, %d fail\n", pass, fail);
        FILE* f = fopen("sol_selftest.txt","w");
        if (f) { fprintf(f,"pass=%d fail=%d dbl=%d inv=%u\n",pass,fail,dbl,inv); fclose(f); }
    }
    else if (strcmp(mode,"bench")==0) {
        int iters = (argc>=3) ? atoi(argv[2]) : 200;
        int blocks = (argc>=4) ? atoi(argv[3]) : 512;
        u64* d_tbl_key; u32* d_tbl_p; u8* d_tbl_occ; u32* d_tbl_off; u32* d_tbl_mask; u32* d_tbl_cnt;
        CUDA_CHECK(cudaMalloc(&d_tbl_key, 64)); CUDA_CHECK(cudaMalloc(&d_tbl_p, 32));
        CUDA_CHECK(cudaMalloc(&d_tbl_occ, 8)); CUDA_CHECK(cudaMalloc(&d_tbl_off, NLMAX*4));
        CUDA_CHECK(cudaMalloc(&d_tbl_mask, NLMAX*4)); CUDA_CHECK(cudaMalloc(&d_tbl_cnt, NLMAX*4));
        CUDA_CHECK(cudaMemset(d_tbl_occ, 0, 8));
        u32 zeros[NLMAX]; memset(zeros, 0, sizeof(zeros));
        CUDA_CHECK(cudaMemset(d_tbl_cnt, 0, NLMAX*4)); CUDA_CHECK(cudaMemset(d_tbl_off, 0, NLMAX*4)); CUDA_CHECK(cudaMemset(d_tbl_mask, 0, NLMAX*4));
        (void)zeros;
        u8* d_pfx; CUDA_CHECK(cudaMalloc(&d_pfx, 1));
        u32* d_found; u32* d_nfound; ResqEntry* d_resq; u32* d_resq_cnt;
        CUDA_CHECK(cudaMalloc(&d_found, 4)); CUDA_CHECK(cudaMalloc(&d_nfound, 4));
        CUDA_CHECK(cudaMalloc(&d_resq, (size_t)(RESQ_CAP*sizeof(ResqEntry)))); CUDA_CHECK(cudaMalloc(&d_resq_cnt, 4));
        CUDA_CHECK(cudaMemset(d_found, 0, 4)); CUDA_CHECK(cudaMemset(d_nfound, 0, 4)); CUDA_CHECK(cudaMemset(d_resq_cnt, 0, 4));
        u8* d_salt; CUDA_CHECK(cudaMalloc(&d_salt, 32)); CUDA_CHECK(cudaMemset(d_salt, 7, 32));
        TblEntry* d_tbl; CUDA_CHECK(cudaMalloc(&d_tbl, (size_t)(NW*TBLN*sizeof(TblEntry))));
        build_table_kernel<<<1,1>>>(d_tbl); CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t t0,t1; CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        sol_search_kernel<<<blocks,256>>>(d_tbl_key, d_tbl_p, d_tbl_occ, d_tbl_off, d_tbl_mask, d_tbl_cnt,
            d_pfx, 0, NLMAX, d_salt, 0, iters, d_tbl, d_found, d_nfound, d_resq, d_resq_cnt);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        u64 total = (u64)iters*blocks*256;
        printf("bench: %llu keys in %.2f s = %.0f keys/s (%.1f k/s)\n",
               (unsigned long long)total, ms/1000.0, total/(ms/1000.0), total/(ms/1000.0)/1000.0);
        CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    }
    else if (strcmp(mode,"file")==0) {
        if (argc<3) { fprintf(stderr,"usage: file <prefix_file> [blocks]\n"); return 1; }
        int blocks = (argc>=4) ? atoi(argv[3]) : 512;
        if (blocks < 1) blocks = 512;
        u64 nskip = 0; u8* pfx = NULL; u8* plen = NULL;
        u64 n = load_prefix_file(argv[2], &pfx, &plen, &nskip);
        if (n == 0) { fprintf(stderr,"no valid prefixes in file\n"); return 1; }
        printf("file mode: read %llu prefixes (skipped %llu with len>%d)\n",
               (unsigned long long)n, (unsigned long long)nskip, NLMAX);
        run_sol_search(pfx, plen, n, blocks);
        free(pfx); free(plen);
    }
    else if (strcmp(mode,"vanity")==0) {
        if (argc<3) { fprintf(stderr,"usage: vanity <prefix> [blocks]\n"); return 1; }
        int blocks = (argc>=4) ? atoi(argv[3]) : 512;
        int L = (int)strlen(argv[2]);
        if (L == 0 || L > NLMAX) { fprintf(stderr,"prefix must be 1..%d base58 chars\n", NLMAX); return 1; }
        for (int i=0;i<L;i++) if (!strchr(B58_ALPHA, argv[2][i])) { fprintf(stderr,"char '%c' not in base58 alphabet\n", argv[2][i]); return 1; }
        u8* pfx = (u8*)calloc(PSTR,1); u8* plen = (u8*)calloc(1,1);
        memcpy(pfx, argv[2], L); plen[0]=(u8)L;
        printf("vanity prefix: \"%s\" (%d chars)\n", argv[2], L);
        run_sol_search(pfx, plen, 1, blocks);
        free(pfx); free(plen);
    }
    else {
        fprintf(stderr,"usage:\n  vanity_multi.exe sol file <prefix_file> [blocks]\n"
                       "  vanity_multi.exe sol vanity <prefix> [blocks]\n"
                       "  vanity_multi.exe sol bench [iters] [blocks]\n"
                       "  vanity_multi.exe sol selftest\n");
        return 1;
    }
    return 0;
}

} // namespace sol
#undef CUDA_CHECK
#undef PSTR
#undef NLMAX
#undef NW
#undef TBLN
#undef RESQ_CAP
#undef RESQ_MASK

// ============================ chain: solbip ============================
#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Err: %s (line %d)\n", cudaGetErrorString(err), __LINE__); exit(1); } } while (0)
#define PSTR 48          // 每行前缀存储宽度（Solana 地址最长 44）
#define NLMAX 44         // 最大前缀长度
#define NW 51            // 5-bit 窗口数 (51*5 = 255 bit)
#define TBLN 32          // 每窗口 32 项
#define RESQ_CAP (1<<19)
#define RESQ_MASK (RESQ_CAP-1)
namespace solbip {
// ============================================================================
//  sol_vanity_bip39_v3.cu  -  Solana BIP39 vanity address generator (v3)
//  BIP39: entropy -> 12-word mnemonic -> PBKDF2 -> SLIP-0010 m/44'/501'/0'/0' -> ed25519 -> base58 address
//
//  v2 相对旧脚本化的增强：
//    * 大前缀文件（2GB+）流式解析，u64 计数
//    * 命中实时落盘 sol_bip39_hits.txt（无缓冲追加，Ctrl+C 强杀不丢）
//    * 熵由 secret 混合：ent = SHA256(.sol_vanity_secret || counter) 前 16B（与 ETH/BTC/TRX v2 同口径，secret 不打印）
//    * 命中不退出：无限跑到全部唯一前缀命中或 Ctrl+C
//    * 不可能前缀预检（长度>44 / 非 base58 字符集）
//    * 输出可导入钱包的 secret key（base58 64B = seed||pubkey，等同 Phantom 私钥）
//    * 命中同时记录助记词：Phantom 里「导入助记词」或导入私钥都能恢复（路径 m/44'/501'/0'/0'、空口令）
//    * 代价：每个候选都要跑满 PBKDF2 2048 次，速度比 v2 随机 seed 版慢几个数量级，先用 bench 摸底
// ============================================================================

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t  i32;
typedef int64_t  i64;



// ============================================================================
//  fe: GF(2^255-19)，radix 2^25.5 (10 limbs)，ref10 风格
// ============================================================================
struct fe { i32 v[10]; };

__device__ __constant__ fe C_D2  = {{ 45281625,27714825,36363642,13898781,229458,15978800,54557047,27058993,29715967,9444199 }};
__device__ __constant__ fe C_BX  = {{ 52811034,25909283,16144682,17082669,27570973,30858332,40966398,8378388,20764389,8758491 }};
__device__ __constant__ fe C_BY  = {{ 40265304,26843545,13421772,20132659,26843545,6710886,53687091,13421772,40265318,26843545 }};

__device__ __forceinline__ void fe_zero(fe& h) {
    #pragma unroll
    for (int i=0;i<10;i++) h.v[i]=0; }
__device__ __forceinline__ void fe_one(fe& h)  {
    #pragma unroll
    for (int i=0;i<10;i++) h.v[i]=0; h.v[0]=1; }
__device__ __forceinline__ void fe_copy(fe& h, const fe& f) {
    #pragma unroll
    for (int i=0;i<10;i++) h.v[i]=f.v[i]; }
__device__ __forceinline__ int fe_iszero(const fe& f) {
    int z=1;
    #pragma unroll
    for (int i=0;i<10;i++) z &= (f.v[i]==0); return z; }

__device__ __forceinline__ void fe_add(fe& h, const fe& f, const fe& g) {
    i32 t0=f.v[0]+g.v[0], t1=f.v[1]+g.v[1], t2=f.v[2]+g.v[2], t3=f.v[3]+g.v[3], t4=f.v[4]+g.v[4];
    i32 t5=f.v[5]+g.v[5], t6=f.v[6]+g.v[6], t7=f.v[7]+g.v[7], t8=f.v[8]+g.v[8], t9=f.v[9]+g.v[9];
    i32 c;
    c=(t0+(1<<25))>>26; t1+=c; t0-=c<<26;
    c=(t1+(1<<24))>>25; t2+=c; t1-=c<<25;
    c=(t2+(1<<25))>>26; t3+=c; t2-=c<<26;
    c=(t3+(1<<24))>>25; t4+=c; t3-=c<<25;
    c=(t4+(1<<25))>>26; t5+=c; t4-=c<<26;
    c=(t5+(1<<24))>>25; t6+=c; t5-=c<<25;
    c=(t6+(1<<25))>>26; t7+=c; t6-=c<<26;
    c=(t7+(1<<24))>>25; t8+=c; t7-=c<<25;
    c=(t8+(1<<25))>>26; t9+=c; t8-=c<<26;
    c=(t9+(1<<24))>>25; t0+=c*19; t9-=c<<25;
    h.v[0]=t0; h.v[1]=t1; h.v[2]=t2; h.v[3]=t3; h.v[4]=t4;
    h.v[5]=t5; h.v[6]=t6; h.v[7]=t7; h.v[8]=t8; h.v[9]=t9;
}

__device__ __forceinline__ void fe_sub(fe& h, const fe& f, const fe& g) {
    i32 t0=f.v[0]-g.v[0], t1=f.v[1]-g.v[1], t2=f.v[2]-g.v[2], t3=f.v[3]-g.v[3], t4=f.v[4]-g.v[4];
    i32 t5=f.v[5]-g.v[5], t6=f.v[6]-g.v[6], t7=f.v[7]-g.v[7], t8=f.v[8]-g.v[8], t9=f.v[9]-g.v[9];
    i32 c;
    c=(t0+(1<<25))>>26; t1+=c; t0-=c<<26;
    c=(t1+(1<<24))>>25; t2+=c; t1-=c<<25;
    c=(t2+(1<<25))>>26; t3+=c; t2-=c<<26;
    c=(t3+(1<<24))>>25; t4+=c; t3-=c<<25;
    c=(t4+(1<<25))>>26; t5+=c; t4-=c<<26;
    c=(t5+(1<<24))>>25; t6+=c; t5-=c<<25;
    c=(t6+(1<<25))>>26; t7+=c; t6-=c<<26;
    c=(t7+(1<<24))>>25; t8+=c; t7-=c<<25;
    c=(t8+(1<<25))>>26; t9+=c; t8-=c<<26;
    c=(t9+(1<<24))>>25; t0+=c*19; t9-=c<<25;
    h.v[0]=t0; h.v[1]=t1; h.v[2]=t2; h.v[3]=t3; h.v[4]=t4;
    h.v[5]=t5; h.v[6]=t6; h.v[7]=t7; h.v[8]=t8; h.v[9]=t9;
}

__device__ __forceinline__ void fe_neg(fe& h, const fe& f) { fe z; fe_zero(z); fe_sub(h, z, f); }

// 10x10 卷积 + 19 折叠。d = weight(i)+weight(j)-weight(i+j) = 1 仅当 i,j 均奇。
__device__ __noinline__ void fe_mul(fe& h, const fe& f, const fe& g) {
    i64 t[19];
    #pragma unroll
    for (int k=0;k<19;k++) t[k]=0;
    #pragma unroll
    for (int i=0;i<10;i++) {
        #pragma unroll
        for (int j=0;j<10;j++) {
            i64 p = (i64)f.v[i] * (i64)g.v[j];
            if ((i&1) && (j&1)) p += p;
            t[i+j] += p;
        }
    }
    #pragma unroll
    for (int k=10;k<19;k++) t[k-10] += 19*t[k];
    i64 c;
    c=(t[0]+(1LL<<25))>>26; t[1]+=c; t[0]-=c*(1LL<<26);
    c=(t[1]+(1LL<<24))>>25; t[2]+=c; t[1]-=c*(1LL<<25);
    c=(t[2]+(1LL<<25))>>26; t[3]+=c; t[2]-=c*(1LL<<26);
    c=(t[3]+(1LL<<24))>>25; t[4]+=c; t[3]-=c*(1LL<<25);
    c=(t[4]+(1LL<<25))>>26; t[5]+=c; t[4]-=c*(1LL<<26);
    c=(t[5]+(1LL<<24))>>25; t[6]+=c; t[5]-=c*(1LL<<25);
    c=(t[6]+(1LL<<25))>>26; t[7]+=c; t[6]-=c*(1LL<<26);
    c=(t[7]+(1LL<<24))>>25; t[8]+=c; t[7]-=c*(1LL<<25);
    c=(t[8]+(1LL<<25))>>26; t[9]+=c; t[8]-=c*(1LL<<26);
    c=(t[9]+(1LL<<24))>>25; t[0]+=19*c; t[9]-=c*(1LL<<25);
    c=(t[0]+(1LL<<25))>>26; t[1]+=c; t[0]-=c*(1LL<<26);
    #pragma unroll
    for (int i=0;i<10;i++) h.v[i]=(i32)t[i];
}

__device__ __forceinline__ void fe_sq(fe& h, const fe& f) { fe_mul(h, f, f); }

// a^(p-2)  (ref10 pow225521 链：255 平方 + 11 乘)
__device__ __noinline__ void fe_invert(fe& out, const fe& z) {
    fe t0,t1,t2,t3;
    fe_sq(t0,z);                    // 2
    fe_sq(t1,t0);                   // 4
    fe_sq(t1,t1);                   // 8
    fe_mul(t1,z,t1);                // 9
    fe_mul(t0,t0,t1);               // 11
    fe_sq(t2,t0);                   // 22
    fe_mul(t1,t1,t2);               // 31 = 2^5-1
    fe_sq(t2,t1);
    #pragma unroll
    for (int i=1;i<5;i++) fe_sq(t2,t2);
    fe_mul(t1,t2,t1);               // 2^10-1
    fe_sq(t2,t1);
    #pragma unroll
    for (int i=1;i<10;i++) fe_sq(t2,t2);
    fe_mul(t2,t2,t1);               // 2^20-1
    fe_sq(t3,t2);
    #pragma unroll
    for (int i=1;i<20;i++) fe_sq(t3,t3);
    fe_mul(t2,t3,t2);               // 2^40-1
    fe_sq(t2,t2);
    #pragma unroll
    for (int i=1;i<10;i++) fe_sq(t2,t2);
    fe_mul(t1,t2,t1);               // 2^50-1
    fe_sq(t2,t1);
    #pragma unroll
    for (int i=1;i<50;i++) fe_sq(t2,t2);
    fe_mul(t2,t2,t1);               // 2^100-1
    fe_sq(t3,t2);
    #pragma unroll
    for (int i=1;i<100;i++) fe_sq(t3,t3);
    fe_mul(t2,t3,t2);               // 2^200-1
    fe_sq(t2,t2);
    #pragma unroll
    for (int i=1;i<50;i++) fe_sq(t2,t2);
    fe_mul(t1,t2,t1);               // 2^250-1
    fe_sq(t1,t1);
    #pragma unroll
    for (int i=1;i<5;i++) fe_sq(t1,t1);
    fe_mul(out,t1,t0);              // 2^255-21 = p-2
}

__device__ __noinline__ void fe_tobytes(u8 s[32], const fe& hin) {
    i32 t0=hin.v[0],t1=hin.v[1],t2=hin.v[2],t3=hin.v[3],t4=hin.v[4];
    i32 t5=hin.v[5],t6=hin.v[6],t7=hin.v[7],t8=hin.v[8],t9=hin.v[9];
    i32 q;
    q = (19*t9 + (1<<24)) >> 25;
    q = (t0 + q) >> 26;
    q = (t1 + q) >> 25;
    q = (t2 + q) >> 26;
    q = (t3 + q) >> 25;
    q = (t4 + q) >> 26;
    q = (t5 + q) >> 25;
    q = (t6 + q) >> 26;
    q = (t7 + q) >> 25;
    q = (t8 + q) >> 26;
    q = (t9 + q) >> 25;
    t0 += 19*q;
    i32 c;
    // 这里必须用截断式进位（不是 ref10 fe_add/fe_mul 的四舍五入式），
    // 否则 limb 会变负、打包时的 << 会带出符号位。
    c=(t0)>>26; t1+=c; t0-=c<<26;
    c=(t1)>>25; t2+=c; t1-=c<<25;
    c=(t2)>>26; t3+=c; t2-=c<<26;
    c=(t3)>>25; t4+=c; t3-=c<<25;
    c=(t4)>>26; t5+=c; t4-=c<<26;
    c=(t5)>>25; t6+=c; t5-=c<<25;
    c=(t6)>>26; t7+=c; t6-=c<<26;
    c=(t7)>>25; t8+=c; t7-=c<<25;
    c=(t8)>>26; t9+=c; t8-=c<<26;
    c=(t9)>>25; t9-=c<<25;
    s[ 0]=(u8)(t0>> 0); s[ 1]=(u8)(t0>> 8); s[ 2]=(u8)(t0>>16);
    s[ 3]=(u8)((t0>>24)|(t1<<2));
    s[ 4]=(u8)(t1>> 6); s[ 5]=(u8)(t1>>14);
    s[ 6]=(u8)((t1>>22)|(t2<<3));
    s[ 7]=(u8)(t2>> 5); s[ 8]=(u8)(t2>>13);
    s[ 9]=(u8)((t2>>21)|(t3<<5));
    s[10]=(u8)(t3>> 3); s[11]=(u8)(t3>>11);
    s[12]=(u8)((t3>>19)|(t4<<6));
    s[13]=(u8)(t4>> 2); s[14]=(u8)(t4>>10); s[15]=(u8)(t4>>18);
    s[16]=(u8)(t5>> 0); s[17]=(u8)(t5>> 8); s[18]=(u8)(t5>>16);
    s[19]=(u8)((t5>>24)|(t6<<1));
    s[20]=(u8)(t6>> 7); s[21]=(u8)(t6>>15);
    s[22]=(u8)((t6>>23)|(t7<<3));
    s[23]=(u8)(t7>> 5); s[24]=(u8)(t7>>13);
    s[25]=(u8)((t7>>21)|(t8<<4));
    s[26]=(u8)(t8>> 4); s[27]=(u8)(t8>>12);
    s[28]=(u8)((t8>>20)|(t9<<6));
    s[29]=(u8)(t9>> 2); s[30]=(u8)(t9>>10); s[31]=(u8)(t9>>18);
}

// ============================================================================
//  Edwards 点运算（extended coords: x=X/Z, y=Y/Z, T=XY/Z）
// ============================================================================
struct ge_p3     { fe X, Y, Z, T; };
struct ge_cached { fe YpX, YmX, Z, T2d; };   // Z 存 2*Z2
typedef struct { ge_p3 p; ge_cached c; } TblEntry;

__device__ __forceinline__ void ge_set_base(ge_p3& p) {
    fe_copy(p.X, C_BX); fe_copy(p.Y, C_BY); fe_one(p.Z); fe_mul(p.T, C_BX, C_BY);
}

__device__ __forceinline__ void ge_p3_to_cached(ge_cached& q, const ge_p3& p) {
    fe_add(q.YpX, p.Y, p.X);
    fe_sub(q.YmX, p.Y, p.X);
    fe_add(q.Z, p.Z, p.Z);          // 2Z
    fe_mul(q.T2d, p.T, C_D2);
}

// add-2008-hwcd-3: 8M
__device__ __forceinline__ void ge_add(ge_p3& r, const ge_p3& p, const ge_cached& q) {
    fe A,B,C,D,E,F,G,H,t0,t1;
    fe_sub(t0, p.Y, p.X); fe_mul(A, t0, q.YmX);
    fe_add(t1, p.Y, p.X); fe_mul(B, t1, q.YpX);
    fe_mul(C, p.T, q.T2d);
    fe_mul(D, p.Z, q.Z);
    fe_add(H, B, A);                 // H = B+A
    fe_sub(E, B, A);                 // E = B-A  (A 复用)
    fe_sub(F, D, C);                 // F = D-C  (C 复用)
    fe_add(G, D, C);                 // G = D+C  (D 复用)
    fe_mul(r.X, E, F);
    fe_mul(r.Y, G, H);
    fe_mul(r.T, E, H);
    fe_mul(r.Z, F, G);
}

// dbl-2008-hwcd (a=-1): 3M+4S
__device__ __forceinline__ void ge_dbl(ge_p3& r, const ge_p3& p) {
    fe A,B,C,E,F,G,H;
    fe_sq(A, p.X);
    fe_sq(B, p.Y);
    fe_sq(C, p.Z); fe_add(C, C, C);      // C = 2Z^2
    fe_add(E, p.X, p.Y); fe_sq(E, E); fe_sub(E, E, A); fe_sub(E, E, B);   // E = (X+Y)^2-A-B
    fe_sub(G, B, A);                     // G = B-A   (D = -A)
    fe_add(H, B, A); fe_neg(H, H);       // H = -A-B
    fe_sub(F, G, C);                     // F = G-C
    fe_mul(r.X, E, F);
    fe_mul(r.Y, G, H);
    fe_mul(r.T, E, H);
    fe_mul(r.Z, F, G);
}

__device__ __forceinline__ void ge_to_affine(fe& x, fe& y, const ge_p3& p) {
    fe zi; fe_invert(zi, p.Z);
    fe_mul(x, p.X, zi);
    fe_mul(y, p.Y, zi);
}

__device__ __forceinline__ int ge_encode(u8 out[32], const ge_p3& p) {
    fe x,y; ge_to_affine(x,y,p);
    u8 yb[32], xb[32];
    fe_tobytes(yb, y);
    fe_tobytes(xb, x);
    #pragma unroll
    for (int i=0;i<32;i++) out[i]=yb[i];
    out[31] |= (u8)((xb[0]&1) << 7);
    return 1;
}

// 固定基点预计算表：tbl[w*TBLN+d] = d * 2^(5w) * B
__global__ void build_table_kernel(TblEntry* tbl) {
    if (blockIdx.x || threadIdx.x) return;
    ge_p3 P; ge_set_base(P);
    ge_p3 Q;
    for (int w=0; w<NW; w++) {
        ge_cached cp; ge_p3_to_cached(cp, P);
        fe_zero(Q.X); fe_zero(Q.Y); fe_one(Q.Z); fe_zero(Q.T);
        for (int dg=1; dg<TBLN; dg++) {
            if (dg==1) Q = P; else ge_add(Q, Q, cp);
            tbl[w*TBLN+dg].p = Q;
            ge_p3_to_cached(tbl[w*TBLN+dg].c, Q);
        }
        // P = 32 * P
        for (int s=0;s<5;s++) ge_dbl(P, P);
    }
}

// ============================================================================
//  SHA-256 / SHA-512
// ============================================================================
__device__ __constant__ u32 K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };

__device__ __forceinline__ u32 rotr32(u32 x, int n) { return (x>>n)|(x<<(32-n)); }

__device__ __forceinline__ void sha256_block(u32 h[8], const u8 blk[64]) {
    u32 w[64];
    #pragma unroll
    for (int i=0;i<16;i++) w[i]=((u32)blk[i*4]<<24)|((u32)blk[i*4+1]<<16)|((u32)blk[i*4+2]<<8)|blk[i*4+3];
    #pragma unroll
    for (int i=16;i<64;i++) {
        u32 s0 = rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);
        u32 s1 = rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    u32 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    #pragma unroll
    for (int i=0;i<64;i++) {
        u32 S1 = rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
        u32 ch = (e&f)^((~e)&g);
        u32 t1 = hh + S1 + ch + K256[i] + w[i];
        u32 S0 = rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
        u32 maj = (a&b)^(a&c)^(b&c);
        u32 t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

__device__ __forceinline__ void sha256(const u8* data, int len, u8 out[32]) {
    u32 h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    u8 blk[64]; int off=0;
    while (len-off >= 64) {
        #pragma unroll
        for (int i=0;i<64;i++) blk[i]=data[off+i];
        sha256_block(h, blk); off += 64;
    }
    #pragma unroll
    for (int i=0;i<64;i++) blk[i]=0;
    int rem = len-off;
    for (int i=0;i<rem;i++) blk[i]=data[off+i];
    blk[rem]=0x80;
    u64 bits = (u64)len*8;
    if (rem >= 56) { sha256_block(h, blk);
        #pragma unroll
        for (int i=0;i<64;i++) blk[i]=0; }
    #pragma unroll
    for (int i=0;i<8;i++) blk[56+i]=(u8)((bits>>(56-8*i))&0xFF);
    sha256_block(h, blk);
    #pragma unroll
    for (int i=0;i<8;i++) { out[i*4]=(u8)(h[i]>>24); out[i*4+1]=(u8)(h[i]>>16); out[i*4+2]=(u8)(h[i]>>8); out[i*4+3]=(u8)h[i]; }
}

__device__ __constant__ u64 K512[80] = {
    0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL,0x12835b0145706fbeULL,0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL,0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL,0xd5a79147930aa725ULL,0x06ca6351e003826fULL,0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,0x53380d139d95b3dfULL,
    0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL,0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL,
    0xca273eceea26619cULL,0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,0x113f9804bef90daeULL,0x1b710b35131c471bULL,
    0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL };

__device__ __forceinline__ u64 rotr64(u64 x, int n) { return (x>>n)|(x<<(64-n)); }
__device__ __forceinline__ u64 S0f(u64 x) { return rotr64(x,28)^rotr64(x,34)^rotr64(x,39); }
__device__ __forceinline__ u64 S1f(u64 x) { return rotr64(x,14)^rotr64(x,18)^rotr64(x,41); }
__device__ __forceinline__ u64 s0f(u64 x) { return rotr64(x,1)^rotr64(x,8)^(x>>7); }
__device__ __forceinline__ u64 s1f(u64 x) { return rotr64(x,19)^rotr64(x,61)^(x>>6); }

__device__ __forceinline__ void sha512_init(u64 st[8]) {
    st[0]=0x6a09e667f3bcc908ULL; st[1]=0xbb67ae8584caa73bULL;
    st[2]=0x3c6ef372fe94f82bULL; st[3]=0xa54ff53a5f1d36f1ULL;
    st[4]=0x510e527fade682d1ULL; st[5]=0x9b05688c2b3e6c1fULL;
    st[6]=0x1f83d9abfb41bd6bULL; st[7]=0x5be0cd19137e2179ULL;
}

__device__ __noinline__ void sha512_compress(u64 st[8], u64 w[16]) {
    u64 a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
    for (int i=0;i<80;i++) {
        u64 wi;
        if (i < 16) wi = w[i];
        else {
            w[i&15] += s1f(w[(i-2)&15]) + w[(i-7)&15] + s0f(w[(i-15)&15]);
            wi = w[i&15];
        }
        u64 T1 = h + S1f(e) + ((e&f)^((~e)&g)) + K512[i] + wi;
        u64 T2 = S0f(a) + ((a&b)^(a&c)^(b&c));
        h=g; g=f; f=e; e=d+T1; d=c; c=b; b=a; a=T1+T2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d; st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}

// SHA-512(32 字节消息)：单块
__device__ __noinline__ void sha512_of32(const u8 seed[32], u8 out[64]) {
    u64 st[8]; sha512_init(st);
    u8 blk[128];
    #pragma unroll
    for (int i=0;i<128;i++) blk[i]=0;
    #pragma unroll
    for (int i=0;i<32;i++) blk[i]=seed[i];
    blk[32]=0x80;
    blk[126]=0x01; blk[127]=0x00;          // 256 bit 长度
    u64 w[16];
    #pragma unroll
    for (int i=0;i<16;i++) {
        u64 v=0;
        #pragma unroll
        for (int j=0;j<8;j++) v = (v<<8) | blk[i*8+j];
        w[i]=v;
    }
    sha512_compress(st, w);
    #pragma unroll
    for (int i=0;i<8;i++) {
        out[i*8+0]=(u8)(st[i]>>56); out[i*8+1]=(u8)(st[i]>>48); out[i*8+2]=(u8)(st[i]>>40); out[i*8+3]=(u8)(st[i]>>32);
        out[i*8+4]=(u8)(st[i]>>24); out[i*8+5]=(u8)(st[i]>>16); out[i*8+6]=(u8)(st[i]>>8);  out[i*8+7]=(u8)st[i];
    }
}

// ============================================================================
//  v3 新增：BIP39 + SLIP-0010 派生（助记词版的核心链路）
//  counter -> 熵(16B) -> 12 词助记词 -> PBKDF2-HMAC-SHA512(2048 次, 空口令)
//          -> SLIP-0010 ed25519 派生 m/44'/501'/0'/0' -> 子私钥即 ed25519 seed
//  与 Phantom 默认导入口径一致：助记词 + 空口令 + 上述路径的第一个地址就是命中地址。
// ============================================================================
__constant__ u8 SOL_SECRET[32];  // 主机端真随机 secret（.sol_vanity_secret），与 counter 混合决定熵序列；绝不打印

// 通用 SHA-512：建立在既有 sha512_init / sha512_compress 之上。
// 注意 sha512_compress 会把 w[16] 当滚动窗口改写，所以每一块都要重新装载。
__device__ __forceinline__ void sha512_load_w(const u8* blk, u64 w[16]) {
    #pragma unroll
    for (int i=0;i<16;i++) { u64 v=0; for (int j=0;j<8;j++) v=(v<<8)|blk[i*8+j]; w[i]=v; }
}
// 从已有状态继续吸收 data 并按 total_len（整条消息总字节数）收尾输出。HMAC 的 ipad/opad 预压缩状态靠它复用。
__device__ void sha512_from_state(u64 st[8], const u8* data, int len, u64 total_len, u8 out[64]) {
    int off=0; u64 w[16];
    while (len-off >= 128) { sha512_load_w(data+off, w); sha512_compress(st, w); off += 128; }
    u8 blk[128];
    for (int i=0;i<128;i++) blk[i]=0;
    int rem = len-off;
    for (int i=0;i<rem;i++) blk[i]=data[off+i];
    blk[rem]=0x80;
    if (rem >= 112) { sha512_load_w(blk, w); sha512_compress(st, w); for (int i=0;i<128;i++) blk[i]=0; }
    u64 bits = total_len*8;
    for (int i=0;i<8;i++) blk[120+i]=(u8)((bits>>(56-8*i))&0xFF);
    sha512_load_w(blk, w); sha512_compress(st, w);
    for (int i=0;i<8;i++) for (int j=0;j<8;j++) out[i*8+j]=(u8)((st[i]>>(56-8*j))&0xFF);
}
__device__ void sha512_gen(const u8* data, int len, u8 out[64]) {
    u64 st[8]; sha512_init(st);
    sha512_from_state(st, data, len, (u64)len, out);
}
// HMAC-SHA512：先预计算 ipad/opad 压缩后状态，之后每次调用只剩消息块 + 收尾块。
__device__ __forceinline__ void hmac512_precompute(const u8* key, int key_len, u64 ipad_st[8], u64 opad_st[8]) {
    u8 kb[128];
    for (int i=0;i<128;i++) kb[i]=0;
    if (key_len > 128) { u8 kh[64]; sha512_gen(key, key_len, kh); for (int i=0;i<64;i++) kb[i]=kh[i]; }
    else for (int i=0;i<key_len;i++) kb[i]=key[i];
    u8 blk[128]; u64 w[16];
    for (int i=0;i<128;i++) blk[i]=kb[i]^0x36;
    sha512_init(ipad_st); sha512_load_w(blk, w); sha512_compress(ipad_st, w);
    for (int i=0;i<128;i++) blk[i]=kb[i]^0x5c;
    sha512_init(opad_st); sha512_load_w(blk, w); sha512_compress(opad_st, w);
}
__device__ __forceinline__ void hmac512_with_states(const u64 ipad_st[8], const u64 opad_st[8], const u8* msg, int msg_len, u8 out[64]) {
    u64 st[8]; for (int i=0;i<8;i++) st[i]=ipad_st[i];
    u8 ih[64];
    sha512_from_state(st, msg, msg_len, (u64)(128+msg_len), ih);
    for (int i=0;i<8;i++) st[i]=opad_st[i];
    sha512_from_state(st, ih, 64, (u64)(128+64), out);
}
__device__ __forceinline__ void hmac512(const u8* key, int key_len, const u8* msg, int msg_len, u8 out[64]) {
    u64 is_[8], os_[8]; hmac512_precompute(key, key_len, is_, os_);
    hmac512_with_states(is_, os_, msg, msg_len, out);
}
// 熵 -> 12 词助记词（BIP39：128 位熵 + 4 位校验和 = 12 x 11 位索引）
__device__ void sol_entropy_to_mnemonic(const u8 ent[16], char* mnem, int* mnem_len) {
    u8 h[32]; sha256(ent, 16, h);
    u16 idx[12];
    for (int wd=0; wd<12; wd++) {
        int bp = wd*11; u32 val=0;
        for (int b=0;b<11;b++) {
            int p=bp+b; int byte=p>>3, sh=7-(p&7);
            int bit = (byte<16) ? ((ent[byte]>>sh)&1) : ((h[0]>>sh)&1);
            val = (val<<1)|bit;
        }
        idx[wd]=(u16)val;
    }
    int pos=0;
    for (int wd=0; wd<12; wd++) {
        const char* word = BIP39_WORDS[idx[wd]];
        int len=0; while (word[len]) len++;
        for (int i=0;i<len;i++) mnem[pos++]=word[i];
        if (wd<11) mnem[pos++]=' ';
    }
    mnem[pos]=0;
    if (mnem_len) *mnem_len=pos;
}
// BIP39 种子：PBKDF2-HMAC-SHA512(password=助记词, salt="mnemonic", 2048 次)，空口令（Phantom 默认）
__device__ void sol_bip39_seed(const char* mnem, int mnem_len, u8 seed[64]) {
    u64 is_[8], os_[8];
    hmac512_precompute((const u8*)mnem, mnem_len, is_, os_);
    const u8 saltmsg[12] = { 'm','n','e','m','o','n','i','c', 0,0,0,1 };   // "mnemonic" || INT32_BE(1)
    u8 U[64], T[64];
    hmac512_with_states(is_, os_, saltmsg, 12, U);
    for (int i=0;i<64;i++) T[i]=U[i];
    for (int it=1; it<2048; it++) {
        hmac512_with_states(is_, os_, U, 64, U);
        for (int i=0;i<64;i++) T[i]^=U[i];
    }
    for (int i=0;i<64;i++) seed[i]=T[i];
}
// SLIP-0010 ed25519 硬化派生一步：I = HMAC-SHA512(chain_code, 0x00 || k || ser32(index'))
__device__ __forceinline__ void slip10_ckd(u8 k[32], u8 c[32], u32 idx_hardened) {
    u8 data[37]; data[0]=0;
    for (int i=0;i<32;i++) data[1+i]=k[i];
    data[33]=(u8)(idx_hardened>>24); data[34]=(u8)(idx_hardened>>16);
    data[35]=(u8)(idx_hardened>>8);  data[36]=(u8)idx_hardened;
    u8 out[64]; hmac512(c, 32, data, 37, out);
    for (int i=0;i<32;i++) { k[i]=out[i]; c[i]=out[32+i]; }
}
// BIP39 种子 -> m/44'/501'/0'/0' 子私钥（该 32B 直接作 ed25519 seed 使用）
__device__ void sol_derive_child(const u8 seed64[64], u8 child[32]) {
    u8 out[64];
    hmac512((const u8*)"ed25519 seed", 12, seed64, 64, out);
    u8 k[32], c[32];
    for (int i=0;i<32;i++) { k[i]=out[i]; c[i]=out[32+i]; }
    slip10_ckd(k, c, 0x80000000u+44u);
    slip10_ckd(k, c, 0x80000000u+501u);
    slip10_ckd(k, c, 0x80000000u+0u);
    slip10_ckd(k, c, 0x80000000u+0u);
    for (int i=0;i<32;i++) child[i]=k[i];
}
// counter -> 熵：ent = SHA256(SOL_SECRET(32B) || counter 小端 8B) 的前 16 字节（与 ETH/BTC/TRX v2 同口径）
__device__ __forceinline__ void sol_counter_to_entropy(u64 n, u8 ent[16]) {
    u8 buf[40];
    for (int i=0;i<32;i++) buf[i]=SOL_SECRET[i];
    for (int i=0;i<8;i++) buf[32+i]=(u8)(n>>(8*i));
    u8 h[32]; sha256(buf, 40, h);
    for (int i=0;i<16;i++) ent[i]=h[i];
}
// 一条龙：counter -> 子私钥 seed，并回填助记词与熵（供命中记录落盘）
__device__ __forceinline__ void sol_derive_from_counter(u64 n, u8 child[32], char* mnem, u8 ent_out[16]) {
    u8 ent[16]; sol_counter_to_entropy(n, ent);
    char m[128]; int mlen=0;
    sol_entropy_to_mnemonic(ent, m, &mlen);
    u8 seed[64]; sol_bip39_seed(m, mlen, seed);
    sol_derive_child(seed, child);
    for (int i=0;i<128;i++) mnem[i]=0;
    for (int i=0;i<=mlen;i++) mnem[i]=m[i];
    for (int i=0;i<16;i++) ent_out[i]=ent[i];
}
// ============================================================================
//  Base58
// ============================================================================
__device__ __constant__ char C_B58[59] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

__device__ __noinline__ int b58_enc32(const u8 in[32], char* out) {
    int zeros=0;
    while (zeros<32 && in[zeros]==0) zeros++;
    u8 num[32];
    #pragma unroll
    for (int i=0;i<32;i++) num[i]=in[i];
    char digs[48]; int nd=0;
    for (;;) {
        int allzero=1;
        for (int i=zeros;i<32;i++) if (num[i]) { allzero=0; break; }
        if (allzero) break;
        int rem=0;
        for (int i=zeros;i<32;i++) {
            int v = rem*256 + num[i];
            num[i] = (u8)(v/58);
            rem = v - (v/58)*58;
        }
        digs[nd++] = C_B58[rem];
        if (nd>=47) break;
    }
    int l=0;
    for (int i=0;i<zeros;i++) out[l++]='1';
    for (int i=nd-1;i>=0;i--) out[l++]=digs[i];
    if (l==0) out[l++]='1';
    out[l]=0;
    return l;
}

// ============================================================================
//  主搜索内核
// ============================================================================
typedef struct { u32 p; u32 pad; u8 addr[PSTR]; u8 seed[32]; u8 pk[32]; u8 ent[16]; char mnem[128]; } ResqEntry;  // v3: seed = SLIP-0010 child key; ent/mnem for mnemonic recovery

__device__ __forceinline__ u32 get_bits5(const u32 w[8], int bit) {
    int idx = bit>>5, sh = bit&31;
    u64 v = w[idx];
    if (sh > 27 && idx < 7) v |= ((u64)w[idx+1])<<32;
    return (u32)((v>>sh) & 31u);
}

__global__ __launch_bounds__(256, 2) void sol_search_kernel(
        const u64* d_tbl_key, const u32* d_tbl_p, const u8* d_tbl_occ,
        const u32* d_tbl_off, const u32* d_tbl_mask, const u32* d_tbl_cnt,
        const u8* d_pfx, u64 uniq, int nl_max,
        u64 start_nonce, int iters,
        const TblEntry* tbl,
        u32* d_found, u32* d_nfound, ResqEntry* d_resq, u32* d_resq_cnt) {

    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x*blockDim.x;

    for (int it=0; it<iters; it++) {
        // 不再“全部命中就 return”：提前返回会让空转迭代也被计入 addr/s，出现天文数字假速度。
        // 已命中的前缀只是不再上报，标量乘照常做满。
        u64 n = start_nonce + tid + (u64)it*stride;

        // v3: counter -> entropy -> mnemonic -> PBKDF2 -> SLIP-0010 child key (= ed25519 seed)
        u8 seed[32]; char mnem[128]; u8 ent[16];
        sol_derive_from_counter(n, seed, mnem, ent);
        u8 hsh[64];  sha512_of32(seed, hsh);

        u8 a[32];
        #pragma unroll
        for (int i=0;i<32;i++) a[i]=hsh[i];
        a[0] &= 248; a[31] &= 127; a[31] |= 64;

        u32 w8[8];
        #pragma unroll
        for (int i=0;i<8;i++) w8[i] = (u32)a[i*4] | ((u32)a[i*4+1]<<8) | ((u32)a[i*4+2]<<16) | ((u32)a[i*4+3]<<24);

        ge_p3 R; int started=0;
        // v2: 每线程每个候选做一次完整的固定基点乘（钱包可导入的 seed 决定标量，无法做增量点加）
        for (int w=NW-1; w>=0; w--) {
            u32 dg = get_bits5(w8, w*5);
            if (!dg) continue;
            const TblEntry& e = tbl[w*TBLN + dg];
            if (!started) { R = e.p; started = 1; }
            else ge_add(R, R, e.c);
        }
        if (!started) { continue; }

        u8 pk[32]; ge_encode(pk, R);

        char ast[PSTR];
        int laddr = b58_enc32(pk, ast);
        if (laddr > PSTR-1) laddr = PSTR-1;

        u64 hv = 0xcbf29ce484222325ull;
        for (int L=1; L<=nl_max; L++) {
            if (L > laddr) break;
            hv = (hv ^ (u8)ast[L-1]) * 0x100000001b3ull;
            int cnt = d_tbl_cnt[L-1];
            if (!cnt) continue;
            u32 mask = d_tbl_mask[L-1];
            u32 base = d_tbl_off[L-1];
            u32 hh2 = (u32)((hv * 0x9E3779B97F4A7C15ull) >> 32);
            int s = (int)(hh2 & mask);
            for (;;) {
                if (!d_tbl_occ[base+s]) break;
                if (d_tbl_key[base+s] == hv) {
                    u32 p = d_tbl_p[base+s];
                    int ok=1;
                    for (int k=0;k<L;k++) if (d_pfx[(u64)p*PSTR+k]!=(u8)ast[k]) { ok=0; break; }
                    if (ok && !d_found[p]) {
                        u32 old = atomicCAS(&d_found[p], 0u, 1u);
                        if (old==0) {
                            u32 q = atomicAdd(d_resq_cnt, 1);
                            ResqEntry* re = &d_resq[q & RESQ_MASK];
                            re->p = p; re->pad = 0;
                            for (int i=0;i<laddr;i++) re->addr[i]=(u8)ast[i];
                            re->addr[laddr]=0;
                            #pragma unroll
                            for (int i=0;i<32;i++) { re->seed[i]=seed[i]; re->pk[i]=pk[i]; }
                            for (int i=0;i<16;i++) re->ent[i]=ent[i];
                            for (int i=0;i<128;i++) re->mnem[i]=mnem[i];
                            atomicAdd(d_nfound, 1);
                        }
                        break;
                    }
                }
                s = (s+1) & mask;
            }
        }
    }
}

// ============================================================================
//  自检内核
// ============================================================================
typedef struct { u8 mode; u8 seed[32]; u8 salt[32]; u64 ctr; u8 ent[16]; } TestCase;   // mode 0=direct seed, 1=salt+ctr (v2 legacy), 2=entropy->mnemonic full chain (v3)

__global__ void selftest_kernel(const TestCase* cases, int ncase, u8* d_pub, u8* d_addr, int* d_dbl, u32* d_invok, const TblEntry* tbl) {
    if (blockIdx.x || threadIdx.x) return;
    for (int t=0;t<ncase;t++) {
        u8 seed[32];
        if (cases[t].mode == 0) {
            #pragma unroll
            for (int i=0;i<32;i++) seed[i]=cases[t].seed[i];
        } else if (cases[t].mode == 2) {
            char m[128]; int mlen=0;
            sol_entropy_to_mnemonic(cases[t].ent, m, &mlen);
            u8 s64[64]; sol_bip39_seed(m, mlen, s64);
            sol_derive_child(s64, seed);
        } else {
            u8 buf[40];
            #pragma unroll
            for (int i=0;i<32;i++) buf[i]=cases[t].salt[i];
            #pragma unroll
            for (int i=0;i<8;i++) buf[32+i]=(u8)(cases[t].ctr>>(8*i));
            sha256(buf, 40, seed);
        }
        u8 hsh[64]; sha512_of32(seed, hsh);
        u8 a[32];
        #pragma unroll
        for (int i=0;i<32;i++) a[i]=hsh[i];
        a[0]&=248; a[31]&=127; a[31]|=64;
        u32 w8[8];
        #pragma unroll
        for (int i=0;i<8;i++) w8[i] = (u32)a[i*4] | ((u32)a[i*4+1]<<8) | ((u32)a[i*4+2]<<16) | ((u32)a[i*4+3]<<24);
        ge_p3 R; int started=0;
        for (int w=NW-1; w>=0; w--) {
            u32 dg = get_bits5(w8, w*5);
            if (!dg) continue;
            const TblEntry& e = tbl[w*TBLN + dg];
            if (!started) { R = e.p; started = 1; }
            else ge_add(R, R, e.c);
        }
        u8 pk[32]; ge_encode(pk, R);
        #pragma unroll
        for (int i=0;i<32;i++) d_pub[t*32+i]=pk[i];
        char ast[PSTR]; int la = b58_enc32(pk, ast);
        for (int i=0;i<PSTR;i++) d_addr[t*PSTR+i]=(u8)(i<la?ast[i]:0);
    }
    // 2B: dbl vs add 一致性
    ge_p3 B, R1, R2;
    ge_set_base(B);
    ge_cached cb; ge_p3_to_cached(cb, B);
    ge_add(R1, B, cb);
    ge_dbl(R2, B);
    u8 e1[32], e2[32];
    ge_encode(e1, R1); ge_encode(e2, R2);
    int same=1;
    #pragma unroll
    for (int i=0;i<32;i++) if (e1[i]!=e2[i]) same=0;
    d_dbl[0] = same;

    // (x^-1)*x == 1
    fe x, xi, y;
    fe z; fe_zero(z);
    x.v[0]=12345678; x.v[1]=-98765; x.v[2]=777; x.v[3]=-4242424; x.v[4]=99;
    x.v[5]=-1234; x.v[6]=5678901; x.v[7]=-77; x.v[8]=3333; x.v[9]=-55555;
    fe_invert(xi, x);
    fe_mul(y, xi, x);
    u8 yb[32]; fe_tobytes(yb, y);
    u8 one[32]; fe_one(z); fe_tobytes(one, z);
    same=1;
    #pragma unroll
    for (int i=0;i<32;i++) if (yb[i]!=one[i]) same=0;
    d_invok[0] = same;
}


// ============================================================================
//  主机侧
// ============================================================================
static const char B58C[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

static int b58_enc(const u8* in, int len, char* out) {
    int zeros=0; while (zeros<len && in[zeros]==0) zeros++;
    u8 num[96]; memcpy(num, in, len);
    char digs[160]; int nd=0;
    for (;;) {
        int allzero=1;
        for (int i=zeros;i<len;i++) if (num[i]) { allzero=0; break; }
        if (allzero) break;
        int rem=0;
        for (int i=zeros;i<len;i++) { int v=rem*256+num[i]; num[i]=(u8)(v/58); rem=v-(v/58)*58; }
        digs[nd++]=B58C[rem];
        if (nd>=159) break;
    }
    int l=0;
    for (int i=0;i<zeros;i++) out[l++]='1';
    for (int i=nd-1;i>=0;i--) out[l++]=digs[i];
    if (l==0) out[l++]='1';
    out[l]=0;
    return l;
}

static const char* B58_ALPHA = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// 流式读前缀文件（支持 2GB+）；返回条数，pfx 行宽 PSTR
static u64 load_prefix_file(const char* fn, u8** pfx_out, u8** plen_out, u64* skip_out) {
    FILE* fp = fopen(fn, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", fn); return 0; }
    u64 cap = 1ull<<20, n = 0, skipped = 0;
    u8* pfx = (u8*)malloc((size_t)(cap*PSTR));
    u8* plen = (u8*)malloc((size_t)cap);
    if (!pfx || !plen) { fprintf(stderr, "out of memory\n"); fclose(fp); return 0; }
    const size_t CH = 32ull<<20;
    u8* buf = (u8*)malloc(CH + 8192);
    if (!buf) { fprintf(stderr, "out of memory\n"); fclose(fp); return 0; }
    size_t carry = 0;
    while (1) {
        size_t r = fread(buf + carry, 1, CH, fp);
        size_t total = carry + r;
        if (total == 0) break;
        size_t start = 0;
        for (size_t i=0;i<total;i++) {
            if (buf[i] != '\n' && buf[i] != '\r') continue;
            size_t s0 = start, e0 = i;
            while (s0 < e0 && (buf[s0]==' '||buf[s0]=='\t')) s0++;
            size_t e1 = s0;
            while (e1 < e0 && buf[e1]!=' ' && buf[e1]!='\t' && buf[e1]!='#') e1++;
            size_t L = e1 - s0;
            if (L > 0) {
                if (L > NLMAX) { skipped++; }
                else {
                    if (n >= cap) { cap <<= 1;
                        pfx = (u8*)realloc(pfx, (size_t)(cap*PSTR));
                        plen = (u8*)realloc(plen, (size_t)cap);
                        if (!pfx || !plen) { fprintf(stderr,"out of memory\n"); fclose(fp); return 0; } }
                    memcpy(pfx + n*PSTR, buf + s0, L);
                    plen[n] = (u8)L;
                    n++;
                }
            }
            start = i + 1;
        }
        carry = total - start;
        if (r == 0) {
            // 文件最后一行可能没有换行符，必须补处理，否则会丢掉最后一条前缀
            size_t s0 = start;
            while (s0 < total && (buf[s0]==' '||buf[s0]=='\t')) s0++;
            size_t e1 = s0;
            while (e1 < total && buf[e1]!=' ' && buf[e1]!='\t' && buf[e1]!='#' &&
                   buf[e1]!='\n' && buf[e1]!='\r') e1++;
            size_t L = e1 - s0;
            if (L > 0) {
                if (L > NLMAX) { skipped++; }
                else {
                    if (n >= cap) { cap <<= 1;
                        u8* np = (u8*)realloc(pfx, (size_t)(cap*PSTR));
                        u8* nl = (u8*)realloc(plen, (size_t)cap);
                        if (!np || !nl) { fprintf(stderr,"out of memory\n"); fclose(fp); return 0; }
                        pfx = np; plen = nl; }
                    memcpy(pfx + n*PSTR, buf + s0, L);
                    plen[n] = (u8)L;
                    n++;
                }
            }
            break;
        }
        if (carry > 4096) { skipped++; carry = 0; }        // 超长行直接丢弃
        else if (carry) memcpy(buf, buf + start, carry);
    }
    free(buf); fclose(fp);
    *pfx_out = pfx; *plen_out = plen; *skip_out = skipped;
    return n;
}

static u64 fnv1a(const u8* p, int L) {
    u64 h = 0xcbf29ce484222325ull;
    for (int i=0;i<L;i++) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

static void make_secret_key_b58(const u8* seed, const u8* pk, char* out) {
    u8 sk[64]; memcpy(sk, seed, 32); memcpy(sk+32, pk, 32);
    b58_enc(sk, 64, out);
}

static void print_json_array(FILE* f, const u8* seed, const u8* pk) {
    u8 sk[64]; memcpy(sk, seed, 32); memcpy(sk+32, pk, 32);
    fprintf(f, "[");
    for (int i=0;i<64;i++) fprintf(f, "%u%s", sk[i], i==63?"":",");
    fprintf(f, "]");
}

static void run_sol_search(u8* pfx, u8* plen, u64 nprefix, int blocks) {
    if (nprefix == 0) { fprintf(stderr, "no valid prefixes\n"); return; }

    // 预检
    u64 impossible = 0;
    for (u64 p=0;p<nprefix;p++) {
        int L = plen[p]; int bad = 0;
        for (int i=0;i<L;i++) if (!strchr(B58_ALPHA, pfx[p*PSTR+i])) { bad=1; break; }
        // 44 字符地址的首位至少是 '2'：以 '1' 开头的 44 字符编码会退化成 43 字符
        if (!bad && L == 44 && pfx[p*PSTR] == '1') bad = 1;
        if (bad) impossible++;
    }
    if (impossible)
        printf("WARNING: %llu prefix(es) contain characters outside the base58 alphabet and can never match\n",
               (unsigned long long)impossible);

    int lcnt[NLMAX+1]; for (int i=0;i<=NLMAX;i++) lcnt[i]=0;
    for (u64 p=0;p<nprefix;p++) if (plen[p]<=NLMAX) lcnt[plen[p]]++;
    printf("length dist: ");
    for (int L=1;L<=NLMAX;L++) if (lcnt[L]) printf("%d-char:%d ", L, lcnt[L]);
    printf("\n");

    // 每长度一条哈希表
    int hcap[NLMAX]; u32 off[NLMAX]; u32 mask[NLMAX];
    u64* hk[NLMAX]; u32* hp[NLMAX]; u8* hoc[NLMAX];
    u64 total_cap = 0;
    for (int L=1;L<=NLMAX;L++) {
        u64 cnt = (u64)lcnt[L];
        u64 c2 = 8; while (c2 < cnt*2+1) c2 <<= 1;
        hcap[L-1] = (int)c2; off[L-1] = (u32)total_cap; mask[L-1] = (u32)(c2-1);
        hk[L-1]  = (u64*)malloc((size_t)(c2*8));
        hp[L-1]  = (u32*)malloc((size_t)(c2*4));
        hoc[L-1] = (u8*)malloc((size_t)c2);
        if (!hk[L-1]||!hp[L-1]||!hoc[L-1]) { fprintf(stderr,"out of memory building table\n"); return; }
        memset(hoc[L-1], 0, (size_t)c2);
        total_cap += c2;
    }
    if (total_cap >= (1ull<<32)) { fprintf(stderr,"hash table too large (>4G slots)\n"); return; }

    u64 uniq = 0;
    for (u64 p=0;p<nprefix;p++) {
        int L = plen[p];
        if (L > NLMAX) continue;
        u64 key = fnv1a(pfx + p*PSTR, L);
        u32 hh = (u32)((key * 0x9E3779B97F4A7C15ull) >> 32);
        int s = (int)(hh & mask[L-1]);
        u8* oc = hoc[L-1]; u64* kk = hk[L-1]; u32* pp = hp[L-1];
        for (;;) {
            if (!oc[s]) {
                oc[s]=1; kk[s]=key; pp[s]=(u32)uniq;
                if (uniq < p) { memmove(pfx + uniq*PSTR, pfx + p*PSTR, PSTR); plen[uniq]=plen[p]; }
                uniq++; break;
            }
            if (kk[s]==key) {
                u32 q = pp[s]; int same=1;
                for (int i=0;i<L;i++) if (pfx[(u64)q*PSTR+i]!=pfx[p*PSTR+i]) { same=0; break; }
                if (same) break;
            }
            s = (s+1) & mask[L-1];
        }
    }
    printf("unique prefixes after dedup: %llu\n", (unsigned long long)uniq);
    if (uniq == 0) return;
    if (uniq >= (1ull<<31)) { fprintf(stderr,"too many unique prefixes (>=2^31)\n"); return; }

    u64* ck = (u64*)malloc((size_t)(total_cap*8));
    u32* cp = (u32*)malloc((size_t)(total_cap*4));
    u8*  co = (u8*)malloc((size_t)total_cap);
    if (!ck||!cp||!co) { fprintf(stderr,"out of memory\n"); return; }
    for (int L=1;L<=NLMAX;L++) {
        memcpy(ck + off[L-1], hk[L-1], (size_t)(hcap[L-1]*8));
        memcpy(cp + off[L-1], hp[L-1], (size_t)(hcap[L-1]*4));
        memcpy(co + off[L-1], hoc[L-1], (size_t)hcap[L-1]);
        free(hk[L-1]); free(hp[L-1]); free(hoc[L-1]);
    }

    u64* d_tbl_key; u32* d_tbl_p; u8* d_tbl_occ;
    u32* d_tbl_off; u32* d_tbl_mask; u32* d_tbl_cnt;
    CUDA_CHECK(cudaMalloc(&d_tbl_key, (size_t)(total_cap*8)));
    CUDA_CHECK(cudaMalloc(&d_tbl_p, (size_t)(total_cap*4)));
    CUDA_CHECK(cudaMalloc(&d_tbl_occ, (size_t)total_cap));
    CUDA_CHECK(cudaMalloc(&d_tbl_off, NLMAX*4));
    CUDA_CHECK(cudaMalloc(&d_tbl_mask, NLMAX*4));
    CUDA_CHECK(cudaMalloc(&d_tbl_cnt, NLMAX*4));
    CUDA_CHECK(cudaMemcpy(d_tbl_key, ck, (size_t)(total_cap*8), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl_p, cp, (size_t)(total_cap*4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl_occ, co, (size_t)total_cap, cudaMemcpyHostToDevice));

    u32 tcnt[NLMAX];
    for (int L=1;L<=NLMAX;L++) tcnt[L-1]=0;
    for (u64 pi=0;pi<uniq;pi++) tcnt[plen[pi]-1]++;
    CUDA_CHECK(cudaMemcpy(d_tbl_off, off, NLMAX*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl_mask, mask, NLMAX*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl_cnt, tcnt, NLMAX*4, cudaMemcpyHostToDevice));
    free(ck); free(cp); free(co);

    u8* d_pfx; CUDA_CHECK(cudaMalloc(&d_pfx, (size_t)(uniq*PSTR)));
    CUDA_CHECK(cudaMemcpy(d_pfx, pfx, (size_t)(uniq*PSTR), cudaMemcpyHostToDevice));

    u32* d_found; u32* d_nfound;
    ResqEntry* d_resq; u32* d_resq_cnt;
    CUDA_CHECK(cudaMalloc(&d_found, (size_t)(uniq*4)));
    CUDA_CHECK(cudaMalloc(&d_nfound, 4));
    CUDA_CHECK(cudaMalloc(&d_resq, (size_t)(RESQ_CAP*sizeof(ResqEntry))));
    CUDA_CHECK(cudaMalloc(&d_resq_cnt, 4));
    CUDA_CHECK(cudaMemset(d_found, 0, (size_t)(uniq*4)));
    CUDA_CHECK(cudaMemset(d_nfound, 0, 4));
    CUDA_CHECK(cudaMemset(d_resq_cnt, 0, 4));

    // v3: secret was loaded by main() into the SOL_SECRET symbol (.sol_vanity_secret); no per-run salt anymore

    TblEntry* d_tbl; CUDA_CHECK(cudaMalloc(&d_tbl, (size_t)(NW*TBLN*sizeof(TblEntry))));
    build_table_kernel<<<1,1>>>(d_tbl);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("derivation: BIP39 12-word mnemonic -> SLIP-0010 m/44'/501'/0'/0' (passphrase: empty)\n");
    printf("  (entropy = first 16B of SHA256(.sol_vanity_secret || counter); secret is never printed; counter starts at 0)\n");

    printf("searching BIP39-derived ed25519 keypairs: %d blocks x 256 threads\n", blocks);
    printf("each key costs 1 full scalar mult; runs forever until Ctrl+C (hits are appended to sol_bip39_hits.txt immediately)\n");

    FILE* hf = fopen("sol_bip39_hits.txt", "a");
    if (hf) setvbuf(hf, NULL, _IONBF, 0);
#ifndef _WIN32
    chmod("sol_bip39_hits.txt", 0600);  // hits contain mnemonic + secret key: tighten permissions
#endif

    u64 start = 0; int batch = 1; double total_s = 0; u32 nfound = 0;
    u64 qprev = 0, printed = 0;
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    char skb58[128];

    for (;;) {
        int launched = batch;
        CUDA_CHECK(cudaEventRecord(t0));
        sol_search_kernel<<<blocks, 256>>>(d_tbl_key, d_tbl_p, d_tbl_occ,
            d_tbl_off, d_tbl_mask, d_tbl_cnt, d_pfx, uniq, NLMAX,
            start, batch, d_tbl,
            d_found, d_nfound, d_resq, d_resq_cnt);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        CUDA_CHECK(cudaMemcpy(&nfound, d_nfound, 4, cudaMemcpyDeviceToHost));
        total_s += ms/1000.0;
        u64 done = (u64)launched*blocks*256;
        if (ms > 1000.0f) { int nb = batch/2; if (nb<1) nb=1; batch = nb; }
        else if (ms < 200.0f) { int nb = batch*2; if (nb>1<<20) nb=1<<20; batch = nb; }

        u32 qcnt = 0; CUDA_CHECK(cudaMemcpy(&qcnt, d_resq_cnt, 4, cudaMemcpyDeviceToHost));
        while (qprev < (u64)qcnt) {
            ResqEntry e;
            CUDA_CHECK(cudaMemcpy(&e, d_resq + (qprev & RESQ_MASK), sizeof(ResqEntry), cudaMemcpyDeviceToHost));
            qprev++;
            char pf[PSTR+1]; int L = plen[e.p]; memcpy(pf, pfx + (u64)e.p*PSTR, L); pf[L]=0;
            make_secret_key_b58(e.seed, e.pk, skb58);
            printf("\nHIT [%5u] prefix \"%s\"\n", e.p, pf);
            printf("      Address:                 %s\n", e.addr);
            printf("      Mnemonic (BIP39 12 words):  %s\n", e.mnem);
            printf("      Path: m/44'/501'/0'/0' (passphrase: empty)\n");
            printf("      Secret key (base58 64B): %s\n", skb58);
            printf("      Seed (base58 32B):       ");
            { char sd[64]; b58_enc(e.seed, 32, sd); printf("%s\n", sd); }
            printf("      Keypair JSON:            ");
            print_json_array(stdout, e.seed, e.pk); printf("\n");
            fflush(stdout);
            if (hf) {
                fprintf(hf, "HIT [%5u] prefix \"%s\"\n", e.p, pf);
                fprintf(hf, "      Address:                 %s\n", e.addr);
                fprintf(hf, "      Mnemonic (BIP39 12 words):  %s\n", e.mnem);
                fprintf(hf, "      Path: m/44'/501'/0'/0' (passphrase: empty)\n");
                fprintf(hf, "      Secret key (base58 64B): %s\n", skb58);
                fprintf(hf, "      Seed (base58 32B):       ");
                { char sd[64]; b58_enc(e.seed, 32, sd); fprintf(hf, "%s\n", sd); }
                fprintf(hf, "      Keypair JSON:            ");
                print_json_array(hf, e.seed, e.pk); fprintf(hf, "\n\n");
            }
            printed++;
        }
        u64 total_keys = start + done;                       // 累计已算过的 key 数（含本批）
        double as = (double)total_keys / (total_s>0?total_s:1.0);
        double ks = as / 1000.0;
        printf("\r  [%7.1f s] %10.0f addr/s (%.0f k/s) | hit %llu | covered %u/%llu  (Ctrl+C stop)",
               total_s, as, ks, (unsigned long long)printed, nfound, (unsigned long long)uniq);
        fflush(stdout);
        start += done;

        // 所有前缀都命中过之后不清表、不停机：清空命中标记后继续无限跑，
        // 这样单个前缀（或小文件）也不会出现“命中一次就结束”的情况。
        if (nfound >= (u64)uniq) {
            CUDA_CHECK(cudaMemset(d_found, 0, (size_t)(uniq*4)));
            CUDA_CHECK(cudaMemset(d_nfound, 0, 4));
            nfound = 0;
        }
    }
    printf("\nall %llu unique prefixes found in %.1f s, %llu hits\n", (unsigned long long)uniq, total_s, (unsigned long long)printed);
    if (hf) fclose(hf);
    CUDA_CHECK(cudaFree(d_tbl_key)); CUDA_CHECK(cudaFree(d_tbl_p)); CUDA_CHECK(cudaFree(d_tbl_occ));
    CUDA_CHECK(cudaFree(d_tbl_off)); CUDA_CHECK(cudaFree(d_tbl_mask)); CUDA_CHECK(cudaFree(d_tbl_cnt));
    CUDA_CHECK(cudaFree(d_pfx)); CUDA_CHECK(cudaFree(d_found)); CUDA_CHECK(cudaFree(d_nfound));
    CUDA_CHECK(cudaFree(d_resq)); CUDA_CHECK(cudaFree(d_resq_cnt));
    CUDA_CHECK(cudaFree(d_tbl));
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
}

// v3: 加载或创建 32B secret 文件（默认 ./.sol_vanity_secret，可用环境变量 SOL_VANITY_SECRET 指定路径）。
// secret 与计数器共同决定助记词序列；secret 绝不打印、不写入日志、不进 hits 文件。
// 注意：丢了 secret 不影响用 sol_bip39_hits.txt 里记录的助记词找回已命中的单个钱包，但无法再复现整条搜索序列。
// 首次创建后请离线备份一份，并保持文件权限 0600。
static int load_or_create_sol_secret(u8 sec[32]) {
    const char* path = getenv("SOL_VANITY_SECRET");
    if (!path || !path[0]) path = ".sol_vanity_secret";
    FILE* f = fopen(path, "rb");
    if (f) {
        size_t r = fread(sec, 1, 32, f); fclose(f);
        if (r == 32) {
#ifndef _WIN32
            chmod(path, 0600);
#endif
            printf("secret: loaded from %s (keep it private, back it up offline)\n", path); return 1; }
        fprintf(stderr, "secret file %s exists but is not 32 bytes; refusing to overwrite it\n", path);
        return 0;
    }
    FILE* u = fopen("/dev/urandom", "rb");
    if (u) {
        size_t r = fread(sec, 1, 32, u); fclose(u);
        if (r != 32) { fprintf(stderr, "short read from /dev/urandom\n"); return 0; }
    } else {
        std::random_device rd;  // 无 /dev/urandom 的平台（如原生 Windows）走系统 CSPRNG
        for (int i=0;i<32;i++) sec[i]=(u8)(rd()&0xFF);
    }
    f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot create secret file %s\n", path); return 0; }
    fwrite(sec, 1, 32, f); fclose(f);
#ifndef _WIN32
    chmod(path, 0600);
#endif
    printf("secret: created %s (0600). Back it up offline; never share it.\n", path);
    return 1;
}
// ============================================================================
static int chain_main(int argc, char** argv) {
    const char* mode = argc>1 ? argv[1] : "bench";
    printf("=== Solana Vanity Generator v3 (BIP39 mnemonic -> SLIP-0010 m/44'/501'/0'/0' -> base58 address) ===\n");
    u8 secret[32];
    if (!load_or_create_sol_secret(secret)) return 1;
    CUDA_CHECK(cudaMemcpyToSymbol(SOL_SECRET, secret, 32));

    if (strcmp(mode,"selftest")==0) {
        TestCase* hc = (TestCase*)calloc(11, sizeof(TestCase));
        const char* rfc[3] = {
            "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
            "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
            "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7" };
        const char* exp_pub[11] = {
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
            "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
            "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
            "fa6fffb4b5eb1d429c0ffcb1a9b7f8594ad7eef7a7c993d7785e4d082070f417",
            "92b25a878a0963dc1152611d4cbe0ee096bd8c67f512b010694d248c565a4bb8",
            "426406b8f41f1513ffd37d4cbd3d646f280f9e7c77f02191eec028e0133087f5",
            "0050ba1c0b5f42fb067644af04c65d1f8db80eaf3a50722fc9ec5f6fa4d8c94a",
            "000f45a82245cd56235f6283466dab50d60eee29ab6d18107254f72549c9d56a",
            "f036276246a75b9de3349ed42b15e232f6518fc20f5fcd4f1d64e81f9bd258f7",
            "c1f54914799306105c9c61e564737fd5553c4bb040e4a32c8d434b1cc3a2d2cb",
            "" };
        const char* exp_addr[11] = {
            "FVen3X669xLzsi6N2V91DoiyzHzg1uAgqiT8jZ9nS96Z",
            "586Z7H2vpX9qNhN2T4e9Utugie3ogjbxzGaMtM3E6HR5",
            "Hyx62wPQGyvXCoihZq1BrbUjBRh2LuNxWiiqMkfAuSZr",
            "HrbyevrPyc5P7AAjpn6xJLfUJVV3mZ7KBUZKSwGGpF34",
            "AseG86D8WSo6VRHsxJ7TqvQFojB1E1JEfzwJPMVtUiA3",
            "5UANZgPx5jeL4FtpMQDu7uL2hc21yQR9qiBVRQM2sLPn",
            "12EPvzPxQ27Gdc8DGZihWWZjYsh4sFxvEpz1wLtsp2Y9",
            "1EWPc7d7HwH25LWP6xTYTzbGvf6fMUDLKbhgrzfsC2M",
            "HAgk14JpMQLgt6rVgv7cBQFJWFto5Dqxi472uT3DKpqk",
            "E48cosDiQZK1iDSsyUzhvW4WxJeoKuDk5qgcdkmANV4N",
            "" };
        for (int i=0;i<3;i++) {
            hc[i].mode = 0;
            for (int j=0;j<32;j++) { int a,b; char c1=rfc[i][2*j], c2=rfc[i][2*j+1];
                a=(c1<='9')?c1-'0':c1-'a'+10; b=(c2<='9')?c2-'0':c2-'a'+10;
                hc[i].seed[j]=(u8)((a<<4)|b); }
        }
        for (int i=0;i<4;i++) { hc[3+i].mode = 1; for (int j=0;j<32;j++) hc[3+i].salt[j]=(u8)j; hc[3+i].ctr=(u64)i; }
        hc[7].mode = 1; for (int j=0;j<32;j++) hc[7].salt[j]=(u8)(32+j); hc[7].ctr = 274;
        hc[8].mode = 2; for (int j=0;j<16;j++) hc[8].ent[j]=0x00;   // entropy 0x00*16 -> abandon x11 + about
        hc[9].mode = 2; for (int j=0;j<16;j++) hc[9].ent[j]=0xFF;   // entropy 0xFF*16 -> zoo x11 + wrong

        TestCase* d_c; CUDA_CHECK(cudaMalloc(&d_c, 11*sizeof(TestCase)));
        CUDA_CHECK(cudaMemcpy(d_c, hc, 11*sizeof(TestCase), cudaMemcpyHostToDevice));
        u8* d_pub; u8* d_addr; int* d_dbl; u32* d_inv;
        CUDA_CHECK(cudaMalloc(&d_pub, 11*32));
        CUDA_CHECK(cudaMalloc(&d_addr, 11*PSTR));
        CUDA_CHECK(cudaMalloc(&d_dbl, 4));
        CUDA_CHECK(cudaMalloc(&d_inv, 4));
        TblEntry* d_tbl; CUDA_CHECK(cudaMalloc(&d_tbl, (size_t)(NW*TBLN*sizeof(TblEntry))));
        build_table_kernel<<<1,1>>>(d_tbl);
        CUDA_CHECK(cudaDeviceSynchronize());
        selftest_kernel<<<1,1>>>(d_c, 10, d_pub, d_addr, d_dbl, d_inv, d_tbl);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        u8 pub[11*32]; u8 addr[11*PSTR]; int dbl=0; u32 inv=0;
        CUDA_CHECK(cudaMemcpy(pub, d_pub, 11*32, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(addr, d_addr, 11*PSTR, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&dbl, d_dbl, 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&inv, d_inv, 4, cudaMemcpyDeviceToHost));

        int pass=0, fail=0;
        printf("case  pubkey                                                            address\n");
        for (int i=0;i<10;i++) {
            char ph[65]; for (int j=0;j<32;j++) sprintf(ph+2*j, "%02x", pub[i*32+j]);
            char ad[PSTR+1]; memcpy(ad, addr+i*PSTR, PSTR); ad[PSTR]=0;
            int okp = (strcmp(ph, exp_pub[i])==0);
            int oka = (strcmp(ad, exp_addr[i])==0);
            printf("[%d] %s %-50s %-44s\n", i, (okp&&oka)?"PASS":"FAIL", ph, ad);
            if (okp&&oka) pass++; else { fail++; if(!okp) printf("    expect pub %s\n", exp_pub[i]); if(!oka) printf("    expect addr %s\n", exp_addr[i]); }
        }
        printf("2B via dbl == via add : %s\n", dbl?"PASS":"FAIL"); if(dbl)pass++; else fail++;
        printf("(x^-1)*x == 1         : %s\n", inv?"PASS":"FAIL"); if(inv)pass++; else fail++;
        printf("\nselftest: %d pass, %d fail\n", pass, fail);
        FILE* f = fopen("sol_selftest.txt","w");
        if (f) { fprintf(f,"pass=%d fail=%d dbl=%d inv=%u\n",pass,fail,dbl,inv); fclose(f); }
    }
    else if (strcmp(mode,"bench")==0) {
        int iters = (argc>=3) ? atoi(argv[2]) : 200;
        int blocks = (argc>=4) ? atoi(argv[3]) : 512;
        u64* d_tbl_key; u32* d_tbl_p; u8* d_tbl_occ; u32* d_tbl_off; u32* d_tbl_mask; u32* d_tbl_cnt;
        CUDA_CHECK(cudaMalloc(&d_tbl_key, 64)); CUDA_CHECK(cudaMalloc(&d_tbl_p, 32));
        CUDA_CHECK(cudaMalloc(&d_tbl_occ, 8)); CUDA_CHECK(cudaMalloc(&d_tbl_off, NLMAX*4));
        CUDA_CHECK(cudaMalloc(&d_tbl_mask, NLMAX*4)); CUDA_CHECK(cudaMalloc(&d_tbl_cnt, NLMAX*4));
        CUDA_CHECK(cudaMemset(d_tbl_occ, 0, 8));
        u32 zeros[NLMAX]; memset(zeros, 0, sizeof(zeros));
        CUDA_CHECK(cudaMemset(d_tbl_cnt, 0, NLMAX*4)); CUDA_CHECK(cudaMemset(d_tbl_off, 0, NLMAX*4)); CUDA_CHECK(cudaMemset(d_tbl_mask, 0, NLMAX*4));
        (void)zeros;
        u8* d_pfx; CUDA_CHECK(cudaMalloc(&d_pfx, 1));
        u32* d_found; u32* d_nfound; ResqEntry* d_resq; u32* d_resq_cnt;
        CUDA_CHECK(cudaMalloc(&d_found, 4)); CUDA_CHECK(cudaMalloc(&d_nfound, 4));
        CUDA_CHECK(cudaMalloc(&d_resq, (size_t)(RESQ_CAP*sizeof(ResqEntry)))); CUDA_CHECK(cudaMalloc(&d_resq_cnt, 4));
        CUDA_CHECK(cudaMemset(d_found, 0, 4)); CUDA_CHECK(cudaMemset(d_nfound, 0, 4)); CUDA_CHECK(cudaMemset(d_resq_cnt, 0, 4));
        { u8 fixed_sec[32]; for (int i=0;i<32;i++) fixed_sec[i]=7; CUDA_CHECK(cudaMemcpyToSymbol(SOL_SECRET, fixed_sec, 32)); }  // bench: fixed secret for reproducible numbers
        TblEntry* d_tbl; CUDA_CHECK(cudaMalloc(&d_tbl, (size_t)(NW*TBLN*sizeof(TblEntry))));
        build_table_kernel<<<1,1>>>(d_tbl); CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t t0,t1; CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        sol_search_kernel<<<blocks,256>>>(d_tbl_key, d_tbl_p, d_tbl_occ, d_tbl_off, d_tbl_mask, d_tbl_cnt,
            d_pfx, 0, NLMAX, 0, iters, d_tbl, d_found, d_nfound, d_resq, d_resq_cnt);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        u64 total = (u64)iters*blocks*256;
        printf("bench: %llu keys in %.2f s = %.0f keys/s (%.1f k/s)\n",
               (unsigned long long)total, ms/1000.0, total/(ms/1000.0), total/(ms/1000.0)/1000.0);
        CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    }
    else if (strcmp(mode,"file")==0) {
        if (argc<3) { fprintf(stderr,"usage: file <prefix_file> [blocks]\n"); return 1; }
        int blocks = (argc>=4) ? atoi(argv[3]) : 512;
        if (blocks < 1) blocks = 512;
        u64 nskip = 0; u8* pfx = NULL; u8* plen = NULL;
        u64 n = load_prefix_file(argv[2], &pfx, &plen, &nskip);
        if (n == 0) { fprintf(stderr,"no valid prefixes in file\n"); return 1; }
        printf("file mode: read %llu prefixes (skipped %llu with len>%d)\n",
               (unsigned long long)n, (unsigned long long)nskip, NLMAX);
        run_sol_search(pfx, plen, n, blocks);
        free(pfx); free(plen);
    }
    else if (strcmp(mode,"vanity")==0) {
        if (argc<3) { fprintf(stderr,"usage: vanity <prefix> [blocks]\n"); return 1; }
        int blocks = (argc>=4) ? atoi(argv[3]) : 512;
        int L = (int)strlen(argv[2]);
        if (L == 0 || L > NLMAX) { fprintf(stderr,"prefix must be 1..%d base58 chars\n", NLMAX); return 1; }
        for (int i=0;i<L;i++) if (!strchr(B58_ALPHA, argv[2][i])) { fprintf(stderr,"char '%c' not in base58 alphabet\n", argv[2][i]); return 1; }
        u8* pfx = (u8*)calloc(PSTR,1); u8* plen = (u8*)calloc(1,1);
        memcpy(pfx, argv[2], L); plen[0]=(u8)L;
        printf("vanity prefix: \"%s\" (%d chars)\n", argv[2], L);
        run_sol_search(pfx, plen, 1, blocks);
        free(pfx); free(plen);
    }
    else {
        fprintf(stderr,"usage:\n  vanity_multi.exe solbip file <prefix_file> [blocks]\n"
                       "  vanity_multi.exe solbip vanity <prefix> [blocks]\n"
                       "  vanity_multi.exe solbip bench [iters] [blocks]\n"
                       "  vanity_multi.exe solbip selftest\n");
        return 1;
    }
    return 0;
}

} // namespace solbip
#undef CUDA_CHECK
#undef PSTR
#undef NLMAX
#undef NW
#undef TBLN
#undef RESQ_CAP
#undef RESQ_MASK

// =============================================================================
//  统一入口：直接分发 + 交互菜单
// =============================================================================
static const char* HELP_LINES[] = {
    "\x76\x61\x6E\x69\x74\x79\x5F\x6D\x75\x6C\x74\x69\x20\x2D\x20\x6D\x75\x6C\x74\x69\x2D\x63\x68\x61\x69\x6E\x20\x42\x49\x50\x33\x39\x20\x76\x61\x6E\x69\x74\x79\x20\x61\x64\x64\x72\x65\x73\x73\x20\x67\x65\x6E\x65\x72\x61\x74\x6F\x72\x20\x28\x73\x69\x6E\x67\x6C\x65\x20\x62\x69\x6E\x61\x72\x79\x29\x0A",
    "\x0A",
    "\x55\x53\x41\x47\x45\x0A",
    "\x20\x20\x76\x61\x6E\x69\x74\x79\x5F\x6D\x75\x6C\x74\x69\x2E\x65\x78\x65\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x6E\x6F\x20\x61\x72\x67\x73\x20\x2D\x3E\x20\x69\x6E\x74\x65\x72\x61\x63\x74\x69\x76\x65\x20\x6D\x65\x6E\x75\x0A",
    "\x20\x20\x76\x61\x6E\x69\x74\x79\x5F\x6D\x75\x6C\x74\x69\x2E\x65\x78\x65\x20\x2D\x68\x20\x7C\x20\x2D\x2D\x68\x65\x6C\x70\x20\x7C\x20\x68\x65\x6C\x70\x20\x20\x20\x20\x20\x20\x20\x73\x68\x6F\x77\x20\x74\x68\x69\x73\x20\x68\x65\x6C\x70\x0A",
    "\x20\x20\x76\x61\x6E\x69\x74\x79\x5F\x6D\x75\x6C\x74\x69\x2E\x65\x78\x65\x20\x3C\x63\x68\x61\x69\x6E\x3E\x20\x3C\x6D\x6F\x64\x65\x3E\x20\x5B\x61\x72\x67\x73\x2E\x2E\x2E\x5D\x20\x72\x75\x6E\x20\x6F\x6E\x65\x20\x63\x68\x61\x69\x6E\x20\x64\x69\x72\x65\x63\x74\x6C\x79\x0A",
    "\x0A",
    "\x20\x20\x3C\x63\x68\x61\x69\x6E\x3E\x0A",
    "\x20\x20\x20\x20\x65\x74\x68\x20\x20\x20\x20\x20\x45\x74\x68\x65\x72\x65\x75\x6D\x20\x20\x20\x42\x49\x50\x33\x39\x20\x2D\x3E\x20\x6D\x2F\x34\x34\x27\x2F\x36\x30\x27\x2F\x30\x27\x2F\x30\x2F\x4E\x20\x20\x2D\x3E\x20\x73\x65\x63\x70\x32\x35\x36\x6B\x31\x20\x2D\x3E\x20\x6B\x65\x63\x63\x61\x6B\x32\x35\x36\x20\x2D\x3E\x20\x30\x78\x20\x61\x64\x64\x72\x65\x73\x73\x0A",
    "\x20\x20\x20\x20\x62\x74\x63\x20\x20\x20\x20\x20\x42\x69\x74\x63\x6F\x69\x6E\x20\x20\x20\x20\x42\x49\x50\x33\x39\x20\x2D\x3E\x20\x6D\x2F\x34\x34\x27\x2F\x30\x27\x2F\x30\x27\x2F\x30\x2F\x4E\x20\x20\x20\x2D\x3E\x20\x73\x65\x63\x70\x32\x35\x36\x6B\x31\x20\x2D\x3E\x20\x42\x61\x73\x65\x35\x38\x20\x2F\x20\x42\x65\x63\x68\x33\x32\x20\x61\x64\x64\x72\x65\x73\x73\x0A",
    "\x20\x20\x20\x20\x74\x72\x78\x20\x20\x20\x20\x20\x54\x52\x4F\x4E\x20\x20\x20\x20\x20\x20\x20\x42\x49\x50\x33\x39\x20\x2D\x3E\x20\x6D\x2F\x34\x34\x27\x2F\x31\x39\x35\x27\x2F\x30\x27\x2F\x30\x2F\x4E\x20\x2D\x3E\x20\x73\x65\x63\x70\x32\x35\x36\x6B\x31\x20\x2D\x3E\x20\x6B\x65\x63\x63\x61\x6B\x32\x35\x36\x20\x2D\x3E\x20\x54\x20\x61\x64\x64\x72\x65\x73\x73\x0A",
    "\x20\x20\x20\x20\x73\x6F\x6C\x20\x20\x20\x20\x20\x53\x6F\x6C\x61\x6E\x61\x20\x66\x61\x73\x74\x20\x20\x20\x20\x20\x20\x72\x61\x6E\x64\x6F\x6D\x20\x33\x32\x42\x20\x73\x65\x65\x64\x20\x2D\x3E\x20\x65\x64\x32\x35\x35\x31\x39\x20\x2D\x3E\x20\x62\x61\x73\x65\x35\x38\x20\x28\x6E\x6F\x20\x50\x42\x4B\x44\x46\x32\x2C\x20\x66\x61\x73\x74\x65\x73\x74\x29\x0A",
    "\x20\x20\x20\x20\x73\x6F\x6C\x62\x69\x70\x20\x20\x53\x6F\x6C\x61\x6E\x61\x20\x6D\x6E\x65\x6D\x6F\x6E\x69\x63\x20\x20\x42\x49\x50\x33\x39\x20\x2D\x3E\x20\x50\x42\x4B\x44\x46\x32\x28\x32\x30\x34\x38\x29\x20\x2D\x3E\x20\x53\x4C\x49\x50\x2D\x30\x30\x31\x30\x20\x6D\x2F\x34\x34\x27\x2F\x35\x30\x31\x27\x2F\x30\x27\x2F\x30\x27\x0A",
    "\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x2D\x3E\x20\x65\x64\x32\x35\x35\x31\x39\x20\x2D\x3E\x20\x62\x61\x73\x65\x35\x38\x20\x28\x69\x6D\x70\x6F\x72\x74\x61\x62\x6C\x65\x20\x69\x6E\x74\x6F\x20\x50\x68\x61\x6E\x74\x6F\x6D\x20\x2F\x20\x53\x6F\x6C\x66\x6C\x61\x72\x65\x29\x0A",
    "\x0A",
    "\x4D\x4F\x44\x45\x53\x0A",
    "\x20\x20\x65\x74\x68\x20\x7C\x20\x62\x74\x63\x20\x7C\x20\x74\x72\x78\x0A",
    "\x20\x20\x20\x20\x76\x61\x6E\x69\x74\x79\x20\x3C\x70\x72\x65\x66\x69\x78\x3E\x20\x5B\x6E\x69\x64\x78\x5D\x20\x5B\x62\x6C\x6F\x63\x6B\x73\x5D\x20\x20\x20\x20\x20\x20\x20\x20\x20\x73\x65\x61\x72\x63\x68\x20\x66\x6F\x72\x65\x76\x65\x72\x2C\x20\x73\x74\x6F\x70\x20\x77\x69\x74\x68\x20\x43\x74\x72\x6C\x2B\x43\x0A",
    "\x20\x20\x20\x20\x66\x69\x6C\x65\x20\x20\x20\x3C\x70\x72\x65\x66\x69\x78\x5F\x66\x69\x6C\x65\x3E\x20\x5B\x6E\x69\x64\x78\x5D\x20\x5B\x62\x6C\x6F\x63\x6B\x73\x5D\x20\x20\x20\x20\x6F\x6E\x65\x20\x70\x72\x65\x66\x69\x78\x20\x70\x65\x72\x20\x6C\x69\x6E\x65\x2C\x20\x75\x6E\x6C\x69\x6D\x69\x74\x65\x64\x20\x73\x69\x7A\x65\x20\x28\x32\x47\x42\x2B\x20\x73\x74\x72\x65\x61\x6D\x65\x64\x29\x0A",
    "\x20\x20\x20\x20\x73\x65\x6C\x66\x74\x65\x73\x74\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x62\x75\x69\x6C\x74\x2D\x69\x6E\x20\x76\x65\x63\x74\x6F\x72\x20\x73\x65\x6C\x66\x2D\x74\x65\x73\x74\x0A",
    "\x20\x20\x20\x20\x62\x65\x6E\x63\x68\x20\x20\x5B\x69\x74\x65\x72\x73\x5D\x20\x5B\x62\x6C\x6F\x63\x6B\x73\x5D\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x74\x68\x72\x6F\x75\x67\x68\x70\x75\x74\x20\x62\x65\x6E\x63\x68\x6D\x61\x72\x6B\x0A",
    "\x0A",
    "\x20\x20\x62\x74\x63\x20\x74\x61\x6B\x65\x73\x20\x6F\x6E\x65\x20\x65\x78\x74\x72\x61\x20\x4E\x55\x4D\x45\x52\x49\x43\x20\x61\x64\x64\x72\x74\x79\x70\x65\x20\x61\x72\x67\x75\x6D\x65\x6E\x74\x2C\x20\x72\x69\x67\x68\x74\x20\x61\x66\x74\x65\x72\x20\x70\x72\x65\x66\x69\x78\x20\x2F\x20\x70\x72\x65\x66\x69\x78\x20\x66\x69\x6C\x65\x20\x28\x64\x65\x66\x61\x75\x6C\x74\x20\x33\x29\x0A",
    "\x20\x20\x20\x20\x62\x74\x63\x20\x76\x61\x6E\x69\x74\x79\x20\x3C\x70\x72\x65\x66\x69\x78\x3E\x20\x20\x20\x20\x20\x20\x5B\x61\x64\x64\x72\x74\x79\x70\x65\x5D\x20\x5B\x6E\x69\x64\x78\x5D\x20\x5B\x62\x6C\x6F\x63\x6B\x73\x5D\x0A",
    "\x20\x20\x20\x20\x62\x74\x63\x20\x66\x69\x6C\x65\x20\x20\x20\x3C\x70\x72\x65\x66\x69\x78\x5F\x66\x69\x6C\x65\x3E\x20\x5B\x61\x64\x64\x72\x74\x79\x70\x65\x5D\x20\x5B\x6E\x69\x64\x78\x5D\x20\x5B\x62\x6C\x6F\x63\x6B\x73\x5D\x0A",
    "\x20\x20\x20\x20\x61\x64\x64\x72\x74\x79\x70\x65\x3A\x20\x30\x20\x3D\x20\x50\x32\x50\x4B\x48\x28\x31\x2E\x2E\x2E\x29\x20\x20\x20\x31\x20\x3D\x20\x50\x32\x53\x48\x2D\x50\x32\x57\x50\x4B\x48\x28\x33\x2E\x2E\x2E\x29\x20\x20\x20\x32\x20\x3D\x20\x50\x32\x57\x50\x4B\x48\x28\x62\x63\x31\x71\x2E\x2E\x2E\x29\x0A",
    "\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x33\x20\x3D\x20\x50\x32\x54\x52\x28\x62\x63\x31\x70\x2E\x2E\x2E\x29\x20\x20\x28\x64\x65\x66\x61\x75\x6C\x74\x29\x0A",
    "\x20\x20\x20\x20\x20\x20\x70\x72\x65\x66\x69\x78\x20\x73\x74\x61\x72\x74\x73\x20\x77\x69\x74\x68\x20\x31\x20\x2D\x3E\x20\x30\x2C\x20\x77\x69\x74\x68\x20\x33\x20\x2D\x3E\x20\x31\x2C\x20\x62\x63\x31\x71\x20\x2D\x3E\x20\x32\x2C\x20\x62\x63\x31\x70\x20\x2D\x3E\x20\x33\x0A",
    "\x20\x20\x20\x20\x20\x20\x61\x20\x6D\x69\x73\x6D\x61\x74\x63\x68\x65\x64\x20\x74\x79\x70\x65\x20\x73\x69\x6D\x70\x6C\x79\x20\x6E\x65\x76\x65\x72\x20\x6D\x61\x74\x63\x68\x65\x73\x0A",
    "\x20\x20\x20\x20\x62\x74\x63\x20\x62\x65\x6E\x63\x68\x20\x61\x6C\x73\x6F\x20\x74\x61\x6B\x65\x73\x20\x61\x64\x64\x72\x74\x79\x70\x65\x20\x66\x69\x72\x73\x74\x3A\x0A",
    "\x20\x20\x20\x20\x62\x74\x63\x20\x62\x65\x6E\x63\x68\x20\x20\x3C\x61\x64\x64\x72\x74\x79\x70\x65\x3E\x20\x5B\x69\x74\x65\x72\x73\x5D\x20\x5B\x62\x6C\x6F\x63\x6B\x73\x5D\x0A",
    "\x0A",
    "\x20\x20\x73\x6F\x6C\x20\x7C\x20\x73\x6F\x6C\x62\x69\x70\x0A",
    "\x20\x20\x20\x20\x76\x61\x6E\x69\x74\x79\x20\x3C\x70\x72\x65\x66\x69\x78\x3E\x20\x5B\x62\x6C\x6F\x63\x6B\x73\x5D\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x73\x65\x61\x72\x63\x68\x20\x66\x6F\x72\x65\x76\x65\x72\x2C\x20\x73\x74\x6F\x70\x20\x77\x69\x74\x68\x20\x43\x74\x72\x6C\x2B\x43\x0A",
    "\x20\x20\x20\x20\x66\x69\x6C\x65\x20\x20\x20\x3C\x70\x72\x65\x66\x69\x78\x5F\x66\x69\x6C\x65\x3E\x20\x5B\x62\x6C\x6F\x63\x6B\x73\x5D\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x70\x72\x65\x66\x69\x78\x20\x66\x69\x6C\x65\x20\x28\x32\x47\x42\x2B\x20\x73\x74\x72\x65\x61\x6D\x65\x64\x29\x0A",
    "\x20\x20\x20\x20\x62\x65\x6E\x63\x68\x20\x20\x5B\x69\x74\x65\x72\x73\x5D\x20\x5B\x62\x6C\x6F\x63\x6B\x73\x5D\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x74\x68\x72\x6F\x75\x67\x68\x70\x75\x74\x20\x62\x65\x6E\x63\x68\x6D\x61\x72\x6B\x20\x28\x73\x6F\x6C\x62\x69\x70\x20\x70\x61\x79\x73\x20\x50\x42\x4B\x44\x46\x32\x29\x0A",
    "\x20\x20\x20\x20\x73\x65\x6C\x66\x74\x65\x73\x74\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x62\x75\x69\x6C\x74\x2D\x69\x6E\x20\x76\x65\x63\x74\x6F\x72\x20\x73\x65\x6C\x66\x2D\x74\x65\x73\x74\x0A",
    "\x0A",
    "\x41\x52\x47\x55\x4D\x45\x4E\x54\x53\x0A",
    "\x20\x20\x70\x72\x65\x66\x69\x78\x20\x20\x20\x20\x61\x64\x64\x72\x65\x73\x73\x20\x70\x72\x65\x66\x69\x78\x20\x74\x6F\x20\x6D\x61\x74\x63\x68\x20\x28\x63\x61\x73\x65\x20\x73\x65\x6E\x73\x69\x74\x69\x76\x65\x29\x2E\x20\x45\x54\x48\x20\x6D\x61\x79\x20\x69\x6E\x63\x6C\x75\x64\x65\x20\x30\x78\x3B\x20\x42\x54\x43\x20\x73\x74\x61\x72\x74\x73\x20\x77\x69\x74\x68\x0A",
    "\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x31\x20\x2F\x20\x33\x20\x2F\x20\x62\x63\x31\x71\x20\x2F\x20\x62\x63\x31\x70\x3B\x20\x54\x52\x58\x20\x6D\x75\x73\x74\x20\x73\x74\x61\x72\x74\x20\x77\x69\x74\x68\x20\x54\x3B\x20\x53\x4F\x4C\x20\x75\x73\x65\x73\x20\x62\x61\x73\x65\x35\x38\x20\x6F\x6E\x6C\x79\x2C\x20\x6D\x61\x78\x20\x34\x34\x20\x63\x68\x61\x72\x73\x2E\x0A",
    "\x20\x20\x6E\x69\x64\x78\x20\x20\x20\x20\x20\x20\x61\x64\x64\x72\x65\x73\x73\x65\x73\x20\x64\x65\x72\x69\x76\x65\x64\x20\x70\x65\x72\x20\x6D\x6E\x65\x6D\x6F\x6E\x69\x63\x20\x28\x48\x44\x20\x69\x6E\x64\x65\x78\x20\x30\x2E\x2E\x4E\x2D\x31\x29\x2C\x20\x64\x65\x66\x61\x75\x6C\x74\x20\x31\x2E\x0A",
    "\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x6C\x61\x72\x67\x65\x72\x20\x2D\x3E\x20\x6D\x6F\x72\x65\x20\x68\x69\x74\x73\x20\x70\x65\x72\x20\x62\x61\x74\x63\x68\x20\x62\x75\x74\x20\x73\x6C\x6F\x77\x65\x72\x20\x70\x65\x72\x20\x61\x64\x64\x72\x65\x73\x73\x0A",
    "\x20\x20\x62\x6C\x6F\x63\x6B\x73\x20\x20\x20\x20\x43\x55\x44\x41\x20\x67\x72\x69\x64\x20\x62\x6C\x6F\x63\x6B\x73\x20\x28\x32\x35\x36\x20\x74\x68\x72\x65\x61\x64\x73\x20\x65\x61\x63\x68\x29\x2C\x20\x64\x65\x66\x61\x75\x6C\x74\x20\x31\x30\x32\x34\x20\x28\x73\x6F\x6C\x20\x2F\x20\x73\x6F\x6C\x62\x69\x70\x3A\x20\x35\x31\x32\x29\x0A",
    "\x20\x20\x69\x74\x65\x72\x73\x20\x20\x20\x20\x20\x62\x65\x6E\x63\x68\x20\x69\x74\x65\x72\x61\x74\x69\x6F\x6E\x73\x20\x28\x65\x61\x63\x68\x20\x69\x74\x65\x72\x61\x74\x69\x6F\x6E\x20\x3D\x20\x62\x6C\x6F\x63\x6B\x73\x2A\x32\x35\x36\x20\x63\x61\x6E\x64\x69\x64\x61\x74\x65\x73\x29\x0A",
    "\x0A",
    "\x48\x49\x54\x53\x20\x41\x4E\x44\x20\x4F\x55\x54\x50\x55\x54\x0A",
    "\x20\x20\x2A\x20\x65\x76\x65\x72\x79\x20\x68\x69\x74\x20\x69\x73\x20\x61\x70\x70\x65\x6E\x64\x65\x64\x20\x69\x6D\x6D\x65\x64\x69\x61\x74\x65\x6C\x79\x20\x74\x6F\x20\x3C\x63\x68\x61\x69\x6E\x3E\x5F\x68\x69\x74\x73\x2E\x74\x78\x74\x20\x28\x75\x6E\x62\x75\x66\x66\x65\x72\x65\x64\x29\x2C\x20\x43\x74\x72\x6C\x2B\x43\x20\x6B\x65\x65\x70\x73\x20\x74\x68\x65\x6D\x0A",
    "\x20\x20\x2A\x20\x74\x68\x65\x20\x73\x65\x61\x72\x63\x68\x20\x6E\x65\x76\x65\x72\x20\x73\x74\x6F\x70\x73\x20\x6F\x6E\x20\x61\x20\x68\x69\x74\x3B\x20\x6F\x6E\x6C\x79\x20\x43\x74\x72\x6C\x2B\x43\x20\x65\x6E\x64\x73\x20\x69\x74\x2E\x20\x46\x6F\x75\x6E\x64\x20\x66\x6C\x61\x67\x73\x20\x61\x72\x65\x20\x72\x65\x73\x65\x74\x20\x65\x76\x65\x72\x79\x0A",
    "\x20\x20\x20\x20\x62\x61\x74\x63\x68\x2C\x20\x73\x6F\x20\x6F\x6E\x65\x20\x70\x72\x65\x66\x69\x78\x20\x6B\x65\x65\x70\x73\x20\x70\x72\x6F\x64\x75\x63\x69\x6E\x67\x20\x6E\x65\x77\x20\x68\x69\x74\x73\x20\x69\x6E\x73\x74\x65\x61\x64\x20\x6F\x66\x20\x6F\x6E\x6C\x79\x20\x6F\x6E\x65\x0A",
    "\x20\x20\x2A\x20\x6C\x69\x76\x65\x20\x72\x61\x74\x65\x20\x69\x73\x20\x6D\x65\x61\x73\x75\x72\x65\x64\x20\x70\x65\x72\x20\x62\x61\x74\x63\x68\x20\x61\x6E\x64\x20\x70\x72\x69\x6E\x74\x65\x64\x20\x61\x73\x20\x61\x20\x63\x75\x6D\x75\x6C\x61\x74\x69\x76\x65\x20\x66\x69\x67\x75\x72\x65\x0A",
    "\x20\x20\x2A\x20\x45\x54\x48\x2F\x42\x54\x43\x2F\x54\x52\x58\x20\x68\x69\x74\x20\x62\x6C\x6F\x63\x6B\x3A\x20\x6D\x6E\x65\x6D\x6F\x6E\x69\x63\x2C\x20\x64\x65\x72\x69\x76\x61\x74\x69\x6F\x6E\x20\x70\x61\x74\x68\x2C\x20\x61\x64\x64\x72\x65\x73\x73\x2C\x20\x70\x72\x69\x76\x61\x74\x65\x20\x6B\x65\x79\x20\x28\x33\x32\x42\x20\x68\x65\x78\x29\x0A",
    "\x20\x20\x2A\x20\x53\x4F\x4C\x20\x66\x61\x73\x74\x20\x68\x69\x74\x20\x62\x6C\x6F\x63\x6B\x3A\x20\x62\x61\x73\x65\x35\x38\x20\x61\x64\x64\x72\x65\x73\x73\x2C\x20\x53\x65\x65\x64\x28\x33\x32\x42\x29\x2C\x20\x53\x65\x63\x72\x65\x74\x20\x6B\x65\x79\x28\x36\x34\x42\x29\x2C\x20\x6B\x65\x79\x70\x61\x69\x72\x20\x4A\x53\x4F\x4E\x0A",
    "\x20\x20\x2A\x20\x53\x4F\x4C\x20\x6D\x6E\x65\x6D\x6F\x6E\x69\x63\x20\x68\x69\x74\x20\x62\x6C\x6F\x63\x6B\x3A\x20\x31\x32\x2D\x77\x6F\x72\x64\x20\x6D\x6E\x65\x6D\x6F\x6E\x69\x63\x2C\x20\x70\x61\x74\x68\x2C\x20\x61\x64\x64\x72\x65\x73\x73\x2C\x20\x53\x65\x65\x64\x2C\x20\x53\x65\x63\x72\x65\x74\x20\x6B\x65\x79\x2C\x20\x6B\x65\x79\x70\x61\x69\x72\x20\x4A\x53\x4F\x4E\x0A",
    "\x0A",
    "\x46\x49\x4C\x45\x53\x20\x43\x52\x45\x41\x54\x45\x44\x20\x41\x54\x20\x52\x55\x4E\x54\x49\x4D\x45\x0A",
    "\x20\x20\x65\x74\x68\x5F\x68\x69\x74\x73\x2E\x74\x78\x74\x20\x2F\x20\x62\x74\x63\x5F\x68\x69\x74\x73\x2E\x74\x78\x74\x20\x2F\x20\x74\x72\x78\x5F\x68\x69\x74\x73\x2E\x74\x78\x74\x20\x20\x20\x20\x20\x68\x69\x74\x20\x72\x65\x63\x6F\x72\x64\x73\x0A",
    "\x20\x20\x73\x6F\x6C\x5F\x68\x69\x74\x73\x2E\x74\x78\x74\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x73\x6F\x6C\x20\x28\x72\x61\x6E\x64\x6F\x6D\x20\x73\x65\x65\x64\x29\x0A",
    "\x20\x20\x73\x6F\x6C\x5F\x62\x69\x70\x33\x39\x5F\x68\x69\x74\x73\x2E\x74\x78\x74\x20\x20\x20\x20\x20\x20\x20\x73\x6F\x6C\x62\x69\x70\x20\x28\x6D\x6E\x65\x6D\x6F\x6E\x69\x63\x29\x0A",
    "\x20\x20\x2E\x65\x74\x68\x5F\x76\x61\x6E\x69\x74\x79\x5F\x73\x65\x63\x72\x65\x74\x20\x2F\x20\x2E\x62\x74\x63\x5F\x76\x61\x6E\x69\x74\x79\x5F\x73\x65\x63\x72\x65\x74\x20\x2F\x20\x2E\x74\x72\x78\x5F\x76\x61\x6E\x69\x74\x79\x5F\x73\x65\x63\x72\x65\x74\x20\x2F\x20\x2E\x73\x6F\x6C\x5F\x76\x61\x6E\x69\x74\x79\x5F\x73\x65\x63\x72\x65\x74\x0A",
    "\x20\x20\x20\x20\x20\x20\x33\x32\x42\x20\x73\x65\x63\x72\x65\x74\x2C\x20\x63\x72\x65\x61\x74\x65\x64\x20\x6F\x6E\x20\x66\x69\x72\x73\x74\x20\x72\x75\x6E\x2C\x20\x64\x65\x63\x69\x64\x65\x73\x20\x74\x68\x65\x20\x6D\x6E\x65\x6D\x6F\x6E\x69\x63\x20\x73\x65\x71\x75\x65\x6E\x63\x65\x2E\x20\x4C\x6F\x73\x69\x6E\x67\x20\x69\x74\x20\x64\x6F\x65\x73\x20\x6E\x6F\x74\x0A",
    "\x20\x20\x20\x20\x20\x20\x6C\x6F\x73\x65\x20\x63\x6F\x69\x6E\x73\x20\x28\x74\x68\x65\x20\x72\x65\x63\x6F\x72\x64\x65\x64\x20\x6D\x6E\x65\x6D\x6F\x6E\x69\x63\x73\x20\x73\x74\x69\x6C\x6C\x20\x72\x65\x63\x6F\x76\x65\x72\x20\x74\x68\x65\x20\x77\x61\x6C\x6C\x65\x74\x29\x20\x62\x75\x74\x20\x74\x68\x65\x20\x73\x65\x71\x75\x65\x6E\x63\x65\x0A",
    "\x20\x20\x20\x20\x20\x20\x63\x61\x6E\x6E\x6F\x74\x20\x62\x65\x20\x72\x65\x70\x72\x6F\x64\x75\x63\x65\x64\x0A",
    "\x20\x20\x73\x6F\x6C\x20\x70\x72\x69\x6E\x74\x73\x20\x61\x20\x66\x72\x65\x73\x68\x20\x33\x32\x42\x20\x72\x75\x6E\x20\x73\x61\x6C\x74\x20\x61\x74\x20\x73\x74\x61\x72\x74\x75\x70\x20\x28\x6E\x6F\x74\x20\x73\x61\x76\x65\x64\x29\x3B\x20\x6B\x65\x65\x70\x20\x69\x74\x20\x74\x6F\x20\x72\x65\x70\x6C\x61\x79\x20\x61\x20\x72\x75\x6E\x0A",
    "\x0A",
    "\x4E\x4F\x54\x45\x53\x0A",
    "\x20\x20\x2A\x20\x57\x69\x6E\x64\x6F\x77\x73\x20\x63\x61\x6E\x6E\x6F\x74\x20\x63\x68\x6D\x6F\x64\x20\x30\x36\x30\x30\x20\x74\x68\x65\x20\x68\x69\x74\x73\x20\x66\x69\x6C\x65\x73\x20\x61\x6E\x64\x20\x74\x68\x65\x79\x20\x68\x6F\x6C\x64\x20\x70\x6C\x61\x69\x6E\x74\x65\x78\x74\x20\x6D\x6E\x65\x6D\x6F\x6E\x69\x63\x73\x20\x61\x6E\x64\x0A",
    "\x20\x20\x20\x20\x70\x72\x69\x76\x61\x74\x65\x20\x6B\x65\x79\x73\x3A\x20\x6B\x65\x65\x70\x20\x74\x68\x65\x6D\x20\x69\x6E\x20\x61\x20\x70\x72\x69\x76\x61\x74\x65\x20\x66\x6F\x6C\x64\x65\x72\x2C\x20\x6E\x65\x76\x65\x72\x20\x69\x6E\x20\x73\x79\x6E\x63\x65\x64\x2F\x73\x68\x61\x72\x65\x64\x20\x64\x69\x72\x65\x63\x74\x6F\x72\x69\x65\x73\x0A",
    "\x20\x20\x2A\x20\x6C\x6F\x6E\x67\x65\x72\x20\x70\x72\x65\x66\x69\x78\x20\x3D\x20\x65\x78\x70\x6F\x6E\x65\x6E\x74\x69\x61\x6C\x6C\x79\x20\x68\x61\x72\x64\x65\x72\x3A\x20\x53\x4F\x4C\x20\x63\x6F\x73\x74\x73\x20\x61\x62\x6F\x75\x74\x20\x35\x38\x78\x20\x70\x65\x72\x20\x65\x78\x74\x72\x61\x20\x63\x68\x61\x72\x61\x63\x74\x65\x72\x0A",
    "\x20\x20\x2A\x20\x70\x72\x65\x66\x69\x78\x20\x66\x69\x6C\x65\x73\x20\x61\x72\x65\x20\x64\x65\x2D\x64\x75\x70\x6C\x69\x63\x61\x74\x65\x64\x20\x61\x6E\x64\x20\x61\x20\x6C\x61\x73\x74\x20\x6C\x69\x6E\x65\x20\x77\x69\x74\x68\x6F\x75\x74\x20\x6E\x65\x77\x6C\x69\x6E\x65\x20\x69\x73\x20\x73\x74\x69\x6C\x6C\x20\x72\x65\x61\x64\x0A",
    "\x20\x20\x2A\x20\x6F\x6E\x6C\x79\x20\x72\x75\x6E\x20\x76\x61\x6E\x69\x74\x79\x20\x66\x6F\x72\x20\x61\x64\x64\x72\x65\x73\x73\x65\x73\x20\x79\x6F\x75\x20\x6F\x77\x6E\x20\x6F\x72\x20\x63\x6F\x6E\x74\x72\x6F\x6C\x0A",
    "\x20\x20\x2A\x20\x63\x6F\x6E\x73\x6F\x6C\x65\x20\x6F\x75\x74\x70\x75\x74\x20\x69\x73\x20\x41\x53\x43\x49\x49\x2D\x6F\x6E\x6C\x79\x20\x6F\x6E\x20\x70\x75\x72\x70\x6F\x73\x65\x2C\x20\x73\x6F\x20\x61\x20\x47\x42\x4B\x2F\x6C\x65\x67\x61\x63\x79\x20\x63\x6F\x64\x65\x20\x70\x61\x67\x65\x20\x73\x68\x6F\x77\x73\x20\x69\x74\x20\x63\x6F\x72\x72\x65\x63\x74\x6C\x79\x0A",
    "\x0A",
    "\x53\x65\x65\x20\x52\x45\x41\x44\x4D\x45\x2E\x6D\x64\x20\x66\x6F\x72\x20\x6D\x65\x61\x73\x75\x72\x65\x64\x20\x72\x61\x74\x65\x73\x2C\x20\x62\x75\x69\x6C\x64\x20\x73\x74\x65\x70\x73\x20\x61\x6E\x64\x20\x76\x65\x72\x69\x66\x69\x63\x61\x74\x69\x6F\x6E\x2E\x0A",
    "\x0A"
};

static const char* MENU_LINES[] = {
    "\x0A",
    "\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x0A",
    "\x20\x6D\x75\x6C\x74\x69\x2D\x63\x68\x61\x69\x6E\x20\x76\x61\x6E\x69\x74\x79\x20\x61\x64\x64\x72\x65\x73\x73\x20\x67\x65\x6E\x65\x72\x61\x74\x6F\x72\x20\x20\x20\x76\x61\x6E\x69\x74\x79\x5F\x6D\x75\x6C\x74\x69\x0A",
    "\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x3D\x0A",
    "\x20\x20\x31\x29\x20\x45\x54\x48\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x42\x49\x50\x33\x39\x20\x2D\x3E\x20\x6D\x2F\x34\x34\x27\x2F\x36\x30\x27\x2F\x30\x27\x2F\x30\x2F\x4E\x20\x20\x2D\x3E\x20\x30\x78\x20\x61\x64\x64\x72\x65\x73\x73\x0A",
    "\x20\x20\x32\x29\x20\x42\x54\x43\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x42\x49\x50\x33\x39\x20\x2D\x3E\x20\x6D\x2F\x34\x34\x27\x2F\x30\x27\x2F\x30\x27\x2F\x30\x2F\x4E\x20\x20\x20\x2D\x3E\x20\x31\x20\x2F\x20\x33\x20\x2F\x20\x62\x63\x31\x71\x20\x2F\x20\x62\x63\x31\x70\x0A",
    "\x20\x20\x33\x29\x20\x54\x52\x58\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x42\x49\x50\x33\x39\x20\x2D\x3E\x20\x6D\x2F\x34\x34\x27\x2F\x31\x39\x35\x27\x2F\x30\x27\x2F\x30\x2F\x4E\x20\x2D\x3E\x20\x54\x20\x61\x64\x64\x72\x65\x73\x73\x0A",
    "\x20\x20\x34\x29\x20\x53\x4F\x4C\x20\x66\x61\x73\x74\x20\x20\x20\x20\x20\x72\x61\x6E\x64\x6F\x6D\x20\x73\x65\x65\x64\x20\x2D\x3E\x20\x65\x64\x32\x35\x35\x31\x39\x20\x2D\x3E\x20\x62\x61\x73\x65\x35\x38\x20\x20\x20\x20\x20\x28\x66\x61\x73\x74\x65\x73\x74\x29\x0A",
    "\x20\x20\x35\x29\x20\x53\x4F\x4C\x20\x6D\x6E\x65\x6D\x6F\x6E\x69\x63\x20\x42\x49\x50\x33\x39\x20\x2D\x3E\x20\x53\x4C\x49\x50\x2D\x30\x30\x31\x30\x20\x2D\x3E\x20\x62\x61\x73\x65\x35\x38\x20\x20\x20\x20\x20\x20\x20\x20\x20\x28\x69\x6D\x70\x6F\x72\x74\x61\x62\x6C\x65\x29\x0A",
    "\x20\x20\x68\x29\x20\x68\x65\x6C\x70\x20\x20\x20\x20\x20\x20\x20\x20\x20\x71\x29\x20\x71\x75\x69\x74\x0A",
    "\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x2D\x0A",
    "\x0A"
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
        printf("select [1-5/h/q]: ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) return 0;
        char c = line[0];
        if (c == 'q' || c == 'Q' || c == '\n' || c == '\r' || c == 0) { printf("bye\n"); return 0; }
        if (c == 'h' || c == 'H') { print_help(); continue; }
        int pick = (c >= '1' && c <= '5') ? (c - '0') : -1;
        if (pick < 0) { printf("invalid choice\n"); continue; }
        const char* chain = chain_of(pick);
        printf("\nselected %s. enter args (e.g. vanity Ase 1024 / file list.txt / selftest / bench 64 1024)\n", chain);
        printf("press Enter to return to the menu.\n> ");
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
        printf("\n[%s finished, back to menu]\n", chain);
    }
}
