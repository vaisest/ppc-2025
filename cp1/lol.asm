Compiling to assembly
Compiling...
Compiled
Output for assembly:
	.file	"cp.cc"
# GNU C++20 (Ubuntu 13.3.0-6ubuntu2~24.04) version 13.3.0 (x86_64-linux-gnu)
#	compiled by GNU C version 13.3.0, GMP version 6.3.0, MPFR version 4.2.1, MPC version 1.3.1, isl version isl-0.26-GMP

# GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
# options passed: -march=znver3 -mmmx -mpopcnt -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -msse4a -mno-fma4 -mno-xop -mfma -mno-avx512f -mbmi -mbmi2 -maes -mpclmul -mno-avx512vl -mno-avx512bw -mno-avx512dq -mno-avx512cd -mno-avx512er -mno-avx512pf -mno-avx512vbmi -mno-avx512ifma -mno-avx5124vnniw -mno-avx5124fmaps -mno-avx512vpopcntdq -mno-avx512vbmi2 -mno-gfni -mvpclmulqdq -mno-avx512vnni -mno-avx512bitalg -mno-avx512bf16 -mno-avx512vp2intersect -mno-3dnow -madx -mabm -mno-cldemote -mclflushopt -mclwb -mclzero -mcx16 -mno-enqcmd -mf16c -mfsgsbase -mfxsr -mno-hle -msahf -mno-lwp -mlzcnt -mmovbe -mno-movdir64b -mno-movdiri -mno-mwaitx -mno-pconfig -mno-pku -mno-prefetchwt1 -mprfchw -mno-ptwrite -mrdpid -mrdrnd -mrdseed -mno-rtm -mno-serialize -mno-sgx -msha -mshstk -mno-tbm -mno-tsxldtrk -mvaes -mno-waitpkg -mno-wbnoinvd -mxsave -mxsavec -mxsaveopt -mxsaves -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-uintr -mno-hreset -mno-kl -mno-widekl -mno-avxvnni -mno-avx512fp16 -mno-avxifma -mno-avxvnniint8 -mno-avxneconvert -mno-cmpccxadd -mno-amx-fp16 -mno-prefetchi -mno-raoint -mno-amx-complex --param=l1-cache-size=32 --param=l1-cache-line-size=64 --param=l2-cache-size=512 -mtune=znver3 -O3 -std=c++20 -fno-tree-vectorize -fasynchronous-unwind-tables -fstack-protector-strong -fstack-clash-protection -fcf-protection
	.text
#APP
	.globl _ZSt21ios_base_library_initv
#NO_APP
	.section	.text.unlikely,"ax",@progbits
.LCOLDB2:
	.text
.LHOTB2:
	.p2align 4
	.globl	_Z9correlateiiPKfPf
	.type	_Z9correlateiiPKfPf, @function
_Z9correlateiiPKfPf:
.LFB6196:
	.cfi_startproc
	endbr64	
# cp.cc:31:     std::unique_ptr<double[]> norm(new double[ny * nx]);
	movl	%edi, %eax	# ny, tmp153
# cp.cc:30: {
	pushq	%r15	#
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14	#
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13	#
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
# cp.cc:31:     std::unique_ptr<double[]> norm(new double[ny * nx]);
	imull	%esi, %eax	# nx, tmp153
# cp.cc:30: {
	pushq	%r12	#
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movq	%rdx, %r12	# tmp188, data
	pushq	%rbp	#
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx	#
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$56, %rsp	#,
	.cfi_def_cfa_offset 112
# cp.cc:31:     std::unique_ptr<double[]> norm(new double[ny * nx]);
	cltq
# cp.cc:30: {
	movq	%rcx, (%rsp)	# tmp189, %sfp
# cp.cc:31:     std::unique_ptr<double[]> norm(new double[ny * nx]);
	movq	%rax, %rdx	# _58, tmp195
	shrq	$60, %rdx	#, tmp195
	jne	.L2	#,
	movl	%edi, %r14d	# tmp186, ny
# cp.cc:31:     std::unique_ptr<double[]> norm(new double[ny * nx]);
	leaq	0(,%rax,8), %rdi	#, tmp155
	movl	%esi, %ebx	# tmp187, nx
# cp.cc:31:     std::unique_ptr<double[]> norm(new double[ny * nx]);
	call	_Znam@PLT	#
	movq	%rax, %rbp	# tmp190, _63
# cp.cc:33:     for (int i = 0; i < ny; i++)
	testl	%r14d, %r14d	# ny
	jle	.L4	#,
	movslq	%ebx, %r13	# nx, _147
	movq	%rax, %rdx	# _63, ivtmp.105
	movq	%r12, %r10	# data, data
# cp.cc:33:     for (int i = 0; i < ny; i++)
	xorl	%esi, %esi	# i
	leaq	0(,%r13,8), %r8	#, _129
	leaq	0(,%r13,4), %r9	#, _146
	movq	%rax, 8(%rsp)	# _63, %sfp
	movl	%esi, %ebp	# i, i
	leaq	(%rax,%r8), %r15	#, ivtmp.107
	movl	%ebx, %eax	# nx, nx
	movq	%rdx, %rbx	# ivtmp.105, ivtmp.105
	movq	%r13, %rdi	# _147, _147
	movl	%r14d, %ecx	# ny, ny
	vxorps	%xmm3, %xmm3, %xmm3	# tmp192
	subq	%r9, %r10	# _146, data
	movq	%r9, %r12	# ivtmp.103, _116
	movq	%r8, %r14	# _129, _129
	movl	%eax, %edx	# nx, nx
	movq	%r9, %r13	# ivtmp.103, ivtmp.103
	movq	%rbx, %rsi	# ivtmp.105, _63
	.p2align 4
	.p2align 3
.L14:
# cp.cc:35:         std::copy(data + nx * i, data + nx * (i + 1), norm.get() + nx * i);
	incl	%ebp	# i
# cp.cc:35:         std::copy(data + nx * i, data + nx * (i + 1), norm.get() + nx * i);
	leaq	(%r10,%r13), %r8	#, _13
# /usr/include/c++/13/bits/stl_algobase.h:386: 	  for(_Distance __n = __last - __first; __n > 0; --__n)
	xorl	%eax, %eax	# ivtmp.92
	testq	%r12, %r12	# _116
	jle	.L8	#,
	.p2align 4
	.p2align 3
.L5:
# /usr/include/c++/13/bits/stl_algobase.h:388: 	      *__result = *__first;
	vcvtss2sd	(%r8,%rax), %xmm3, %xmm0	# MEM[(const float *)_13 + ivtmp.92_166 * 1], tmp192, tmp193
	vmovsd	%xmm0, (%rbx,%rax,2)	# tmp162, MEM[(double *)_6 + ivtmp.92_166 * 2]
# /usr/include/c++/13/bits/stl_algobase.h:386: 	  for(_Distance __n = __last - __first; __n > 0; --__n)
	addq	$4, %rax	#, ivtmp.92
	cmpq	%rax, %r12	# ivtmp.92, _116
	jne	.L5	#,
.L8:
# cp.cc:38:         for (int x = 0; x < nx; x++)
	movq	%rbx, %rax	# ivtmp.105, ivtmp.87
# cp.cc:37:         double sum = 0;
	vxorpd	%xmm2, %xmm2, %xmm2	# sum
# cp.cc:38:         for (int x = 0; x < nx; x++)
	testl	%edx, %edx	# nx
	jle	.L7	#,
	movq	%r15, %r8	# ivtmp.107, tmp181
	subq	%rbx, %r8	# ivtmp.105, tmp181
	andl	$8, %r8d	#, tmp181
	je	.L6	#,
	leaq	8(%rbx), %rax	#, ivtmp.87
# cp.cc:40:             sum += norm[x + i * nx];
	vaddsd	(%rbx), %xmm2, %xmm2	# MEM[(double &)_173], sum, sum
# cp.cc:38:         for (int x = 0; x < nx; x++)
	cmpq	%rax, %r15	# ivtmp.87, ivtmp.107
	je	.L41	#,
	.p2align 4
	.p2align 3
.L6:
# cp.cc:40:             sum += norm[x + i * nx];
	vaddsd	(%rax), %xmm2, %xmm2	# MEM[(double &)_173], sum, sum
# cp.cc:38:         for (int x = 0; x < nx; x++)
	addq	$16, %rax	#, ivtmp.87
# cp.cc:40:             sum += norm[x + i * nx];
	vaddsd	-8(%rax), %xmm2, %xmm2	# MEM[(double &)_173], sum, sum
# cp.cc:38:         for (int x = 0; x < nx; x++)
	cmpq	%rax, %r15	# ivtmp.87, ivtmp.107
	jne	.L6	#,
.L41:
# cp.cc:43:         double mean = sum / (double)nx;
	vcvtsi2sdl	%edx, %xmm3, %xmm1	# nx, tmp192, tmp194
# cp.cc:43:         double mean = sum / (double)nx;
	movq	%rbx, %rax	# ivtmp.105, ivtmp.82
	vdivsd	%xmm1, %xmm2, %xmm2	# tmp163, sum, mean
# cp.cc:44:         double sum_sq = 0;
	vxorpd	%xmm1, %xmm1, %xmm1	# sum_sq
	.p2align 4
	.p2align 3
.L9:
# cp.cc:47:             double asd = norm[x + i * nx] - mean;
	vmovsd	(%rax), %xmm0	# MEM[(double &)_187], MEM[(double &)_187]
# cp.cc:45:         for (int x = 0; x < nx; x++)
	addq	$8, %rax	#, ivtmp.82
# cp.cc:47:             double asd = norm[x + i * nx] - mean;
	vsubsd	%xmm2, %xmm0, %xmm0	# mean, MEM[(double &)_187], asd
# cp.cc:48:             norm[x + i * nx] = asd;
	vmovsd	%xmm0, -8(%rax)	# asd, MEM[(double &)_187]
# cp.cc:49:             sum_sq += asd * asd;
	vmulsd	%xmm0, %xmm0, %xmm0	# asd, asd, tmp165
# cp.cc:49:             sum_sq += asd * asd;
	vaddsd	%xmm0, %xmm1, %xmm1	# tmp165, sum_sq, sum_sq
# cp.cc:45:         for (int x = 0; x < nx; x++)
	cmpq	%rax, %r15	# ivtmp.82, ivtmp.107
	jne	.L9	#,
	vxorpd	%xmm0, %xmm0, %xmm0	# tmp166
	vucomisd	%xmm1, %xmm0	# sum_sq, tmp166
	ja	.L39	#,
# cp.cc:51:         double sq_sqrt = sqrt(sum_sq);
	vsqrtsd	%xmm1, %xmm1, %xmm1	# sum_sq, sq_sqrt
.L12:
# cp.cc:44:         double sum_sq = 0;
	movq	%rbx, %rax	# ivtmp.105, ivtmp.77
	.p2align 4
	.p2align 3
.L13:
# cp.cc:54:             norm[x + i * nx] /= sq_sqrt;
	vmovsd	(%rax), %xmm0	# MEM[(double &)_201], MEM[(double &)_201]
# cp.cc:52:         for (int x = 0; x < nx; x++)
	addq	$8, %rax	#, ivtmp.77
# cp.cc:54:             norm[x + i * nx] /= sq_sqrt;
	vdivsd	%xmm1, %xmm0, %xmm0	# sq_sqrt, MEM[(double &)_201], tmp167
	vmovsd	%xmm0, -8(%rax)	# tmp167, MEM[(double &)_201]
# cp.cc:52:         for (int x = 0; x < nx; x++)
	cmpq	%rax, %r15	# ivtmp.77, ivtmp.107
	jne	.L13	#,
.L7:
# cp.cc:33:     for (int i = 0; i < ny; i++)
	addq	%r9, %r13	# _146, ivtmp.103
	addq	%r14, %rbx	# _129, ivtmp.105
	addq	%r14, %r15	# _129, ivtmp.107
	cmpl	%ecx, %ebp	# ny, i
	jne	.L14	#,
	movq	(%rsp), %rax	# %sfp, ivtmp.72
	movslq	%ecx, %r11	# ny,
	movl	%edx, %ebx	# nx, nx
	movq	%rsi, %rbp	# _63, _63
	movq	%r11, %r14	#,
	movq	%rdi, %r13	# _147, _147
	salq	$2, %r11	#, _221
	xorl	%ecx, %ecx	# ivtmp.73
# cp.cc:57:     for (int x = 0; x < ny; x++)
	xorl	%r10d, %r10d	# x
	movl	%r14d, %edx	# ny, ny
	.p2align 4
	.p2align 3
.L16:
# cp.cc:44:         double sum_sq = 0;
	movq	%rax, %r12	# ivtmp.72, ivtmp.66
	xorl	%r14d, %r14d	# ivtmp.68
# cp.cc:59:         for (int y = 0; y <= x; y++)
	xorl	%r9d, %r9d	# y
	movslq	%ecx, %r15	# ivtmp.73, _29
	movl	%ecx, (%rsp)	# ivtmp.73, %sfp
	.p2align 4
	.p2align 3
.L19:
# cp.cc:63:             asm("# k-loop");
#APP
# 63 "cp.cc" 1
	# k-loop
# 0 "" 2
# cp.cc:64:             for (int k = 0; k < nx; k++)
#NO_APP
	testl	%ebx, %ebx	# nx
	jle	.L22	#,
	movslq	%r14d, %r8	# ivtmp.68, _242
# cp.cc:62:             double sum = 0;
	vxorpd	%xmm1, %xmm1, %xmm1	# sum
	leaq	0(%r13,%r8), %rsi	#, tmp171
	leaq	0(%rbp,%r8,8), %rcx	#, ivtmp.58
	leaq	0(%rbp,%rsi,8), %rdi	#, _227
# cp.cc:66:                 sum += norm[y * nx + k] * norm[x * nx + k];
	movq	%r15, %rsi	# _29, tmp174
	subq	%r8, %rsi	# _242, tmp174
	.p2align 4
	.p2align 3
.L18:
	vmovsd	(%rcx,%rsi,8), %xmm0	# MEM[(double &)_239 + _234 * 8], MEM[(double &)_239 + _234 * 8]
# cp.cc:64:             for (int k = 0; k < nx; k++)
	addq	$8, %rcx	#, ivtmp.58
# cp.cc:66:                 sum += norm[y * nx + k] * norm[x * nx + k];
	vmulsd	-8(%rcx), %xmm0, %xmm0	# MEM[(double &)_239], MEM[(double &)_239 + _234 * 8], tmp175
# cp.cc:66:                 sum += norm[y * nx + k] * norm[x * nx + k];
	vaddsd	%xmm0, %xmm1, %xmm1	# tmp175, sum, sum
# cp.cc:64:             for (int k = 0; k < nx; k++)
	cmpq	%rcx, %rdi	# ivtmp.58, _227
	jne	.L18	#,
# cp.cc:68:             result[y * ny + x] = (float)sum;
	vcvtsd2ss	%xmm1, %xmm1, %xmm1	# sum, _249
.L17:
# cp.cc:59:         for (int y = 0; y <= x; y++)
	incl	%r9d	# y
# cp.cc:68:             result[y * ny + x] = (float)sum;
	vmovss	%xmm1, (%r12)	# _249, MEM[(float *)_213]
# cp.cc:59:         for (int y = 0; y <= x; y++)
	addl	%ebx, %r14d	# nx, ivtmp.68
	addq	%r11, %r12	# _221, ivtmp.66
	cmpl	%r10d, %r9d	# x, y
	jle	.L19	#,
# cp.cc:57:     for (int x = 0; x < ny; x++)
	movl	(%rsp), %ecx	# %sfp, ivtmp.73
	incl	%r10d	# x
# cp.cc:57:     for (int x = 0; x < ny; x++)
	addq	$4, %rax	#, ivtmp.72
	addl	%ebx, %ecx	# nx, ivtmp.73
	cmpl	%r10d, %edx	# x, ny
	jne	.L16	#,
.L4:
# cp.cc:71: }
	addq	$56, %rsp	#,
	.cfi_remember_state
	.cfi_def_cfa_offset 56
# /usr/include/c++/13/bits/unique_ptr.h:140: 	  delete [] __ptr;
	movq	%rbp, %rdi	# _63,
# cp.cc:71: }
	popq	%rbx	#
	.cfi_def_cfa_offset 48
	popq	%rbp	#
	.cfi_def_cfa_offset 40
	popq	%r12	#
	.cfi_def_cfa_offset 32
	popq	%r13	#
	.cfi_def_cfa_offset 24
	popq	%r14	#
	.cfi_def_cfa_offset 16
	popq	%r15	#
	.cfi_def_cfa_offset 8
# /usr/include/c++/13/bits/unique_ptr.h:140: 	  delete [] __ptr;
	jmp	_ZdaPv@PLT	#
	.p2align 4
	.p2align 3
.L22:
	.cfi_restore_state
# cp.cc:64:             for (int k = 0; k < nx; k++)
	vxorps	%xmm1, %xmm1, %xmm1	# _249
	jmp	.L17	#
.L39:
# cp.cc:51:         double sq_sqrt = sqrt(sum_sq);
	vmovsd	%xmm1, %xmm1, %xmm0	# sum_sq,
	movl	%edx, 44(%rsp)	# nx, %sfp
	movl	%ecx, 40(%rsp)	# ny, %sfp
	movq	%rdi, 32(%rsp)	# _147, %sfp
	movq	%r9, 24(%rsp)	# _146, %sfp
	movq	%rsi, 16(%rsp)	# _63, %sfp
	movq	%r10, 8(%rsp)	# _55, %sfp
	call	sqrt@PLT	#
	movl	44(%rsp), %edx	# %sfp, nx
	vxorps	%xmm3, %xmm3, %xmm3	# tmp192
	movl	40(%rsp), %ecx	# %sfp, ny
	movq	32(%rsp), %rdi	# %sfp, _147
	vmovsd	%xmm0, %xmm0, %xmm1	# tmp191, sq_sqrt
	movq	24(%rsp), %r9	# %sfp, _146
	movq	16(%rsp), %rsi	# %sfp, _63
	movq	8(%rsp), %r10	# %sfp, _55
	jmp	.L12	#
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.type	_Z9correlateiiPKfPf.cold, @function
_Z9correlateiiPKfPf.cold:
.LFSB6196:
.L2:
	.cfi_def_cfa_offset 112
	.cfi_offset 3, -56
	.cfi_offset 6, -48
	.cfi_offset 12, -40
	.cfi_offset 13, -32
	.cfi_offset 14, -24
	.cfi_offset 15, -16
# cp.cc:31:     std::unique_ptr<double[]> norm(new double[ny * nx]);
	call	__cxa_throw_bad_array_new_length@PLT	#
	.cfi_endproc
.LFE6196:
	.text
	.size	_Z9correlateiiPKfPf, .-_Z9correlateiiPKfPf
	.section	.text.unlikely
	.size	_Z9correlateiiPKfPf.cold, .-_Z9correlateiiPKfPf.cold
.LCOLDE2:
	.text
.LHOTE2:
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
