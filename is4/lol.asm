Compiling to assembly
Compiling...
Compiled
Output for assembly:
	.file	"is.cc"
# GNU C++20 (Ubuntu 13.3.0-6ubuntu2~24.04) version 13.3.0 (x86_64-linux-gnu)
#	compiled by GNU C version 13.3.0, GMP version 6.3.0, MPFR version 4.2.1, MPC version 1.3.1, isl version isl-0.26-GMP

# GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
# options passed: -march=znver3 -mmmx -mpopcnt -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -msse4a -mno-fma4 -mno-xop -mfma -mno-avx512f -mbmi -mbmi2 -maes -mpclmul -mno-avx512vl -mno-avx512bw -mno-avx512dq -mno-avx512cd -mno-avx512er -mno-avx512pf -mno-avx512vbmi -mno-avx512ifma -mno-avx5124vnniw -mno-avx5124fmaps -mno-avx512vpopcntdq -mno-avx512vbmi2 -mno-gfni -mvpclmulqdq -mno-avx512vnni -mno-avx512bitalg -mno-avx512bf16 -mno-avx512vp2intersect -mno-3dnow -madx -mabm -mno-cldemote -mclflushopt -mclwb -mclzero -mcx16 -mno-enqcmd -mf16c -mfsgsbase -mfxsr -mno-hle -msahf -mno-lwp -mlzcnt -mmovbe -mno-movdir64b -mno-movdiri -mno-mwaitx -mno-pconfig -mno-pku -mno-prefetchwt1 -mprfchw -mno-ptwrite -mrdpid -mrdrnd -mrdseed -mno-rtm -mno-serialize -mno-sgx -msha -mshstk -mno-tbm -mno-tsxldtrk -mvaes -mno-waitpkg -mno-wbnoinvd -mxsave -mxsavec -mxsaveopt -mxsaves -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-uintr -mno-hreset -mno-kl -mno-widekl -mno-avxvnni -mno-avx512fp16 -mno-avxifma -mno-avxvnniint8 -mno-avxneconvert -mno-cmpccxadd -mno-amx-fp16 -mno-prefetchi -mno-raoint -mno-amx-complex --param=l1-cache-size=32 --param=l1-cache-line-size=64 --param=l2-cache-size=512 -mtune=znver3 -O3 -std=c++20 -fopenmp -fasynchronous-unwind-tables -fstack-protector-strong -fstack-clash-protection -fcf-protection
	.text
#APP
	.globl _ZSt21ios_base_library_initv
#NO_APP
	.section	.text._ZNKSt5ctypeIcE8do_widenEc,"axG",@progbits,_ZNKSt5ctypeIcE8do_widenEc,comdat
	.align 2
	.p2align 4
	.weak	_ZNKSt5ctypeIcE8do_widenEc
	.type	_ZNKSt5ctypeIcE8do_widenEc, @function
_ZNKSt5ctypeIcE8do_widenEc:
.LFB2376:
	.cfi_startproc
	endbr64	
# /usr/include/c++/13/bits/locale_facets.h:1092:       do_widen(char __c) const
	movl	%esi, %eax	# tmp87, __c
# /usr/include/c++/13/bits/locale_facets.h:1093:       { return __c; }
	ret	
	.cfi_endproc
.LFE2376:
	.size	_ZNKSt5ctypeIcE8do_widenEc, .-_ZNKSt5ctypeIcE8do_widenEc
	.text
	.p2align 4
	.type	_Z7segmentiiPKf._omp_fn.0, @function
_Z7segmentiiPKf._omp_fn.0:
.LFB14110:
	.cfi_startproc
	endbr64	
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	vxorps	%xmm2, %xmm2, %xmm2	# vect_local_result_inner_1_14.150
	vxorps	%xmm1, %xmm1, %xmm1	# vect__165.111
	movq	%rsp, %rbp	#,
	.cfi_def_cfa_register 6
	pushq	%r15	#
	pushq	%r14	#
	pushq	%r13	#
	pushq	%r12	#
	pushq	%rbx	#
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	%rdi, %rbx	# tmp331, .omp_data_i
	andq	$-32, %rsp	#,
	subq	$192, %rsp	#,
# is.cc:92: #pragma omp parallel
	vmovapd	(%rdi), %ymm7	# *.omp_data_i_27(D).image_totals, image_totals
	movl	56(%rdi), %r12d	# *.omp_data_i_27(D).ny, ny
	vmovlps	%xmm2, 160(%rsp)	# vect_local_result_inner_1_14.150, %sfp
	vmovaps	%xmm1, 112(%rsp)	# vect__165.111, %sfp
	movl	60(%rdi), %r15d	# *.omp_data_i_27(D).nx, nx
	vmovapd	%ymm7, 128(%rsp)	# image_totals, %sfp
	vzeroupper
	call	omp_get_num_threads@PLT	#
	movl	%eax, %r14d	# tmp332, _33
	call	omp_get_thread_num@PLT	#
	vmovaps	112(%rsp), %xmm1	# %sfp, vect__165.111
	leal	1(%rax), %edi	#, _39
	cmpl	%eax, %r12d	# _34, ny
	vmovq	160(%rsp), %xmm2	# %sfp, vect_local_result_inner_1_14.150
	movl	%edi, 168(%rsp)	# _39, %sfp
	jle	.L27	#,
# is.cc:129:                         inner_sums = sums[br];
	movq	40(%rbx), %rdi	# *.omp_data_i_27(D).sums, _77
# is.cc:102:                 double outer_area = nx * ny - inner_area;
	movl	%r15d, %ecx	# nx, _52
# is.cc:169:                         total_error += pix_sum(vec_fmadd(outer_color, outer_sums, inner_errors));
	movl	%r14d, 40(%rsp)	# _33, %sfp
	vxorps	%xmm8, %xmm8, %xmm8	# tmp334
# is.cc:94:         double local_min_error = std::numeric_limits<double>::min();
	vmovsd	.LC0(%rip), %xmm3	#, local_min_error
# is.cc:102:                 double outer_area = nx * ny - inner_area;
	imull	%r12d, %ecx	# ny, _52
# is.cc:169:                         total_error += pix_sum(vec_fmadd(outer_color, outer_sums, inner_errors));
	vxorpd	%xmm6, %xmm6, %xmm6	# tmp324
	movl	%r12d, 44(%rsp)	# ny, %sfp
	vmovapd	128(%rsp), %ymm7	# %sfp, image_totals
	movq	%rbx, 16(%rsp)	# .omp_data_i, %sfp
# is.cc:129:                         inner_sums = sums[br];
	movq	%rdi, 56(%rsp)	# _77, %sfp
	movl	%r12d, %edi	# ny, ivtmp.213
# is.cc:102:                 double outer_area = nx * ny - inner_area;
	movl	%ecx, %r9d	# _52, _52
	movl	%r15d, %ecx	# nx, nx
	subl	%eax, %edi	# _34, ivtmp.213
	imull	%r15d, %eax	# nx, _34
# is.cc:169:                         total_error += pix_sum(vec_fmadd(outer_color, outer_sums, inner_errors));
	movl	%r9d, 36(%rsp)	# _52, %sfp
	movl	%edi, 112(%rsp)	# ivtmp.213, %sfp
	movl	%r15d, %edi	# nx, _550
	imull	%r14d, %edi	# _33, _550
	movl	%r13d, %r14d	# local_result$x1, local_result$x1
	movl	%eax, 76(%rsp)	# _34, %sfp
	movl	%r15d, %eax	# nx, tmp243
	negl	%eax	# tmp243
	cltq
	movl	%edi, 32(%rsp)	# _550, %sfp
	leaq	1(%rax), %rdx	#, tmp244
	negq	%rax	# tmp250
	salq	$5, %rdx	#, tmp244
	salq	$5, %rax	#, tmp250
	movq	%rdx, %r10	# tmp244, ivtmp.201
	movslq	%r15d, %rdx	# nx, nx
	movq	%rax, 48(%rsp)	# tmp250, %sfp
	negq	%rdx	# tmp247
	movq	%r10, 24(%rsp)	# ivtmp.201, %sfp
	salq	$5, %rdx	#, tmp247
	movq	%rdx, 160(%rsp)	# tmp247, %sfp
.L6:
# is.cc:99:             for (int w = 1; w <= nx; w++)
	testl	%ecx, %ecx	# nx
	jle	.L7	#,
# is.cc:104:                 if (w == nx && h == ny)
	movl	168(%rsp), %r12d	# %sfp, _39
	cmpl	%r12d, 44(%rsp)	# _39, %sfp
# is.cc:102:                 double outer_area = nx * ny - inner_area;
	vcvtsi2sdl	36(%rsp), %xmm8, %xmm0	# %sfp, tmp334, tmp335
	leal	-1(%rcx), %edx	#, ivtmp.195
	movl	112(%rsp), %eax	# %sfp, ivtmp.213
	movslq	76(%rsp), %r8	# %sfp, ivtmp.214
	vmovsd	%xmm0, %xmm0, %xmm14	# tmp335, _53
# is.cc:104:                 if (w == nx && h == ny)
	movl	$1, %r15d	#, ivtmp.197
	movq	24(%rsp), %r10	# %sfp, ivtmp.201
	movl	%ecx, %esi	# nx, nx
	sete	95(%rsp)	#, %sfp
	decl	%eax	# _560
	salq	$5, %r8	#, ivtmp.214
	movl	%eax, 72(%rsp)	# _560, %sfp
	movq	%r8, 64(%rsp)	# ivtmp.188, %sfp
	movq	%r10, %r11	# ivtmp.201, ivtmp.201
	.p2align 4
	.p2align 3
.L10:
# is.cc:104:                 if (w == nx && h == ny)
	cmpl	%r15d, %esi	# ivtmp.197, nx
	jne	.L9	#,
	cmpb	$0, 95(%rsp)	#, %sfp
	je	.L9	#,
.L42:
	movl	%esi, %ecx	# nx, nx
.L7:
	movl	40(%rsp), %ebx	# %sfp, _33
	addl	%ebx, 168(%rsp)	# _33, %sfp
	movl	168(%rsp), %eax	# %sfp, _39
	movl	32(%rsp), %edi	# %sfp, _550
	subl	%ebx, 112(%rsp)	# _33, %sfp
	addl	%edi, 76(%rsp)	# _550, %sfp
	leal	-1(%rax), %edx	#, tmp252
	cmpl	%edx, 44(%rsp)	# tmp252, %sfp
	jg	.L6	#,
	movq	16(%rsp), %rbx	# %sfp, .omp_data_i
	movl	%r14d, %r13d	# local_result$x1, local_result$x1
	vzeroupper
.L4:
	vmovaps	%xmm1, 112(%rsp)	# vect__165.111, %sfp
	vmovlps	%xmm2, 168(%rsp)	# vect_local_result_inner_1_14.150, %sfp
	vmovsd	%xmm3, 128(%rsp)	# local_min_error, %sfp
# is.cc:188: #pragma omp critical
	call	GOMP_critical_start@PLT	#
# is.cc:191:             if (local_min_error > final_error)
	vmovsd	128(%rsp), %xmm3	# %sfp, local_min_error
	vmovq	168(%rsp), %xmm2	# %sfp, vect_local_result_inner_1_14.150
	vcomisd	48(%rbx), %xmm3	# *.omp_data_i_27(D).final_error, local_min_error
	vmovaps	112(%rsp), %xmm1	# %sfp, vect__165.111
	jbe	.L26	#,
# is.cc:194:                 final_result = local_result;
	movq	32(%rbx), %rax	# *.omp_data_i_27(D).final_result, _43
# is.cc:193:                 final_error = local_min_error;
	vmovsd	%xmm3, 48(%rbx)	# local_min_error, *.omp_data_i_27(D).final_error
# is.cc:194:                 final_result = local_result;
	movl	188(%rsp), %ebx	# %sfp, local_result$y0
	movl	%ebx, (%rax)	# local_result$y0, *_43.y0
	movl	184(%rsp), %ebx	# %sfp, local_result$x0
	movl	%r13d, 12(%rax)	# local_result$x1, *_43.x1
	vmovups	%xmm1, 16(%rax)	# vect__165.111, MEM <vector(4) float> [(float *)_43 + 16B]
	vmovlps	%xmm2, 32(%rax)	# vect_local_result_inner_1_14.150, MEM <vector(2) float> [(float *)_43 + 32B]
	movl	%ebx, 4(%rax)	# local_result$x0, *_43.x0
	movl	180(%rsp), %ebx	# %sfp, local_result$y1
	movl	%ebx, 8(%rax)	# local_result$y1, *_43.y1
.L26:
# is.cc:92: #pragma omp parallel
	leaq	-40(%rbp), %rsp	#,
	popq	%rbx	#
	popq	%r12	#
	popq	%r13	#
	popq	%r14	#
	popq	%r15	#
	popq	%rbp	#
	.cfi_remember_state
	.cfi_def_cfa 7, 8
# is.cc:188: #pragma omp critical
	jmp	GOMP_critical_end@PLT	#
	.p2align 4
	.p2align 3
.L9:
	.cfi_restore_state
# is.cc:108:                 for (int y = 0; y <= ny - h; y++)
	movl	72(%rsp), %eax	# %sfp,
	testl	%eax, %eax	#
	js	.L11	#,
# /usr/include/c++/13/bits/unique_ptr.h:199:       pointer    _M_ptr() const noexcept { return std::get<0>(_M_t); }
	movq	56(%rsp), %rax	# %sfp, _77
	movq	64(%rsp), %rdi	# %sfp, ivtmp.188
# is.cc:101:                 double inner_area = h * w;
	vcvtsi2sdl	%r12d, %xmm8, %xmm4	# ivtmp.200, tmp334, tmp336
	movq	%r15, %rcx	# ivtmp.197, tmp260
# is.cc:150:                         inner_color = inner_sums / (double)inner_area;
	vbroadcastsd	%xmm4, %ymm5	# inner_area, pretmp_346
# is.cc:152:                         outer_color = outer_sums / (double)outer_area;
	movq	48(%rsp), %r10	# %sfp, ivtmp.187
# is.cc:102:                 double outer_area = nx * ny - inner_area;
	vsubsd	%xmm4, %xmm14, %xmm4	# inner_area, _53, outer_area
# is.cc:152:                         outer_color = outer_sums / (double)outer_area;
	movl	76(%rsp), %ebx	# %sfp, ivtmp.181
	salq	$5, %rcx	#, tmp260
# is.cc:108:                 for (int y = 0; y <= ny - h; y++)
	xorl	%r9d, %r9d	# y
	movl	%r12d, 88(%rsp)	# ivtmp.200, %sfp
	movq	%r11, 80(%rsp)	# ivtmp.201, %sfp
# /usr/include/c++/13/bits/unique_ptr.h:199:       pointer    _M_ptr() const noexcept { return std::get<0>(_M_t); }
	movq	(%rax), %r8	# MEM[(vector(4) double * const &)_77], _17
	movl	%esi, %eax	# nx, ivtmp.184
# is.cc:152:                         outer_color = outer_sums / (double)outer_area;
	vbroadcastsd	%xmm4, %ymm4	# outer_area, pretmp_347
	negl	%eax	# ivtmp.184
	movl	%eax, 128(%rsp)	# ivtmp.184, %sfp
	movq	%r8, %rax	# _17, tmp259
	leaq	-32(%r8,%rcx), %rcx	#, tmp314
	subq	%rdi, %rax	# ivtmp.188, tmp259
	movq	%rcx, 104(%rsp)	# tmp314, %sfp
	leaq	-32(%r11,%rax), %rax	#, tmp318
	movq	%rax, 96(%rsp)	# tmp318, %sfp
	.p2align 4
	.p2align 3
.L18:
# is.cc:111:                     const int bot_row = nx * (y + h - 1);
	movl	168(%rsp), %eax	# %sfp, _39
	leal	(%r9,%rax), %r11d	#, _62
	testl	%r9d, %r9d	# y
	je	.L12	#,
	movq	96(%rsp), %rax	# %sfp, tmp318
# is.cc:117:                         const int tl = top_row + left;
	movl	128(%rsp), %r12d	# %sfp, ivtmp.184
# is.cc:119:                         const int bl = bot_row + left;
	leal	-1(%rbx), %r13d	#, tmp310
	movl	%esi, 176(%rsp)	# nx, %sfp
	leaq	(%rax,%rdi), %rcx	#, ivtmp.170
# is.cc:117:                         const int tl = top_row + left;
	decl	%r12d	# tmp311
# is.cc:112:                     for (int x = 0; x <= nx - w; x++)
	xorl	%eax, %eax	# x
	jmp	.L13	#
	.p2align 4
	.p2align 3
.L46:
# is.cc:138:                             inner_sums -= sums[tr];
	vsubpd	(%rcx), %ymm0, %ymm0	# MEM[(vector(4) double *)_453], inner_sums, inner_sums
.L24:
# is.cc:147:                         outer_sums = image_totals - inner_sums;
	vsubpd	%ymm0, %ymm7, %ymm9	# inner_sums, image_totals, outer_sums
# is.cc:150:                         inner_color = inner_sums / (double)inner_area;
	vdivpd	%ymm5, %ymm0, %ymm10	# pretmp_346, inner_sums, inner_color.4_104
# is.cc:152:                         outer_color = outer_sums / (double)outer_area;
	vdivpd	%ymm4, %ymm9, %ymm11	# pretmp_347, outer_sums, outer_color.6_108
# is.cc:54:     return (a * b) + c;
	vmulpd	%ymm11, %ymm9, %ymm9	# outer_color.6_108, outer_sums, tmp283
# is.cc:54:     return (a * b) + c;
	vfmadd132pd	%ymm10, %ymm9, %ymm0	# inner_color.4_104, tmp283, _134
# is.cc:46:     return ((v[0] + v[1]) + (v[2]));
	vunpckhpd	%xmm0, %xmm0, %xmm12	# tmp285, tmp286
# is.cc:46:     return ((v[0] + v[1]) + (v[2]));
	vaddsd	%xmm12, %xmm0, %xmm9	# tmp286, tmp284, tmp288
# is.cc:46:     return ((v[0] + v[1]) + (v[2]));
	vextractf128	$0x1, %ymm0, %xmm0	# _134, tmp290
# is.cc:46:     return ((v[0] + v[1]) + (v[2]));
	vaddsd	%xmm0, %xmm9, %xmm9	# tmp289, tmp288, tmp291
# is.cc:169:                         total_error += pix_sum(vec_fmadd(outer_color, outer_sums, inner_errors));
	vaddsd	%xmm6, %xmm9, %xmm9	# tmp324, tmp291, total_error
# is.cc:171:                         if (total_error > local_min_error)
	vcomisd	%xmm3, %xmm9	# local_min_error, total_error
	ja	.L44	#,
# is.cc:112:                     for (int x = 0; x <= nx - w; x++)
	incl	%eax	# x
# is.cc:112:                     for (int x = 0; x <= nx - w; x++)
	addq	$32, %rcx	#, ivtmp.170
	cmpl	%edx, %eax	# ivtmp.195, x
	jg	.L45	#,
.L13:
# is.cc:129:                         inner_sums = sums[br];
	leaq	(%rcx,%r10), %rsi	#, tmp282
	vmovapd	(%rsi,%rdi), %ymm0	# MEM[(vector(4) double &)_447 + ivtmp.188_481 * 1], inner_sums
# is.cc:131:                         if (x != 0)
	testl	%eax, %eax	# x
	je	.L46	#,
# is.cc:119:                         const int bl = bot_row + left;
	leal	0(%r13,%rax), %esi	#, bl
# is.cc:133:                             inner_sums -= sums[bl];
	movslq	%esi, %rsi	# bl, bl
# is.cc:133:                             inner_sums -= sums[bl];
	salq	$5, %rsi	#, tmp300
	vsubpd	(%r8,%rsi), %ymm0, %ymm0	# *_143, inner_sums, inner_sums
# is.cc:117:                         const int tl = top_row + left;
	leal	(%r12,%rax), %esi	#, tl
# is.cc:143:                             inner_sums += sums[tl];
	movslq	%esi, %rsi	# tl, tl
# is.cc:143:                             inner_sums += sums[tl];
	salq	$5, %rsi	#, tmp305
# is.cc:138:                             inner_sums -= sums[tr];
	vsubpd	(%rcx), %ymm0, %ymm0	# MEM[(vector(4) double *)_454], inner_sums, inner_sums
# is.cc:143:                             inner_sums += sums[tl];
	vaddpd	(%r8,%rsi), %ymm0, %ymm0	# *_137, inner_sums, inner_sums
	jmp	.L24	#
	.p2align 4
	.p2align 3
.L44:
# is.cc:180:                                 local_result.inner[c] = inner_color[c];
	vpermpd	$233, %ymm10, %ymm2	#, inner_color.4_104, tmp293
# is.cc:181:                                 local_result.outer[c] = outer_color[c];
	vpermpd	$36, %ymm10, %ymm10	#, inner_color.4_104, tmp296
# is.cc:116:                         const int right = (x + w - 1);
	leal	(%rax,%r15), %r14d	#, local_result$x1
# is.cc:175:                             local_result.x0 = x;
	movl	%eax, 184(%rsp)	# x, %sfp
# is.cc:181:                                 local_result.outer[c] = outer_color[c];
	vblendpd	$8, %ymm10, %ymm11, %ymm1	#, tmp296, outer_color.6_108, tmp295
# is.cc:112:                     for (int x = 0; x <= nx - w; x++)
	incl	%eax	# x
# is.cc:180:                                 local_result.inner[c] = inner_color[c];
	vcvtpd2psx	%xmm2, %xmm2	# tmp294, vect_local_result_inner_1_14.150
# is.cc:176:                             local_result.y1 = y + h;
	movl	%r11d, 180(%rsp)	# _62, %sfp
# is.cc:181:                                 local_result.outer[c] = outer_color[c];
	vcvtpd2psy	%ymm1, %xmm1	# tmp295, vect__165.111
# is.cc:174:                             local_result.y0 = y;
	movl	%r9d, 188(%rsp)	# y, %sfp
# is.cc:173:                             local_min_error = total_error;
	vmovsd	%xmm9, %xmm9, %xmm3	# total_error, local_min_error
# is.cc:112:                     for (int x = 0; x <= nx - w; x++)
	addq	$32, %rcx	#, ivtmp.170
	cmpl	%edx, %eax	# ivtmp.195, x
	jle	.L13	#,
.L45:
	movl	176(%rsp), %esi	# %sfp, nx
.L23:
# is.cc:108:                 for (int y = 0; y <= ny - h; y++)
	movq	160(%rsp), %rax	# %sfp, tmp248
# is.cc:108:                 for (int y = 0; y <= ny - h; y++)
	incl	%r9d	# y
# is.cc:108:                 for (int y = 0; y <= ny - h; y++)
	addl	%esi, 128(%rsp)	# nx, %sfp
	addl	%esi, %ebx	# nx, ivtmp.181
	addq	%rax, %r10	# tmp248, ivtmp.187
	subq	%rax, %rdi	# tmp248, ivtmp.188
	movl	112(%rsp), %eax	# %sfp, ivtmp.213
	cmpl	%eax, %r9d	# ivtmp.213, y
	jne	.L18	#,
	movl	88(%rsp), %r12d	# %sfp, ivtmp.200
	movq	80(%rsp), %r11	# %sfp, ivtmp.201
.L11:
# is.cc:99:             for (int w = 1; w <= nx; w++)
	movl	168(%rsp), %eax	# %sfp, _39
	incq	%r15	# ivtmp.197
	decl	%edx	# ivtmp.195
	addq	$32, %r11	#, ivtmp.201
	addl	%eax, %r12d	# _39, ivtmp.200
	cmpl	%r15d, %esi	# ivtmp.197, nx
	jge	.L10	#,
	jmp	.L42	#
	.p2align 4
	.p2align 3
.L12:
	movq	104(%rsp), %rax	# %sfp, tmp314
# is.cc:119:                         const int bl = bot_row + left;
	leal	-1(%rbx), %r13d	#, tmp307
	leaq	(%rax,%rdi), %rcx	#, ivtmp.156
# is.cc:112:                     for (int x = 0; x <= nx - w; x++)
	xorl	%eax, %eax	# x
	jmp	.L17	#
	.p2align 4
	.p2align 3
.L15:
# is.cc:112:                     for (int x = 0; x <= nx - w; x++)
	incl	%eax	# x
# is.cc:112:                     for (int x = 0; x <= nx - w; x++)
	addq	$32, %rcx	#, ivtmp.156
	cmpl	%edx, %eax	# ivtmp.195, x
	jg	.L23	#,
.L17:
# is.cc:129:                         inner_sums = sums[br];
	vmovapd	(%rcx), %ymm9	# MEM[(vector(4) double &)_357], inner_sums
# is.cc:131:                         if (x != 0)
	testl	%eax, %eax	# x
	je	.L14	#,
# is.cc:119:                         const int bl = bot_row + left;
	leal	0(%r13,%rax), %r12d	#, bl
# is.cc:133:                             inner_sums -= sums[bl];
	movslq	%r12d, %r12	# bl, bl
# is.cc:133:                             inner_sums -= sums[bl];
	salq	$5, %r12	#, tmp266
	vsubpd	(%r8,%r12), %ymm9, %ymm9	# *_342, inner_sums, inner_sums
.L14:
# is.cc:147:                         outer_sums = image_totals - inner_sums;
	vsubpd	%ymm9, %ymm7, %ymm12	# inner_sums, image_totals, outer_sums
# is.cc:150:                         inner_color = inner_sums / (double)inner_area;
	vdivpd	%ymm5, %ymm9, %ymm10	# pretmp_346, inner_sums, inner_color.4_320
# is.cc:152:                         outer_color = outer_sums / (double)outer_area;
	vdivpd	%ymm4, %ymm12, %ymm11	# pretmp_347, outer_sums, outer_color.6_319
# is.cc:54:     return (a * b) + c;
	vmulpd	%ymm12, %ymm11, %ymm12	# outer_sums, outer_color.6_319, tmp268
# is.cc:54:     return (a * b) + c;
	vfmadd132pd	%ymm10, %ymm12, %ymm9	# inner_color.4_320, tmp268, _316
# is.cc:46:     return ((v[0] + v[1]) + (v[2]));
	vunpckhpd	%xmm9, %xmm9, %xmm0	# tmp270, tmp269
# is.cc:46:     return ((v[0] + v[1]) + (v[2]));
	vaddsd	%xmm9, %xmm0, %xmm0	# tmp271, tmp269, tmp273
# is.cc:46:     return ((v[0] + v[1]) + (v[2]));
	vextractf128	$0x1, %ymm9, %xmm9	# _316, tmp275
# is.cc:46:     return ((v[0] + v[1]) + (v[2]));
	vaddsd	%xmm9, %xmm0, %xmm0	# tmp274, tmp273, tmp276
# is.cc:169:                         total_error += pix_sum(vec_fmadd(outer_color, outer_sums, inner_errors));
	vaddsd	%xmm6, %xmm0, %xmm0	# tmp324, tmp276, total_error
# is.cc:171:                         if (total_error > local_min_error)
	vcomisd	%xmm3, %xmm0	# local_min_error, total_error
	jbe	.L15	#,
# is.cc:180:                                 local_result.inner[c] = inner_color[c];
	vpermpd	$233, %ymm10, %ymm2	#, inner_color.4_320, tmp278
# is.cc:181:                                 local_result.outer[c] = outer_color[c];
	vpermpd	$36, %ymm10, %ymm10	#, inner_color.4_320, tmp281
# is.cc:116:                         const int right = (x + w - 1);
	leal	(%rax,%r15), %r14d	#, local_result$x1
# is.cc:176:                             local_result.y1 = y + h;
	movl	%r11d, 180(%rsp)	# _62, %sfp
# is.cc:181:                                 local_result.outer[c] = outer_color[c];
	vblendpd	$8, %ymm10, %ymm11, %ymm1	#, tmp281, outer_color.6_319, tmp280
# is.cc:180:                                 local_result.inner[c] = inner_color[c];
	vcvtpd2psx	%xmm2, %xmm2	# tmp279, vect_local_result_inner_1_14.150
# is.cc:175:                             local_result.x0 = x;
	movl	%eax, 184(%rsp)	# x, %sfp
# is.cc:174:                             local_result.y0 = y;
	movl	$0, 188(%rsp)	#, %sfp
# is.cc:181:                                 local_result.outer[c] = outer_color[c];
	vcvtpd2psy	%ymm1, %xmm1	# tmp280, vect__165.111
# is.cc:173:                             local_min_error = total_error;
	vmovsd	%xmm0, %xmm0, %xmm3	# total_error, local_min_error
	jmp	.L15	#
.L27:
# is.cc:94:         double local_min_error = std::numeric_limits<double>::min();
	vmovsd	.LC0(%rip), %xmm3	#, local_min_error
	jmp	.L4	#
	.cfi_endproc
.LFE14110:
	.size	_Z7segmentiiPKf._omp_fn.0, .-_Z7segmentiiPKf._omp_fn.0
	.section	.text.unlikely,"ax",@progbits
.LCOLDB4:
	.text
.LHOTB4:
	.p2align 4
	.globl	_Z7segmentiiPKf
	.type	_Z7segmentiiPKf, @function
_Z7segmentiiPKf:
.LFB12214:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA12214
	endbr64	
	leaq	8(%rsp), %r10	#,
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp	#,
	pushq	-8(%r10)	#
	pushq	%rbp	#
	movq	%rsp, %rbp	#,
	.cfi_escape 0x10,0x6,0x2,0x76,0
	pushq	%r15	#
	pushq	%r14	#
	pushq	%r13	#
	pushq	%r12	#
	pushq	%r10	#
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx	#
	subq	$160, %rsp	#,
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
# is.cc:58: {
	movq	%rdi, -184(%rbp)	# tmp295, %sfp
# is.cc:59:     std::unique_ptr<f64x4[]> sums(new f64x4[ny * nx]);
	movq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], _1
	movq	%rax, -56(%rbp)	# _1, D.205486
	movl	%esi, %eax	# ny, _1
	imull	%edx, %eax	# nx, _1
	movslq	%eax, %rdi	# _1, _38
	movl	%eax, -188(%rbp)	# _1, %sfp
# is.cc:59:     std::unique_ptr<f64x4[]> sums(new f64x4[ny * nx]);
	movq	%rdi, %rax	# _38, tmp321
	shrq	$58, %rax	#, tmp321
	jne	.L48	#,
# is.cc:59:     std::unique_ptr<f64x4[]> sums(new f64x4[ny * nx]);
	salq	$5, %rdi	#, tmp208
	movl	%esi, %r12d	# tmp296, ny
# is.cc:59:     std::unique_ptr<f64x4[]> sums(new f64x4[ny * nx]);
	movl	$32, %esi	#,
	movl	%edx, %ebx	# tmp297, nx
	movq	%rcx, %r13	# tmp298, data
.LEHB0:
	call	_ZnamSt11align_val_t@PLT	#
.LEHE0:
# /usr/include/c++/13/bits/unique_ptr.h:176:       __uniq_ptr_impl(pointer __p) : _M_t() { _M_ptr() = __p; }
	movq	%rax, -152(%rbp)	# tmp299, MEM[(vector(4) double * &)&sums]
# is.cc:27:     Timer() : beg_(clock_::now()) {}
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT	#
	movq	%rax, -200(%rbp)	# tmp300, %sfp
# is.cc:66:     for (int y = 0; y < ny; y++)
	testl	%r12d, %r12d	# ny
	jle	.L56	#,
# /usr/include/c++/13/bits/unique_ptr.h:199:       pointer    _M_ptr() const noexcept { return std::get<0>(_M_t); }
	movq	-152(%rbp), %rdx	# MEM[(vector(4) double * const &)&sums], _45
	movq	%rdx, -176(%rbp)	# _45, %sfp
	testl	%ebx, %ebx	# nx
	jle	.L56	#,
	movslq	%ebx, %r14	# nx, _267
	movl	%ebx, %r11d	# nx, ivtmp.251
# is.cc:68:         f64x4 sum = zero_pix;
	vxorpd	%xmm0, %xmm0, %xmm0	# sum
	movq	%r13, %rax	# data, ivtmp.232
	leaq	(%r14,%r14,2), %r15	#, tmp219
	negl	%r11d	# ivtmp.251
	salq	$5, %r14	#, _281
# is.cc:78:             sums[x + nx * y] = prev + sum;
	vmovapd	%ymm0, %ymm4	#, tmp237
	salq	$2, %r15	#, tmp220
	leaq	0(%r13,%r15), %rsi	#, ivtmp.249
	.p2align 4
	.p2align 3
.L53:
# is.cc:73:                 double pix = data[c + 3 * x + 3 * nx * y];
	vxorpd	%xmm6, %xmm6, %xmm6	# tmp367
# is.cc:74:                 sum[c] += pix;
	vunpckhpd	%xmm0, %xmm0, %xmm3	# tmp223, tmp227
# is.cc:69:         for (int x = 0; x < nx; x++)
	addq	$12, %rax	#, ivtmp.232
	addq	$32, %rdx	#, ivtmp.233
# is.cc:73:                 double pix = data[c + 3 * x + 3 * nx * y];
	vcvtss2sd	-12(%rax), %xmm6, %xmm1	# MEM[(const float *)_200], tmp367, tmp313
	vcvtss2sd	-8(%rax), %xmm6, %xmm2	# MEM[(const float *)_200 + 4B], tmp368, tmp314
# is.cc:74:                 sum[c] += pix;
	vaddsd	%xmm1, %xmm0, %xmm1	# pix, tmp223, tmp225
	vaddsd	%xmm3, %xmm2, %xmm2	# tmp227, pix, _235
# is.cc:74:                 sum[c] += pix;
	vextractf128	$0x1, %ymm0, %xmm3	# sum, tmp233
# is.cc:74:                 sum[c] += pix;
	vextractf128	$0x1, %ymm0, %xmm0	# sum, tmp235
	vunpcklpd	%xmm2, %xmm1, %xmm1	# _235, tmp225, tmp230
# is.cc:73:                 double pix = data[c + 3 * x + 3 * nx * y];
	vcvtss2sd	-4(%rax), %xmm6, %xmm2	# MEM[(const float *)_200 + 8B], tmp369, tmp315
# is.cc:74:                 sum[c] += pix;
	vaddsd	%xmm3, %xmm2, %xmm2	# tmp232, pix, _225
	vmovsd	%xmm2, %xmm0, %xmm0	# _225, tmp235, tmp235
	vinsertf128	$0x1, %xmm0, %ymm1, %ymm0	# tmp235, tmp230, sum
# is.cc:78:             sums[x + nx * y] = prev + sum;
	vaddpd	%ymm4, %ymm0, %ymm1	# tmp237, sum, tmp236
# is.cc:78:             sums[x + nx * y] = prev + sum;
	vmovapd	%ymm1, -32(%rdx)	# tmp236, MEM[(vector(4) double &)_197]
# is.cc:69:         for (int x = 0; x < nx; x++)
	cmpq	%rsi, %rax	# ivtmp.249, ivtmp.232
	jne	.L53	#,
	xorl	%r8d, %r8d	# ivtmp.252
	xorl	%r9d, %r9d	# ivtmp.248
	xorl	%r10d, %r10d	# y
.L55:
# is.cc:66:     for (int y = 0; y < ny; y++)
	incl	%r10d	# y
# is.cc:66:     for (int y = 0; y < ny; y++)
	addl	%ebx, %r9d	# nx, ivtmp.248
	addq	%r15, %rsi	# tmp220, ivtmp.249
	addl	%ebx, %r11d	# nx, ivtmp.251
	addq	%r14, %r8	# _281, ivtmp.252
	cmpl	%r10d, %r12d	# y, ny
	je	.L83	#,
	movq	-176(%rbp), %rdx	# %sfp, _45
# is.cc:73:                 double pix = data[c + 3 * x + 3 * nx * y];
	leal	(%r9,%r9,2), %eax	#, tmp240
	movslq	%r11d, %rdi	# ivtmp.251, ivtmp.251
# is.cc:68:         f64x4 sum = zero_pix;
	vxorpd	%xmm0, %xmm0, %xmm0	# sum
	cltq
	salq	$5, %rdi	#, _67
	leaq	0(%r13,%rax,4), %rax	#, ivtmp.238
	addq	%rdi, %rdx	# _67, ivtmp.239
	.p2align 4
	.p2align 3
.L54:
# is.cc:73:                 double pix = data[c + 3 * x + 3 * nx * y];
	vxorpd	%xmm5, %xmm5, %xmm5	# tmp371
# is.cc:74:                 sum[c] += pix;
	vunpckhpd	%xmm0, %xmm0, %xmm3	# tmp246, tmp250
# is.cc:78:             sums[x + nx * y] = prev + sum;
	movq	%rdx, %rcx	# ivtmp.239, tmp259
# is.cc:69:         for (int x = 0; x < nx; x++)
	addq	$12, %rax	#, ivtmp.238
# is.cc:73:                 double pix = data[c + 3 * x + 3 * nx * y];
	vcvtss2sd	-12(%rax), %xmm5, %xmm1	# MEM[(const float *)_65], tmp371, tmp316
# is.cc:78:             sums[x + nx * y] = prev + sum;
	subq	%rdi, %rcx	# _67, tmp259
# is.cc:69:         for (int x = 0; x < nx; x++)
	addq	$32, %rdx	#, ivtmp.239
# is.cc:73:                 double pix = data[c + 3 * x + 3 * nx * y];
	vcvtss2sd	-8(%rax), %xmm5, %xmm2	# MEM[(const float *)_65 + 4B], tmp372, tmp317
# is.cc:74:                 sum[c] += pix;
	vaddsd	%xmm1, %xmm0, %xmm1	# pix, tmp246, tmp248
	vaddsd	%xmm3, %xmm2, %xmm2	# tmp250, pix, _127
# is.cc:74:                 sum[c] += pix;
	vextractf128	$0x1, %ymm0, %xmm3	# sum, tmp256
# is.cc:74:                 sum[c] += pix;
	vextractf128	$0x1, %ymm0, %xmm0	# sum, tmp258
	vunpcklpd	%xmm2, %xmm1, %xmm1	# _127, tmp248, tmp253
# is.cc:73:                 double pix = data[c + 3 * x + 3 * nx * y];
	vcvtss2sd	-4(%rax), %xmm5, %xmm2	# MEM[(const float *)_65 + 8B], tmp373, tmp318
# is.cc:74:                 sum[c] += pix;
	vaddsd	%xmm3, %xmm2, %xmm2	# tmp255, pix, _194
	vmovsd	%xmm2, %xmm0, %xmm0	# _194, tmp258, tmp258
	vinsertf128	$0x1, %xmm0, %ymm1, %ymm0	# tmp258, tmp253, sum
# is.cc:78:             sums[x + nx * y] = prev + sum;
	vaddpd	-32(%rdx), %ymm0, %ymm1	# MEM[(const f64x4 &)_13], sum, tmp260
# is.cc:78:             sums[x + nx * y] = prev + sum;
	vmovapd	%ymm1, (%rcx,%r8)	# tmp260, MEM[(vector(4) double &)_8 + ivtmp.252_279 * 1]
# is.cc:69:         for (int x = 0; x < nx; x++)
	cmpq	%rsi, %rax	# ivtmp.249, ivtmp.238
	jne	.L54	#,
	jmp	.L55	#
.L83:
	vzeroupper
.L56:
# is.cc:31:         return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT	#
# /usr/include/c++/13/bits/chrono.h:716: 	return __cd(__cd(__lhs).count() - __cd(__rhs).count());
	movq	-200(%rbp), %rdi	# %sfp, tmp210
# /usr/include/c++/13/bits/chrono.h:212: 	      static_cast<_CR>(__d.count()) / static_cast<_CR>(_CF::den)));
	vxorpd	%xmm7, %xmm7, %xmm7	# tmp364
# /usr/include/c++/13/ostream:223:       { return _M_insert(__f); }
	leaq	_ZSt4cout(%rip), %r15	#, tmp292
# /usr/include/c++/13/bits/chrono.h:716: 	return __cd(__cd(__lhs).count() - __cd(__rhs).count());
	subq	%rdi, %rax	# tmp210, tmp212
# /usr/include/c++/13/ostream:223:       { return _M_insert(__f); }
	movq	%r15, %rdi	# tmp292,
# /usr/include/c++/13/bits/chrono.h:212: 	      static_cast<_CR>(__d.count()) / static_cast<_CR>(_CF::den)));
	vcvtsi2sdq	%rax, %xmm7, %xmm0	# tmp212, tmp364, tmp312
# /usr/include/c++/13/bits/chrono.h:212: 	      static_cast<_CR>(__d.count()) / static_cast<_CR>(_CF::den)));
	vdivsd	.LC3(%rip), %xmm0, %xmm0	#, tmp213, _75
.LEHB1:
# /usr/include/c++/13/ostream:223:       { return _M_insert(__f); }
	call	_ZNSo9_M_insertIdEERSoT_@PLT	#
	movq	%rax, %r13	# tmp302, _71
# /usr/include/c++/13/ostream:736:     { return flush(__os.put(__os.widen('\n'))); }
	movq	(%rax), %rax	# MEM[(struct basic_ostream *)_71]._vptr.basic_ostream, MEM[(struct basic_ostream *)_71]._vptr.basic_ostream
	movq	-24(%rax), %rax	# MEM[(long int *)_105 + -24B], MEM[(long int *)_105 + -24B]
	movq	240(%r13,%rax), %r14	# MEM[(const struct __ctype_type * *)_108 + 240B], _113
# /usr/include/c++/13/bits/basic_ios.h:49:       if (!__f)
	testq	%r14, %r14	# _113
	je	.L86	#,
# /usr/include/c++/13/bits/locale_facets.h:882: 	if (_M_widen_ok)
	cmpb	$0, 56(%r14)	#, MEM[(const struct ctype *)_113]._M_widen_ok
	je	.L59	#,
# /usr/include/c++/13/ostream:736:     { return flush(__os.put(__os.widen('\n'))); }
	movsbl	67(%r14), %esi	# MEM[(const struct ctype *)_113]._M_widen[10], _254
.L60:
	movq	%r13, %rdi	# _71,
	call	_ZNSo3putEc@PLT	#
	vmovq	-184(%rbp), %xmm7	# %sfp, .result_ptr
	movq	%rax, %rdi	# tmp304, _111
	vmovd	%r12d, %xmm4	# ny, ny
	leaq	-152(%rbp), %rax	#, tmp267
	vpinsrd	$1, %ebx, %xmm4, %xmm4	# nx, ny, _221
	vmovq	%xmm4, %rbx	# _221, _221
	vpinsrq	$1, %rax, %xmm7, %xmm7	# tmp267, .result_ptr, _209
	vmovdqa	%xmm7, -176(%rbp)	# _209, %sfp
# /usr/include/c++/13/ostream:758:     { return __os.flush(); }
	call	_ZNSo5flushEv@PLT	#
# is.cc:28:     void reset() { beg_ = clock_::now(); }
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT	#
	xorl	%ecx, %ecx	#
	xorl	%edx, %edx	#
	leaq	_Z7segmentiiPKf._omp_fn.0(%rip), %rdi	#, tmp278
	movq	%rax, %r12	# tmp305, tmp269
# is.cc:86:     f64x4 image_totals = sums[(nx - 1) + (nx) * (ny - 1)];
	movl	-188(%rbp), %eax	# %sfp, _1
# is.cc:92: #pragma omp parallel
	vmovdqa	-176(%rbp), %xmm7	# %sfp, _209
	leaq	-144(%rbp), %rsi	#, tmp277
# is.cc:86:     f64x4 image_totals = sums[(nx - 1) + (nx) * (ny - 1)];
	decl	%eax	# _1
# is.cc:86:     f64x4 image_totals = sums[(nx - 1) + (nx) * (ny - 1)];
	cltq
# is.cc:86:     f64x4 image_totals = sums[(nx - 1) + (nx) * (ny - 1)];
	salq	$5, %rax	#, tmp274
	addq	-152(%rbp), %rax	# MEM[(vector(4) double * const &)&sums], tmp275
	vmovapd	(%rax), %ymm4	# *_83, tmp380
# is.cc:92: #pragma omp parallel
	movq	.LC0(%rip), %rax	#, tmp382
	vmovdqa	%xmm7, -112(%rbp)	# _209, MEM <vector(2) long unsigned int> [(void *)&.omp_data_o.11 + 32B]
	movq	%rbx, -88(%rbp)	# _221, MEM <vector(2) int> [(int *)&.omp_data_o.11 + 56B]
	movq	%rax, -96(%rbp)	# tmp382, .omp_data_o.11.final_error
# is.cc:86:     f64x4 image_totals = sums[(nx - 1) + (nx) * (ny - 1)];
	vmovapd	%ymm4, -144(%rbp)	# tmp380, .omp_data_o.11.image_totals
	vzeroupper
	call	GOMP_parallel@PLT	#
# is.cc:31:         return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT	#
# /usr/include/c++/13/bits/chrono.h:212: 	      static_cast<_CR>(__d.count()) / static_cast<_CR>(_CF::den)));
	vxorpd	%xmm7, %xmm7, %xmm7	# tmp383
# /usr/include/c++/13/bits/chrono.h:716: 	return __cd(__cd(__lhs).count() - __cd(__rhs).count());
	subq	%r12, %rax	# tmp269, tmp279
# /usr/include/c++/13/ostream:223:       { return _M_insert(__f); }
	movq	%r15, %rdi	# tmp292,
# /usr/include/c++/13/bits/chrono.h:212: 	      static_cast<_CR>(__d.count()) / static_cast<_CR>(_CF::den)));
	vcvtsi2sdq	%rax, %xmm7, %xmm0	# tmp279, tmp383, tmp319
# /usr/include/c++/13/bits/chrono.h:212: 	      static_cast<_CR>(__d.count()) / static_cast<_CR>(_CF::den)));
	vdivsd	.LC3(%rip), %xmm0, %xmm0	#, tmp280, _80
# /usr/include/c++/13/ostream:223:       { return _M_insert(__f); }
	call	_ZNSo9_M_insertIdEERSoT_@PLT	#
	movq	%rax, %rbx	# tmp307, _76
# /usr/include/c++/13/ostream:736:     { return flush(__os.put(__os.widen('\n'))); }
	movq	(%rax), %rax	# MEM[(struct basic_ostream *)_76]._vptr.basic_ostream, MEM[(struct basic_ostream *)_76]._vptr.basic_ostream
	movq	-24(%rax), %rax	# MEM[(long int *)_133 + -24B], MEM[(long int *)_133 + -24B]
	movq	240(%rbx,%rax), %r12	# MEM[(const struct __ctype_type * *)_136 + 240B], _141
# /usr/include/c++/13/bits/basic_ios.h:49:       if (!__f)
	testq	%r12, %r12	# _141
	je	.L87	#,
# /usr/include/c++/13/bits/locale_facets.h:882: 	if (_M_widen_ok)
	cmpb	$0, 56(%r12)	#, MEM[(const struct ctype *)_141]._M_widen_ok
	je	.L63	#,
# /usr/include/c++/13/ostream:736:     { return flush(__os.put(__os.widen('\n'))); }
	movsbl	67(%r12), %esi	# MEM[(const struct ctype *)_141]._M_widen[10], _257
.L64:
	movq	%rbx, %rdi	# _76,
	call	_ZNSo3putEc@PLT	#
	movq	%rax, %rdi	# tmp309, _139
# /usr/include/c++/13/ostream:758:     { return __os.flush(); }
	call	_ZNSo5flushEv@PLT	#
# /usr/include/c++/13/bits/unique_ptr.h:673: 	if (__ptr != nullptr)
	movq	-152(%rbp), %rdi	# MEM[(vector(4) double * &)&sums], _84
# /usr/include/c++/13/bits/unique_ptr.h:673: 	if (__ptr != nullptr)
	testq	%rdi, %rdi	# _84
	je	.L47	#,
# /usr/include/c++/13/bits/unique_ptr.h:140: 	  delete [] __ptr;
	movl	$32, %esi	#,
	call	_ZdaPvSt11align_val_t@PLT	#
.L47:
# is.cc:202: }
	movq	-56(%rbp), %rax	# D.205486, tmp326
	subq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp326
	jne	.L88	#,
	movq	-184(%rbp), %rax	# %sfp,
	addq	$160, %rsp	#,
	popq	%rbx	#
	popq	%r10	#
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12	#
	popq	%r13	#
	popq	%r14	#
	popq	%r15	#
	popq	%rbp	#
	leaq	-8(%r10), %rsp	#,
	.cfi_def_cfa 7, 8
	ret	
.L59:
	.cfi_restore_state
# /usr/include/c++/13/bits/locale_facets.h:884: 	this->_M_widen_init();
	movq	%r14, %rdi	# _113,
	call	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT	#
# /usr/include/c++/13/bits/locale_facets.h:885: 	return this->do_widen(__c);
	movq	(%r14), %rax	# MEM[(const struct ctype *)_113].D.55982._vptr.facet, MEM[(const struct ctype *)_113].D.55982._vptr.facet
	leaq	_ZNKSt5ctypeIcE8do_widenEc(%rip), %rdx	#, tmp265
	movl	$10, %esi	#, _254
	movq	48(%rax), %rax	# MEM[(int (*) () *)_123 + 48B], _124
	cmpq	%rdx, %rax	# tmp265, _124
	je	.L60	#,
	movq	%r14, %rdi	# _113,
	call	*%rax	# _124
# /usr/include/c++/13/ostream:736:     { return flush(__os.put(__os.widen('\n'))); }
	movsbl	%al, %esi	# tmp303, _254
	jmp	.L60	#
.L63:
# /usr/include/c++/13/bits/locale_facets.h:884: 	this->_M_widen_init();
	movq	%r12, %rdi	# _141,
	call	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT	#
# /usr/include/c++/13/bits/locale_facets.h:885: 	return this->do_widen(__c);
	movq	(%r12), %rax	# MEM[(const struct ctype *)_141].D.55982._vptr.facet, MEM[(const struct ctype *)_141].D.55982._vptr.facet
	leaq	_ZNKSt5ctypeIcE8do_widenEc(%rip), %rdx	#, tmp287
	movl	$10, %esi	#, _257
	movq	48(%rax), %rax	# MEM[(int (*) () *)_151 + 48B], _152
	cmpq	%rdx, %rax	# tmp287, _152
	je	.L64	#,
	movq	%r12, %rdi	# _141,
	call	*%rax	# _152
# /usr/include/c++/13/ostream:736:     { return flush(__os.put(__os.widen('\n'))); }
	movsbl	%al, %esi	# tmp308, _257
	jmp	.L64	#
.L88:
# is.cc:202: }
	call	__stack_chk_fail@PLT	#
.L87:
# /usr/include/c++/13/bits/basic_ios.h:50: 	__throw_bad_cast();
	movq	-56(%rbp), %rax	# D.205486, tmp324
	subq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp324
	jne	.L89	#,
	call	_ZSt16__throw_bad_castv@PLT	#
.L86:
	movq	-56(%rbp), %rax	# D.205486, tmp323
	subq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp323
	jne	.L90	#,
	call	_ZSt16__throw_bad_castv@PLT	#
.LEHE1:
.L89:
	call	__stack_chk_fail@PLT	#
.L90:
	call	__stack_chk_fail@PLT	#
.L72:
	endbr64	
# /usr/include/c++/13/bits/unique_ptr.h:673: 	if (__ptr != nullptr)
	movq	%rax, %rbx	# tmp310, tmp288
	jmp	.L66	#
	.globl	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
.LLSDA12214:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE12214-.LLSDACSB12214
.LLSDACSB12214:
	.uleb128 .LEHB0-.LFB12214
	.uleb128 .LEHE0-.LEHB0
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB1-.LFB12214
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L72-.LFB12214
	.uleb128 0
.LLSDACSE12214:
	.text
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC12214
	.type	_Z7segmentiiPKf.cold, @function
_Z7segmentiiPKf.cold:
.LFSB12214:
.L48:
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	.cfi_escape 0x10,0x6,0x2,0x76,0
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
# is.cc:59:     std::unique_ptr<f64x4[]> sums(new f64x4[ny * nx]);
	movq	-56(%rbp), %rax	# D.205486, tmp322
	subq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp322
	jne	.L91	#,
.LEHB2:
	call	__cxa_throw_bad_array_new_length@PLT	#
.L66:
# /usr/include/c++/13/bits/unique_ptr.h:673: 	if (__ptr != nullptr)
	movq	-152(%rbp), %rdi	# MEM[(vector(4) double * &)&sums], _85
# /usr/include/c++/13/bits/unique_ptr.h:673: 	if (__ptr != nullptr)
	testq	%rdi, %rdi	# _85
	jne	.L92	#,
	vzeroupper
.L67:
	movq	-56(%rbp), %rax	# D.205486, tmp325
	subq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp325
	jne	.L93	#,
	movq	%rbx, %rdi	# tmp288,
	call	_Unwind_Resume@PLT	#
.LEHE2:
.L91:
# is.cc:59:     std::unique_ptr<f64x4[]> sums(new f64x4[ny * nx]);
	call	__stack_chk_fail@PLT	#
.L92:
# /usr/include/c++/13/bits/unique_ptr.h:140: 	  delete [] __ptr;
	movl	$32, %esi	#,
	vzeroupper
	call	_ZdaPvSt11align_val_t@PLT	#
# /usr/include/c++/13/bits/unique_ptr.h:141: 	}
	jmp	.L67	#
.L93:
	call	__stack_chk_fail@PLT	#
	.cfi_endproc
.LFE12214:
	.section	.gcc_except_table
.LLSDAC12214:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC12214-.LLSDACSBC12214
.LLSDACSBC12214:
	.uleb128 .LEHB2-.LCOLDB4
	.uleb128 .LEHE2-.LEHB2
	.uleb128 0
	.uleb128 0
.LLSDACSEC12214:
	.section	.text.unlikely
	.text
	.size	_Z7segmentiiPKf, .-_Z7segmentiiPKf
	.section	.text.unlikely
	.size	_Z7segmentiiPKf.cold, .-_Z7segmentiiPKf.cold
.LCOLDE4:
	.text
.LHOTE4:
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC0:
	.long	0
	.long	1048576
	.align 8
.LC3:
	.long	0
	.long	1104006501
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
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
