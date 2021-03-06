// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section	.note.GNU-stack, "", @progbits
#endif

#include "regs.h"
#define ARG0 RUSTRT_ARG0_S
#define ARG1 RUSTRT_ARG1_S

        .text

/*
According to ABI documentation found at
http://www.x86-64.org/documentation.html
and Microsoft discussion at
http://msdn.microsoft.com/en-US/library/9z1stfyw%28v=VS.80%29.aspx.

BOTH CALLING CONVENTIONS

Callee save registers:
	R12--R15, RDI, RSI, RBX, RBP, RSP
        XMM0--XMM5

Caller save registers:
	RAX, RCX, RDX, R8--R11
        XMM6--XMM15
        Floating point stack

MAC/AMD CALLING CONVENTIONS

Integer arguments go in registers:
        rdi, rsi, rdx, rcx, r8, r9

User flags have no specified role and are not preserved
        across calls, with the exception of DF in %rFLAGS,
        which must be clear (set to "forward" direction)
        on function entry and return.

MICROSOFT CALLING CONVENTIONS

Return value: RAX

First four arguments:
        RCX, RDX, R8, R9
        XMM0, XMM1, XMM2, XMM3
*/

/*
        Stores current registers into arg0/RCX and restores
        registers found in arg1/RDX. This is used by our
	    implementation of getcontext.  Only saves/restores nonvolatile
        registers and the register used for the first argument.
        Volatile registers in general ought to be saved by the caller
        anyhow.
*/

#if defined(__APPLE__)
#define SWAP_REGISTERS _rust_swap_registers
#else
#define SWAP_REGISTERS rust_swap_registers
#endif

// swap_registers(registers_t *oregs, registers_t *regs)
.globl SWAP_REGISTERS
SWAP_REGISTERS:
        // n.b. when we enter, the return address is at the top of
        // the stack (i.e., 0(%RSP)) and the argument is in
        // RUSTRT_ARG0_S.  We
        // simply save all NV registers into oregs.
        // We then restore all NV registers from regs.  This restores
        // the old stack pointer, which should include the proper
        // return address. We can therefore just return normally to
        // jump back into the old code.

        // Save instruction pointer:
        pop %rax
        mov %rax, (RUSTRT_IP*8)(RUSTRT_ARG0_S)

        // Save non-volatile integer registers:
        //   (including RSP)
        mov %rbx, (RUSTRT_RBX*8)(ARG0)
        mov %rsp, (RUSTRT_RSP*8)(ARG0)
        mov %rbp, (RUSTRT_RBP*8)(ARG0)
        mov %r12, (RUSTRT_R12*8)(ARG0)
        mov %r13, (RUSTRT_R13*8)(ARG0)
        mov %r14, (RUSTRT_R14*8)(ARG0)
        mov %r15, (RUSTRT_R15*8)(ARG0)

#if defined(__MINGW32__) || defined(_WINDOWS)
        mov %rdi, (RUSTRT_RDI*8)(ARG0)
        mov %rsi, (RUSTRT_RSI*8)(ARG0)
#endif

        // Save 0th argument register:
        mov ARG0, (RUSTRT_ARG0*8)(ARG0)

        // Save non-volatile XMM registers:
#if defined(__MINGW32__) || defined(_WINDOWS)
        movapd %xmm6, (RUSTRT_XMM6*8)(ARG0)
        movapd %xmm7, (RUSTRT_XMM7*8)(ARG0)
        movapd %xmm8, (RUSTRT_XMM8*8)(ARG0)
        movapd %xmm9, (RUSTRT_XMM9*8)(ARG0)
        movapd %xmm10, (RUSTRT_XMM10*8)(ARG0)
        movapd %xmm11, (RUSTRT_XMM11*8)(ARG0)
        movapd %xmm12, (RUSTRT_XMM12*8)(ARG0)
        movapd %xmm13, (RUSTRT_XMM13*8)(ARG0)
        movapd %xmm14, (RUSTRT_XMM14*8)(ARG0)
        movapd %xmm15, (RUSTRT_XMM15*8)(ARG0)
#else
        movapd %xmm0, (RUSTRT_XMM0*8)(ARG0)
        movapd %xmm1, (RUSTRT_XMM1*8)(ARG0)
        movapd %xmm2, (RUSTRT_XMM2*8)(ARG0)
        movapd %xmm3, (RUSTRT_XMM3*8)(ARG0)
        movapd %xmm4, (RUSTRT_XMM4*8)(ARG0)
        movapd %xmm5, (RUSTRT_XMM5*8)(ARG0)
#endif

        // Restore non-volatile integer registers:
        //   (including RSP)
        mov (RUSTRT_RBX*8)(ARG1), %rbx
        mov (RUSTRT_RSP*8)(ARG1), %rsp
        mov (RUSTRT_RBP*8)(ARG1), %rbp
        mov (RUSTRT_R12*8)(ARG1), %r12
        mov (RUSTRT_R13*8)(ARG1), %r13
        mov (RUSTRT_R14*8)(ARG1), %r14
        mov (RUSTRT_R15*8)(ARG1), %r15

#if defined(__MINGW32__) || defined(_WINDOWS)
        mov (RUSTRT_RDI*8)(ARG1), %rdi
        mov (RUSTRT_RSI*8)(ARG1), %rsi
#endif

        // Restore 0th argument register:
        mov (RUSTRT_ARG0*8)(ARG1), ARG0

        // Restore non-volatile XMM registers:
#if defined(__MINGW32__) || defined(_WINDOWS)
        movapd (RUSTRT_XMM6*8)(ARG1), %xmm6
        movapd (RUSTRT_XMM7*8)(ARG1), %xmm7
        movapd (RUSTRT_XMM8*8)(ARG1), %xmm8
        movapd (RUSTRT_XMM9*8)(ARG1), %xmm9
        movapd (RUSTRT_XMM10*8)(ARG1), %xmm10
        movapd (RUSTRT_XMM11*8)(ARG1), %xmm11
        movapd (RUSTRT_XMM12*8)(ARG1), %xmm12
        movapd (RUSTRT_XMM13*8)(ARG1), %xmm13
        movapd (RUSTRT_XMM14*8)(ARG1), %xmm14
        movapd (RUSTRT_XMM15*8)(ARG1), %xmm15
#else
        movapd (RUSTRT_XMM0*8)(ARG1), %xmm0
        movapd (RUSTRT_XMM1*8)(ARG1), %xmm1
        movapd (RUSTRT_XMM2*8)(ARG1), %xmm2
        movapd (RUSTRT_XMM3*8)(ARG1), %xmm3
        movapd (RUSTRT_XMM4*8)(ARG1), %xmm4
        movapd (RUSTRT_XMM5*8)(ARG1), %xmm5
#endif

        // Jump to the instruction pointer
        // found in regs:
        jmp *(RUSTRT_IP*8)(ARG1)

/*
        Save current registers into arg0/RCX
        Only saves nonvolatile registers and
        the register used for the first argument.
        Volatile registers in general ought
        to be saved by the caller anyhow.
*/

#if defined(__APPLE__)
#define SAVE_REGISTERS _rust_save_registers
#else
#define SAVE_REGISTERS rust_save_registers
#endif

// save_registers(registers_t *regs)
.globl SAVE_REGISTERS
SAVE_REGISTERS:
        // n.b. when we enter, the return address is at the top of
        // the stack (i.e., 0(%RSP)) and the argument is in
        // RUSTRT_ARG0_S.  We
        // simply save all NV registers into oregs.
        // We then restore all NV registers from regs.  This restores
        // the old stack pointer, which should include the proper
        // return address. We can therefore just return normally to
        // jump back into the old code.

        // Save instruction pointer:
        pop %rax
        mov %rax, (RUSTRT_IP*8)(RUSTRT_ARG0_S)

        // Save non-volatile integer registers:
        //   (including RSP)
        mov %rbx, (RUSTRT_RBX*8)(ARG0)
        mov %rsp, (RUSTRT_RSP*8)(ARG0)
        mov %rbp, (RUSTRT_RBP*8)(ARG0)
        mov %r12, (RUSTRT_R12*8)(ARG0)
        mov %r13, (RUSTRT_R13*8)(ARG0)
        mov %r14, (RUSTRT_R14*8)(ARG0)
        mov %r15, (RUSTRT_R15*8)(ARG0)

#if defined(__MINGW32__) || defined(_WINDOWS)
        mov %rdi, (RUSTRT_RDI*8)(ARG0)
        mov %rsi, (RUSTRT_RSI*8)(ARG0)
#endif

        // Save 0th argument register:
        mov ARG0, (RUSTRT_ARG0*8)(ARG0)

        // Save non-volatile XMM registers:
#if defined(__MINGW32__) || defined(_WINDOWS)
        movapd %xmm6, (RUSTRT_XMM6*8)(ARG0)
        movapd %xmm7, (RUSTRT_XMM7*8)(ARG0)
        movapd %xmm8, (RUSTRT_XMM8*8)(ARG0)
        movapd %xmm9, (RUSTRT_XMM9*8)(ARG0)
        movapd %xmm10, (RUSTRT_XMM10*8)(ARG0)
        movapd %xmm11, (RUSTRT_XMM11*8)(ARG0)
        movapd %xmm12, (RUSTRT_XMM12*8)(ARG0)
        movapd %xmm13, (RUSTRT_XMM13*8)(ARG0)
        movapd %xmm14, (RUSTRT_XMM14*8)(ARG0)
        movapd %xmm15, (RUSTRT_XMM15*8)(ARG0)
#else
        movapd %xmm0, (RUSTRT_XMM0*8)(ARG0)
        movapd %xmm1, (RUSTRT_XMM1*8)(ARG0)
        movapd %xmm2, (RUSTRT_XMM2*8)(ARG0)
        movapd %xmm3, (RUSTRT_XMM3*8)(ARG0)
        movapd %xmm4, (RUSTRT_XMM4*8)(ARG0)
        movapd %xmm5, (RUSTRT_XMM5*8)(ARG0)
#endif

        // Jump to the instruction pointer
        // found in regs:
        jmp *(RUSTRT_IP*8)(ARG0)

#if defined(__APPLE__)
#define LOAD_REGISTERS _rust_load_registers
#else
#define LOAD_REGISTERS rust_load_registers
#endif

// load_registers(registers_t *regs)
.globl LOAD_REGISTERS
LOAD_REGISTERS:
    // Should be exactly the same as the code snippet above
    // Restore non-volatile integer registers:
    //   (including RSP)
    mov (RUSTRT_RBX*8)(ARG0), %rbx
    mov (RUSTRT_RSP*8)(ARG0), %rsp
    mov (RUSTRT_RBP*8)(ARG0), %rbp
    mov (RUSTRT_R12*8)(ARG0), %r12
    mov (RUSTRT_R13*8)(ARG0), %r13
    mov (RUSTRT_R14*8)(ARG0), %r14
    mov (RUSTRT_R15*8)(ARG0), %r15

#if defined(__MINGW32__) || defined(_WINDOWS)
    mov (RUSTRT_RDI*8)(ARG0), %rdi
    mov (RUSTRT_RSI*8)(ARG0), %rsi
#endif


    // Restore non-volatile XMM registers:
#if defined(__MINGW32__) || defined(_WINDOWS)
    movapd (RUSTRT_XMM6*8)(ARG0), %xmm6
    movapd (RUSTRT_XMM7*8)(ARG0), %xmm7
    movapd (RUSTRT_XMM8*8)(ARG0), %xmm8
    movapd (RUSTRT_XMM9*8)(ARG0), %xmm9
    movapd (RUSTRT_XMM10*8)(ARG0), %xmm10
    movapd (RUSTRT_XMM11*8)(ARG0), %xmm11
    movapd (RUSTRT_XMM12*8)(ARG0), %xmm12
    movapd (RUSTRT_XMM13*8)(ARG0), %xmm13
    movapd (RUSTRT_XMM14*8)(ARG0), %xmm14
    movapd (RUSTRT_XMM15*8)(ARG0), %xmm15
#else
    movapd (RUSTRT_XMM0*8)(ARG0), %xmm0
    movapd (RUSTRT_XMM1*8)(ARG0), %xmm1
    movapd (RUSTRT_XMM2*8)(ARG0), %xmm2
    movapd (RUSTRT_XMM3*8)(ARG0), %xmm3
    movapd (RUSTRT_XMM4*8)(ARG0), %xmm4
    movapd (RUSTRT_XMM5*8)(ARG0), %xmm5
#endif

    //  save ARG0
    mov ARG0, %rax
    // Restore 0th argument register:
    mov (RUSTRT_ARG0*8)(ARG0), ARG0
    // Jump to the instruction pointer
    // found in regs:
    jmp *(RUSTRT_IP*8)(%rax)


// This function below, rust_bootstrap_green_task, is used to initialize a green
// task. This code is the very first code that is run whenever a green task
// starts. The only assumptions that this code makes is that it has a register
// context previously set up by Context::new() and some values are in some
// special registers.
//
// In theory the register context could be set up and then the context switching
// would plop us directly into some 'extern "C" fn', but not all platforms have
// the argument registers saved throughout a context switch (linux doesn't save
// rdi/rsi, the first two argument registers). Instead of modifying all context
// switches, instead the initial data for starting a green thread is shoved into
// unrelated registers (r12/13, etc) which always need to be saved on context
// switches anyway.
//
// With this strategy we get the benefit of being able to pass a fair bit of
// contextual data from the start of a green task to its init function, as well
// as not hindering any context switches.
//
// If you alter this code in any way, you likely need to update
// src/libgreen/context.rs as well.

#if defined(__APPLE__)
#define BOOTSTRAP _rust_bootstrap_green_task
#else
#define BOOTSTRAP rust_bootstrap_green_task
#endif
.globl BOOTSTRAP
BOOTSTRAP:
	mov %r12, RUSTRT_ARG0_S
	mov %r13, RUSTRT_ARG1_S
// no need for the second argument
//    mov %r14, RUSTRT_ARG2_S
	jmpq *%r14
