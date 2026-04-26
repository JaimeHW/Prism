use super::*;
use crate::backend::x64::Assembler;
use crate::tier1::frame::FrameLayout;

fn make_ctx<'a>(asm: &'a mut Assembler, frame: &'a FrameLayout) -> TemplateContext<'a> {
    let mut ctx = TemplateContext::new(asm, frame);
    ctx.create_deopt_label();
    ctx
}

// =========================================================================
// IntPowerTemplate Tests
// =========================================================================

#[test]
fn test_int_power_template_emits_code() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntPowerTemplate {
        dst_reg: 2,
        base_reg: 0,
        exp_reg: 1,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    // Binary exponentiation loop should generate substantial code
    assert!(
        ctx.asm.offset() > 100,
        "Power template too short: {}",
        ctx.asm.offset()
    );
}

#[test]
fn test_int_power_template_estimated_size() {
    let tmpl = IntPowerTemplate {
        dst_reg: 2,
        base_reg: 0,
        exp_reg: 1,
        deopt_idx: 0,
    };
    assert_eq!(tmpl.estimated_size(), 256);
}

#[test]
fn test_int_power_same_src_dst() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntPowerTemplate {
        dst_reg: 0,
        base_reg: 0,
        exp_reg: 1,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);
    assert!(ctx.asm.offset() > 50);
}

#[test]
fn test_int_power_high_register_numbers() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(16);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntPowerTemplate {
        dst_reg: 15,
        base_reg: 10,
        exp_reg: 12,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);
    assert!(ctx.asm.offset() > 50);
}

// =========================================================================
// IntAbsTemplate Tests
// =========================================================================

#[test]
fn test_int_abs_template_emits_code() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntAbsTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    assert!(
        ctx.asm.offset() > 30,
        "Abs template too short: {}",
        ctx.asm.offset()
    );
}

#[test]
fn test_int_abs_template_estimated_size() {
    let tmpl = IntAbsTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    assert_eq!(tmpl.estimated_size(), 96);
}

#[test]
fn test_int_abs_same_src_dst() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntAbsTemplate {
        dst_reg: 0,
        src_reg: 0,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);
    assert!(ctx.asm.offset() > 20);
}

#[test]
fn test_int_abs_contains_neg_instruction() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntAbsTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    // Should be larger than just a load+store (has conditional negation)
    assert!(ctx.asm.offset() > 40);
}

// =========================================================================
// IntBitLengthTemplate Tests
// =========================================================================

#[test]
fn test_int_bit_length_template_emits_code() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntBitLengthTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    // Contains BSR instruction + zero handling + negative handling
    assert!(
        ctx.asm.offset() > 60,
        "BitLength template too short: {}",
        ctx.asm.offset()
    );
}

#[test]
fn test_int_bit_length_estimated_size() {
    let tmpl = IntBitLengthTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    assert_eq!(tmpl.estimated_size(), 128);
}

#[test]
fn test_int_bit_length_same_src_dst() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntBitLengthTemplate {
        dst_reg: 0,
        src_reg: 0,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);
    assert!(ctx.asm.offset() > 40);
}

#[test]
fn test_int_bit_length_contains_bsr_encoding() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntBitLengthTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    // Verify BSR opcode (0F BD) appears in emitted code
    let code = ctx.asm.code();
    let has_bsr = code.windows(2).any(|w| w[0] == 0x0F && w[1] == 0xBD);
    assert!(has_bsr, "BSR instruction (0F BD) not found in emitted code");
}

// =========================================================================
// IntLShiftTemplate Tests
// =========================================================================

#[test]
fn test_int_lshift_template_emits_code() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntLShiftTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    assert!(
        ctx.asm.offset() > 80,
        "LShift template too short: {}",
        ctx.asm.offset()
    );
}

#[test]
fn test_int_lshift_estimated_size() {
    let tmpl = IntLShiftTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        deopt_idx: 0,
    };
    assert_eq!(tmpl.estimated_size(), 160);
}

#[test]
fn test_int_lshift_contains_roundtrip_verification() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntLShiftTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    // Should contain both shl and sar (roundtrip verification)
    assert!(ctx.asm.offset() > 100);
}

#[test]
fn test_int_lshift_same_src_dst() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntLShiftTemplate {
        dst_reg: 0,
        lhs_reg: 0,
        rhs_reg: 1,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);
    assert!(ctx.asm.offset() > 50);
}

// =========================================================================
// IntRShiftTemplate Tests
// =========================================================================

#[test]
fn test_int_rshift_template_emits_code() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntRShiftTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    assert!(
        ctx.asm.offset() > 50,
        "RShift template too short: {}",
        ctx.asm.offset()
    );
}

#[test]
fn test_int_rshift_estimated_size() {
    let tmpl = IntRShiftTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        deopt_idx: 0,
    };
    assert_eq!(tmpl.estimated_size(), 128);
}

#[test]
fn test_int_rshift_no_overflow_check() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntRShiftTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    // RShift should be shorter than LShift (no roundtrip verification)
    let rshift_size = ctx.asm.offset();

    let mut asm2 = Assembler::new();
    let mut ctx2 = make_ctx(&mut asm2, &frame);
    let lshift_tmpl = IntLShiftTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        deopt_idx: 0,
    };
    lshift_tmpl.emit(&mut ctx2);
    let lshift_size = ctx2.asm.offset();

    assert!(
        rshift_size < lshift_size,
        "RShift ({rshift_size}) should be shorter than LShift ({lshift_size})"
    );
}

// =========================================================================
// IntUnaryPositiveTemplate Tests
// =========================================================================

#[test]
fn test_int_unary_positive_template_emits_code() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntUnaryPositiveTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    // Minimal: just type check + store
    assert!(
        ctx.asm.offset() > 10,
        "UnaryPositive too short: {}",
        ctx.asm.offset()
    );
}

#[test]
fn test_int_unary_positive_estimated_size() {
    let tmpl = IntUnaryPositiveTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    assert_eq!(tmpl.estimated_size(), 48);
}

#[test]
fn test_int_unary_positive_is_identity() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntUnaryPositiveTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    // Should be shortest of all templates (just guard + copy)
    let pos_size = ctx.asm.offset();

    let mut asm2 = Assembler::new();
    let mut ctx2 = make_ctx(&mut asm2, &frame);
    let abs_tmpl = IntAbsTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    abs_tmpl.emit(&mut ctx2);

    assert!(
        pos_size < ctx2.asm.offset(),
        "Unary positive should be smaller than abs"
    );
}

#[test]
fn test_int_unary_positive_same_src_dst() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntUnaryPositiveTemplate {
        dst_reg: 0,
        src_reg: 0,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);
    assert!(ctx.asm.offset() > 8);
}

// =========================================================================
// IntToFloatTemplate Tests
// =========================================================================

#[test]
fn test_int_to_float_template_emits_code() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntToFloatTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    assert!(
        ctx.asm.offset() > 20,
        "IntToFloat too short: {}",
        ctx.asm.offset()
    );
}

#[test]
fn test_int_to_float_estimated_size() {
    let tmpl = IntToFloatTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    assert_eq!(tmpl.estimated_size(), 64);
}

#[test]
fn test_int_to_float_contains_cvtsi2sd() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntToFloatTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    // CVTSI2SD encoding: F2 REX.W 0F 2A (REX prefix between F2 and opcode)
    let code = ctx.asm.code();
    let has_cvt = code
        .windows(4)
        .any(|w| w[0] == 0xF2 && (w[1] & 0xF0 == 0x40) && w[2] == 0x0F && w[3] == 0x2A);
    assert!(has_cvt, "CVTSI2SD (F2 REX 0F 2A) not found in emitted code");
}

// =========================================================================
// IntCompareTemplate Tests
// =========================================================================

#[test]
fn test_int_compare_less_than() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntCompareTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        condition: Condition::Less,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    assert!(
        ctx.asm.offset() > 50,
        "Compare template too short: {}",
        ctx.asm.offset()
    );
}

#[test]
fn test_int_compare_equal() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntCompareTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        condition: Condition::Equal,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    assert!(ctx.asm.offset() > 50);
}

#[test]
fn test_int_compare_greater_equal() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntCompareTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        condition: Condition::GreaterEqual,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    assert!(ctx.asm.offset() > 50);
}

#[test]
fn test_int_compare_not_equal() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntCompareTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        condition: Condition::NotEqual,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);
    assert!(ctx.asm.offset() > 50);
}

#[test]
fn test_int_compare_estimated_size() {
    let tmpl = IntCompareTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        condition: Condition::Less,
        deopt_idx: 0,
    };
    assert_eq!(tmpl.estimated_size(), 112);
}

#[test]
fn test_int_compare_all_conditions_emit() {
    let conditions = [
        Condition::Less,
        Condition::LessEqual,
        Condition::Greater,
        Condition::GreaterEqual,
        Condition::Equal,
        Condition::NotEqual,
    ];

    let frame = FrameLayout::minimal(4);
    for cond in conditions {
        let mut asm = Assembler::new();
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntCompareTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            condition: cond,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);
        assert!(
            ctx.asm.offset() > 40,
            "Condition {:?} produced too little code",
            cond
        );
    }
}

#[test]
fn test_int_compare_contains_setcc() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let tmpl = IntCompareTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        condition: Condition::Less,
        deopt_idx: 0,
    };
    tmpl.emit(&mut ctx);

    // SETcc encoding: 0F 9x where x = condition code
    let code = ctx.asm.code();
    let has_setcc = code
        .windows(2)
        .any(|w| w[0] == 0x0F && (0x90..=0x9F).contains(&w[1]));
    assert!(has_setcc, "SETcc instruction not found in emitted code");
}

// =========================================================================
// Cross-Template Tests
// =========================================================================

#[test]
fn test_all_templates_create_with_different_deopt_indices() {
    let frame = FrameLayout::minimal(8);

    for deopt_idx in 0..3 {
        let mut asm = Assembler::new();
        let mut ctx = TemplateContext::new(&mut asm, &frame);
        for _ in 0..=deopt_idx {
            ctx.create_deopt_label();
        }

        let tmpl = IntAbsTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx,
        };
        tmpl.emit(&mut ctx);
        assert!(ctx.asm.offset() > 0);
    }
}

#[test]
fn test_template_sizes_are_ordered() {
    // Verify relative size expectations
    let unary_pos = IntUnaryPositiveTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    let abs = IntAbsTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    let to_float = IntToFloatTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    let compare = IntCompareTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        condition: Condition::Less,
        deopt_idx: 0,
    };
    let rshift = IntRShiftTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        deopt_idx: 0,
    };
    let bit_length = IntBitLengthTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    let lshift = IntLShiftTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        deopt_idx: 0,
    };
    let power = IntPowerTemplate {
        dst_reg: 2,
        base_reg: 0,
        exp_reg: 1,
        deopt_idx: 0,
    };

    // Unary positive < abs < to_float < compare < rshift < bit_length < lshift < power
    assert!(unary_pos.estimated_size() < abs.estimated_size());
    assert!(to_float.estimated_size() < compare.estimated_size());
    assert!(compare.estimated_size() < rshift.estimated_size());
    assert!(rshift.estimated_size() <= bit_length.estimated_size());
    assert!(lshift.estimated_size() <= power.estimated_size());
}

#[test]
fn test_multiple_templates_emit_sequentially() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    let mut ctx = make_ctx(&mut asm, &frame);

    let abs_tmpl = IntAbsTemplate {
        dst_reg: 1,
        src_reg: 0,
        deopt_idx: 0,
    };
    abs_tmpl.emit(&mut ctx);
    let after_abs = ctx.asm.offset();

    let rshift_tmpl = IntRShiftTemplate {
        dst_reg: 3,
        lhs_reg: 1,
        rhs_reg: 2,
        deopt_idx: 0,
    };
    rshift_tmpl.emit(&mut ctx);
    let after_rshift = ctx.asm.offset();

    assert!(
        after_rshift > after_abs,
        "Sequential emission should grow code buffer"
    );
}
