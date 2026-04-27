//! Python range object implementation.
//!
//! Prism keeps the fast-path compact `i64` representation for ordinary ranges
//! while transparently promoting to bigint-backed bounds when Python semantics
//! require arbitrary-precision integers.

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use crate::types::int::{bigint_to_value, value_to_i64};
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use prism_core::Value;
use std::fmt;

#[derive(Clone, PartialEq, Eq)]
enum RangeRepr {
    Small {
        start: i64,
        stop: i64,
        step: i64,
    },
    Big {
        start: BigInt,
        stop: BigInt,
        step: BigInt,
    },
}

#[inline]
fn small_len(start: i64, stop: i64, step: i64) -> usize {
    let len = if step > 0 {
        if stop <= start {
            return 0;
        }
        let distance = i128::from(stop) - i128::from(start) - 1;
        distance / i128::from(step) + 1
    } else {
        if stop >= start {
            return 0;
        }
        let distance = i128::from(start) - i128::from(stop) - 1;
        distance / -i128::from(step) + 1
    };

    usize::try_from(len).unwrap_or(usize::MAX)
}

#[inline]
fn big_is_empty(start: &BigInt, stop: &BigInt, step: &BigInt) -> bool {
    if step.is_positive() {
        stop <= start
    } else {
        stop >= start
    }
}

fn big_len(start: &BigInt, stop: &BigInt, step: &BigInt) -> BigInt {
    if big_is_empty(start, stop, step) {
        return BigInt::zero();
    }

    if step.is_positive() {
        ((stop - start - BigInt::one()) / step) + BigInt::one()
    } else {
        ((start - stop - BigInt::one()) / (-step)) + BigInt::one()
    }
}

/// Python range object.
#[repr(C)]
pub struct RangeObject {
    /// Object header.
    pub header: ObjectHeader,
    repr: RangeRepr,
}

impl RangeObject {
    /// Create a new range with start, stop, and step.
    #[inline]
    pub fn new(start: i64, stop: i64, step: i64) -> Self {
        assert!(step != 0, "range() arg 3 must not be zero");
        Self {
            header: ObjectHeader::new(TypeId::RANGE),
            repr: RangeRepr::Small { start, stop, step },
        }
    }

    /// Create a range with arbitrary-precision bounds.
    #[inline]
    pub fn from_bigints(start: BigInt, stop: BigInt, step: BigInt) -> Self {
        assert!(!step.is_zero(), "range() arg 3 must not be zero");

        match (start.to_i64(), stop.to_i64(), step.to_i64()) {
            (Some(start), Some(stop), Some(step)) => Self::new(start, stop, step),
            _ => Self {
                header: ObjectHeader::new(TypeId::RANGE),
                repr: RangeRepr::Big { start, stop, step },
            },
        }
    }

    /// Create a range from 0 to stop with step 1.
    #[inline]
    pub fn from_stop(stop: i64) -> Self {
        Self::new(0, stop, 1)
    }

    /// Create a range from start to stop with step 1.
    #[inline]
    pub fn from_start_stop(start: i64, stop: i64) -> Self {
        Self::new(start, stop, 1)
    }

    /// Returns true when the range uses compact `i64` bounds.
    #[inline]
    pub fn is_small(&self) -> bool {
        matches!(self.repr, RangeRepr::Small { .. })
    }

    /// Return the start value when it fits in `i64`.
    #[inline]
    pub fn start_i64(&self) -> Option<i64> {
        match &self.repr {
            RangeRepr::Small { start, .. } => Some(*start),
            RangeRepr::Big { start, .. } => start.to_i64(),
        }
    }

    /// Return the start value as an arbitrary-precision integer.
    #[inline]
    pub fn start_bigint(&self) -> BigInt {
        match &self.repr {
            RangeRepr::Small { start, .. } => BigInt::from(*start),
            RangeRepr::Big { start, .. } => start.clone(),
        }
    }

    /// Return the stop value when it fits in `i64`.
    #[inline]
    pub fn stop_i64(&self) -> Option<i64> {
        match &self.repr {
            RangeRepr::Small { stop, .. } => Some(*stop),
            RangeRepr::Big { stop, .. } => stop.to_i64(),
        }
    }

    /// Return the stop value as an arbitrary-precision integer.
    #[inline]
    pub fn stop_bigint(&self) -> BigInt {
        match &self.repr {
            RangeRepr::Small { stop, .. } => BigInt::from(*stop),
            RangeRepr::Big { stop, .. } => stop.clone(),
        }
    }

    /// Return the step value when it fits in `i64`.
    #[inline]
    pub fn step_i64(&self) -> Option<i64> {
        match &self.repr {
            RangeRepr::Small { step, .. } => Some(*step),
            RangeRepr::Big { step, .. } => step.to_i64(),
        }
    }

    /// Return the step value as an arbitrary-precision integer.
    #[inline]
    pub fn step_bigint(&self) -> BigInt {
        match &self.repr {
            RangeRepr::Small { step, .. } => BigInt::from(*step),
            RangeRepr::Big { step, .. } => step.clone(),
        }
    }

    /// Get the number of elements when it fits in `usize`.
    #[inline]
    pub fn try_len(&self) -> Option<usize> {
        match &self.repr {
            RangeRepr::Small { start, stop, step } => Some(small_len(*start, *stop, *step)),
            RangeRepr::Big { start, stop, step } => big_len(start, stop, step).to_usize(),
        }
    }

    /// Get the number of elements as an arbitrary-precision integer.
    #[inline]
    pub fn len_bigint(&self) -> BigInt {
        match &self.repr {
            RangeRepr::Small { start, stop, step } => BigInt::from(small_len(*start, *stop, *step)),
            RangeRepr::Big { start, stop, step } => big_len(start, stop, step),
        }
    }

    /// Get the number of elements in the range.
    ///
    /// Big ranges that exceed `usize` saturate to `usize::MAX`; callers that
    /// need overflow detection should use [`Self::try_len`].
    #[inline]
    pub fn len(&self) -> usize {
        self.try_len().unwrap_or(usize::MAX)
    }

    /// Check if the range is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        match &self.repr {
            RangeRepr::Small { start, stop, step } => small_len(*start, *stop, *step) == 0,
            RangeRepr::Big { start, stop, step } => big_is_empty(start, stop, step),
        }
    }

    /// Get the element at index as a Python value.
    pub fn get_value(&self, index: i64) -> Option<Value> {
        self.get_value_bigint(&BigInt::from(index))
    }

    /// Get the element at an arbitrary-precision index as a Python value.
    pub fn get_value_bigint(&self, index: &BigInt) -> Option<Value> {
        match &self.repr {
            RangeRepr::Small { start, stop, step } => {
                let len = BigInt::from(small_len(*start, *stop, *step));
                let mut idx = index.clone();
                if idx.is_negative() {
                    idx += &len;
                }
                if idx.is_negative() || idx >= len {
                    return None;
                }
                Some(bigint_to_value(
                    BigInt::from(*start) + idx * BigInt::from(*step),
                ))
            }
            RangeRepr::Big { start, stop, step } => {
                let len = big_len(start, stop, step);
                let mut idx = index.clone();
                if idx.is_negative() {
                    idx += &len;
                }
                if idx.is_negative() || idx >= len {
                    return None;
                }

                let value = start + idx * step;
                Some(bigint_to_value(value))
            }
        }
    }

    /// Compute the value at a non-normalized integer index.
    ///
    /// Callers are responsible for ensuring the index is semantically valid for
    /// their operation. This is used by range slicing, where CPython preserves
    /// arithmetic range structure without materializing elements.
    #[inline]
    pub fn value_at_index_bigint(&self, index: &BigInt) -> BigInt {
        self.start_bigint() + index * self.step_bigint()
    }

    /// Get the element at index when it fits in `i64`.
    #[inline]
    pub fn get(&self, index: i64) -> Option<i64> {
        self.get_value(index).and_then(value_to_i64)
    }

    /// Check if a small integer is in the range.
    #[inline]
    pub fn contains(&self, value: i64) -> bool {
        self.contains_bigint(&BigInt::from(value))
    }

    /// Check if an arbitrary-precision integer is in the range.
    #[inline]
    pub fn contains_bigint(&self, value: &BigInt) -> bool {
        match &self.repr {
            RangeRepr::Small { start, stop, step } => {
                let start = BigInt::from(*start);
                let stop = BigInt::from(*stop);
                let step = BigInt::from(*step);
                if step.is_positive() {
                    if value < &start || value >= &stop {
                        return false;
                    }
                } else if value > &start || value <= &stop {
                    return false;
                }
                ((value - start) % step).is_zero()
            }
            RangeRepr::Big { start, stop, step } => {
                if step.is_positive() {
                    if value < start || value >= stop {
                        return false;
                    }
                } else if value > start || value <= stop {
                    return false;
                }
                ((value - start) % step).is_zero()
            }
        }
    }

    /// Return the position of an arbitrary-precision integer in the range.
    #[inline]
    pub fn index_of_bigint(&self, value: &BigInt) -> Option<BigInt> {
        if !self.contains_bigint(value) {
            return None;
        }

        Some((value - self.start_bigint()) / self.step_bigint())
    }

    /// Create an iterator over this range.
    #[inline]
    pub fn iter(&self) -> RangeIterator {
        match &self.repr {
            RangeRepr::Small { start, stop, step } => RangeIterator::Small {
                start: *start,
                current: *start,
                stop: *stop,
                step: *step,
            },
            RangeRepr::Big { start, stop, step } => RangeIterator::Big {
                start: start.clone(),
                current: start.clone(),
                stop: stop.clone(),
                step: step.clone(),
            },
        }
    }

    /// Get the first element when it fits in `i64`.
    #[inline]
    pub fn first(&self) -> Option<i64> {
        if self.is_empty() { None } else { self.get(0) }
    }

    /// Get the last element when it fits in `i64`.
    #[inline]
    pub fn last(&self) -> Option<i64> {
        let len = self.try_len()?;
        if len == 0 {
            None
        } else {
            self.get((len - 1) as i64)
        }
    }

    /// Reverse the range.
    pub fn reverse(&self) -> RangeObject {
        if self.is_empty() {
            return RangeObject::new(0, 0, 1);
        }

        let step = self.step_bigint();
        let len = self.len_bigint();
        let last = self.value_at_index_bigint(&(len - BigInt::one()));
        RangeObject::from_bigints(last, self.start_bigint() - &step, -step)
    }

    /// Materialize the range into a vector of Python values.
    pub fn to_vec(&self) -> Vec<Value> {
        self.iter().collect()
    }
}

impl Clone for RangeObject {
    fn clone(&self) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::RANGE),
            repr: self.repr.clone(),
        }
    }
}

impl PartialEq for RangeObject {
    fn eq(&self, other: &Self) -> bool {
        let self_len = self.len_bigint();
        let other_len = other.len_bigint();
        if self_len != other_len {
            return false;
        }
        if self_len.is_zero() {
            return true;
        }

        if self.start_bigint() != other.start_bigint() {
            return false;
        }

        self_len == BigInt::one() || self.step_bigint() == other.step_bigint()
    }
}

impl Eq for RangeObject {}

impl fmt::Debug for RangeObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for RangeObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.repr {
            RangeRepr::Small { start, stop, step } => {
                if *step == 1 {
                    write!(f, "range({}, {})", start, stop)
                } else {
                    write!(f, "range({}, {}, {})", start, stop, step)
                }
            }
            RangeRepr::Big { start, stop, step } => {
                if step == &BigInt::one() {
                    write!(f, "range({}, {})", start, stop)
                } else {
                    write!(f, "range({}, {}, {})", start, stop, step)
                }
            }
        }
    }
}

impl PyObject for RangeObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

/// Iterator over a range.
#[derive(Clone, Debug)]
pub enum RangeIterator {
    Small {
        start: i64,
        current: i64,
        stop: i64,
        step: i64,
    },
    Big {
        start: BigInt,
        current: BigInt,
        stop: BigInt,
        step: BigInt,
    },
}

impl RangeIterator {
    /// Remaining element count when it fits in `usize`.
    #[inline]
    pub fn remaining_len(&self) -> Option<usize> {
        match self {
            Self::Small {
                start: _,
                current,
                stop,
                step,
            } => Some(small_len(*current, *stop, *step)),
            Self::Big {
                start: _,
                current,
                stop,
                step,
            } => big_len(current, stop, step).to_usize(),
        }
    }

    /// Return the original range bounds represented by this iterator.
    #[inline]
    pub fn bounds_bigint(&self) -> (BigInt, BigInt, BigInt) {
        match self {
            Self::Small {
                start, stop, step, ..
            } => (
                BigInt::from(*start),
                BigInt::from(*stop),
                BigInt::from(*step),
            ),
            Self::Big {
                start, stop, step, ..
            } => (start.clone(), stop.clone(), step.clone()),
        }
    }

    /// Return the current zero-based iterator position.
    #[inline]
    pub fn state_bigint(&self) -> BigInt {
        match self {
            Self::Small {
                start,
                current,
                stop,
                step,
            } => {
                if (*step > 0 && *current >= *stop) || (*step < 0 && *current <= *stop) {
                    BigInt::from(small_len(*start, *stop, *step))
                } else {
                    (BigInt::from(*current) - BigInt::from(*start)) / BigInt::from(*step)
                }
            }
            Self::Big {
                start,
                current,
                stop,
                step,
            } => {
                if (step.is_positive() && current >= stop)
                    || (step.is_negative() && current <= stop)
                {
                    big_len(start, stop, step)
                } else {
                    (current - start) / step
                }
            }
        }
    }

    /// Restore an absolute iterator position and report whether it is exhausted.
    pub fn set_state_bigint(&mut self, raw_state: &BigInt) -> bool {
        let state = if raw_state.is_negative() {
            BigInt::zero()
        } else {
            raw_state.clone()
        };

        match self {
            Self::Small {
                start,
                current,
                stop,
                step,
            } => {
                let len = BigInt::from(small_len(*start, *stop, *step));
                if state >= len {
                    *current = *stop;
                    return true;
                }

                let next = BigInt::from(*start) + state * BigInt::from(*step);
                *current = next
                    .to_i64()
                    .expect("valid compact range element should fit in i64");
                false
            }
            Self::Big {
                start,
                current,
                stop,
                step,
            } => {
                let len = big_len(start, stop, step);
                if state >= len {
                    *current = stop.clone();
                    return true;
                }

                *current = start.clone() + state * step.clone();
                false
            }
        }
    }
}

impl Iterator for RangeIterator {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Small {
                start: _,
                current,
                stop,
                step,
            } => {
                if (*step > 0 && *current >= *stop) || (*step < 0 && *current <= *stop) {
                    return None;
                }
                let value = *current;
                match current.checked_add(*step) {
                    Some(next) => *current = next,
                    None => *current = *stop,
                }
                Some(bigint_to_value(BigInt::from(value)))
            }
            Self::Big {
                start: _,
                current,
                stop,
                step,
            } => {
                if (step.is_positive() && *current >= *stop)
                    || (step.is_negative() && *current <= *stop)
                {
                    return None;
                }
                let value = current.clone();
                *current += step.clone();
                Some(bigint_to_value(value))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.remaining_len() {
            Some(len) => (len, Some(len)),
            None => (0, None),
        }
    }
}

impl IntoIterator for &RangeObject {
    type Item = Value;
    type IntoIter = RangeIterator;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
