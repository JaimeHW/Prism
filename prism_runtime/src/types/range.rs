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
    if step > 0 {
        if stop <= start {
            0
        } else {
            ((stop - start - 1) / step + 1) as usize
        }
    } else if stop >= start {
        0
    } else {
        ((start - stop - 1) / (-step) + 1) as usize
    }
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

    /// Return the stop value when it fits in `i64`.
    #[inline]
    pub fn stop_i64(&self) -> Option<i64> {
        match &self.repr {
            RangeRepr::Small { stop, .. } => Some(*stop),
            RangeRepr::Big { stop, .. } => stop.to_i64(),
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

    /// Get the number of elements when it fits in `usize`.
    #[inline]
    pub fn try_len(&self) -> Option<usize> {
        match &self.repr {
            RangeRepr::Small { start, stop, step } => Some(small_len(*start, *stop, *step)),
            RangeRepr::Big { start, stop, step } => big_len(start, stop, step).to_usize(),
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
        match &self.repr {
            RangeRepr::Small { start, stop, step } => {
                let len = small_len(*start, *stop, *step) as i64;
                let idx = if index < 0 { len + index } else { index };
                if idx < 0 || idx >= len {
                    return None;
                }
                Some(bigint_to_value(
                    BigInt::from(*start) + BigInt::from(idx) * BigInt::from(*step),
                ))
            }
            RangeRepr::Big { start, stop, step } => {
                let idx = if index < 0 {
                    let len = self.try_len()?.to_i64()?;
                    len + index
                } else {
                    index
                };
                if idx < 0 {
                    return None;
                }

                let value = start + BigInt::from(idx) * step;
                if step.is_positive() {
                    if value < *start || value >= *stop {
                        return None;
                    }
                } else if value > *start || value <= *stop {
                    return None;
                }
                Some(bigint_to_value(value))
            }
        }
    }

    /// Get the element at index when it fits in `i64`.
    #[inline]
    pub fn get(&self, index: i64) -> Option<i64> {
        self.get_value(index).and_then(value_to_i64)
    }

    /// Check if a small integer is in the range.
    #[inline]
    pub fn contains(&self, value: i64) -> bool {
        match &self.repr {
            RangeRepr::Small { start, stop, step } => {
                if *step > 0 {
                    if value < *start || value >= *stop {
                        return false;
                    }
                } else if value > *start || value <= *stop {
                    return false;
                }
                (value - *start) % *step == 0
            }
            RangeRepr::Big { start, stop, step } => {
                let value = BigInt::from(value);
                if step.is_positive() {
                    if value < *start || value >= *stop {
                        return false;
                    }
                } else if value > *start || value <= *stop {
                    return false;
                }
                ((value - start) % step).is_zero()
            }
        }
    }

    /// Create an iterator over this range.
    #[inline]
    pub fn iter(&self) -> RangeIterator {
        match &self.repr {
            RangeRepr::Small { start, stop, step } => RangeIterator::Small {
                current: *start,
                stop: *stop,
                step: *step,
            },
            RangeRepr::Big { start, stop, step } => RangeIterator::Big {
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

        match &self.repr {
            RangeRepr::Small {
                start,
                stop: _,
                step,
            } => {
                let len = small_len(*start, self.stop_i64().unwrap(), *step);
                let last = self.get((len - 1) as i64).unwrap();
                RangeObject::new(last, start - step, -step)
            }
            RangeRepr::Big {
                start,
                stop: _,
                step,
            } => {
                let len = big_len(
                    start,
                    match &self.repr {
                        RangeRepr::Big { stop, .. } => stop,
                        RangeRepr::Small { .. } => unreachable!(),
                    },
                    step,
                );
                let last = start + (&len - BigInt::one()) * step;
                RangeObject::from_bigints(last, start - step, -step.clone())
            }
        }
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
        match (self.try_len(), other.try_len()) {
            (Some(a), Some(b)) if a != b => return false,
            _ => {}
        }

        if self.is_empty() && other.is_empty() {
            return true;
        }

        self.iter().next() == other.iter().next()
            && self.repr_step_string() == other.repr_step_string()
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

impl RangeObject {
    #[inline]
    fn repr_step_string(&self) -> String {
        match &self.repr {
            RangeRepr::Small { step, .. } => step.to_string(),
            RangeRepr::Big { step, .. } => step.to_string(),
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
        current: i64,
        stop: i64,
        step: i64,
    },
    Big {
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
                current,
                stop,
                step,
            } => Some(small_len(*current, *stop, *step)),
            Self::Big {
                current,
                stop,
                step,
            } => big_len(current, stop, step).to_usize(),
        }
    }
}

impl Iterator for RangeIterator {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Small {
                current,
                stop,
                step,
            } => {
                if (*step > 0 && *current >= *stop) || (*step < 0 && *current <= *stop) {
                    return None;
                }
                let value = *current;
                *current += *step;
                Some(bigint_to_value(BigInt::from(value)))
            }
            Self::Big {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_i64(range: &RangeObject) -> Vec<i64> {
        range
            .iter()
            .map(|value| value_to_i64(value).expect("range item should fit in i64"))
            .collect()
    }

    #[test]
    fn test_range_from_stop() {
        let r = RangeObject::from_stop(5);
        assert_eq!(r.start_i64(), Some(0));
        assert_eq!(r.stop_i64(), Some(5));
        assert_eq!(r.step_i64(), Some(1));
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn test_range_from_start_stop() {
        let r = RangeObject::from_start_stop(2, 8);
        assert_eq!(r.start_i64(), Some(2));
        assert_eq!(r.stop_i64(), Some(8));
        assert_eq!(r.len(), 6);
    }

    #[test]
    fn test_range_with_step() {
        let r = RangeObject::new(0, 10, 2);
        assert_eq!(r.len(), 5);
        assert_eq!(collect_i64(&r), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_range_negative_step() {
        let r = RangeObject::new(10, 0, -1);
        assert_eq!(r.len(), 10);
        assert_eq!(collect_i64(&r), vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_empty_range() {
        let r = RangeObject::from_start_stop(5, 5);
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert!(r.iter().next().is_none());
    }

    #[test]
    fn test_range_get_and_contains() {
        let r = RangeObject::new(10, 20, 2);
        assert_eq!(r.get(0), Some(10));
        assert_eq!(r.get(4), Some(18));
        assert_eq!(r.get(-1), Some(18));
        assert_eq!(r.get(5), None);
        assert!(r.contains(12));
        assert!(!r.contains(13));
    }

    #[test]
    fn test_range_reverse() {
        let r = RangeObject::new(1, 6, 1);
        let reversed = r.reverse();
        assert_eq!(collect_i64(&reversed), vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_big_range_iterates_lazily() {
        let stop = BigInt::from(1_u8) << 1000_u32;
        let range = RangeObject::from_bigints(BigInt::zero(), stop, BigInt::one());

        let mut iter = range.iter();
        assert_eq!(value_to_i64(iter.next().unwrap()), Some(0));
        assert_eq!(value_to_i64(iter.next().unwrap()), Some(1));
        assert_eq!(value_to_i64(iter.next().unwrap()), Some(2));
    }

    #[test]
    fn test_big_range_try_len_reports_overflow() {
        let stop = BigInt::from(1_u8) << 1000_u32;
        let range = RangeObject::from_bigints(BigInt::zero(), stop, BigInt::one());
        assert_eq!(range.try_len(), None);
        assert_eq!(range.len(), usize::MAX);
    }

    #[test]
    fn test_big_range_display() {
        let stop = BigInt::from(1_u8) << 80_u32;
        let range = RangeObject::from_bigints(BigInt::zero(), stop.clone(), BigInt::one());
        assert_eq!(range.to_string(), format!("range(0, {stop})"));
    }
}
