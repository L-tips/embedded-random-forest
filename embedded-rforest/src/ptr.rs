use core::fmt;
use zerocopy::{byteorder::little_endian::U16, FromBytes, Immutable, IntoBytes, KnownLayout};

/// A specialized relative pointer for use with optimized trees.
///
/// It contains an `u32`, and can hold up to 31 bits of data. The data is
/// encoded in the follwing form:
///
/// * If the first bit is 1, the next node in the tree is a leaf.
/// * If the first bit is 0, the next node in the tree is a branch.
#[repr(transparent)]
#[derive(Clone, Copy, IntoBytes, KnownLayout, Immutable, FromBytes)]
pub struct NodePointer(U16);

impl NodePointer {
    pub fn new_ptr(ptr: u16) -> Self {
        Self(U16::new(ptr))
    }

    // pub fn new_f32(float: f32) -> Self {
    //     let float = F32::new(float);
    //     Self(U16::from_bytes(float.to_bytes()))
    // }

    /// Return the pointer representation as a raw integer.
    pub fn as_ptr(&self) -> u16 {
        self.0.get()
    }

    // pub fn as_f32(&self) -> F32 {
    //     let bytes = self.0.to_bytes();
    //     F32::from_bytes(bytes)
    // }
}

impl fmt::Debug for NodePointer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NodePointer: {{ bytes: {:?}, (as_u32: {}, as_f32: {}) }}",
            self.0.as_bytes(),
            self.as_ptr(),
            "N/I",
            // self.as_f32()
        )
    }
}

impl fmt::Display for NodePointer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NodePointer: {:?} (u32: {}, f32: {})",
            self.0.as_bytes(),
            self.as_ptr(),
            "N/I",
            // self.as_f32()
        )
    }
}
