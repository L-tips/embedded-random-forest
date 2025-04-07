use core::{
    fmt::{self, Debug},
    marker::PhantomData,
    num::NonZeroU8,
};

use heapless::LinearMap;
use zerocopy::{
    FromBytes, Immutable, IntoBytes, KnownLayout, TryFromBytes,
    byteorder::little_endian::{F32, U32},
};

use crate::{Error, ptr::NodePointer};

pub mod deserialize;

#[cfg(feature = "std")]
pub mod serialize;

pub trait ProblemType {
    type Output: Copy;
    const HAS_TARGETS: bool;
}

pub trait Predict {
    type ProblemType: ProblemType;

    /// Make a prediction based on input values (features)
    fn predict(&self, features: &[f32]) -> <Self::ProblemType as ProblemType>::Output;
}

pub struct Classification {
    num_targets: NonZeroU8,
}

impl Classification {
    pub fn new(num_targets: u8) -> Result<Self, Error> {
        let num_targets = NonZeroU8::new(num_targets).ok_or(Error::MalformedForest)?;
        Ok(Self { num_targets })
    }
}

impl ProblemType for Classification {
    type Output = u32;
    const HAS_TARGETS: bool = true;
}

pub struct Regression;

impl ProblemType for Regression {
    type Output = f32;
    const HAS_TARGETS: bool = false;
}

#[repr(transparent)]
#[derive(IntoBytes, Clone, KnownLayout, Immutable, FromBytes)]
pub struct Flags(U32);

impl Flags {
    fn new(split_var_idx: u32, left_is_prediction: bool, right_is_prediction: bool) -> Self {
        assert!(split_var_idx <= u32::MAX >> 2);

        let val = split_var_idx
            | ((left_is_prediction as u32) << (32 - 1))
            | ((right_is_prediction as u32) << (32 - 2));
        Self(U32::new(val))
    }

    fn left_prediction(&self) -> bool {
        (self.0 >> (32 - 1)) & 1 != 0
    }

    fn right_prediction(&self) -> bool {
        (self.0 >> (32 - 2)) & 1 != 0
    }

    fn split_var_idx(&self) -> u32 {
        (self.0 & (u32::MAX >> 2)).get()
    }
}

impl Debug for Flags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Flags {{ left is leaf: {}, right is leaf: {}, split var: {} }}",
            self.left_prediction(),
            self.right_prediction(),
            self.split_var_idx()
        )
    }
}

#[derive(Debug, Clone, IntoBytes, KnownLayout, Immutable, FromBytes)]
#[repr(C, align(4))]
pub struct Branch {
    left: NodePointer,
    right: NodePointer,
    split_at: F32,
    flags: Flags,
}

impl Branch {
    #[inline]
    pub fn new(
        split_with: u32,
        split_at: f32,
        left: NodePointer,
        right: NodePointer,
        left_leaf: bool,
        right_leaf: bool,
    ) -> Self {
        let flags = Flags::new(split_with, left_leaf, right_leaf);
        Self {
            flags,
            split_at: F32::new(split_at),
            left,
            right,
        }
    }

    #[inline]
    pub fn split_with(&self) -> u32 {
        self.flags.split_var_idx()
    }

    #[inline]
    pub fn split_at(&self) -> f32 {
        self.split_at.get()
    }

    #[inline]
    pub fn left_ptr(&self) -> NodePointer {
        self.left
    }

    #[inline]
    pub fn right_ptr(&self) -> NodePointer {
        self.right
    }
}

impl fmt::Display for Branch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Branch | split var: {}, split: {}, left: {}, right: {}",
            self.flags.split_var_idx(),
            self.split_at,
            self.left,
            self.right
        )
    }
}

/// An array-backed, optimized random forest model
#[repr(C, align(4))]
#[derive(TryFromBytes, KnownLayout, Immutable)]
pub struct OptimizedForest<'data, P: ProblemType> {
    num_trees: U32,
    num_features: u8,
    /// If num_targets is Some, we have a classification problem.
    /// Otherwise, we have a regression problem.
    num_targets: Option<NonZeroU8>,
    _padding: [u8; 2],
    nodes: &'data [Branch],
    _problem: PhantomData<P>,
}

impl<P: ProblemType> OptimizedForest<'_, P> {
    pub fn nodes(&self) -> &[Branch] {
        self.nodes
    }

    pub fn num_features(&self) -> u8 {
        self.num_features
    }

    fn next_left(&self, branch: &Branch) -> &Branch {
        unsafe { self.nodes.get_unchecked(branch.left_ptr().as_ptr() as usize) }
    }

    fn next_right(&self, branch: &Branch) -> &Branch {
        unsafe { self.nodes.get_unchecked(branch.right_ptr().as_ptr() as usize) }
    }
}

impl<'data> OptimizedForest<'data, Classification> {
    pub fn new(
        num_trees: u32,
        nodes: &'data [Branch],
        num_features: u8,
        problem: Classification,
    ) -> Result<Self, Error> {
        Ok(Self {
            num_trees: U32::new(num_trees),
            nodes,
            num_features,
            num_targets: Some(problem.num_targets),
            _padding: [0; 2],
            _problem: PhantomData,
        })
    }

    pub fn num_targets(&self) -> Option<NonZeroU8> {
        self.num_targets
    }
}

impl Predict for OptimizedForest<'_, Classification> {
    type ProblemType = Classification;

    #[must_use]
    #[inline(never)]
    fn predict(&self, features: &[f32]) -> <Self::ProblemType as ProblemType>::Output {
        let mut votes = LinearMap::<_, _, 255>::new();
        unsafe {
            for tree_id in 0..self.num_trees.get() {
                let mut node = self.nodes.get_unchecked(tree_id as usize);

                let prediction = loop {
                    let test = *features.get_unchecked(node.split_with() as usize) <= node.split_at();

                    if test {
                        if node.flags.left_prediction() {
                            break node.left_ptr().as_ptr();
                        } else {
                            node = self.next_left(node);
                        }
                    } else if node.flags.right_prediction() {
                        break node.right_ptr().as_ptr();
                    } else {
                        node = self.next_right(node);
                    }
                };

                // Register the vote for this tree's prediction
                let vote = votes.get_mut(&prediction);
                if let Some(v) = vote {
                    *v += 1;
                } else {
                    votes.insert(prediction, 0).unwrap();
                }
            }
        }

        votes
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(num, _)| num)
            .copied()
            .unwrap()
    }
}

impl<'data> OptimizedForest<'data, Regression> {
    pub fn new(num_trees: u32, nodes: &'data [Branch], num_features: u8) -> Result<Self, Error> {
        Ok(Self {
            num_trees: U32::new(num_trees),
            nodes,
            num_features,
            num_targets: None,
            _padding: [0; 2],
            _problem: PhantomData,
        })
    }
}

impl Predict for OptimizedForest<'_, Regression> {
    type ProblemType = Regression;

    #[must_use]
    #[inline(never)]
    fn predict(&self, features: &[f32]) -> f32 {
        let mut result = 0.0;

        for tree_id in 0..self.num_trees.get() {
            let mut node = &self.nodes[tree_id as usize];

            let prediction = loop {
                let test = features[node.split_with() as usize] <= node.split_at();

                if test {
                    if node.flags.left_prediction() {
                        break node.left_ptr().as_f32();
                    } else {
                        node = self.next_left(node);
                    }
                } else if node.flags.right_prediction() {
                    break node.right_ptr().as_f32();
                } else {
                    node = self.next_right(node);
                }
            };

            // Register the vote for this tree's prediction
            result += prediction;
        }

        result / self.num_trees.get() as f32
    }
}

impl<P: ProblemType> fmt::Display for OptimizedForest<'_, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tgts) = self.num_targets {
            writeln!(
                f,
                "OPTIMIZED CLASSIFICATION Forest: {} trees, size {}, {} features, {} targets\n------------",
                self.num_trees,
                self.nodes.len(),
                self.num_features,
                tgts
            )?;
        } else {
            writeln!(
                f,
                "OPTIMIZED REGRESSION Forest: {} trees, size {}, {} features\n------------",
                self.num_trees,
                self.nodes.len(),
                self.num_features,
            )?;
        }

        for (i, node) in self.nodes.iter().enumerate() {
            writeln!(f, "\t{i}: {node}")?;
        }
        writeln!(f, "------------")?;
        Ok(())
    }
}
